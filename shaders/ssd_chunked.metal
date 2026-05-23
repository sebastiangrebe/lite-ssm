// shaders/ssd_chunked.metal
// Mamba-2 State Space Duality kernel — PREFILL path.
//
// Reference implementation: one threadgroup per (batch, head), processes
// all chunks of the sequence sequentially, carrying the (P, N) head state
// in threadgroup memory.
//
// Per-chunk we do three things:
//   1. Inter-chunk: Y_inter[t, p] += C[t] · (exp(L_t) ⊙ S)
//   2. Intra-chunk: Y_intra[t, p] += Σ_{j<=t} (C[t]·B[j]) · exp(L_t - L_j) · dt[j] · X[j, p]
//   3. State update: S ← exp(L_{C-1}) ⊙ S + Σ_j exp(L_{C-1} - L_j) · dt[j] · B[j] ⊗ X[j]
//
// Assumptions (match HF AntonV/mamba2-*-hf checkpoints):
//   * n_groups = 1 — B and C are shared across heads (per-timestep length N).
//   * batch B = 1 in the inference path (lifting this is trivial: extend grid.z).
//
// Optimization notes (deferred to a later phase):
//   * The (C[t]·B[j]) factor is a C × C matrix M[t,j]. Computing it as a
//     simdgroup_matrix gemm (C,N)·(N,C)^T over 8×8 tiles in registers would
//     replace the inner-N dot product with the M-series matrix unit.
//   * Cross-threadgroup parallel scan: this kernel keeps a single threadgroup
//     per head. Splitting the sequence into stripes and reducing chunk-final
//     states via a Blelloch scan (with simdgroup_broadcast carrying partials
//     between stripes) parallelizes across the L axis. For decode and short
//     prefills (L < 1024) the single-threadgroup form already saturates ALUs.

#include <metal_stdlib>
using namespace metal;

struct SSDParams {
    uint  L;            // sequence length
    uint  H;            // heads
    uint  P;            // d_head
    uint  N;            // d_state
    uint  n_groups;     // Phase 11: B/C broadcast width
    uint  chunk_size;
    float dt_min;       // dt clamp (mamba-2 default: 0.001)
    float dt_max;       // dt clamp (mamba-2 default: 100.0)
    uint  has_D;        // 0/1
};

// Caps so we can size threadgroup arrays at compile time.
constant uint SSD_MAX_P  = 64;     // d_head <= 64 covers all canonical Mamba-2 sizes
constant uint SSD_MAX_N  = 128;    // d_state <= 128 ditto
constant uint SSD_MAX_CK = 256;    // chunk_size <= 256

inline float softplus_fast(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void ssd_chunked_f16(
    device const half*  X        [[ buffer(0) ]],   // (L, H, P)
    device const half*  Bproj    [[ buffer(1) ]],   // (L, N)   shared across heads
    device const half*  Cproj    [[ buffer(2) ]],   // (L, N)
    device const half*  dt_raw   [[ buffer(3) ]],   // (L, H)   pre-softplus
    device const half*  dt_bias  [[ buffer(4) ]],   // (H)
    device const half*  A_log    [[ buffer(5) ]],   // (H)      A = -exp(A_log)
    device const half*  D_skip   [[ buffer(6) ]],   // (H)      optional, gated by has_D
    device       half*  Y        [[ buffer(7) ]],   // (L, H, P)
    device       float* state_io [[ buffer(8) ]],   // (H, P, N) carried; written back as final state
    constant SSDParams& p        [[ buffer(9) ]],
    uint  h                      [[ threadgroup_position_in_grid ]],
    uint  tid                    [[ thread_position_in_threadgroup ]],
    uint  tg_size                [[ threads_per_threadgroup ]],
    uint  simd_lane              [[ thread_index_in_simdgroup ]],
    uint  simd_index             [[ simdgroup_index_in_threadgroup ]]
) {
    if (h >= p.H) return;

    const uint P = p.P;
    const uint N = p.N;
    const uint C = p.chunk_size;
    const uint n_chunks = (p.L + C - 1) / C;

    // Phase 11 B/C broadcasting. Compute the group index for this
    // threadgroup once — every B/C load below uses it.
    //   heads_per_group = H / n_groups
    //   group_id        = h / heads_per_group
    //   GN              = n_groups * d_state  (row stride of B and C)
    const uint heads_per_group = max(1u, p.H / p.n_groups);
    const uint g               = h / heads_per_group;
    const uint GN              = p.n_groups * N;

    // Per-head scalars
    const float A_h = -exp(float(A_log[h]));
    const float dt_b = float(dt_bias[h]);
    const float D_h = (p.has_D ? float(D_skip[h]) : 0.0f);

    // State S[p, n] lives in threadgroup memory. Stored as fp16 to fit under
    // the 32 KB threadgroup-memory cap (fp32 would be 32 KB by itself for
    // P=64, N=128). Math still runs in fp32 — every update upcasts on read
    // and downcasts on write. One-round fp16 quantization per chunk is well
    // within Mamba-2's tolerance bounds; if numerical drift ever bites we
    // can move S back to device memory.
    threadgroup half S[SSD_MAX_P * SSD_MAX_N];

    // Load initial state from device buffer (zero on first call from caller).
    device const float* state_in = state_io + h * P * N;
    for (uint i = tid; i < P * N; i += tg_size) {
        S[i] = half(state_in[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-chunk scratch.
    threadgroup float Lcum[SSD_MAX_CK];    // cumulative log-decay over chunk
    threadgroup float DT[SSD_MAX_CK];      // softplused dt per timestep

    for (uint c = 0; c < n_chunks; ++c) {
        const uint t0   = c * C;
        const uint tlen = min(C, p.L - t0);

        // ------------------------------------------------------------------
        // Step A: load dt and compute cumulative log-decays for this chunk.
        // Lcum[t] = Σ_{s=0..t} A_h · softplus(dt_raw[t0+s] + dt_b)
        // Only one thread does the prefix sum (sequential dep over tlen).
        // We then broadcast the final value via simdgroup_broadcast so all
        // lanes can use it without reading threadgroup memory again.
        // ------------------------------------------------------------------
        for (uint t = tid; t < tlen; t += tg_size) {
            float d = float(dt_raw[(t0 + t) * p.H + h]) + dt_b;
            float dt = softplus_fast(d);
            dt = clamp(dt, p.dt_min, p.dt_max);
            DT[t] = dt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float acc = 0.0f;
            for (uint t = 0; t < tlen; ++t) {
                acc += A_h * DT[t];
                Lcum[t] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Broadcast Lcum[tlen-1] across simdgroup-0 lane 0 to all lanes. Even
        // though Lcum is in threadgroup memory, simdgroup_broadcast keeps the
        // hot value in registers for the state-update pass below.
        float L_end = Lcum[tlen - 1];
        L_end = simd_broadcast(L_end, 0);
        const float decay_chunk = exp(L_end);

        // ------------------------------------------------------------------
        // Step B: per-timestep output. Each thread owns a slab of (t, p)
        // output coordinates spread across the chunk.
        //
        //   y_inter[t, p] = exp(Lcum[t]) · Σ_n C[t,n] · S[p,n]
        //   y_intra[t, p] = Σ_{j<=t} (C[t]·B[j]) · exp(Lcum[t]-Lcum[j]) · dt[j] · X[j,p]
        //
        // Reference implementation: two nested loops per output element. The
        // intra term is the dominant work (O(C²·N) per head). simdgroup_matrix
        // tiling of (C·B^T) is the eventual optimization.
        // ------------------------------------------------------------------
        const uint out_elems = tlen * P;
        for (uint idx = tid; idx < out_elems; idx += tg_size) {
            const uint t   = idx / P;
            const uint pp  = idx % P;

            // Inter-chunk piece.
            float y_inter = 0.0f;
            const float exp_Lt = exp(Lcum[t]);
            for (uint n = 0; n < N; ++n) {
                float c_tn = float(Cproj[(t0 + t) * GN + g * N + n]);
                y_inter += c_tn * float(S[pp * N + n]);
            }
            y_inter *= exp_Lt;

            // Intra-chunk piece.
            float y_intra = 0.0f;
            for (uint j = 0; j <= t; ++j) {
                // (C[t] · B[j])  — both indexed via the current head's group g
                float cb = 0.0f;
                for (uint n = 0; n < N; ++n) {
                    cb += float(Cproj[(t0 + t) * GN + g * N + n]) *
                          float(Bproj[(t0 + j) * GN + g * N + n]);
                }
                float decay = exp(Lcum[t] - Lcum[j]);
                float xj    = float(X[(t0 + j) * (p.H * P) + h * P + pp]);
                y_intra += cb * decay * DT[j] * xj;
            }

            float x_cur = float(X[(t0 + t) * (p.H * P) + h * P + pp]);
            float y     = y_intra + y_inter + D_h * x_cur;
            Y[(t0 + t) * (p.H * P) + h * P + pp] = half(y);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------------------------------------------------------------------
        // Step C: update state for the next chunk.
        //   S_new[p, n] = decay_chunk · S[p, n]
        //              + Σ_j exp(Lcum[end] - Lcum[j]) · dt[j] · B[j,n] · X[j,p]
        // ------------------------------------------------------------------
        const uint sn_elems = P * N;
        for (uint idx = tid; idx < sn_elems; idx += tg_size) {
            const uint pp = idx / N;
            const uint nn = idx % N;

            float acc = decay_chunk * float(S[idx]);
            for (uint j = 0; j < tlen; ++j) {
                float w  = exp(Lcum[tlen - 1] - Lcum[j]) * DT[j];
                float bj = float(Bproj[(t0 + j) * GN + g * N + nn]);
                float xj = float(X[(t0 + j) * (p.H * P) + h * P + pp]);
                acc += w * bj * xj;
            }
            S[idx] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist final state for the caller (decode resumes from here). Promote
    // back to fp32 since state_io is declared as float; the SSD step kernel
    // also reads it as float.
    device float* state_out = state_io + h * P * N;
    for (uint i = tid; i < P * N; i += tg_size) {
        state_out[i] = float(S[i]);
    }
}
