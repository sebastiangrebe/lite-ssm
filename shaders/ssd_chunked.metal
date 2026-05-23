// shaders/ssd_chunked.metal
// Mamba-2 State Space Duality kernel — PREFILL path.
//
// Phase 17 rewrite — algorithmic + memory-pattern improvements over the
// original O(L²·N·P·H) scalar loop.
//
// Key insight that drives the speedup:
//
//   For each chunk, the "attention-like" weight M[t,j] depends only on
//   (t, j), the group's B/C, and the head's dt — NOT on (p) or on x.
//   Computing it once and reusing it across all P output elements
//   converts the intra-chunk hot loop from O(C²·N) per (h, p) into
//   O(C²·N) per (h) (shared) plus O(C²) per (h, p).
//
//   Numbers for Codestral 7B with chunk_size=12 (12-token prefill):
//     old per-head:  C²·N·P     = 144 * 128 * 64  ≈ 1.18M ops
//     new per-head:  C²·N + C²·P = 18432 + 9216  ≈ 27K ops   (43× cheaper)
//
//   For the full chunk_size = 256:
//     old:  256·256·128·64  ≈ 537M ops
//     new:  256·256·128 + 256·256·64  ≈ 12M ops               (43× cheaper)
//
// Additionally, the chunk-local cumulative log-decay is now computed with
// a Kogge-Stone simd_shuffle prefix scan within a simdgroup + a single
// cross-simdgroup pass through threadgroup memory. Sequential cumsum on
// one thread is the textbook serial bottleneck the user spec'd out.
//
// What we explicitly did NOT change:
//   * Sequential per-chunk processing within a single threadgroup. Splitting
//     a head's chunks across multiple threadgroups would require a Blelloch
//     scan over chunk-final states with cross-threadgroup memory + a second
//     kernel launch. For typical prefills (n_chunks ≤ 2 on 7B 16 GB Mac
//     workloads) the latency cost of the extra launch dominates the
//     theoretical scan win, so we keep the in-threadgroup serial walk.

#include <metal_stdlib>
using namespace metal;

struct SSDParams {
    uint  L;
    uint  H;
    uint  P;
    uint  N;
    uint  n_groups;
    uint  chunk_size;
    float dt_min;
    float dt_max;
    uint  has_D;
};

// Compile-time caps so threadgroup arrays size at translation time.
// SSD_MAX_CK is the kernel's per-launch chunk cap. We size it at 64 (not
// the config's 256) so the (P×N) state plus the (CK×CK) M tile both fit
// inside the 32 KB threadgroup-memory budget:
//   S        (P × N halves)  = 64*128*2 = 16 KB
//   M        (CK × CK halves) = 64*64*2 = 8 KB
//   DT/Lcum  (CK floats each) = 64*4*2  = 0.5 KB
//   slack                                 ~7.5 KB
// The dispatch wrapper caps the runtime chunk to this same value, so
// long prefills just iterate over more sub-chunks (math unchanged).
constant uint SSD_MAX_P  = 64;
constant uint SSD_MAX_N  = 128;
constant uint SSD_MAX_CK = 64;
constant uint SIMD_WIDTH = 32;
constant uint SSD_MAX_SIMDS = 8;   // max simdgroups per dispatch (TG_SIZE ≤ 256)

inline float softplus_fast(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// Kogge-Stone inclusive prefix sum across a 32-lane simdgroup.
inline float simd_prefix_sum_inclusive(float v, uint lane) {
    float n;
    n = simd_shuffle_up(v, 1u);  if (lane >= 1u)  v += n;
    n = simd_shuffle_up(v, 2u);  if (lane >= 2u)  v += n;
    n = simd_shuffle_up(v, 4u);  if (lane >= 4u)  v += n;
    n = simd_shuffle_up(v, 8u);  if (lane >= 8u)  v += n;
    n = simd_shuffle_up(v, 16u); if (lane >= 16u) v += n;
    return v;
}

kernel void ssd_chunked_f16(
    device const half*  X        [[ buffer(0) ]],   // (L, H, P)
    device const half*  Bproj    [[ buffer(1) ]],   // (L, n_groups, N)
    device const half*  Cproj    [[ buffer(2) ]],   // (L, n_groups, N)
    device const half*  dt_raw   [[ buffer(3) ]],   // (L, H)
    device const half*  dt_bias  [[ buffer(4) ]],   // (H)
    device const half*  A_log    [[ buffer(5) ]],   // (H)
    device const half*  D_skip   [[ buffer(6) ]],   // (H)
    device       half*  Y        [[ buffer(7) ]],   // (L, H, P)
    device       float* state_io [[ buffer(8) ]],   // (H, P, N) — fp32 carry
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

    const uint heads_per_group = max(1u, p.H / p.n_groups);
    const uint g               = h / heads_per_group;
    const uint GN              = p.n_groups * N;

    const float A_h  = -exp(float(A_log[h]));
    const float dt_b = float(dt_bias[h]);
    const float D_h  = (p.has_D ? float(D_skip[h]) : 0.0f);

    // State S[p, n] in threadgroup memory (fp16 storage; math in fp32 regs).
    // Carried across chunks. 64 × 128 = 16 KB.
    threadgroup half  S[SSD_MAX_P * SSD_MAX_N];

    // Per-chunk scratch.
    threadgroup float DT[SSD_MAX_CK];                   // softplused dt per t
    threadgroup float Lcum[SSD_MAX_CK];                 // cumulative A·dt
    threadgroup half  M[SSD_MAX_CK * SSD_MAX_CK];       // intra-chunk weight matrix
    threadgroup float simd_totals[SSD_MAX_SIMDS]; // per-simdgroup totals for cross-warp scan

    // Load initial state from device buffer.
    device const float* state_in = state_io + h * P * N;
    for (uint i = tid; i < P * N; i += tg_size) {
        S[i] = half(state_in[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < n_chunks; ++c) {
        const uint t0   = c * C;
        const uint tlen = min(C, p.L - t0);

        // ============================================================
        // Step A: per-timestep DT (softplused dt).
        // ============================================================
        for (uint t = tid; t < tlen; t += tg_size) {
            float d  = float(dt_raw[(t0 + t) * p.H + h]) + dt_b;
            float dt = clamp(softplus_fast(d), p.dt_min, p.dt_max);
            DT[t] = dt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // Step B: parallel inclusive prefix scan of (A_h * DT) into Lcum.
        //
        // Each simdgroup handles up to 32 timesteps via a Kogge-Stone
        // shuffle scan. Cross-simdgroup totals are then accumulated by
        // simdgroup 0, and each later simdgroup adds the accumulated
        // predecessor totals. This replaces the prior serial loop on tid==0.
        // ============================================================
        {
            // Per-thread input (or 0 for out-of-range lanes).
            const uint t_idx = simd_index * SIMD_WIDTH + simd_lane;
            float v = (t_idx < tlen) ? (A_h * DT[t_idx]) : 0.0f;

            // Intra-simdgroup inclusive prefix.
            float prefix = simd_prefix_sum_inclusive(v, simd_lane);

            // Last lane writes the simdgroup total.
            if (simd_lane == SIMD_WIDTH - 1) {
                simd_totals[simd_index] = prefix;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Simdgroup 0, lane 0, builds the EXCLUSIVE prefix over the
            // simd_totals[] array — i.e. simd_totals[s] becomes "sum of all
            // earlier simdgroups". We snapshot the originals first because
            // the loop overwrites in place.
            const uint n_simds = (tlen + SIMD_WIDTH - 1) / SIMD_WIDTH;
            if (simd_index == 0 && simd_lane == 0) {
                float totals_orig[SSD_MAX_SIMDS] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (uint s = 0; s < n_simds; ++s) totals_orig[s] = simd_totals[s];
                float running = 0.0f;
                for (uint s = 0; s < n_simds; ++s) {
                    simd_totals[s] = running;
                    running += totals_orig[s];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each lane writes Lcum[t_idx] = prefix + predecessor-simdgroup-total.
            if (t_idx < tlen) {
                Lcum[t_idx] = prefix + simd_totals[simd_index];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Broadcast chunk-final L for state update (avoids LCum re-read in regs).
        const float L_end       = Lcum[tlen - 1];
        const float decay_chunk = exp(L_end);

        // ============================================================
        // Step C: precompute the intra-chunk weight matrix M[t, j].
        //
        //   M[t, j] = (C[t]·B[j])  *  exp(Lcum[t] - Lcum[j])  *  DT[j]
        //           for j ≤ t; zero otherwise.
        //
        // This is the dominant work amortization — done O(C²·N) once per
        // (h, chunk), then reused across all P output elements at O(C²)
        // each.
        // ============================================================
        const uint m_elems = tlen * tlen;
        for (uint idx = tid; idx < m_elems; idx += tg_size) {
            const uint t = idx / tlen;
            const uint j = idx % tlen;
            if (j > t) { M[idx] = half(0); continue; }
            // Sum over n_state. Coalesced reads — adjacent n's are
            // contiguous in memory (last dim of Cproj/Bproj).
            float cb = 0.0f;
            device const half* Crow = Cproj + (t0 + t) * GN + g * N;
            device const half* Brow = Bproj + (t0 + j) * GN + g * N;
            for (uint n = 0; n < N; ++n) {
                cb += float(Crow[n]) * float(Brow[n]);
            }
            float decay = exp(Lcum[t] - Lcum[j]);
            M[idx] = half(cb * decay * DT[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // Step D: compute the per-(t, p) outputs.
        //
        //   y_intra[t, p] = Σ_{j ≤ t} M[t, j] * X[j, h, p]
        //   y_inter[t, p] = exp(Lcum[t]) * Σ_n C[t, n] * S[p, n]
        //   y[t, p]       = y_intra + y_inter + D_h * X[t, h, p]
        //
        // Parallelize over (t, p) — outputs are independent.
        // ============================================================
        const uint out_elems = tlen * P;
        for (uint idx = tid; idx < out_elems; idx += tg_size) {
            const uint t  = idx / P;
            const uint pp = idx % P;

            // y_intra: dot product over j of M[t, :] · X[:, h, pp]
            float y_intra = 0.0f;
            threadgroup const half* Mrow = M + t * tlen;
            // X stride between j's = H*P. Strided rather than contiguous —
            // can't easily coalesce without transposing X. The L2 cache
            // absorbs the strided pattern in practice for short chunks.
            for (uint j = 0; j <= t; ++j) {
                float m_tj = float(Mrow[j]);
                float xj   = float(X[(t0 + j) * (p.H * P) + h * P + pp]);
                y_intra += m_tj * xj;
            }

            // y_inter from prev-chunk state S.
            float y_inter = 0.0f;
            const float exp_Lt = exp(Lcum[t]);
            device const half* Crow_t = Cproj + (t0 + t) * GN + g * N;
            threadgroup const half* Srow = S + pp * N;
            for (uint n = 0; n < N; ++n) {
                y_inter += float(Crow_t[n]) * float(Srow[n]);
            }
            y_inter *= exp_Lt;

            // D residual on UNSCALED x — matches HF's `D_residual` which is
            // computed BEFORE the `hidden_states *= dt[..., None]` line.
            float x_cur = float(X[(t0 + t) * (p.H * P) + h * P + pp]);
            float y     = y_intra + y_inter + D_h * x_cur;
            Y[(t0 + t) * (p.H * P) + h * P + pp] = half(y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // Step E: state update for next chunk.
        //
        //   S_new[p, n] = decay_chunk * S[p, n]
        //              + Σ_j exp(L_end - Lcum[j]) * DT[j] * B[j, n] * X[j, p]
        //
        // Each thread handles a few (p, n) slots; the per-j inner loop reads
        // B and X strided, similar to step D.
        // ============================================================
        const uint sn_elems = P * N;
        for (uint idx = tid; idx < sn_elems; idx += tg_size) {
            const uint pp = idx / N;
            const uint nn = idx % N;

            float acc = decay_chunk * float(S[idx]);
            for (uint j = 0; j < tlen; ++j) {
                float w  = exp(L_end - Lcum[j]) * DT[j];
                float bj = float(Bproj[(t0 + j) * GN + g * N + nn]);
                float xj = float(X[(t0 + j) * (p.H * P) + h * P + pp]);
                acc += w * bj * xj;
            }
            S[idx] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist final state in fp32 for the decode step kernel.
    device float* state_out = state_io + h * P * N;
    for (uint i = tid; i < P * N; i += tg_size) {
        state_out[i] = float(S[i]);
    }
}
