// shaders/fused.metal
//
// Phase 20 — in-proj mega-kernel.
//
// Fuses the prefill chain
//     in_proj(3 matmul slices) -> conv1d (xBC slice) -> SiLU+split
// into a single Metal kernel that issues ONE `dispatchThreadgroups` and
// writes the 5 downstream slots (proj_z, ssd_x, ssd_B, ssd_C, proj_dt)
// directly. Cuts the prefill chain by 4 launches per layer (× 64 layers
// of Codestral 7B = 256 fewer GPU command transitions).
//
// Notes:
//   * RMSNorm is NOT fused inside this kernel — `normed` is the existing
//     RMSNorm output slot, computed by the unchanged `rmsnorm_f16` launch
//     just before this one. Pulling RMSNorm in would force every col-chunk
//     threadgroup to recompute the per-row sum-of-squares across `d_model`,
//     bloating global-memory traffic on `hidden`.
//   * The MM uses `simdgroup_matrix<half, 8, 8>` tiles — same AMX path as
//     `linear_f16_gemm` from Phase 18.
//   * Requires L <= FUSED_TILE_M (16) so the conv1d step can see the full
//     time dim of the matmul output for the col chunk in threadgroup memory.
//     For longer prefills the caller must split into <=16-token sub-chunks.

#include <metal_stdlib>
using namespace metal;

struct FusedInProjParams {
    uint L;
    uint d_model;
    uint d_inner;
    uint d_state;
    uint n_groups;
    uint n_heads;
    uint d_conv;
    uint proj_dim;        // d_inner + xBC_dim + n_heads
};

constant uint FUSED_TILE_M  = 16;
constant uint FUSED_TILE_N  = 32;
constant uint FUSED_TILE_K  = 32;
constant uint FUSED_TG_SIZE = 256;  // 8 simdgroups

kernel void inproj_fused_f16(
    device const half* normed     [[ buffer(0) ]],   // (L, d_model)
    device const half* W_in_proj  [[ buffer(1) ]],   // (proj_dim, d_model)
    device const half* W_conv1d   [[ buffer(2) ]],   // (xBC_dim, d_conv)
    device const half* conv1d_b   [[ buffer(3) ]],   // (xBC_dim)
    device       half* proj_z     [[ buffer(4) ]],   // (L, d_inner)
    device       half* ssd_x      [[ buffer(5) ]],   // (L, d_inner)
    device       half* ssd_B      [[ buffer(6) ]],   // (L, n_groups*d_state)
    device       half* ssd_C      [[ buffer(7) ]],   // (L, n_groups*d_state)
    device       half* proj_dt    [[ buffer(8) ]],   // (L, n_heads)
    device       half* proj_xBC   [[ buffer(10) ]],  // (L, xBC_dim) pre-conv copy
    constant FusedInProjParams& p [[ buffer(9) ]],
    uint  gid                     [[ threadgroup_position_in_grid ]],
    uint  tid                     [[ thread_position_in_threadgroup ]],
    uint  simd_index              [[ simdgroup_index_in_threadgroup ]]
) {
    const uint col_start = gid * FUSED_TILE_N;
    if (col_start >= p.proj_dim) return;
    const uint col_end = min(col_start + FUSED_TILE_N, p.proj_dim);
    const uint chunk_w = col_end - col_start;

    const uint L        = p.L;
    const uint K        = p.d_model;
    const uint d_inner  = p.d_inner;
    const uint G_N      = p.n_groups * p.d_state;
    const uint xBC_dim  = d_inner + 2u * G_N;
    const uint d_conv   = p.d_conv;

    // Threadgroup-memory layout: A and B staging for the MM, plus an fp32
    // matmul output that doubles as conv1d input. 16*32*4 + 16*32*2 + 32*32*2
    // = 2KB + 1KB + 2KB = 5 KB. Plenty of slack.
    threadgroup half  As[FUSED_TILE_M * FUSED_TILE_K];
    threadgroup half  Bs[FUSED_TILE_N * FUSED_TILE_K];
    threadgroup float Cmm[FUSED_TILE_M * FUSED_TILE_N];

    // 8 simdgroups × one 8x8 acc tile each → covers 16×32 (= 2 row × 4 col).
    // sg layout: sg s → (ri = s/4, ci = s%4).
    const uint ri = simd_index / 4u;
    const uint ci = simd_index % 4u;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < K; k += FUSED_TILE_K) {
        // Load X tile (16 × 32 halves = 512 halves). 256 threads → 2/thread.
        for (uint t = tid; t < FUSED_TILE_M * FUSED_TILE_K; t += FUSED_TG_SIZE) {
            uint m  = t / FUSED_TILE_K;
            uint kk = t % FUSED_TILE_K;
            uint kg = k + kk;
            As[t] = (m < L && kg < K) ? normed[m * K + kg] : half(0);
        }
        // Load W tile (32 × 32 halves = 1024 halves). 4/thread.
        for (uint t = tid; t < FUSED_TILE_N * FUSED_TILE_K; t += FUSED_TG_SIZE) {
            uint c  = t / FUSED_TILE_K;
            uint kk = t % FUSED_TILE_K;
            uint kg = k + kk;
            uint w_row = col_start + c;
            Bs[t] = (c < chunk_w && kg < K) ? W_in_proj[w_row * K + kg] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per simdgroup: 4 K-sub-tile multiply-accumulates over (ri, ci).
        // Bs is stored row-major as (c, kk); we transpose-load on the way
        // into the simdgroup_matrix so the AMX op sees (kk, c) = standard
        // K×N right-hand operand. simdgroup_multiply_accumulate then
        // computes acc += A @ B^T directly.
        for (uint ki8 = 0; ki8 < FUSED_TILE_K / 8; ++ki8) {
            simdgroup_half8x8 a_tile, b_tile;
            simdgroup_load(a_tile, &As[(ri * 8) * FUSED_TILE_K + ki8 * 8],
                           FUSED_TILE_K);
            simdgroup_load(b_tile, &Bs[(ci * 8) * FUSED_TILE_K + ki8 * 8],
                           FUSED_TILE_K, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each sg stores its 8x8 acc tile into Cmm.
    simdgroup_store(acc, &Cmm[(ri * 8) * FUSED_TILE_N + ci * 8], FUSED_TILE_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final per-element pass: conv1d + silu for xBC; raw write for z/dt.
    for (uint t = tid; t < L * chunk_w; t += FUSED_TG_SIZE) {
        uint m  = t / chunk_w;
        uint c  = t % chunk_w;
        uint gc = col_start + c;     // global col in proj space

        if (gc < d_inner) {
            // z slice — raw matmul output.
            float v = Cmm[m * FUSED_TILE_N + c];
            proj_z[m * d_inner + gc] = half(v);
        } else if (gc < d_inner + xBC_dim) {
            // xBC slice — also dump pre-conv matmul output to proj_xBC so the
            // subsequent seed_conv_state_f16 step (for prefill -> decode
            // handoff) can read the last d_conv rows.
            uint xbc_c = gc - d_inner;
            float pre_conv = Cmm[m * FUSED_TILE_N + c];
            proj_xBC[m * xBC_dim + xbc_c] = half(pre_conv);

            // conv1d (depthwise, time-domain) + SiLU
            float sum = float(conv1d_b[xbc_c]);
            for (uint kk = 0; kk < d_conv; ++kk) {
                int src_m = int(m) - int(d_conv - 1 - kk);
                if (src_m >= 0) {
                    sum += float(W_conv1d[xbc_c * d_conv + kk]) *
                           Cmm[uint(src_m) * FUSED_TILE_N + c];
                }
            }
            float silu_v = sum / (1.0f + exp(-sum));

            if (xbc_c < d_inner) {
                ssd_x[m * d_inner + xbc_c] = half(silu_v);
            } else if (xbc_c < d_inner + G_N) {
                ssd_B[m * G_N + (xbc_c - d_inner)] = half(silu_v);
            } else {
                ssd_C[m * G_N + (xbc_c - d_inner - G_N)] = half(silu_v);
            }
        } else {
            // dt slice — raw matmul output.
            float v = Cmm[m * FUSED_TILE_N + c];
            uint dt_c = gc - d_inner - xBC_dim;
            proj_dt[m * p.n_heads + dt_c] = half(v);
        }
    }
}
