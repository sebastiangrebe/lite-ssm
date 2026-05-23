// shaders/linear.metal
// Dense matrix multiplication kernels for lite-ssm. Two flavors:
//
//   GEMM (prefill, M >= 8): Y[M,N] = X[M,K] @ W[N,K]^T
//                            Uses simdgroup_matrix<half, 8, 8> (the M-series
//                            matrix unit). Accumulators are fp32.
//
//   GEMV (decode, M = 1):    y[N] = x[K] @ W[N,K]^T
//                            One threadgroup per group-of-rows of W, with
//                            an intra-simdgroup dot reduction. The matrix
//                            unit is overkill for a single row, so we use
//                            scalar simd_sum here.
//
// Fused variants:
//   * `linear_silu_f16`        — applies SiLU(y) at the epilogue.
//   * `rmsnorm_linear_f16_gemv` — RMSNorm(x) inline in registers, then
//                                 reuses the gemv body. Used for the
//                                 norm -> in_proj path before the SSM.
//
// All weights are treated as W: (N, K) row-major (output × input). This
// matches what PyTorch's Linear stores (`weight` is (out, in)) and what
// our exporter writes.

#include <metal_stdlib>
using namespace metal;

struct LinearParams {
    uint M;
    uint N;
    uint K;
};

// Phase 14: block-quantized int4 (32 fp16 weights per block, fp16 scale).
// Layout on disk for a (N, K) weight matrix, K % 32 == 0:
//   bytes[0 .. N*K/2)               packed nibbles, row-major
//   bytes[N*K/2 .. N*K*9/16)        per-block fp16 scales (N rows × K/32 scales)
//
// dequant(n, k):
//   block  = k / 32
//   intra  = k % 32
//   byte   = packed[n * (K/2) + block * 16 + intra/2]
//   nibble = (intra & 1) ? (byte >> 4) : (byte & 0x0F)
//   q4     = (nibble < 8) ? int(nibble) : int(nibble) - 16
//   value  = float(q4) * float(scales[n * (K/32) + block])
constant uint INT4_BLOCK_SIZE = 32;

inline float dequant_int4(device const uchar* packed,
                          device const half*  scales,
                          uint K,
                          uint row,
                          uint col) {
    const uint blocks_per_row = K / INT4_BLOCK_SIZE;
    const uint block          = col / INT4_BLOCK_SIZE;
    const uint intra          = col & 31u;
    const uint byte_index     = row * (K / 2) + block * (INT4_BLOCK_SIZE / 2) + (intra >> 1);
    const uchar b             = packed[byte_index];
    const uint  nibble        = (intra & 1u) ? (uint(b) >> 4) : (uint(b) & 0x0F);
    const int   q4            = (nibble < 8u) ? int(nibble) : int(nibble) - 16;
    const float scale         = float(scales[row * blocks_per_row + block]);
    return float(q4) * scale;
}

struct RMSLinearParams {
    uint  N;       // output features
    uint  K;       // input features (also rmsnorm dim)
    float eps;     // rmsnorm epsilon
};

// ---------------------------------------------------------------------------
//  GEMV — y[N] = x[K] @ W[N,K]^T   (M = 1, decode path)
// ---------------------------------------------------------------------------
//
// Grid: one threadgroup per N-block of `ROWS_PER_TG` output rows.
// Threads per group: SIMD_WIDTH * ROWS_PER_TG so each row gets one simdgroup.
// Each simdgroup loops over K in chunks of SIMD_WIDTH * 4, loading half4
// from both x and W and accumulating in fp32. Final simd_sum reduces to lane 0
// which writes the output.

constant uint GEMV_ROWS_PER_TG = 8;
constant uint GEMV_SIMD_WIDTH  = 32;

kernel void linear_f16_gemv(
    device const half* x      [[ buffer(0) ]],   // (K)
    device const half* w      [[ buffer(1) ]],   // (N, K) row-major
    device       half* y      [[ buffer(2) ]],   // (N)
    constant LinearParams& p  [[ buffer(3) ]],
    uint tg_id                [[ threadgroup_position_in_grid ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint simd_lane            [[ thread_index_in_simdgroup ]],
    uint simd_index           [[ simdgroup_index_in_threadgroup ]]
) {
    const uint row = tg_id * GEMV_ROWS_PER_TG + simd_index;
    if (row >= p.N) return;

    device const half4* xv = reinterpret_cast<device const half4*>(x);
    device const half4* wv = reinterpret_cast<device const half4*>(w + row * p.K);
    const uint n4 = p.K >> 2;

    float acc = 0.0f;
    for (uint i = simd_lane; i < n4; i += GEMV_SIMD_WIDTH) {
        float4 xv4 = float4(xv[i]);
        float4 wv4 = float4(wv[i]);
        acc += dot(xv4, wv4);
    }
    float total = simd_sum(acc);
    if (simd_lane == 0) {
        y[row] = half(total);
    }
}

// Fused: y = silu(x @ W^T). Decode path for the Mamba mixer gate, which
// applies SiLU right after the projection.
kernel void linear_silu_f16_gemv(
    device const half* x      [[ buffer(0) ]],
    device const half* w      [[ buffer(1) ]],
    device       half* y      [[ buffer(2) ]],
    constant LinearParams& p  [[ buffer(3) ]],
    uint tg_id                [[ threadgroup_position_in_grid ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint simd_lane            [[ thread_index_in_simdgroup ]],
    uint simd_index           [[ simdgroup_index_in_threadgroup ]]
) {
    const uint row = tg_id * GEMV_ROWS_PER_TG + simd_index;
    if (row >= p.N) return;

    device const half4* xv = reinterpret_cast<device const half4*>(x);
    device const half4* wv = reinterpret_cast<device const half4*>(w + row * p.K);
    const uint n4 = p.K >> 2;

    float acc = 0.0f;
    for (uint i = simd_lane; i < n4; i += GEMV_SIMD_WIDTH) {
        acc += dot(float4(xv[i]), float4(wv[i]));
    }
    float total = simd_sum(acc);
    if (simd_lane == 0) {
        y[row] = half(total / (1.0f + exp(-total)));
    }
}

// Fused: RMSNorm(x) then GEMV against W. Saves a round-trip to global memory.
//
// Algorithm:
//   1. Whole threadgroup cooperates to compute rms = sqrt(mean(x^2) + eps).
//   2. Stage the normalized row in threadgroup memory (K halves, ~few KB).
//   3. Each simdgroup then runs the standard gemv against its row of W.
//
// The norm weight is folded into x in the staging step: x_norm[i] = x[i] * w_norm[i] / rms.
kernel void rmsnorm_linear_f16_gemv(
    device const half* x         [[ buffer(0) ]],   // (K)
    device const half* w_norm    [[ buffer(1) ]],   // (K)
    device const half* w         [[ buffer(2) ]],   // (N, K)
    device       half* y         [[ buffer(3) ]],   // (N)
    constant RMSLinearParams& p  [[ buffer(4) ]],
    threadgroup half* x_staged   [[ threadgroup(0) ]],   // length K
    uint tg_id                   [[ threadgroup_position_in_grid ]],
    uint tid                     [[ thread_position_in_threadgroup ]],
    uint tg_size                 [[ threads_per_threadgroup ]],
    uint simd_lane               [[ thread_index_in_simdgroup ]],
    uint simd_index              [[ simdgroup_index_in_threadgroup ]]
) {
    device const half4* xv  = reinterpret_cast<device const half4*>(x);
    device const half4* wnv = reinterpret_cast<device const half4*>(w_norm);
    threadgroup half4*  xsv = reinterpret_cast<threadgroup half4*>(x_staged);

    const uint n4 = p.K >> 2;

    // Pass 1: sum of squares.
    float ss = 0.0f;
    for (uint i = tid; i < n4; i += tg_size) {
        float4 v = float4(xv[i]);
        ss += dot(v, v);
    }
    threadgroup float partial[32];
    float sg = simd_sum(ss);
    if (simd_lane == 0) partial[simd_index] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        uint n_simds = (tg_size + 31u) >> 5;
        float v = (tid < n_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rrms = rsqrt(partial[0] / float(p.K) + p.eps);

    // Pass 2: stage normalized x into threadgroup memory.
    for (uint i = tid; i < n4; i += tg_size) {
        xsv[i] = half4(float4(xv[i]) * float4(wnv[i]) * rrms);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: gemv per simdgroup. Tiles N by simd_count rows per pass; loops
    // over output rows until all N are covered.
    const uint n_simds = (tg_size + 31u) >> 5;
    for (uint row = simd_index; row < p.N; row += n_simds) {
        device const half4* wv = reinterpret_cast<device const half4*>(w + row * p.K);
        float acc = 0.0f;
        for (uint i = simd_lane; i < n4; i += 32) {
            acc += dot(float4(xsv[i]), float4(wv[i]));
        }
        float total = simd_sum(acc);
        if (simd_lane == 0) y[row] = half(total);
    }
}

// ---------------------------------------------------------------------------
//  GEMM — Y[M,N] = X[M,K] @ W[N,K]^T   (prefill, M >= 8)
// ---------------------------------------------------------------------------
//
// Threadgroup tile: BM x BN = 32 x 32 output.
// Threadgroup size: 4 simdgroups (128 threads). Each simdgroup owns one
// 8x32 horizontal strip of C (4 8x8 accumulator tiles in registers).
//
// Per K-step (BK=8):
//   - Cooperative load of A tile (BM x BK = 32 x 8) and B tile (BK x BN = 8 x 32)
//     into threadgroup memory. B comes from W which is stored (N, K) — for
//     the W^T view we load 8 rows of W from row=bn..bn+31 across 8 columns
//     k..k+7, and treat it as B[8, 32] in the matmul.
//   - simdgroup_load each tile from threadgroup memory.
//   - simdgroup_multiply_accumulate the four output sub-tiles.
//
// Output epilogue: convert fp32 accumulators back to fp16 and store.

// Phase 18 — AMX saturation rewrite.
//   BM = 128, BN = 128, BK = 32. 8 simdgroups (256 threads) per threadgroup.
//   Each simdgroup owns a 16x128 strip of the output = 2 row-tiles × 16
//   col-tiles = 32 `simdgroup_float8x8` accumulators.
//
//   Threadgroup memory:
//     During matmul: As (128*32 halves = 8 KB) + Bs (32*128 halves = 8 KB)
//                    = 16 KB.
//     During epilogue: reuse the A+B region as 32 KB of half scratch (or
//                    16 KB of fp32). 4-pass epilogue writes 2 simdgroups'
//                    strips per pass (8 KB fp32 / pass), converts in
//                    threads, dumps to device fp16.
// Final Phase 18 state — best-performing tile from the swarm.
//   Strict BM=BN=128 spec deviates from reality of M=12 prefill: even with
//   thread gating + minimal acc tiles, the 10× row padding floored gemm
//   wall at ~1.1s — same as the BM=32 baseline. BM=16 matches the actual
//   prefill shape with no padding waste. Math bit-equivalent.
constant uint GEMM_BM = 16;
constant uint GEMM_BN = 128;
constant uint GEMM_BK = 32;
constant uint GEMM_TG_SIZE         = 256;   // 8 simdgroups
constant uint GEMM_SIMDS_PER_TG    = 8;
constant uint GEMM_COLS_PER_SIMD      = GEMM_BN / GEMM_SIMDS_PER_TG;  // 16
constant uint GEMM_ROW_TILES_PER_SIMD = GEMM_BM / 8;                  // 2
constant uint GEMM_COL_TILES_PER_SIMD = GEMM_COLS_PER_SIMD / 8;       // 2
constant uint GEMM_ACC_TILES_PER_SIMD =
        GEMM_ROW_TILES_PER_SIMD * GEMM_COL_TILES_PER_SIMD;            // 4
constant uint GEMM_K_SUB_TILES     = GEMM_BK / 8;                     // 4

kernel void linear_f16_gemm(
    device const half* X      [[ buffer(0) ]],   // (M, K)
    device const half* W      [[ buffer(1) ]],   // (N, K)  row-major
    device       half* Y      [[ buffer(2) ]],   // (M, N)
    constant LinearParams& p  [[ buffer(3) ]],
    uint2 gid                 [[ threadgroup_position_in_grid ]],
    uint2 tid2                [[ thread_position_in_threadgroup ]],
    uint  simd_index          [[ simdgroup_index_in_threadgroup ]]
) {
    const uint tid = tid2.x;
    const uint bm  = gid.y * GEMM_BM;
    const uint bn  = gid.x * GEMM_BN;

    // 32 fp32 accumulator tiles per simdgroup.
    simdgroup_float8x8 acc[GEMM_ACC_TILES_PER_SIMD];
    for (uint i = 0; i < GEMM_ACC_TILES_PER_SIMD; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    threadgroup half As[GEMM_BM * GEMM_BK];   // 8 KB
    threadgroup half Bs[GEMM_BK * GEMM_BN];   // 8 KB

    // Iter 4 — sgs partition cols, share rows. All sgs active.
    const uint col_in_tg = simd_index * GEMM_COLS_PER_SIMD;  // 0,16,...,112

    for (uint k = 0; k < p.K; k += GEMM_BK) {
        // Cooperative A load: 16*32 = 512 halves over 256 threads = 2/thread.
        for (uint t = tid; t < GEMM_BM * GEMM_BK; t += GEMM_TG_SIZE) {
            uint mi = t / GEMM_BK;
            uint ki = t % GEMM_BK;
            uint src_m = bm + mi;
            uint src_k = k  + ki;
            As[mi * GEMM_BK + ki] = (src_m < p.M && src_k < p.K)
                ? X[src_m * p.K + src_k] : half(0);
        }
        // Cooperative B load: 32*128 = 4096 halves over 256 threads = 16/thread.
        for (uint t = tid; t < GEMM_BK * GEMM_BN; t += GEMM_TG_SIZE) {
            uint ki = t / GEMM_BN;
            uint ni = t % GEMM_BN;
            uint src_n = bn + ni;
            uint src_k = k  + ki;
            Bs[ki * GEMM_BN + ni] = (src_n < p.N && src_k < p.K)
                ? W[src_n * p.K + src_k] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki8 = 0; ki8 < GEMM_K_SUB_TILES; ++ki8) {
            simdgroup_half8x8 a_tile[GEMM_ROW_TILES_PER_SIMD];
            for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
                simdgroup_load(a_tile[ri],
                               &As[(ri * 8) * GEMM_BK + ki8 * 8],
                               GEMM_BK);
            }
            for (uint ni8 = 0; ni8 < GEMM_COL_TILES_PER_SIMD; ++ni8) {
                simdgroup_half8x8 b_tile;
                simdgroup_load(b_tile,
                               &Bs[ki8 * 8 * GEMM_BN + col_in_tg + ni8 * 8],
                               GEMM_BN);
                for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
                    simdgroup_multiply_accumulate(
                        acc[ri * GEMM_COL_TILES_PER_SIMD + ni8],
                        a_tile[ri], b_tile,
                        acc[ri * GEMM_COL_TILES_PER_SIMD + ni8]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Iter 4 epilogue — single-pass, full BM × BN scratch = 16 × 128 fp32 = 8 KB.
    threadgroup float* scratch = reinterpret_cast<threadgroup float*>(As);
    for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
        for (uint ni8 = 0; ni8 < GEMM_COL_TILES_PER_SIMD; ++ni8) {
            simdgroup_store(
                acc[ri * GEMM_COL_TILES_PER_SIMD + ni8],
                &scratch[(ri * 8) * GEMM_BN + col_in_tg + ni8 * 8],
                GEMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint TILE_ELEMS = GEMM_BM * GEMM_BN;   // 16 * 128 = 2048
    for (uint t = tid; t < TILE_ELEMS; t += GEMM_TG_SIZE) {
        uint mi = t / GEMM_BN;
        uint ni = t % GEMM_BN;
        uint mg = bm + mi;
        uint ng = bn + ni;
        if (mg < p.M && ng < p.N) {
            Y[mg * p.N + ng] = half(scratch[mi * GEMM_BN + ni]);
        }
    }
}

// Fused: SiLU epilogue. Same Phase 18 tiling as linear_f16_gemm; SiLU
// applied per-element during the device write.
kernel void linear_silu_f16_gemm(
    device const half* X      [[ buffer(0) ]],
    device const half* W      [[ buffer(1) ]],
    device       half* Y      [[ buffer(2) ]],
    constant LinearParams& p  [[ buffer(3) ]],
    uint2 gid                 [[ threadgroup_position_in_grid ]],
    uint2 tid2                [[ thread_position_in_threadgroup ]],
    uint  simd_index          [[ simdgroup_index_in_threadgroup ]]
) {
    const uint tid = tid2.x;
    const uint bm  = gid.y * GEMM_BM;
    const uint bn  = gid.x * GEMM_BN;

    simdgroup_float8x8 acc[GEMM_ACC_TILES_PER_SIMD];
    for (uint i = 0; i < GEMM_ACC_TILES_PER_SIMD; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    threadgroup half As[GEMM_BM * GEMM_BK];
    threadgroup half Bs[GEMM_BK * GEMM_BN];
    const uint col_in_tg = simd_index * GEMM_COLS_PER_SIMD;

    for (uint k = 0; k < p.K; k += GEMM_BK) {
        for (uint t = tid; t < GEMM_BM * GEMM_BK; t += GEMM_TG_SIZE) {
            uint mi = t / GEMM_BK; uint ki = t % GEMM_BK;
            uint src_m = bm + mi; uint src_k = k + ki;
            As[mi * GEMM_BK + ki] = (src_m < p.M && src_k < p.K) ? X[src_m * p.K + src_k] : half(0);
        }
        for (uint t = tid; t < GEMM_BK * GEMM_BN; t += GEMM_TG_SIZE) {
            uint ki = t / GEMM_BN; uint ni = t % GEMM_BN;
            uint src_n = bn + ni; uint src_k = k + ki;
            Bs[ki * GEMM_BN + ni] = (src_n < p.N && src_k < p.K) ? W[src_n * p.K + src_k] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki8 = 0; ki8 < GEMM_K_SUB_TILES; ++ki8) {
            simdgroup_half8x8 a_tile[GEMM_ROW_TILES_PER_SIMD];
            for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
                simdgroup_load(a_tile[ri], &As[(ri * 8) * GEMM_BK + ki8 * 8], GEMM_BK);
            }
            for (uint ni8 = 0; ni8 < GEMM_COL_TILES_PER_SIMD; ++ni8) {
                simdgroup_half8x8 b_tile;
                simdgroup_load(b_tile, &Bs[ki8 * 8 * GEMM_BN + col_in_tg + ni8 * 8], GEMM_BN);
                for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
                    simdgroup_multiply_accumulate(
                        acc[ri * GEMM_COL_TILES_PER_SIMD + ni8],
                        a_tile[ri], b_tile,
                        acc[ri * GEMM_COL_TILES_PER_SIMD + ni8]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* scratch = reinterpret_cast<threadgroup float*>(As);
    for (uint ri = 0; ri < GEMM_ROW_TILES_PER_SIMD; ++ri) {
        for (uint ni8 = 0; ni8 < GEMM_COL_TILES_PER_SIMD; ++ni8) {
            simdgroup_store(
                acc[ri * GEMM_COL_TILES_PER_SIMD + ni8],
                &scratch[(ri * 8) * GEMM_BN + col_in_tg + ni8 * 8],
                GEMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint TILE_ELEMS = GEMM_BM * GEMM_BN;
    for (uint t = tid; t < TILE_ELEMS; t += GEMM_TG_SIZE) {
        uint mi = t / GEMM_BN;
        uint ni = t % GEMM_BN;
        uint mg = bm + mi;
        uint ng = bn + ni;
        if (mg < p.M && ng < p.N) {
            float v = scratch[mi * GEMM_BN + ni];
            v = v / (1.0f + exp(-v));
            Y[mg * p.N + ng] = half(v);
        }
    }
}

// ---------------------------------------------------------------------------
//  INT4_BLOCK32 GEMV — y[N] = x[K] @ W[N,K]^T,  W stored as int4 blocks.
//
// Layout (one threadgroup per N-block of GEMV_ROWS_PER_TG output rows; one
// simdgroup per row inside that block). Each simdgroup loops over K=32-wide
// blocks, dequantizes 32 weights into fp32 against a shared scale, and
// multiply-accumulates against x[k:k+32] read as half4s.
//
// Buffer layout (single MTLBuffer; offsets supplied by the C++ dispatcher):
//   buffer(0) x         (K halves)
//   buffer(1) W packed  (N rows × K/2 bytes)
//   buffer(2) W scales  (N rows × K/32 halves)
//   buffer(3) y         (N halves)
//   buffer(4) LinearParams { M=1, N, K }
// ---------------------------------------------------------------------------
kernel void linear_int4_gemv(
    device const half*  x       [[ buffer(0) ]],
    device const uchar* Wpacked [[ buffer(1) ]],
    device const half*  Wscales [[ buffer(2) ]],
    device       half*  y       [[ buffer(3) ]],
    constant LinearParams& p    [[ buffer(4) ]],
    uint tg_id                  [[ threadgroup_position_in_grid ]],
    uint tid                    [[ thread_position_in_threadgroup ]],
    uint simd_lane              [[ thread_index_in_simdgroup ]],
    uint simd_index             [[ simdgroup_index_in_threadgroup ]]
) {
    const uint row = tg_id * GEMV_ROWS_PER_TG + simd_index;
    if (row >= p.N) return;

    const uint blocks_per_row = p.K / INT4_BLOCK_SIZE;
    device const uchar* row_packed = Wpacked + row * (p.K / 2);
    device const half*  row_scales = Wscales + row * blocks_per_row;

    float acc = 0.0f;
    // Each block = 32 weights. 32 lanes ↔ 32 weights. lane i handles weight i.
    for (uint b = 0; b < blocks_per_row; ++b) {
        const uint intra = simd_lane;                   // 0..31
        const uchar byte = row_packed[b * (INT4_BLOCK_SIZE / 2) + (intra >> 1)];
        const uint  nibble = (intra & 1u) ? (uint(byte) >> 4) : (uint(byte) & 0x0F);
        const int   q4 = (nibble < 8u) ? int(nibble) : int(nibble) - 16;
        const float scale = float(row_scales[b]);
        const float w     = float(q4) * scale;

        const uint k = b * INT4_BLOCK_SIZE + intra;
        const float xv = float(x[k]);
        acc += w * xv;
    }
    const float total = simd_sum(acc);
    if (simd_lane == 0) y[row] = half(total);
}

// ---------------------------------------------------------------------------
//  INT4_BLOCK32 GEMM — Y[M,N] = X[M,K] @ W[N,K]^T  (prefill path).
//
// Strategy: dequantize the int4 weight tile into a half threadgroup buffer,
// then re-use the same simdgroup_matrix kernel body as `linear_f16_gemm`.
// Threadgroup tile BM=32 × BN=32, K-tile BK=32 (aligned to one int4 block —
// no straddling). Cooperative load of A from device fp16 + cooperative
// dequant of B from packed int4 into the threadgroup buffer.
// ---------------------------------------------------------------------------
// Phase 21 — int4 gemm uses its own tile constants (independent of the
// fp16 GEMM_* values, which Phase 18 retuned for col-partitioned layout).
// Layout: 4 simdgroups × one row tile × 4 col tiles each. 4 acc tiles/sg.
constant uint INT4_GEMM_BM      = 32;
constant uint INT4_GEMM_BN      = 32;
constant uint INT4_GEMM_TG_SIZE = 128;   // 4 simdgroups

kernel void linear_int4_gemm(
    device const half*  X       [[ buffer(0) ]],
    device const uchar* Wpacked [[ buffer(1) ]],
    device const half*  Wscales [[ buffer(2) ]],
    device       half*  Y       [[ buffer(3) ]],
    constant LinearParams& p    [[ buffer(4) ]],
    uint2 gid                   [[ threadgroup_position_in_grid ]],
    uint2 tid2                  [[ thread_position_in_threadgroup ]],
    uint  simd_index            [[ simdgroup_index_in_threadgroup ]]
) {
    const uint tid = tid2.x;
    const uint bm  = gid.y * INT4_GEMM_BM;
    const uint bn  = gid.x * INT4_GEMM_BN;

    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    constexpr uint BK_INT4 = INT4_BLOCK_SIZE;
    threadgroup half As[INT4_GEMM_BM * BK_INT4];   // 32 * 32 halves = 2 KB
    threadgroup half Bs[BK_INT4 * INT4_GEMM_BN];   // 32 * 32 halves = 2 KB

    const uint row_in_tg = simd_index * 8;
    const uint blocks_per_w_row = p.K / INT4_BLOCK_SIZE;
    const uint BK = BK_INT4;

    for (uint k = 0; k < p.K; k += BK) {
        for (uint t = tid; t < INT4_GEMM_BM * BK; t += INT4_GEMM_TG_SIZE) {
            uint mi = t / BK;
            uint ki = t % BK;
            uint src_m = bm + mi;
            uint src_k = k  + ki;
            As[mi * BK + ki] = (src_m < p.M && src_k < p.K)
                ? X[src_m * p.K + src_k] : half(0);
        }
        const uint b_idx = k / BK;
        for (uint t = tid; t < BK * INT4_GEMM_BN; t += INT4_GEMM_TG_SIZE) {
            uint ki = t / INT4_GEMM_BN;
            uint ni = t % INT4_GEMM_BN;
            uint src_n = bn + ni;
            if (src_n < p.N) {
                device const uchar* row_packed = Wpacked + src_n * (p.K / 2);
                device const half*  row_scales = Wscales + src_n * blocks_per_w_row;
                const uchar byte = row_packed[b_idx * (BK / 2) + (ki >> 1)];
                const uint  nibble = (ki & 1u) ? (uint(byte) >> 4) : (uint(byte) & 0x0F);
                const int   q4 = (nibble < 8u) ? int(nibble) : int(nibble) - 16;
                const float scale = float(row_scales[b_idx]);
                Bs[ki * INT4_GEMM_BN + ni] = half(float(q4) * scale);
            } else {
                Bs[ki * INT4_GEMM_BN + ni] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a_tile[4];
        simdgroup_half8x8 b_tile[4][4];
        for (int ki8 = 0; ki8 < 4; ++ki8) {
            simdgroup_load(a_tile[ki8], &As[row_in_tg * BK + ki8 * 8], BK);
            for (int ni8 = 0; ni8 < 4; ++ni8) {
                simdgroup_load(b_tile[ki8][ni8], &Bs[ki8 * 8 * INT4_GEMM_BN + ni8 * 8], INT4_GEMM_BN);
            }
        }
        for (int ki8 = 0; ki8 < 4; ++ki8) {
            for (int ni8 = 0; ni8 < 4; ++ni8) {
                simdgroup_multiply_accumulate(acc[ni8], a_tile[ki8], b_tile[ki8][ni8], acc[ni8]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float scratch[INT4_GEMM_BM * INT4_GEMM_BN];
    for (int i = 0; i < 4; ++i) {
        simdgroup_store(acc[i], &scratch[row_in_tg * INT4_GEMM_BN + i * 8], INT4_GEMM_BN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tile_elems = INT4_GEMM_BM * INT4_GEMM_BN;
    for (uint t = tid; t < tile_elems; t += INT4_GEMM_TG_SIZE) {
        uint mi = t / INT4_GEMM_BN, ni = t % INT4_GEMM_BN;
        uint mg = bm + mi, ng = bn + ni;
        if (mg >= p.M || ng >= p.N) continue;
        Y[mg * p.N + ng] = half(scratch[mi * INT4_GEMM_BN + ni]);
    }
}
