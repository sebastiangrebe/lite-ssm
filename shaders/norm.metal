// shaders/norm.metal
// RMSNorm in fp16 with fp32 accumulators.
//
// Mamba-2 uses RMSNorm in three places per block: a pre-mixer LN, a
// pre-out_proj LN (norm), and a final LN before lm_head. All operate on
// rows of `dim` halves. Layout: input is (rows, dim) row-major, output
// matches shape.
//
// Strategy:
//   * One threadgroup per row.
//   * Threadgroup size matches the row width up to 1024; we vectorize as
//     half4 loads (4 elements per thread) so the per-thread work scales
//     to large hidden sizes without exceeding the 1024-thread cap.
//
// Math:
//   y[i] = x[i] * w[i] / sqrt(mean(x[j]^2) + eps)

#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {
    uint  rows;
    uint  dim;        // must be a multiple of 4 for the vectorized path
    float eps;
};

// Vectorized RMSNorm: dim must be a multiple of 4.
//
// Threads per group: tg_size. Each thread handles `dim/(4*tg_size)` half4
// stripes spaced by `tg_size`. Reduction uses an intra-simd sum, then a
// per-simd partial in threadgroup memory, then a final simd sum.
kernel void rmsnorm_f16(
    device const half* x         [[ buffer(0) ]],   // (rows, dim)
    device const half* w         [[ buffer(1) ]],   // (dim)
    device       half* y         [[ buffer(2) ]],   // (rows, dim)
    constant RMSNormParams& p    [[ buffer(3) ]],
    uint  row                    [[ threadgroup_position_in_grid ]],
    uint  tid                    [[ thread_position_in_threadgroup ]],
    uint  tg_size                [[ threads_per_threadgroup ]],
    uint  simd_lane              [[ thread_index_in_simdgroup ]],
    uint  simd_index             [[ simdgroup_index_in_threadgroup ]]
) {
    if (row >= p.rows) return;

    device const half4* xr = reinterpret_cast<device const half4*>(x + row * p.dim);
    device       half4* yr = reinterpret_cast<device       half4*>(y + row * p.dim);
    device const half4* wr = reinterpret_cast<device const half4*>(w);

    const uint n4 = p.dim >> 2;

    // 1. Per-thread sum of squares in fp32.
    float ss = 0.0f;
    for (uint i = tid; i < n4; i += tg_size) {
        float4 v = float4(xr[i]);
        ss += dot(v, v);
    }

    // 2. Threadgroup reduction.
    threadgroup float partial[32];   // up to 32 simdgroups (1024 threads)
    float sg_total = simd_sum(ss);
    if (simd_lane == 0) partial[simd_index] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0) {
        uint n_simds = (tg_size + 31u) >> 5;
        float v = (tid < n_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = partial[0];
    float rrms  = rsqrt(total / float(p.dim) + p.eps);

    // 3. Apply: y = x * w * rrms.
    for (uint i = tid; i < n4; i += tg_size) {
        float4 v = float4(xr[i]) * float4(wr[i]) * rrms;
        yr[i] = half4(v);
    }
}
