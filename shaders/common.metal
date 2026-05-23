// shaders/common.metal
// Shared types and small helpers for all lite-ssm kernels.
//
// Conventions:
//   * Weights live in fp16. Activations in fp16. Accumulators in fp32.
//   * Threadgroup memory is precious — we sized tile constants so the
//     hottest kernels fit well under 32 KB of SRAM on M-series.
//   * For simdgroup matrix ops we use the 8x8 form (the only one the
//     M-series matrix unit accelerates directly).

#include <metal_stdlib>
using namespace metal;

// Convenience: simdgroup width on Apple Silicon is always 32 threads.
constant uint SIMD_WIDTH = 32;

// Reduce a per-thread fp32 value across one simdgroup, broadcast result to lane 0.
inline float simd_sum_f32(float v) {
    return simd_sum(v);
}

// Reduce a per-thread fp32 value across an entire threadgroup. tg_scratch must
// have at least `n_simds` floats. Returns the sum to all threads (broadcast).
inline float tg_sum_f32(float v,
                        threadgroup float* tg_scratch,
                        uint tg_size,
                        uint thread_index,
                        uint simd_lane,
                        uint simd_index) {
    float sg_total = simd_sum(v);
    uint n_simds = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;
    if (simd_lane == 0) tg_scratch[simd_index] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // First simdgroup reduces the partial sums; broadcast via scratch slot 0.
    float total;
    if (simd_index == 0) {
        float v2 = (thread_index < n_simds) ? tg_scratch[thread_index] : 0.0f;
        v2 = simd_sum(v2);
        if (thread_index == 0) tg_scratch[0] = v2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = tg_scratch[0];
    return total;
}

// SiLU = x * sigmoid(x). Computed in fp32 because half-precision saturates
// in the tails and Mamba uses SiLU on gate paths where precision matters.
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline half silu_h(half x) {
    float f = float(x);
    return half(silu_f32(f));
}

// Softplus = ln(1 + exp(x)) with the standard log1p/exp trick to avoid
// blow-up. Used by Mamba-2 for the dt parameter.
inline float softplus_f32(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// noop kernel kept so the metallib still has a stable export from this TU.
kernel void lite_ssm_common_noop(uint tid [[thread_position_in_grid]]) {
    (void)tid;
}
