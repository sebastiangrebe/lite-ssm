// shaders/activations.metal
// Elementwise activations. Standalone versions live here; the fused
// linear+silu / linear+gated_silu live in linear.metal where the math
// can stay in registers.

#include <metal_stdlib>
using namespace metal;

inline float silu_fp32(float x) { return x / (1.0f + exp(-x)); }

// y = silu(x), shape (n)
kernel void silu_f16(
    device const half* x   [[ buffer(0) ]],
    device       half* y   [[ buffer(1) ]],
    constant uint& n       [[ buffer(2) ]],
    uint gid               [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    float v = float(x[gid]);
    y[gid] = half(silu_fp32(v));
}

// y[i] += x[i], shape (n). Residual add for the Mamba block.
kernel void add_inplace_f16(
    device       half* y   [[ buffer(0) ]],
    device const half* x   [[ buffer(1) ]],
    constant uint& n       [[ buffer(2) ]],
    uint gid               [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    y[gid] = half(float(y[gid]) + float(x[gid]));
}

// Split + SiLU for the Mamba-2 mixer's xBC tensor.
//
// Input: xBC (L, d_inner + 2 * n_groups * d_state) — conv1d output, pre-SiLU.
// Output: x (L, d_inner), B (L, n_groups * d_state), C (L, n_groups * d_state)
//         each contiguous, with SiLU applied element-wise during the copy.
//
// Phase 11: B/C now hold one (d_state,) slot per group. For n_groups=1 the
// behaviour collapses to the original Mamba-2 layout.
struct SplitSiluParams {
    uint L;
    uint d_inner;
    uint d_state;
    uint n_groups;
};

kernel void split_silu_xBC_f16(
    device const half* xBC          [[ buffer(0) ]],
    device       half* x_out        [[ buffer(1) ]],
    device       half* B_out        [[ buffer(2) ]],
    device       half* C_out        [[ buffer(3) ]],
    constant SplitSiluParams& p     [[ buffer(4) ]],
    uint2 gid                       [[ thread_position_in_grid ]]
) {
    const uint f       = gid.x;       // feature index within the xBC row
    const uint t       = gid.y;       // timestep
    const uint GN      = p.n_groups * p.d_state;
    const uint xBC_dim = p.d_inner + 2 * GN;
    if (f >= xBC_dim || t >= p.L) return;

    float v = float(xBC[t * xBC_dim + f]);
    v = v / (1.0f + exp(-v));   // SiLU

    if (f < p.d_inner) {
        x_out[t * p.d_inner + f] = half(v);
    } else if (f < p.d_inner + GN) {
        B_out[t * GN + (f - p.d_inner)] = half(v);
    } else {
        C_out[t * GN + (f - p.d_inner - GN)] = half(v);
    }
}

// y = silu(x) * gate, shape (n). Used as the Mamba mixer output gate.
kernel void silu_gated_f16(
    device const half* x    [[ buffer(0) ]],
    device const half* gate [[ buffer(1) ]],
    device       half* y    [[ buffer(2) ]],
    constant uint& n        [[ buffer(3) ]],
    uint gid                [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    float xv = float(x[gid]);
    float gv = float(gate[gid]);
    y[gid] = half(silu_fp32(xv) * gv);
}
