// shaders/conv1d.metal
// Causal depthwise 1-D convolution used by the Mamba mixer just before
// the SSD block. Two kernels:
//
//   causal_conv1d_f16        — prefill (process L tokens at once)
//   causal_conv1d_update_f16 — decode (single token + rolling state buffer)
//
// Layout (matching HF Mamba-2):
//   x      : (B, L, D)        input activations
//   w      : (D, 1, K)        depthwise filter (K = d_conv)
//   bias   : (D)              optional bias (always present in Mamba-2)
//   y      : (B, L, D)        output
//
// "Causal" means y[t] only depends on x[t], x[t-1], ..., x[t-(K-1)] — we
// pad on the left with zeros so no future information leaks backward.

#include <metal_stdlib>
using namespace metal;

struct Conv1dParams {
    uint B;       // batch
    uint L;       // sequence length
    uint D;       // channels (== d_inner_ssm in Mamba-2)
    uint K;       // kernel width (d_conv, usually 4)
};

// Phase 16: seed the conv-state rolling window from the last K timesteps
// of pre-conv xBC. Keeps the prefill -> decode handoff entirely on-GPU,
// so we don't need a CPU memcpy between layers and can batch all 64 layers
// of forward_prefill into a single MTLCommandBuffer.
//
//   state[d, k] = (L - K + k) >= 0 ? xBC[L - K + k, d] : 0
//
// Layout: state is (B=1, D, K) row-major, K-contiguous. Grid: (D, K).
kernel void seed_conv_state_f16(
    device const half* xBC    [[ buffer(0) ]],    // (L, D)
    device       half* state  [[ buffer(1) ]],    // (B=1, D, K)
    constant Conv1dParams& p  [[ buffer(2) ]],
    uint2 gid                 [[ thread_position_in_grid ]]
) {
    const uint d = gid.x;
    const uint k = gid.y;
    if (d >= p.D || k >= p.K) return;

    const int src_t = int(p.L) - int(p.K) + int(k);
    half v = half(0);
    if (src_t >= 0) {
        v = xBC[uint(src_t) * p.D + d];
    }
    state[d * p.K + k] = v;
}

// ---------------------------------------------------------------------------
//  Prefill: full causal conv over the whole sequence.
//
// Grid:    (D, L, B)
// Threads: one per (batch, time, channel) output element.
//
// Each thread reads K weights + up to K inputs (left-zero-padded) and writes
// one output. D × L × B threads in total. Bandwidth-bound; relies on the L2
// cache to absorb the strided weight reads (weights are tiny: D × K halves).
// ---------------------------------------------------------------------------
kernel void causal_conv1d_f16(
    device const half* x        [[ buffer(0) ]],     // (B, L, D)
    device const half* w        [[ buffer(1) ]],     // (D, 1, K)
    device const half* bias     [[ buffer(2) ]],     // (D)
    device       half* y        [[ buffer(3) ]],     // (B, L, D)
    constant Conv1dParams& p    [[ buffer(4) ]],
    uint3 gid                   [[ thread_position_in_grid ]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    const uint b = gid.z;
    if (d >= p.D || t >= p.L || b >= p.B) return;

    device const half* wd = w + d * p.K;
    const uint x_time_stride = p.D;          // step from x[t] to x[t+1] within a batch
    const uint x_batch_stride = p.L * p.D;

    float acc = (bias ? float(bias[d]) : 0.0f);

    // Sum over kernel taps. tap = 0 is the current step; tap = K-1 reaches
    // furthest into the past. Past samples are zero-padded.
    for (uint tap = 0; tap < p.K; ++tap) {
        int src_t = int(t) - int(p.K - 1 - tap);
        float v = 0.0f;
        if (src_t >= 0) {
            v = float(x[b * x_batch_stride + uint(src_t) * x_time_stride + d]);
        }
        acc += float(wd[tap]) * v;
    }
    y[b * x_batch_stride + t * x_time_stride + d] = half(acc);
}

// ---------------------------------------------------------------------------
//  Decode: single-step recurrent update.
//
// State layout: (B, D, K) — for each channel we keep the last K samples,
// oldest first. Per step we:
//   1. Shift the state left by one (drop the oldest, append `x_new`).
//   2. Compute y[d] = bias[d] + dot(state[b, d, :], w[d, 0, :]).
//
// We compute against the state laid out so element K-1 is the freshest
// sample, matching the prefill formula's tap = K-1 == current step.
//
// Grid: (D, B). One thread per (batch, channel).
// ---------------------------------------------------------------------------
kernel void causal_conv1d_update_f16(
    device const half* x_new    [[ buffer(0) ]],     // (B, D)
    device       half* state    [[ buffer(1) ]],     // (B, D, K) mutable in place
    device const half* w        [[ buffer(2) ]],     // (D, 1, K)
    device const half* bias     [[ buffer(3) ]],     // (D)
    device       half* y        [[ buffer(4) ]],     // (B, D)
    constant Conv1dParams& p    [[ buffer(5) ]],
    uint2 gid                   [[ thread_position_in_grid ]]
) {
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= p.D || b >= p.B) return;

    device       half* sd = state + (b * p.D + d) * p.K;
    device const half* wd = w     + d * p.K;
    const half  x_in = x_new[b * p.D + d];

    // Shift the state window: state[k] = state[k+1]. The new sample lands at
    // index K-1 so tap[K-1] of the prefill formula matches.
    for (uint k = 0; k + 1 < p.K; ++k) {
        sd[k] = sd[k + 1];
    }
    sd[p.K - 1] = x_in;

    float acc = (bias ? float(bias[d]) : 0.0f);
    for (uint k = 0; k < p.K; ++k) {
        acc += float(sd[k]) * float(wd[k]);
    }
    y[b * p.D + d] = half(acc);
}
