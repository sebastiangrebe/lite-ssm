// shaders/ssd_step.metal
// Mamba-2 SSD — DECODE path (single-token recurrent step).
//
//   state[h, p, n] = exp(A_h · dt[h]) · state[h, p, n]
//                  + dt[h] · B[n] · x[h, p]
//   y[h, p]        = Σ_n C[n] · state[h, p, n]   +  D[h] · x[h, p]
//
// One token, no chunking, no sequential dependency across t. This is the
// hot inner loop of autoregressive generation, so it's a single fused
// kernel with everything held in registers.
//
// Grid:    (H, 1)
// Threads: P per group (one thread per d_head element). Threads collaborate
// over the d_state dimension via in-loop sum.

#include <metal_stdlib>
using namespace metal;

struct SSDStepParams {
    uint  H;
    uint  P;
    uint  N;
    uint  n_groups;     // Phase 11
    float dt_min;
    float dt_max;
    uint  has_D;
};

inline float softplus_fast(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void ssd_step_f16(
    device const half*  x        [[ buffer(0) ]],   // (H, P)
    device const half*  Bproj    [[ buffer(1) ]],   // (N)
    device const half*  Cproj    [[ buffer(2) ]],   // (N)
    device const half*  dt_raw   [[ buffer(3) ]],   // (H)
    device const half*  dt_bias  [[ buffer(4) ]],   // (H)
    device const half*  A_log    [[ buffer(5) ]],   // (H)
    device const half*  D_skip   [[ buffer(6) ]],   // (H) optional
    device       float* state    [[ buffer(7) ]],   // (H, P, N) — mutated in place
    device       half*  y        [[ buffer(8) ]],   // (H, P)
    constant SSDStepParams& p    [[ buffer(9) ]],
    uint  h                      [[ threadgroup_position_in_grid ]],
    uint  tid                    [[ thread_position_in_threadgroup ]]
) {
    if (h >= p.H) return;
    const uint pp = tid;
    if (pp >= p.P) return;

    // Phase 11: pick the group this head reads B/C from.
    const uint heads_per_group = max(1u, p.H / p.n_groups);
    const uint g               = h / heads_per_group;

    const float A_h  = -exp(float(A_log[h]));
    const float dt   = clamp(
        softplus_fast(float(dt_raw[h]) + float(dt_bias[h])),
        p.dt_min, p.dt_max);
    const float decay = exp(A_h * dt);
    const float D_h   = (p.has_D ? float(D_skip[h]) : 0.0f);
    const float x_hp  = float(x[h * p.P + pp]);

    device float*       S  = state + (h * p.P + pp) * p.N;   // this thread's row
    device const half*  Bg = Bproj + g * p.N;                // this group's B
    device const half*  Cg = Cproj + g * p.N;                // this group's C

    float y_sum = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        float bn = float(Bg[n]);
        float cn = float(Cg[n]);
        float s  = decay * S[n] + dt * bn * x_hp;
        S[n]     = s;
        y_sum   += cn * s;
    }

    y[h * p.P + pp] = half(y_sum + D_h * x_hp);
}
