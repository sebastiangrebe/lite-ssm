#pragma once
// lite_ssm/ops.hpp — public dispatch interface for the Metal compute backend.
//
// MetalOps wraps:
//   * a lazy MTLDevice + MTLCommandQueue
//   * the compiled lite_ssm.metallib (loaded on first use)
//   * a cache of MTLComputePipelineState objects keyed by kernel name
//
// In Phase 4 the surface is intentionally narrow: callers ask for a pipeline
// by name (smoke test for the build), or invoke one of the typed dispatch
// methods, which build a command buffer, encode the kernel, and commit.
// Phase 5 will refactor to batch many ops into one command buffer for the
// whole forward pass.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "lite_ssm/allocator.hpp"

namespace lite_ssm {

// Opaque handle for an id<MTLBuffer>. Callers normally hand us one via
// `UnifiedBuffer::metal_buffer()`. The dispatch layer also accepts plain
// device buffers allocated on the fly (e.g. activations, SSM state).
using MetalBufferHandle = void*;

struct LinearDims     { uint32_t M; uint32_t N; uint32_t K; };
struct RMSNormDims    { uint32_t rows; uint32_t dim; float eps; };
struct RMSLinearDims  { uint32_t N; uint32_t K; float eps; };
struct Conv1dDims     { uint32_t B; uint32_t L; uint32_t D; uint32_t K; };

struct SSDDims {
    uint32_t L;
    uint32_t H;
    uint32_t P;
    uint32_t N;
    uint32_t n_groups;     // Phase 11
    uint32_t chunk_size;
    float    dt_min;
    float    dt_max;
    uint32_t has_D;        // 0/1
};

struct SSDStepDims {
    uint32_t H;
    uint32_t P;
    uint32_t N;
    uint32_t n_groups;     // Phase 11
    float    dt_min;
    float    dt_max;
    uint32_t has_D;
};

class MetalOps {
public:
    MetalOps();
    ~MetalOps();
    MetalOps(const MetalOps&)            = delete;
    MetalOps& operator=(const MetalOps&) = delete;

    // True iff the metallib was loaded successfully. If false, kernel
    // dispatch will throw — useful for tests that want to skip cleanly.
    bool metallib_loaded() const;

    // Returns true if a pipeline for the named kernel exists in the loaded
    // metallib. Builds and caches the MTLComputePipelineState on first call.
    bool has_pipeline(std::string_view name);

    // Path on disk that was loaded (or attempted). Empty if construction
    // hasn't reached the metallib step yet.
    const std::string& metallib_path() const;

    // ------------------------------------------------------------------
    // Batching.
    //
    // By default each dispatch creates its own command buffer + encoder
    // and waits on completion (handy for tests / smoke checks). For real
    // inference the whole forward pass should land in one command buffer:
    //
    //   ops.begin_batch();
    //   // ... dispatch many kernels ...
    //   ops.commit_and_wait();
    //
    // While a batch is active, dispatches share one encoder; no per-op
    // commit/wait. Apple Silicon's serial compute encoder inserts the
    // necessary memory barriers between dispatches on the same buffer.
    // ------------------------------------------------------------------
    void begin_batch();
    void commit_and_wait();
    bool batch_active() const;

    // ------------------------------------------------------------------
    // Op dispatch.
    // ------------------------------------------------------------------

    void rmsnorm_f16(MetalBufferHandle x, std::size_t x_off,
                     MetalBufferHandle w, std::size_t w_off,
                     MetalBufferHandle y, std::size_t y_off,
                     RMSNormDims dims);

    void linear_f16_gemv(MetalBufferHandle x, std::size_t x_off,
                         MetalBufferHandle w, std::size_t w_off,
                         MetalBufferHandle y, std::size_t y_off,
                         LinearDims dims);

    void linear_silu_f16_gemv(MetalBufferHandle x, std::size_t x_off,
                              MetalBufferHandle w, std::size_t w_off,
                              MetalBufferHandle y, std::size_t y_off,
                              LinearDims dims);

    void rmsnorm_linear_f16_gemv(MetalBufferHandle x,      std::size_t x_off,
                                 MetalBufferHandle w_norm, std::size_t w_norm_off,
                                 MetalBufferHandle w,      std::size_t w_off,
                                 MetalBufferHandle y,      std::size_t y_off,
                                 RMSLinearDims dims);

    void linear_f16_gemm(MetalBufferHandle x, std::size_t x_off,
                         MetalBufferHandle w, std::size_t w_off,
                         MetalBufferHandle y, std::size_t y_off,
                         LinearDims dims);

    void linear_silu_f16_gemm(MetalBufferHandle x, std::size_t x_off,
                              MetalBufferHandle w, std::size_t w_off,
                              MetalBufferHandle y, std::size_t y_off,
                              LinearDims dims);

    // Phase 14 — INT4_BLOCK32 variants.
    // `w_packed_off` points at the packed-nibble region; `w_scales_off`
    // points at the per-block fp16 scale array (caller computes from
    // tensor.offset_bytes + numel/2). Caller responsible for routing only
    // weights stored as DTYPE_INT4_B32 here.
    void linear_int4_gemv(MetalBufferHandle x,       std::size_t x_off,
                          MetalBufferHandle w,       std::size_t w_packed_off,
                          MetalBufferHandle w_scl,   std::size_t w_scales_off,
                          MetalBufferHandle y,       std::size_t y_off,
                          LinearDims dims);

    void linear_int4_gemm(MetalBufferHandle x,       std::size_t x_off,
                          MetalBufferHandle w,       std::size_t w_packed_off,
                          MetalBufferHandle w_scl,   std::size_t w_scales_off,
                          MetalBufferHandle y,       std::size_t y_off,
                          LinearDims dims);

    void silu_f16(MetalBufferHandle x, std::size_t x_off,
                  MetalBufferHandle y, std::size_t y_off,
                  uint32_t n);

    void add_inplace_f16(MetalBufferHandle y, std::size_t y_off,
                         MetalBufferHandle x, std::size_t x_off,
                         uint32_t n);

    // De-interleave conv1d output (L, d_inner + 2*n_groups*d_state) into
    // contiguous x (L, d_inner), B (L, n_groups*d_state),
    // C (L, n_groups*d_state) — applying SiLU.
    void split_silu_xBC_f16(MetalBufferHandle xBC,   std::size_t xBC_off,
                            MetalBufferHandle x_out, std::size_t x_off,
                            MetalBufferHandle B_out, std::size_t B_off,
                            MetalBufferHandle C_out, std::size_t C_off,
                            uint32_t L, uint32_t d_inner, uint32_t d_state,
                            uint32_t n_groups);

    void silu_gated_f16(MetalBufferHandle x,    std::size_t x_off,
                        MetalBufferHandle gate, std::size_t gate_off,
                        MetalBufferHandle y,    std::size_t y_off,
                        uint32_t n);

    void causal_conv1d_f16(MetalBufferHandle x,    std::size_t x_off,
                           MetalBufferHandle w,    std::size_t w_off,
                           MetalBufferHandle bias, std::size_t bias_off,
                           MetalBufferHandle y,    std::size_t y_off,
                           Conv1dDims dims);

    void causal_conv1d_update_f16(MetalBufferHandle x_new, std::size_t x_off,
                                  MetalBufferHandle state, std::size_t state_off,
                                  MetalBufferHandle w,     std::size_t w_off,
                                  MetalBufferHandle bias,  std::size_t bias_off,
                                  MetalBufferHandle y,     std::size_t y_off,
                                  Conv1dDims dims);

    void ssd_chunked_f16(MetalBufferHandle X,        std::size_t X_off,
                         MetalBufferHandle B,        std::size_t B_off,
                         MetalBufferHandle C,        std::size_t C_off,
                         MetalBufferHandle dt_raw,   std::size_t dt_off,
                         MetalBufferHandle dt_bias,  std::size_t db_off,
                         MetalBufferHandle A_log,    std::size_t A_off,
                         MetalBufferHandle D_skip,   std::size_t D_off,
                         MetalBufferHandle Y,        std::size_t Y_off,
                         MetalBufferHandle state_io, std::size_t S_off,
                         SSDDims dims);

    void ssd_step_f16(MetalBufferHandle x,        std::size_t x_off,
                      MetalBufferHandle B,        std::size_t B_off,
                      MetalBufferHandle C,        std::size_t C_off,
                      MetalBufferHandle dt_raw,   std::size_t dt_off,
                      MetalBufferHandle dt_bias,  std::size_t db_off,
                      MetalBufferHandle A_log,    std::size_t A_off,
                      MetalBufferHandle D_skip,   std::size_t D_off,
                      MetalBufferHandle state,    std::size_t S_off,
                      MetalBufferHandle y,        std::size_t y_off,
                      SSDStepDims dims);

    // Public so the .mm-internal dispatch helpers can take an Impl*. The
    // pointer itself is still private — callers can name the type but
    // can't reach the instance.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace lite_ssm
