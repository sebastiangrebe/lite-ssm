// src/backend/metal/ops.mm
// C++ -> Metal dispatch wrappers. One method per kernel exported by
// shaders/*.metal. Each method:
//   1. Looks up the cached MTLComputePipelineState via KernelRegistry.
//   2. Spins up a MTLCommandBuffer + MTLComputeCommandEncoder.
//   3. Binds buffers (with offsets) and a small `setBytes:` params struct.
//   4. Picks a threadgroup size matching the kernel's design.
//   5. Commits and waits.
//
// The wait-per-call pattern is intentionally simple for Phase 4. Phase 5
// will refactor to amortize the encoder across the entire forward pass.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "lite_ssm/ops.hpp"
#include "lite_ssm/detail/metal_backend.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace lite_ssm {

// ---------------------------------------------------------------------------
//  Impl
// ---------------------------------------------------------------------------
struct MetalOps::Impl {
    // Active batch state. `batch_cmd` is a bridge-retained id<MTLCommandBuffer>;
    // `batch_enc` is a bridge-retained id<MTLComputeCommandEncoder>. Both
    // null when no batch is active.
    void* batch_cmd = nullptr;
    void* batch_enc = nullptr;
};

namespace {

inline id<MTLComputePipelineState> get_pso(const char* name) {
    void* raw = detail::registry_pipeline(name);
    if (!raw) {
        throw std::runtime_error(std::string("Metal: kernel '") + name + "' not in metallib");
    }
    return (__bridge id<MTLComputePipelineState>)raw;
}

inline id<MTLCommandQueue> queue() {
    return (__bridge id<MTLCommandQueue>)detail::metal_default_queue();
}

inline void bind_buf(id<MTLComputeCommandEncoder> enc, MetalBufferHandle raw,
                     std::size_t offset, NSUInteger idx) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)raw;
    [enc setBuffer:b offset:offset atIndex:idx];
}

inline NSUInteger round_up(NSUInteger n, NSUInteger m) {
    return ((n + m - 1) / m) * m;
}

}  // namespace

MetalOps::MetalOps()  : impl_(std::make_unique<Impl>()) {}
MetalOps::~MetalOps() {
    // If a batch was opened and never committed, drop it cleanly.
    if (impl_->batch_enc) CFBridgingRelease(impl_->batch_enc);
    if (impl_->batch_cmd) CFBridgingRelease(impl_->batch_cmd);
}

// ---------------------------------------------------------------------------
// Batch lifecycle
//
// Every public entry that builds Metal API objects is wrapped in an
// `@autoreleasepool`. Without this, autoreleased temporaries produced by
// the Metal framework (NSError descriptions, internal pool buffers backing
// `setBytes:`, the `MTLCommandBuffer.error` accessor, etc.) accumulate on
// the calling thread's pool until the next runloop drain — which for our
// non-Cocoa decode loop NEVER happens. Phase 9 measured ~1.4 KB/token of
// leakage from exactly this path. With the pool in place each dispatch's
// temporaries die immediately when the block ends.
// ---------------------------------------------------------------------------
void MetalOps::begin_batch() {
    if (impl_->batch_enc) {
        throw std::runtime_error("MetalOps: begin_batch called while batch already active");
    }
    @autoreleasepool {
        id<MTLCommandQueue>          q   = queue();
        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        impl_->batch_cmd = (__bridge_retained void*)cmd;
        impl_->batch_enc = (__bridge_retained void*)enc;
    }
}

void MetalOps::commit_and_wait() {
    if (!impl_->batch_enc) {
        throw std::runtime_error("MetalOps: commit_and_wait without an active batch");
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)impl_->batch_enc;
        id<MTLCommandBuffer>         cmd = (__bridge_transfer id<MTLCommandBuffer>)impl_->batch_cmd;
        impl_->batch_enc = nullptr;
        impl_->batch_cmd = nullptr;
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (NSError* err = cmd.error) {
            throw std::runtime_error(std::string("Metal: forward batch failed: ") +
                                     [[err localizedDescription] UTF8String]);
        }
    }
}

bool MetalOps::batch_active() const { return impl_->batch_enc != nullptr; }

// run() — uses the active batch encoder if any, otherwise builds a one-shot
// command buffer / encoder / commit / wait. setup() configures buffers + dispatch.
namespace {

template <class Setup>
inline void run_impl(MetalOps::Impl* impl, const char* kernel_name, Setup setup) {
    id<MTLComputePipelineState> pso = get_pso(kernel_name);

    if (impl->batch_enc) {
        id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)impl->batch_enc;
        [enc setComputePipelineState:pso];
        setup(enc, pso);
        return;
    }

    // Non-batched path — wrap the whole dispatch in an autoreleasepool so
    // the per-call command buffer + encoder + their autoreleased Metal
    // temporaries are reclaimed immediately on scope exit. See Phase 10
    // notes in begin_batch() for why this matters in our decode loop.
    @autoreleasepool {
        id<MTLCommandQueue>          q   = queue();
        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        setup(enc, pso);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (NSError* err = cmd.error) {
            throw std::runtime_error(std::string("Metal: ") + kernel_name +
                                     " failed: " + [[err localizedDescription] UTF8String]);
        }
    }
}

}  // namespace

// Variadic so the lambda body (which may contain commas: e.g.
// `const NSUInteger BM = 32, BN = 32, TG = 128;`) doesn't split macro args.
#define run(name, ...) run_impl(impl_.get(), name, __VA_ARGS__)

bool MetalOps::metallib_loaded() const {
    // Force a load attempt by asking the registry.
    (void)detail::registry_pipeline("lite_ssm_common_noop");
    return detail::registry_library_loaded();
}

bool MetalOps::has_pipeline(std::string_view name) {
    std::string s(name);
    return detail::registry_pipeline(s.c_str()) != nullptr;
}

const std::string& MetalOps::metallib_path() const {
    return detail::registry_metallib_path();
}

// ---------------------------------------------------------------------------
//  RMSNorm
// ---------------------------------------------------------------------------
void MetalOps::rmsnorm_f16(MetalBufferHandle x, std::size_t x_off,
                           MetalBufferHandle w, std::size_t w_off,
                           MetalBufferHandle y, std::size_t y_off,
                           RMSNormDims dims) {
    if (dims.dim % 4 != 0) {
        throw std::runtime_error("rmsnorm_f16: dim must be a multiple of 4");
    }
    struct Params { uint32_t rows; uint32_t dim; float eps; } p{dims.rows, dims.dim, dims.eps};

    run("rmsnorm_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        bind_buf(enc, x, x_off, 0);
        bind_buf(enc, w, w_off, 1);
        bind_buf(enc, y, y_off, 2);
        [enc setBytes:&p length:sizeof(p) atIndex:3];

        // One threadgroup per row; threads cover the (dim/4) half4 columns.
        NSUInteger tg = std::min<NSUInteger>(round_up(dims.dim / 4, 32), pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake(dims.rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  Linear gemv / gemv fused
// ---------------------------------------------------------------------------
namespace {

void dispatch_gemv(MetalOps::Impl* impl, const char* kernel_name,
                   MetalBufferHandle x, std::size_t x_off,
                   MetalBufferHandle w, std::size_t w_off,
                   MetalBufferHandle y, std::size_t y_off,
                   LinearDims dims) {
    if (dims.M != 1) {
        throw std::runtime_error("dispatch_gemv: M must be 1");
    }
    if (dims.K % 4 != 0) {
        throw std::runtime_error("dispatch_gemv: K must be a multiple of 4");
    }
    struct Params { uint32_t M, N, K; } p{dims.M, dims.N, dims.K};

    run_impl(impl, kernel_name, [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x, x_off, 0);
        bind_buf(enc, w, w_off, 1);
        bind_buf(enc, y, y_off, 2);
        [enc setBytes:&p length:sizeof(p) atIndex:3];

        const NSUInteger rows_per_tg = 8;
        const NSUInteger tg_size     = 32 * rows_per_tg;
        const NSUInteger groups      = (dims.N + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    });
}

}  // namespace

void MetalOps::linear_f16_gemv(MetalBufferHandle x, std::size_t x_off,
                               MetalBufferHandle w, std::size_t w_off,
                               MetalBufferHandle y, std::size_t y_off,
                               LinearDims dims) {
    dispatch_gemv(impl_.get(), "linear_f16_gemv", x, x_off, w, w_off, y, y_off, dims);
}

void MetalOps::linear_silu_f16_gemv(MetalBufferHandle x, std::size_t x_off,
                                    MetalBufferHandle w, std::size_t w_off,
                                    MetalBufferHandle y, std::size_t y_off,
                                    LinearDims dims) {
    dispatch_gemv(impl_.get(), "linear_silu_f16_gemv", x, x_off, w, w_off, y, y_off, dims);
}

void MetalOps::rmsnorm_linear_f16_gemv(MetalBufferHandle x,      std::size_t x_off,
                                       MetalBufferHandle w_norm, std::size_t w_norm_off,
                                       MetalBufferHandle w,      std::size_t w_off,
                                       MetalBufferHandle y,      std::size_t y_off,
                                       RMSLinearDims dims) {
    if (dims.K % 4 != 0) {
        throw std::runtime_error("rmsnorm_linear_f16_gemv: K must be a multiple of 4");
    }
    struct Params { uint32_t N; uint32_t K; float eps; } p{dims.N, dims.K, dims.eps};

    run("rmsnorm_linear_f16_gemv", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x,      x_off,      0);
        bind_buf(enc, w_norm, w_norm_off, 1);
        bind_buf(enc, w,      w_off,      2);
        bind_buf(enc, y,      y_off,      3);
        [enc setBytes:&p length:sizeof(p) atIndex:4];

        // tg memory for staged-normalized x: K * sizeof(half).
        [enc setThreadgroupMemoryLength:dims.K * sizeof(uint16_t) atIndex:0];

        // 256 threads (8 simdgroups). Threadgroups: 1 (kernel loops over N).
        const NSUInteger tg_size = 256;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  Linear gemm (prefill)
// ---------------------------------------------------------------------------
namespace {

void dispatch_gemm(MetalOps::Impl* impl, const char* kernel_name,
                   MetalBufferHandle x, std::size_t x_off,
                   MetalBufferHandle w, std::size_t w_off,
                   MetalBufferHandle y, std::size_t y_off,
                   LinearDims dims) {
    struct Params { uint32_t M, N, K; } p{dims.M, dims.N, dims.K};
    run_impl(impl, kernel_name, [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x, x_off, 0);
        bind_buf(enc, w, w_off, 1);
        bind_buf(enc, y, y_off, 2);
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        const NSUInteger BM = 16, BN = 128, TG = 256;   // Phase 18 final (post-MPS revert)
        const NSUInteger gx = (dims.N + BN - 1) / BN;
        const NSUInteger gy = (dims.M + BM - 1) / BM;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
    });
}

}  // namespace

void MetalOps::linear_f16_gemm(MetalBufferHandle x, std::size_t x_off,
                               MetalBufferHandle w, std::size_t w_off,
                               MetalBufferHandle y, std::size_t y_off,
                               LinearDims dims) {
    dispatch_gemm(impl_.get(), "linear_f16_gemm", x, x_off, w, w_off, y, y_off, dims);
}

void MetalOps::linear_silu_f16_gemm(MetalBufferHandle x, std::size_t x_off,
                                    MetalBufferHandle w, std::size_t w_off,
                                    MetalBufferHandle y, std::size_t y_off,
                                    LinearDims dims) {
    dispatch_gemm(impl_.get(), "linear_silu_f16_gemm", x, x_off, w, w_off, y, y_off, dims);
}

// ---------------------------------------------------------------------------
//  INT4_BLOCK32 dispatchers
// ---------------------------------------------------------------------------
void MetalOps::linear_int4_gemv(MetalBufferHandle x,     std::size_t x_off,
                                MetalBufferHandle w,     std::size_t w_packed_off,
                                MetalBufferHandle w_scl, std::size_t w_scales_off,
                                MetalBufferHandle y,     std::size_t y_off,
                                LinearDims dims) {
    if (dims.M != 1) throw std::runtime_error("linear_int4_gemv: M must be 1");
    if (dims.K % 32 != 0) throw std::runtime_error("linear_int4_gemv: K must be a multiple of 32");
    struct Params { uint32_t M, N, K; } p{dims.M, dims.N, dims.K};

    run("linear_int4_gemv", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x,     x_off,        0);
        bind_buf(enc, w,     w_packed_off, 1);
        bind_buf(enc, w_scl, w_scales_off, 2);
        bind_buf(enc, y,     y_off,        3);
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        const NSUInteger rows_per_tg = 8;
        const NSUInteger tg_size     = 32 * rows_per_tg;
        const NSUInteger groups      = (dims.N + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    });
}

void MetalOps::linear_int4_gemm(MetalBufferHandle x,     std::size_t x_off,
                                MetalBufferHandle w,     std::size_t w_packed_off,
                                MetalBufferHandle w_scl, std::size_t w_scales_off,
                                MetalBufferHandle y,     std::size_t y_off,
                                LinearDims dims) {
    if (dims.K % 32 != 0) throw std::runtime_error("linear_int4_gemm: K must be a multiple of 32");
    struct Params { uint32_t M, N, K; } p{dims.M, dims.N, dims.K};

    run("linear_int4_gemm", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x,     x_off,        0);
        bind_buf(enc, w,     w_packed_off, 1);
        bind_buf(enc, w_scl, w_scales_off, 2);
        bind_buf(enc, y,     y_off,        3);
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        const NSUInteger BM = 32, BN = 32, TG = 128;
        const NSUInteger gx = (dims.N + BN - 1) / BN;
        const NSUInteger gy = (dims.M + BM - 1) / BM;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  Elementwise activations
// ---------------------------------------------------------------------------
void MetalOps::silu_f16(MetalBufferHandle x, std::size_t x_off,
                        MetalBufferHandle y, std::size_t y_off,
                        uint32_t n) {
    run("silu_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        bind_buf(enc, x, x_off, 0);
        bind_buf(enc, y, y_off, 1);
        [enc setBytes:&n length:sizeof(n) atIndex:2];

        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger groups = (n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

void MetalOps::add_inplace_f16(MetalBufferHandle y, std::size_t y_off,
                               MetalBufferHandle x, std::size_t x_off,
                               uint32_t n) {
    run("add_inplace_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        bind_buf(enc, y, y_off, 0);
        bind_buf(enc, x, x_off, 1);
        [enc setBytes:&n length:sizeof(n) atIndex:2];

        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger groups = (n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

void MetalOps::split_silu_xBC_f16(MetalBufferHandle xBC,   std::size_t xBC_off,
                                  MetalBufferHandle x_out, std::size_t x_off,
                                  MetalBufferHandle B_out, std::size_t B_off,
                                  MetalBufferHandle C_out, std::size_t C_off,
                                  uint32_t L, uint32_t d_inner, uint32_t d_state,
                                  uint32_t n_groups) {
    struct Params {
        uint32_t L; uint32_t d_inner; uint32_t d_state; uint32_t n_groups;
    } p{L, d_inner, d_state, n_groups};
    const uint32_t xBC_dim = d_inner + 2u * n_groups * d_state;

    run("split_silu_xBC_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, xBC,   xBC_off, 0);
        bind_buf(enc, x_out, x_off,   1);
        bind_buf(enc, B_out, B_off,   2);
        bind_buf(enc, C_out, C_off,   3);
        [enc setBytes:&p length:sizeof(p) atIndex:4];

        // 2D grid: (features, timesteps). Threadgroup 32x8 — fits Apple's
        // recommended max for 2D dispatch.
        [enc dispatchThreads:MTLSizeMake(xBC_dim, L, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    });
}

void MetalOps::silu_gated_f16(MetalBufferHandle x,    std::size_t x_off,
                              MetalBufferHandle gate, std::size_t gate_off,
                              MetalBufferHandle y,    std::size_t y_off,
                              uint32_t n) {
    run("silu_gated_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        bind_buf(enc, x,    x_off,    0);
        bind_buf(enc, gate, gate_off, 1);
        bind_buf(enc, y,    y_off,    2);
        [enc setBytes:&n length:sizeof(n) atIndex:3];

        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger groups = (n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  Conv1d (causal)
// ---------------------------------------------------------------------------
void MetalOps::causal_conv1d_f16(MetalBufferHandle x,    std::size_t x_off,
                                 MetalBufferHandle w,    std::size_t w_off,
                                 MetalBufferHandle bias, std::size_t bias_off,
                                 MetalBufferHandle y,    std::size_t y_off,
                                 Conv1dDims dims) {
    struct Params { uint32_t B, L, D, K; } p{dims.B, dims.L, dims.D, dims.K};

    run("causal_conv1d_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x,    x_off,    0);
        bind_buf(enc, w,    w_off,    1);
        bind_buf(enc, bias, bias_off, 2);
        bind_buf(enc, y,    y_off,    3);
        [enc setBytes:&p length:sizeof(p) atIndex:4];

        // Grid: (D, L, B). dispatchThreads counts threads (not groups), which
        // requires non-uniform threadgroup support — every Apple Silicon GPU
        // supports it.
        const NSUInteger tg = 64;
        [enc dispatchThreads:MTLSizeMake(dims.D, dims.L, dims.B)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

void MetalOps::seed_conv_state_f16(MetalBufferHandle xBC,   std::size_t xBC_off,
                                   MetalBufferHandle state, std::size_t state_off,
                                   uint32_t L, uint32_t D, uint32_t K) {
    struct Params { uint32_t B, L, D, K; } p{1u, L, D, K};
    run("seed_conv_state_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, xBC,   xBC_off,   0);
        bind_buf(enc, state, state_off, 1);
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(D, K, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    });
}

void MetalOps::causal_conv1d_update_f16(MetalBufferHandle x_new, std::size_t x_off,
                                        MetalBufferHandle state, std::size_t state_off,
                                        MetalBufferHandle w,     std::size_t w_off,
                                        MetalBufferHandle bias,  std::size_t bias_off,
                                        MetalBufferHandle y,     std::size_t y_off,
                                        Conv1dDims dims) {
    struct Params { uint32_t B, L, D, K; } p{dims.B, /*L=*/1u, dims.D, dims.K};

    run("causal_conv1d_update_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x_new, x_off,     0);
        bind_buf(enc, state, state_off, 1);
        bind_buf(enc, w,     w_off,     2);
        bind_buf(enc, bias,  bias_off,  3);
        bind_buf(enc, y,     y_off,     4);
        [enc setBytes:&p length:sizeof(p) atIndex:5];

        const NSUInteger tg = 64;
        [enc dispatchThreads:MTLSizeMake(dims.D, dims.B, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  Phase 20 — Fused in_proj + conv1d + silu_split
// ---------------------------------------------------------------------------
void MetalOps::inproj_fused_f16(MetalBufferHandle normed,   std::size_t normed_off,
                                MetalBufferHandle W_in,     std::size_t w_in_off,
                                MetalBufferHandle W_conv,   std::size_t w_conv_off,
                                MetalBufferHandle conv_b,   std::size_t conv_b_off,
                                MetalBufferHandle proj_z,   std::size_t z_off,
                                MetalBufferHandle ssd_x,    std::size_t x_off,
                                MetalBufferHandle ssd_B,    std::size_t b_off,
                                MetalBufferHandle ssd_C,    std::size_t c_off,
                                MetalBufferHandle proj_dt,  std::size_t dt_off,
                                MetalBufferHandle proj_xBC, std::size_t xBC_off,
                                FusedInProjDims dims) {
    struct Params {
        uint32_t L, d_model, d_inner, d_state, n_groups, n_heads, d_conv, proj_dim;
    } p{dims.L, dims.d_model, dims.d_inner, dims.d_state, dims.n_groups,
        dims.n_heads, dims.d_conv, dims.proj_dim};

    run("inproj_fused_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, normed,   normed_off, 0);
        bind_buf(enc, W_in,     w_in_off,   1);
        bind_buf(enc, W_conv,   w_conv_off, 2);
        bind_buf(enc, conv_b,   conv_b_off, 3);
        bind_buf(enc, proj_z,   z_off,      4);
        bind_buf(enc, ssd_x,    x_off,      5);
        bind_buf(enc, ssd_B,    b_off,      6);
        bind_buf(enc, ssd_C,    c_off,      7);
        bind_buf(enc, proj_dt,  dt_off,     8);
        [enc setBytes:&p length:sizeof(p) atIndex:9];
        bind_buf(enc, proj_xBC, xBC_off,    10);

        const NSUInteger TILE_N = 32, TG = 256;
        const NSUInteger groups = (dims.proj_dim + TILE_N - 1) / TILE_N;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  SSD prefill (chunked)
// ---------------------------------------------------------------------------
void MetalOps::ssd_chunked_f16(MetalBufferHandle X,        std::size_t X_off,
                               MetalBufferHandle B,        std::size_t B_off,
                               MetalBufferHandle C,        std::size_t C_off,
                               MetalBufferHandle dt_raw,   std::size_t dt_off,
                               MetalBufferHandle dt_bias,  std::size_t db_off,
                               MetalBufferHandle A_log,    std::size_t A_off,
                               MetalBufferHandle D_skip,   std::size_t D_off,
                               MetalBufferHandle Y,        std::size_t Y_off,
                               MetalBufferHandle state_io, std::size_t S_off,
                               SSDDims dims) {
    struct Params {
        uint32_t L, H, P, N, n_groups, chunk_size;
        float    dt_min, dt_max;
        uint32_t has_D;
    } p{dims.L, dims.H, dims.P, dims.N, dims.n_groups, dims.chunk_size,
        dims.dt_min, dims.dt_max, dims.has_D};

    run("ssd_chunked_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, X,        X_off,  0);
        bind_buf(enc, B,        B_off,  1);
        bind_buf(enc, C,        C_off,  2);
        bind_buf(enc, dt_raw,   dt_off, 3);
        bind_buf(enc, dt_bias,  db_off, 4);
        bind_buf(enc, A_log,    A_off,  5);
        bind_buf(enc, D_skip,   D_off,  6);
        bind_buf(enc, Y,        Y_off,  7);
        bind_buf(enc, state_io, S_off,  8);
        [enc setBytes:&p length:sizeof(p) atIndex:9];

        // One threadgroup per (head, batch). 128 threads is a reasonable
        // default — kernel internally distributes (P, N) work across threads.
        const NSUInteger tg = 128;
        [enc dispatchThreadgroups:MTLSizeMake(dims.H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    });
}

// ---------------------------------------------------------------------------
//  SSD decode step
// ---------------------------------------------------------------------------
void MetalOps::ssd_step_f16(MetalBufferHandle x,        std::size_t x_off,
                            MetalBufferHandle B,        std::size_t B_off,
                            MetalBufferHandle C,        std::size_t C_off,
                            MetalBufferHandle dt_raw,   std::size_t dt_off,
                            MetalBufferHandle dt_bias,  std::size_t db_off,
                            MetalBufferHandle A_log,    std::size_t A_off,
                            MetalBufferHandle D_skip,   std::size_t D_off,
                            MetalBufferHandle state,    std::size_t S_off,
                            MetalBufferHandle y,        std::size_t y_off,
                            SSDStepDims dims) {
    struct Params { uint32_t H, P, N, n_groups; float dt_min, dt_max; uint32_t has_D; }
        p{dims.H, dims.P, dims.N, dims.n_groups, dims.dt_min, dims.dt_max, dims.has_D};

    run("ssd_step_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
        (void)pso;
        bind_buf(enc, x,       x_off,  0);
        bind_buf(enc, B,       B_off,  1);
        bind_buf(enc, C,       C_off,  2);
        bind_buf(enc, dt_raw,  dt_off, 3);
        bind_buf(enc, dt_bias, db_off, 4);
        bind_buf(enc, A_log,   A_off,  5);
        bind_buf(enc, D_skip,  D_off,  6);
        bind_buf(enc, state,   S_off,  7);
        bind_buf(enc, y,       y_off,  8);
        [enc setBytes:&p length:sizeof(p) atIndex:9];

        // One threadgroup per head, P threads per group.
        [enc dispatchThreadgroups:MTLSizeMake(dims.H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(dims.P, 1, 1)];
    });
}

}  // namespace lite_ssm
