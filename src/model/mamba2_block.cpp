// src/model/mamba2_block.cpp — per-block forward + top-level driver.
//
// HF Mamba-2 block layout (matches `Mamba2Mixer.forward` in transformers):
//
//   residual = x
//   x      = RMSNorm(x, norm_w)
//   proj   = x @ in_proj_w.T                  # split into [z, xBC, dt]
//   xBC    = conv1d(proj_xBC) + bias          # depthwise causal, kernel d_conv
//   xBC    = SiLU(xBC)                         # in our split_silu fuse
//   x_ssm, B, C = split(xBC, [d_inner, d_state, d_state])
//   y      = SSD(x_ssm, B, C, dt, dt_bias, A_log, D)
//   y      = RMSNorm(y, mixer_norm_w)
//   y      = SiLU(z) * y                       # gated mixer output
//   y      = y @ out_proj_w.T
//   return residual + y
//
// In_proj is run as THREE separate matmuls (offsetting into the weight
// tensor's row dim) so each component lands in a contiguous workspace slot.
// This avoids a stride dance through interleaved [z, xBC, dt] columns.

#include "lite_ssm/model.hpp"

#include "lite_ssm/ops.hpp"
#include "lite_ssm/state.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace lite_ssm {

// ModelStash forward-decl & accessors mirror the ones in mamba2_model.cpp.
// We can't include each other's TU; the type just needs to lay out identically.
namespace {
struct ModelStash {
    void*               metal_buf;
    const std::uint8_t* cpu_base;
};
inline void* gpu_buf(void* p)            { return static_cast<ModelStash*>(p)->metal_buf; }
inline const std::uint8_t* cpu(void* p)  { return static_cast<ModelStash*>(p)->cpu_base; }

constexpr float DT_MIN = 1e-6f;   // effectively no clamp
constexpr float DT_MAX = 1e+6f;
constexpr float RMS_EPS = 1e-5f;  // HF Mamba2 default
}  // namespace

// ---------------------------------------------------------------------------
//  Block forward
// ---------------------------------------------------------------------------
void Mamba2Model::run_block(MetalOps& ops,
                            Mamba2Workspace& ws,
                            Mamba2State& state,
                            uint32_t layer_idx,
                            uint32_t L,
                            bool decode) const {
    const auto& cfg  = cfg_;
    const auto& bw   = blocks_[layer_idx];
    const auto& wsl  = ws.layout();

    const uint32_t d_model  = cfg.d_model;
    const uint32_t d_inner  = cfg.expand * cfg.d_model;
    const uint32_t d_state  = cfg.d_state;
    const uint32_t n_heads  = cfg.n_heads;
    const uint32_t n_groups = cfg.n_groups;
    const uint32_t d_head   = cfg.d_head;
    const uint32_t d_conv   = cfg.d_conv;
    const uint32_t xBC_dim  = d_inner + 2u * n_groups * d_state;

    void* W  = gpu_buf(weights_buffer_);
    void* WS = ws.metal_buffer();
    void* ST = state.metal_buffer();

    // In_proj weight layout (rows): [z (d_inner), xBC (xBC_dim), dt (n_heads)]
    const std::size_t inp_off    = bw.in_proj_w.offset_bytes;
    const std::size_t row_bytes  = std::size_t(d_model) * 2;
    const std::size_t W_z_off    = inp_off + 0;
    const std::size_t W_xBC_off  = inp_off + std::size_t(d_inner) * row_bytes;
    const std::size_t W_dt_off   = inp_off + std::size_t(d_inner + xBC_dim) * row_bytes;

    // ------------------------------------------------------------------
    // 1. RMSNorm(hidden, norm_w) -> normed
    // ------------------------------------------------------------------
    ops.rmsnorm_f16(WS, wsl.hidden_off,
                    W,  bw.norm_w.offset_bytes,
                    WS, wsl.normed_off,
                    {L, d_model, RMS_EPS});

    // ------------------------------------------------------------------
    // 2. Three in_proj matmuls -> proj_z, proj_xBC, proj_dt
    //
    // Phase 14 — the in_proj weight may be FP16 or INT4_BLOCK32. The three
    // slices share the SAME packing scheme (we either pack the whole row
    // range at quantize time or none of it), so a single dtype check picks
    // the pipeline for all three matmuls. Per int4 layout:
    //   packed nibbles start at in_proj_w.offset_bytes
    //   scales start at offset + (N_total * K / 2)
    // and each slice has its own row sub-range. The scale region for slice
    // [row0, row1) starts at scales_base + row0 * (K/32) * 2.
    // ------------------------------------------------------------------
    const bool ip_int4 = (bw.in_proj_w.dtype == DTYPE_INT4_B32);
    const uint32_t ip_total_N = static_cast<uint32_t>(bw.in_proj_w.shape[0]);
    const std::size_t ip_scales_base = bw.in_proj_w.offset_bytes
                                     + std::size_t(ip_total_N) * (d_model / 2);
    const std::size_t scale_row_bytes = std::size_t(d_model / 32) * 2;

    auto linear_in_proj = [&](uint32_t slice_row_start, uint32_t slice_N,
                              std::size_t x_off, std::size_t y_off) {
        if (ip_int4) {
            const std::size_t pack_off = bw.in_proj_w.offset_bytes
                                       + std::size_t(slice_row_start) * (d_model / 2);
            const std::size_t scale_off = ip_scales_base
                                        + std::size_t(slice_row_start) * scale_row_bytes;
            if (decode || L == 1) {
                ops.linear_int4_gemv(WS, x_off, W, pack_off, W, scale_off,
                                     WS, y_off, {1u, slice_N, d_model});
            } else {
                ops.linear_int4_gemm(WS, x_off, W, pack_off, W, scale_off,
                                     WS, y_off, {L, slice_N, d_model});
            }
        } else {
            const std::size_t W_off = bw.in_proj_w.offset_bytes
                                    + std::size_t(slice_row_start) * d_model * 2;
            if (decode || L == 1) {
                ops.linear_f16_gemv(WS, x_off, W, W_off, WS, y_off,
                                    {1u, slice_N, d_model});
            } else {
                ops.linear_f16_gemm(WS, x_off, W, W_off, WS, y_off,
                                    {L, slice_N, d_model});
            }
        }
    };
    linear_in_proj(0,                       d_inner, wsl.normed_off, wsl.proj_z_off);
    linear_in_proj(d_inner,                 xBC_dim, wsl.normed_off, wsl.proj_xBC_off);
    linear_in_proj(d_inner + xBC_dim,       n_heads, wsl.normed_off, wsl.proj_dt_off);

    // ------------------------------------------------------------------
    // 3. Conv1D over proj_xBC -> conv_out  (PREFILL or DECODE-update)
    //
    // For PREFILL we ALSO need to seed the conv state with the LAST d_conv
    // pre-conv samples so a subsequent decode picks up where we left off.
    // The seed is done CPU-side after the encoder is committed (we use the
    // mmap'd unified memory so the bytes are simultaneously visible).
    // ------------------------------------------------------------------
    if (decode || L == 1) {
        ops.causal_conv1d_update_f16(
            WS, wsl.proj_xBC_off,
            ST, state.conv_state_offset(layer_idx),
            W,  bw.conv1d_w.offset_bytes,
            W,  bw.conv1d_b.offset_bytes,
            WS, wsl.conv_out_off,
            {1u, 1u, xBC_dim, d_conv});
    } else {
        ops.causal_conv1d_f16(
            WS, wsl.proj_xBC_off,
            W,  bw.conv1d_w.offset_bytes,
            W,  bw.conv1d_b.offset_bytes,
            WS, wsl.conv_out_off,
            {1u, L, xBC_dim, d_conv});
    }

    // ------------------------------------------------------------------
    // 4. SiLU + split: conv_out -> ssd_x, ssd_B, ssd_C
    // ------------------------------------------------------------------
    ops.split_silu_xBC_f16(WS, wsl.conv_out_off,
                           WS, wsl.ssd_x_off,
                           WS, wsl.ssd_B_off,
                           WS, wsl.ssd_C_off,
                           L, d_inner, d_state, n_groups);

    // ------------------------------------------------------------------
    // 5. SSD (prefill = chunked / decode = single fused step)
    // ------------------------------------------------------------------
    if (decode || L == 1) {
        SSDStepDims dims{n_heads, d_head, d_state, n_groups,
                         DT_MIN, DT_MAX, /*has_D*/1u};
        ops.ssd_step_f16(WS, wsl.ssd_x_off,
                         WS, wsl.ssd_B_off,
                         WS, wsl.ssd_C_off,
                         WS, wsl.proj_dt_off,
                         W,  bw.dt_bias.offset_bytes,
                         W,  bw.A_log.offset_bytes,
                         W,  bw.D.offset_bytes,
                         ST, state.ssd_state_offset(layer_idx),
                         WS, wsl.ssd_y_off,
                         dims);
    } else {
        SSDDims dims{L, n_heads, d_head, d_state, n_groups,
                     std::min<uint32_t>(cfg.chunk_size, L),
                     DT_MIN, DT_MAX, /*has_D*/1u};
        ops.ssd_chunked_f16(WS, wsl.ssd_x_off,
                            WS, wsl.ssd_B_off,
                            WS, wsl.ssd_C_off,
                            WS, wsl.proj_dt_off,
                            W,  bw.dt_bias.offset_bytes,
                            W,  bw.A_log.offset_bytes,
                            W,  bw.D.offset_bytes,
                            WS, wsl.ssd_y_off,
                            ST, state.ssd_state_offset(layer_idx),
                            dims);
    }

    // ------------------------------------------------------------------
    // 6. gated mixer output (HF order: GATE THEN NORM):
    //    gated = ssd_y * silu(z)
    //    HF's Mamba2RMSNormGated does `y = y * silu(gate); y = rmsnorm(y, w)`.
    //    silu_gated_f16(x, gate) computes silu(x) * gate, so pass x=z, gate=y.
    // ------------------------------------------------------------------
    ops.silu_gated_f16(WS, wsl.proj_z_off,
                       WS, wsl.ssd_y_off,
                       WS, wsl.gated_off,
                       L * d_inner);

    // ------------------------------------------------------------------
    // 7. mixer RMSNorm(gated) -> ssd_normed
    // ------------------------------------------------------------------
    ops.rmsnorm_f16(WS, wsl.gated_off,
                    W,  bw.mixer_norm_w.offset_bytes,
                    WS, wsl.ssd_normed_off,
                    {L, d_inner, RMS_EPS});

    // ------------------------------------------------------------------
    // 8. out_proj: block_out = ssd_normed @ W_out_proj.T  (FP16 or INT4)
    // ------------------------------------------------------------------
    if (bw.out_proj_w.dtype == DTYPE_INT4_B32) {
        const uint32_t op_N = d_model, op_K = d_inner;
        const std::size_t pack_off = bw.out_proj_w.offset_bytes;
        const std::size_t scale_off = pack_off + std::size_t(op_N) * (op_K / 2);
        if (decode || L == 1) {
            ops.linear_int4_gemv(WS, wsl.ssd_normed_off,
                                 W, pack_off, W, scale_off,
                                 WS, wsl.block_out_off,
                                 {1u, op_N, op_K});
        } else {
            ops.linear_int4_gemm(WS, wsl.ssd_normed_off,
                                 W, pack_off, W, scale_off,
                                 WS, wsl.block_out_off,
                                 {L, op_N, op_K});
        }
    } else if (decode || L == 1) {
        ops.linear_f16_gemv(WS, wsl.ssd_normed_off,
                            W,  bw.out_proj_w.offset_bytes,
                            WS, wsl.block_out_off,
                            {1u, d_model, d_inner});
    } else {
        ops.linear_f16_gemm(WS, wsl.ssd_normed_off,
                            W,  bw.out_proj_w.offset_bytes,
                            WS, wsl.block_out_off,
                            {L, d_model, d_inner});
    }

    // ------------------------------------------------------------------
    // 9. Residual add: hidden += block_out  (in place at hidden_off)
    // ------------------------------------------------------------------
    ops.add_inplace_f16(WS, wsl.hidden_off,
                        WS, wsl.block_out_off,
                        L * d_model);
}

// ---------------------------------------------------------------------------
//  Top-level forward drivers
// ---------------------------------------------------------------------------

namespace {

// CPU helper: after a prefill kernel batch has been committed and waited on,
// seed each layer's conv_state with the LAST d_conv pre-conv xBC samples.
// Required so the next decode step's causal_conv1d_update_f16 sees the right
// window of past samples.
void seed_conv_state_from_prefill(Mamba2Workspace& ws,
                                  Mamba2State& state,
                                  const Mamba2Config& cfg,
                                  uint32_t layer_idx,
                                  uint32_t L) {
    // The block forward overwrites proj_xBC for every layer — so we must
    // call this RIGHT AFTER that layer's prefill, before the next layer
    // clobbers it. We work around the encoder batching by doing this
    // outside the batch (callers split the batch per layer for the seed
    // pass — see forward_prefill below).
    const uint32_t d_inner = cfg.expand * cfg.d_model;
    const uint32_t d_state = cfg.d_state;
    const uint32_t d_conv  = cfg.d_conv;
    const uint32_t xBC_dim = d_inner + 2 * d_state;

    const auto& wsl = ws.layout();
    const auto* xBC_base = static_cast<const std::uint8_t*>(ws.contents())
                         + wsl.proj_xBC_off;

    auto* state_base = static_cast<std::uint8_t*>(state.contents())
                     + state.conv_state_offset(layer_idx);

    // Conv state layout per layer: (B=1, xBC_dim, d_conv) fp16, row-major
    // where row stride = d_conv * 2. Element [d, k] lives at d*d_conv + k.
    // We seed state[d, k] = xBC[L - d_conv + k, d]  for k in [0, d_conv)
    // (the last d_conv timesteps' samples for channel d).
    for (uint32_t d = 0; d < xBC_dim; ++d) {
        for (uint32_t k = 0; k < d_conv; ++k) {
            const int32_t src_t = static_cast<int32_t>(L) - static_cast<int32_t>(d_conv) + k;
            std::uint16_t val_h = 0;
            if (src_t >= 0) {
                const auto* row = xBC_base + std::size_t(src_t) * xBC_dim * 2;
                std::memcpy(&val_h, row + d * 2, 2);
            }
            auto* slot = state_base + (std::size_t(d) * d_conv + k) * 2;
            std::memcpy(slot, &val_h, 2);
        }
    }
}

}  // namespace

void Mamba2Model::forward_prefill(MetalOps& ops,
                                  Mamba2Workspace& ws,
                                  Mamba2State& state,
                                  uint32_t L) const {
    if (L == 0) throw std::invalid_argument("forward_prefill: L must be >= 1");
    if (L > ws.max_seq_len()) {
        throw std::invalid_argument("forward_prefill: L exceeds workspace max_seq_len");
    }

    // Per layer we need to seed the conv_state CPU-side between layers (the
    // workspace's proj_xBC slot is overwritten by the next layer). So the
    // simplest correct path is to commit + wait per layer, then seed.
    // This negates some of the batching win but only inside the prefill
    // phase — decode steps still batch the whole forward into one buffer.
    for (uint32_t i = 0; i < cfg_.n_layer; ++i) {
        ops.begin_batch();
        run_block(ops, ws, state, /*layer_idx*/ i, L, /*decode*/ false);
        ops.commit_and_wait();
        seed_conv_state_from_prefill(ws, state, cfg_, i, L);
    }
}

void Mamba2Model::forward_step(MetalOps& ops,
                               Mamba2Workspace& ws,
                               Mamba2State& state) const {
    ops.begin_batch();
    for (uint32_t i = 0; i < cfg_.n_layer; ++i) {
        run_block(ops, ws, state, /*layer_idx*/ i, /*L*/ 1, /*decode*/ true);
    }
    ops.commit_and_wait();
    // No conv-state seeding needed — causal_conv1d_update_f16 advances the
    // window in place every step.
}

}  // namespace lite_ssm
