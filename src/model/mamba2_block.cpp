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
        // Phase 17 — the rewritten ssd_chunked_f16 stages a (CK × CK) M
        // matrix in threadgroup memory, which forces a hard cap on the
        // runtime chunk size. The config's `chunk_size` (256 for HF Mamba-2)
        // stays in the .ssm header for forward-compat; longer prefills
        // just iterate over more 64-token sub-chunks inside one kernel
        // launch (math unchanged).
        constexpr uint32_t SSD_KERNEL_MAX_CHUNK = 64;
        SSDDims dims{L, n_heads, d_head, d_state, n_groups,
                     std::min<uint32_t>({cfg.chunk_size, L, SSD_KERNEL_MAX_CHUNK}),
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
    // 6+7. Mamba2RMSNormGated — gate THEN norm.
    //
    // Phase 16 audit finding: HF's `MambaRMSNormGated.forward` hardcodes
    // gate-then-norm regardless of the `config.norm_before_gate` flag —
    // the flag is only honored on the fused training path. So our default
    // (gate-then-norm) is the right thing for BOTH 130M (parity passes)
    // and Codestral (HF's runtime ignores its own norm_before_gate=True
    // config). We still parse the flag for forward-compat with future
    // Mamba2 variants that might respect it, but don't branch on it.
    // ------------------------------------------------------------------
    (void)cfg.norm_before_gate;
    ops.silu_gated_f16(WS, wsl.proj_z_off,
                       WS, wsl.ssd_y_off,
                       WS, wsl.gated_off,
                       L * d_inner);
    ops.rmsnorm_f16(WS, wsl.gated_off,
                    W,  bw.mixer_norm_w.offset_bytes,
                    WS, wsl.ssd_normed_off,
                    {L, d_inner, RMS_EPS});
    const std::size_t mixer_out_off = wsl.ssd_normed_off;

    // ------------------------------------------------------------------
    // 8. out_proj: block_out = mixer_out @ W_out_proj.T  (FP16 or INT4)
    // ------------------------------------------------------------------
    if (bw.out_proj_w.dtype == DTYPE_INT4_B32) {
        const uint32_t op_N = d_model, op_K = d_inner;
        const std::size_t pack_off = bw.out_proj_w.offset_bytes;
        const std::size_t scale_off = pack_off + std::size_t(op_N) * (op_K / 2);
        if (decode || L == 1) {
            ops.linear_int4_gemv(WS, mixer_out_off,
                                 W, pack_off, W, scale_off,
                                 WS, wsl.block_out_off,
                                 {1u, op_N, op_K});
        } else {
            ops.linear_int4_gemm(WS, mixer_out_off,
                                 W, pack_off, W, scale_off,
                                 WS, wsl.block_out_off,
                                 {L, op_N, op_K});
        }
    } else if (decode || L == 1) {
        ops.linear_f16_gemv(WS, mixer_out_off,
                            W,  bw.out_proj_w.offset_bytes,
                            WS, wsl.block_out_off,
                            {1u, d_model, d_inner});
    } else {
        ops.linear_f16_gemm(WS, mixer_out_off,
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

    // ------------------------------------------------------------------
    // 10. Phase 16 — seed conv state from pre-conv xBC for the next decode
    //     step. This is now an in-batch kernel rather than a CPU memcpy,
    //     so forward_prefill commits all 64 layers in ONE command buffer.
    //     Skipped for the decode path (causal_conv1d_update_f16 advances
    //     the window in place).
    // ------------------------------------------------------------------
    if (!decode && L >= 1) {
        ops.seed_conv_state_f16(WS, wsl.proj_xBC_off,
                                ST, state.conv_state_offset(layer_idx),
                                L, xBC_dim, d_conv);
    }
}

// ---------------------------------------------------------------------------
//  Top-level forward drivers
// ---------------------------------------------------------------------------

void Mamba2Model::forward_prefill(MetalOps& ops,
                                  Mamba2Workspace& ws,
                                  Mamba2State& state,
                                  uint32_t L) const {
    if (L == 0) throw std::invalid_argument("forward_prefill: L must be >= 1");
    if (L > ws.max_seq_len()) {
        throw std::invalid_argument("forward_prefill: L exceeds workspace max_seq_len");
    }

    // Phase 16: conv state is now seeded by an in-kernel `seed_conv_state_f16`
    // call inside run_block, so the whole prefill batches into ONE command
    // buffer. Prior code committed + waited per layer (64 round-trips on
    // Codestral 7B) to do the seed CPU-side; that overhead was the dominant
    // cost of short prefills.
    ops.begin_batch();
    for (uint32_t i = 0; i < cfg_.n_layer; ++i) {
        run_block(ops, ws, state, /*layer_idx*/ i, L, /*decode*/ false);
    }
    ops.commit_and_wait();
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
