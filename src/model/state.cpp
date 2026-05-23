// src/model/state.cpp — recurrent state + activations arena layout.
//
// Sizing decisions (mamba2-130m at fp16, batch=1, max_seq_len=256):
//   conv1d state per layer  = batch * conv_dim * d_conv * 2 (fp16)
//                           = 1 * 1792 * 4 * 2 = ~14 KB
//   SSD state per layer     = batch * n_heads * d_head * d_state * 4 (fp32)
//                           = 1 * 24 * 64 * 128 * 4 = 768 KB
//   Workspace (per pass)    = ~5 MB peak across slots
//   lm_head logits (1 row)  = vocab * 2 = ~100 KB
//
// Total per-session footprint ~20 MB on top of the 335 MB weights map.

#include "lite_ssm/state.hpp"

#include <cstring>
#include <stdexcept>

namespace lite_ssm {

namespace {

constexpr std::size_t F16 = 2;
constexpr std::size_t F32 = 4;

}  // namespace

// ---------------------------------------------------------------------------
//  Mamba2State
// ---------------------------------------------------------------------------
Mamba2State::Mamba2State(const Mamba2Config& cfg, uint32_t batch)
    : batch_(batch), n_layer_(cfg.n_layer) {
    if (batch != 1) {
        throw std::runtime_error("Mamba2State: only batch=1 is supported in Phase 5");
    }

    // conv_dim = d_inner + 2 * n_groups * d_state. Codestral Mamba uses
    // n_groups=8; HF AntonV's port has n_groups=1.
    const std::size_t conv_dim = static_cast<std::size_t>(cfg.expand) * cfg.d_model
                               + 2u * cfg.n_groups * cfg.d_state;
    conv_layer_nbytes_ = batch_ * conv_dim * cfg.d_conv * F16;
    ssd_layer_nbytes_  = batch_ * cfg.n_heads * cfg.d_head * cfg.d_state * F32;
    conv_block_total_nbytes_ = n_layer_ * conv_layer_nbytes_;

    const std::size_t total = conv_block_total_nbytes_ + n_layer_ * ssd_layer_nbytes_;
    buf_ = DeviceBuffer(total);
    buf_.zero();
}

void Mamba2State::reset() {
    buf_.zero();
}

std::size_t Mamba2State::conv_state_offset(uint32_t layer) const {
    return static_cast<std::size_t>(layer) * conv_layer_nbytes_;
}

std::size_t Mamba2State::ssd_state_offset(uint32_t layer) const {
    return conv_block_total_nbytes_ + static_cast<std::size_t>(layer) * ssd_layer_nbytes_;
}

// ---------------------------------------------------------------------------
//  Mamba2Workspace
// ---------------------------------------------------------------------------
namespace {

// 64-byte alignment so every slot satisfies the GPU vector-load requirements
// (matches our .ssm tensor alignment policy).
std::size_t align_up(std::size_t v, std::size_t a) {
    return (v + a - 1) & ~(a - 1);
}

std::size_t aligned_slot(std::size_t& cursor, std::size_t nbytes) {
    cursor = align_up(cursor, 64);
    std::size_t off = cursor;
    cursor += nbytes;
    return off;
}

}  // namespace

Mamba2Workspace::Mamba2Workspace(const Mamba2Config& cfg, uint32_t max_seq_len)
    : max_seq_len_(max_seq_len) {

    const std::size_t L         = max_seq_len;
    const std::size_t d_model   = cfg.d_model;
    const std::size_t d_inner   = static_cast<std::size_t>(cfg.expand) * d_model;
    const std::size_t d_state   = cfg.d_state;
    const std::size_t n_heads   = cfg.n_heads;
    const std::size_t n_groups  = cfg.n_groups;
    const std::size_t vocab     = cfg.vocab_size;
    const std::size_t xBC_dim   = d_inner + 2 * n_groups * d_state;

    std::size_t cur = 0;

    layout_.hidden_off     = aligned_slot(cur, L * d_model * F16);
    layout_.normed_off     = aligned_slot(cur, L * d_model * F16);

    layout_.proj_z_off     = aligned_slot(cur, L * d_inner * F16);
    layout_.proj_xBC_off   = aligned_slot(cur, L * xBC_dim * F16);
    layout_.proj_dt_off    = aligned_slot(cur, L * n_heads * F16);

    layout_.conv_out_off   = aligned_slot(cur, L * xBC_dim * F16);

    layout_.ssd_x_off      = aligned_slot(cur, L * d_inner * F16);
    layout_.ssd_B_off      = aligned_slot(cur, L * n_groups * d_state * F16);
    layout_.ssd_C_off      = aligned_slot(cur, L * n_groups * d_state * F16);

    layout_.ssd_y_off      = aligned_slot(cur, L * d_inner * F16);
    layout_.ssd_normed_off = aligned_slot(cur, L * d_inner * F16);
    layout_.gated_off      = aligned_slot(cur, L * d_inner * F16);

    layout_.block_out_off  = aligned_slot(cur, L * d_model * F16);
    layout_.logits_off     = aligned_slot(cur, 1 * vocab * F16);

    layout_.total_nbytes   = cur;

    buf_ = DeviceBuffer(layout_.total_nbytes);
    // No need to zero — the forward pass overwrites every slot it reads.

    // Pre-allocate per-token fp32 logits scratch so the decode loop never
    // hits the heap (Phase 10 fix; previously this vector was created and
    // destroyed every step, leaking via ARC-adjacent allocator churn).
    logits_scratch_.assign(vocab, 0.0f);
}

}  // namespace lite_ssm
