#pragma once
// lite_ssm/state.hpp — recurrent state + per-block activations arena.
//
// Two distinct lifetimes:
//
//   Mamba2State      One per generation session. Holds the per-layer
//                    conv1d rolling window (fp16) and per-layer SSD
//                    hidden state (fp32). Persists across the prefill →
//                    decode transition and across every decode step.
//
//   Mamba2Workspace  Pre-sized scratch arena for one forward pass.
//                    Named offsets for every intermediate slot so the
//                    block forward can encode kernels without ever
//                    allocating mid-pass. Slots are sized for the worst
//                    case max_seq_len; decode (L=1) just uses smaller
//                    sub-regions.

#include <cstddef>
#include <cstdint>

#include <vector>

#include "lite_ssm/allocator.hpp"
#include "lite_ssm/model.hpp"

namespace lite_ssm {

class Mamba2State {
public:
    Mamba2State() = default;
    explicit Mamba2State(const Mamba2Config& cfg, uint32_t batch = 1);

    void reset();   // zero conv + SSD state — start of a fresh prompt

    void*       metal_buffer() const { return buf_.metal_buffer(); }
    void*       contents()     const { return buf_.contents(); }
    std::size_t length()       const { return buf_.length(); }

    // Layout: [conv_state_layer_0, conv_state_layer_1, ..., conv_state_layer_{N-1},
    //          ssd_state_layer_0, ssd_state_layer_1, ..., ssd_state_layer_{N-1}]
    std::size_t conv_state_offset(uint32_t layer) const;
    std::size_t conv_state_nbytes_per_layer() const { return conv_layer_nbytes_; }

    std::size_t ssd_state_offset(uint32_t layer) const;
    std::size_t ssd_state_nbytes_per_layer() const { return ssd_layer_nbytes_; }

    uint32_t batch() const { return batch_; }

private:
    DeviceBuffer buf_;
    uint32_t     batch_                   = 1;
    uint32_t     n_layer_                 = 0;
    std::size_t  conv_layer_nbytes_       = 0;
    std::size_t  ssd_layer_nbytes_        = 0;
    std::size_t  conv_block_total_nbytes_ = 0;
};

// Workspace slot offsets. All offsets are in bytes from the workspace's
// MTLBuffer base. Slots are sized for `max_seq_len`; pass L<=max_seq_len
// to any kernel and the rest of the slot is ignored.
//
// The in_proj output is interleaved per row in PyTorch's layout
// ([z, xBC, dt] back-to-back along the feature dim), but our downstream
// kernels want contiguous (L, ...) tensors. We fix this by running in_proj
// as three separate matmuls — sliced over the WEIGHT's leading dim, which
// IS contiguous on disk — so each output lands in its own slot.
struct WorkspaceLayout {
    std::size_t hidden_off    = 0;   // (L, d_model) fp16  — residual carrier, also block input
    std::size_t normed_off    = 0;   // (L, d_model) fp16

    std::size_t proj_z_off    = 0;   // (L, d_inner) fp16  — z (gate) projection output
    std::size_t proj_xBC_off  = 0;   // (L, xBC_dim) fp16  — xBC projection output
    std::size_t proj_dt_off   = 0;   // (L, n_heads) fp16  — dt projection output

    std::size_t conv_out_off  = 0;   // (L, xBC_dim) fp16  — conv1d output, pre-SiLU

    std::size_t ssd_x_off     = 0;   // (L, d_inner) fp16  — SiLU(conv_out)[:,:d_inner], SSD X
    std::size_t ssd_B_off     = 0;   // (L, d_state) fp16  — SiLU(conv_out)[:,d_inner:d_inner+N]
    std::size_t ssd_C_off     = 0;   // (L, d_state) fp16  — SiLU(conv_out)[:,...]

    std::size_t ssd_y_off     = 0;   // (L, d_inner) fp16  — SSD output (reshape (L,H,P)->(L,d_inner))
    std::size_t ssd_normed_off= 0;   // (L, d_inner) fp16  — mixer RMSNorm
    std::size_t gated_off     = 0;   // (L, d_inner) fp16  — silu(z) * ssd_normed

    std::size_t block_out_off = 0;   // (L, d_model) fp16  — out_proj output
    std::size_t logits_off    = 0;   // (1, vocab)   fp16  — last-token logits only

    std::size_t total_nbytes  = 0;
};

class Mamba2Workspace {
public:
    Mamba2Workspace() = default;
    Mamba2Workspace(const Mamba2Config& cfg, uint32_t max_seq_len);

    void*       metal_buffer() const { return buf_.metal_buffer(); }
    void*       contents()     const { return buf_.contents(); }
    std::size_t length()       const { return buf_.length(); }

    const WorkspaceLayout& layout()       const { return layout_; }
    uint32_t               max_seq_len()  const { return max_seq_len_; }

    // Pre-allocated fp32 logits scratch — sized for one vocab row. The
    // decode loop fills this every step via Mamba2Model::read_logits_fp32(),
    // never re-allocating. Owning it here is the Phase 10 fix for the
    // per-token ~200 KB churn the bleed test caught.
    std::vector<float>&       logits_scratch()       { return logits_scratch_; }
    const std::vector<float>& logits_scratch() const { return logits_scratch_; }

private:
    DeviceBuffer       buf_;
    WorkspaceLayout    layout_;
    uint32_t           max_seq_len_ = 0;
    std::vector<float> logits_scratch_;
};

}  // namespace lite_ssm
