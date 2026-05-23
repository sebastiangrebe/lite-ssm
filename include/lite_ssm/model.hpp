#pragma once
// lite_ssm/model.hpp — model-level types.
//
// Mamba2Config  parsed hyperparameters from a .ssm file header.
// SSMFile       owns an mmap region + UnifiedBuffer + a name -> Tensor map.
//               This is what the inference loop binds against. The actual
//               Mamba2Model (forward pass) comes in Phase 5 and will hold
//               a reference to one of these.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lite_ssm/allocator.hpp"
#include "lite_ssm/tensor.hpp"

namespace lite_ssm {

struct Mamba2Config {
    uint32_t d_model       = 0;
    uint32_t n_layer       = 0;
    uint32_t d_state       = 0;
    uint32_t d_conv        = 0;
    uint32_t expand        = 0;
    uint32_t vocab_size    = 0;
    uint32_t n_heads       = 0;
    uint32_t d_head        = 0;
    uint32_t chunk_size    = 0;
    uint32_t n_groups      = 1;   // Phase 11 — B/C broadcast width
    DType    default_dtype = DTYPE_F16;
};

class SSMFile {
public:
    SSMFile() = default;
    explicit SSMFile(const std::string& path);

    SSMFile(const SSMFile&)            = delete;
    SSMFile& operator=(const SSMFile&) = delete;
    SSMFile(SSMFile&&)                 = default;
    SSMFile& operator=(SSMFile&&)      = default;

    const Mamba2Config&  config()         const { return config_; }
    std::size_t          num_tensors()    const { return tensors_.size(); }
    const auto&          tensors()        const { return tensors_; }

    const Tensor*        find(std::string_view name) const;
    const Tensor&        at(std::string_view name)   const;  // throws std::out_of_range
    bool                 contains(std::string_view name) const { return find(name) != nullptr; }

    const MmapFile&      mapping() const { return mapping_; }
    UnifiedBuffer&       buffer()        { return buffer_; }
    const UnifiedBuffer& buffer()  const { return buffer_; }

private:
    // Declaration order matters: mapping_ destructs LAST so the MTLBuffer
    // inside buffer_ is released before munmap runs. See allocator.hpp.
    MmapFile        mapping_;
    UnifiedBuffer   buffer_;
    Mamba2Config    config_;
    std::unordered_map<std::string, Tensor> tensors_;
};

// ---------------------------------------------------------------------------
//  Mamba2Model — forward pass driver.
//
// Constructs once per loaded SSMFile, holding shape constants and tensor
// handles for every weight. forward_prefill() / forward_step() encode the
// per-pass kernels into the supplied MetalOps batch. Caller owns the
// MetalOps, the Mamba2State, and the Mamba2Workspace lifetimes.
// ---------------------------------------------------------------------------
class MetalOps;
class Mamba2Workspace;
class Mamba2State;

struct Mamba2BlockWeights {
    Tensor norm_w;         // (d_model)
    Tensor in_proj_w;      // (3352, d_model) — split into z/xBC/dt slices
    Tensor conv1d_w;       // (xBC_dim, 1, d_conv)
    Tensor conv1d_b;       // (xBC_dim)
    Tensor dt_bias;        // (n_heads)
    Tensor A_log;          // (n_heads)
    Tensor D;              // (n_heads)
    Tensor mixer_norm_w;   // (d_inner)
    Tensor out_proj_w;     // (d_model, d_inner)
};

class Mamba2Model {
public:
    Mamba2Model() = default;
    explicit Mamba2Model(const SSMFile& weights);

    const Mamba2Config& config() const { return cfg_; }
    uint32_t            n_layer() const { return cfg_.n_layer; }

    // Look up the embedding row for `token_id` and copy d_model halves into
    // workspace.hidden_off + (row * d_model * 2). Pure CPU (unified memory).
    void embed(Mamba2Workspace& ws, uint32_t row, uint32_t token_id) const;

    // Prefill: L >= 1 tokens already embedded at hidden_off. Updates state_
    // in-place. After this returns, hidden_off contains post-block hidden
    // states for every position (the FINAL block's residual stream).
    void forward_prefill(MetalOps& ops,
                         Mamba2Workspace& ws,
                         Mamba2State& state,
                         uint32_t L) const;

    // Decode step: one token already embedded at hidden_off (single row).
    void forward_step(MetalOps& ops,
                      Mamba2Workspace& ws,
                      Mamba2State& state) const;

    // After forward_*, apply norm_f + lm_head to the row at `hidden_row` and
    // leave (vocab) fp16 logits at workspace.logits_off. Used at the end of
    // prefill (on the last token) and once per decode step.
    void compute_logits(MetalOps& ops,
                        Mamba2Workspace& ws,
                        uint32_t hidden_row) const;

    // Materialize the fp16 logits at workspace.logits_off into the
    // workspace's pre-allocated fp32 scratch vector. Returns a const
    // reference to that vector so the caller can read without copying.
    // The scratch is reused for the lifetime of the workspace — zero
    // heap allocations per decode step.
    const std::vector<float>& read_logits_fp32(Mamba2Workspace& ws) const;

private:
    void run_block(MetalOps& ops,
                   Mamba2Workspace& ws,
                   Mamba2State& state,
                   uint32_t layer_idx,
                   uint32_t L,
                   bool decode) const;

    Mamba2Config                       cfg_;
    Tensor                             embed_w_;
    Tensor                             norm_f_w_;
    Tensor                             lm_head_w_;   // may equal embed_w_ when tied
    std::vector<Mamba2BlockWeights>    blocks_;
    void*                              weights_buffer_ = nullptr;   // id<MTLBuffer>
};

}  // namespace lite_ssm
