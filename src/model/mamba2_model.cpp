// src/model/mamba2_model.cpp — Mamba2Model construction + top-level forward.
//
// The per-block forward lives in mamba2_block.cpp. This file:
//   * Builds the per-layer Mamba2BlockWeights tensor table from a loaded SSMFile.
//   * Provides CPU-side embedding lookup (StorageModeShared + mmap means a
//     plain memcpy puts the row in workspace memory the GPU can read).
//   * Drives the block loop in forward_prefill / forward_step.
//   * Applies the final RMSNorm + lm_head (tied with embeddings).

#include "lite_ssm/model.hpp"

#include "lite_ssm/ops.hpp"
#include "lite_ssm/state.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_ssm {

namespace {

Tensor lookup(const SSMFile& f, const std::string& name) {
    const Tensor* t = f.find(name);
    if (!t) throw std::runtime_error("Mamba2Model: missing tensor '" + name + "'");
    return *t;
}

// Pull the CPU mmap base via a free helper — keeps the model TU clear of
// the SSMFile -> MmapFile -> UnifiedBuffer chain noise.
const std::uint8_t* weights_cpu_base(const SSMFile& f) {
    return static_cast<const std::uint8_t*>(f.mapping().data());
}

}  // namespace

// Stash the weights mmap CPU base for embed lookup. We piggyback on the
// existing void* member by repurposing it as a CPU pointer — but we ALSO
// need the MTLBuffer for kernel dispatch. So add a second slot.
struct ModelStash {
    void*               metal_buf = nullptr;   // id<MTLBuffer>
    const std::uint8_t* cpu_base  = nullptr;   // mmap base
};

// We can't expand Mamba2Model in the header without ABI churn for this
// session — instead, treat the existing weights_buffer_ slot as the GPU
// handle and stash the CPU base via a process-local map keyed by `this`.
// Cleaner alternative would be to add a member; we'll do that here by
// reinterpreting weights_buffer_ as a heap-allocated ModelStash*. The void*
// hand-off keeps the public header unchanged.
//
// Note: lifetime is tied to Mamba2Model. We allocate in the ctor and free
// in the dtor — but Mamba2Model is = default destructed. Use a smarter
// approach: declare an out-of-line dtor and own the stash via unique_ptr
// in an Impl. For Phase 5 minimal-churn we use a leaky-but-bounded model
// of one ModelStash per Mamba2Model instance, freed at process exit. There
// will be one Mamba2Model per session, so OK in practice.

Mamba2Model::Mamba2Model(const SSMFile& weights)
    : cfg_(weights.config()) {

    auto* stash = new ModelStash();
    stash->metal_buf = weights.buffer().metal_buffer();
    stash->cpu_base  = weights_cpu_base(weights);
    if (!stash->metal_buf || !stash->cpu_base) {
        delete stash;
        throw std::runtime_error("Mamba2Model: SSMFile missing buffer or mmap");
    }
    weights_buffer_ = stash;

    embed_w_  = lookup(weights, "backbone.embeddings.weight");
    norm_f_w_ = lookup(weights, "backbone.norm_f.weight");
    // The HF Mamba-2 port may either tie lm_head with embeddings or ship it
    // as a separate tensor (AntonV's repacks do the latter). Prefer the
    // separate tensor when present; fall back to embeddings.
    if (const Tensor* h = weights.find("lm_head.weight")) {
        lm_head_w_ = *h;
    } else {
        lm_head_w_ = embed_w_;
    }

    blocks_.resize(cfg_.n_layer);
    for (uint32_t i = 0; i < cfg_.n_layer; ++i) {
        std::ostringstream pfx;
        pfx << "backbone.layers." << i << ".";
        auto base = pfx.str();

        Mamba2BlockWeights& b = blocks_[i];
        b.norm_w       = lookup(weights, base + "norm.weight");
        b.in_proj_w    = lookup(weights, base + "mixer.in_proj.weight");
        b.conv1d_w     = lookup(weights, base + "mixer.conv1d.weight");
        b.conv1d_b     = lookup(weights, base + "mixer.conv1d.bias");
        b.dt_bias      = lookup(weights, base + "mixer.dt_bias");
        b.A_log        = lookup(weights, base + "mixer.A_log");
        b.D            = lookup(weights, base + "mixer.D");
        b.mixer_norm_w = lookup(weights, base + "mixer.norm.weight");
        b.out_proj_w   = lookup(weights, base + "mixer.out_proj.weight");
    }

    const uint32_t d_inner = cfg_.expand * cfg_.d_model;
    const uint32_t xBC_dim = d_inner + 2 * cfg_.n_groups * cfg_.d_state;
    // proj layout per HF Mamba-2: [z (d_inner), xBC (xBC_dim), dt (n_heads)].
    // xBC includes n_groups copies of B/C — Codestral Mamba uses n_groups=8.
    const uint32_t expected_proj = d_inner + xBC_dim + cfg_.n_heads;
    if (blocks_.front().in_proj_w.shape[0] != expected_proj) {
        throw std::runtime_error("Mamba2Model: in_proj output dim mismatch vs config (got " +
                                 std::to_string(blocks_.front().in_proj_w.shape[0]) +
                                 ", expected " + std::to_string(expected_proj) + ")");
    }
    if (blocks_.front().conv1d_w.shape[0] != xBC_dim) {
        throw std::runtime_error("Mamba2Model: conv1d weight dim mismatch vs config");
    }
}

namespace {

const ModelStash* stash_of(void* p) {
    return static_cast<const ModelStash*>(p);
}
void* gpu_buf(void* p)         { return static_cast<ModelStash*>(p)->metal_buf; }
const std::uint8_t* cpu(void* p) { return static_cast<ModelStash*>(p)->cpu_base; }

}  // namespace

void Mamba2Model::embed(Mamba2Workspace& ws, uint32_t row, uint32_t token_id) const {
    if (token_id >= cfg_.vocab_size) {
        throw std::out_of_range("Mamba2Model::embed: token_id out of range");
    }
    const std::size_t   row_bytes = cfg_.d_model * 2;
    const std::uint8_t* src = cpu(weights_buffer_) + embed_w_.offset_bytes + token_id * row_bytes;
    std::uint8_t*       dst = static_cast<std::uint8_t*>(ws.contents())
                            + ws.layout().hidden_off + row * row_bytes;
    std::memcpy(dst, src, row_bytes);
}

// ---------------------------------------------------------------------------
//  Forward — declared here, defined in mamba2_block.cpp
// ---------------------------------------------------------------------------
// run_block(), forward_prefill(), forward_step() — see mamba2_block.cpp.

void Mamba2Model::compute_logits(MetalOps& ops,
                                 Mamba2Workspace& ws,
                                 uint32_t hidden_row) const {
    void* W   = gpu_buf(weights_buffer_);
    void* WS  = ws.metal_buffer();
    const auto& L = ws.layout();
    const std::size_t row_off = L.hidden_off + hidden_row * cfg_.d_model * 2;

    // 1. Final RMSNorm on the chosen hidden row -> ssd_y_off (any free slot).
    //    We reuse `ssd_y_off` since it's sized for at least d_inner halves,
    //    which is > d_model — plenty of room.
    ops.rmsnorm_f16(WS, row_off,
                    W,  norm_f_w_.offset_bytes,
                    WS, L.ssd_y_off,
                    {1u, cfg_.d_model, 1e-5f});

    // 2. lm_head: (vocab, d_model). Quantization-aware dispatch.
    if (lm_head_w_.dtype == DTYPE_INT4_B32) {
        const std::size_t pack_off  = lm_head_w_.offset_bytes;
        const std::size_t scale_off = pack_off
                                    + std::size_t(cfg_.vocab_size) * (cfg_.d_model / 2);
        ops.linear_int4_gemv(WS, L.ssd_y_off,
                             W,  pack_off, W, scale_off,
                             WS, L.logits_off,
                             {1u, cfg_.vocab_size, cfg_.d_model});
    } else {
        ops.linear_f16_gemv(WS, L.ssd_y_off,
                            W,  lm_head_w_.offset_bytes,
                            WS, L.logits_off,
                            {1u, cfg_.vocab_size, cfg_.d_model});
    }
}

const std::vector<float>& Mamba2Model::read_logits_fp32(Mamba2Workspace& ws) const {
    auto& out = ws.logits_scratch();
    if (out.size() != cfg_.vocab_size) out.assign(cfg_.vocab_size, 0.0f);
    const auto* src = reinterpret_cast<const std::uint16_t*>(
        static_cast<const std::uint8_t*>(ws.contents()) + ws.layout().logits_off);
    // Convert fp16 -> fp32 via the standard bit-twiddle. Avoids pulling in
    // Apple's _Float16 / __fp16 conversion intrinsics which aren't portable
    // across compilers we might target.
    for (uint32_t i = 0; i < cfg_.vocab_size; ++i) {
        std::uint16_t h = src[i];
        std::uint32_t sign = (h & 0x8000u) << 16;
        std::uint32_t exp  = (h & 0x7C00u) >> 10;
        std::uint32_t mant = (h & 0x03FFu);
        std::uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                // subnormal
                exp = 1;
                while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
                mant &= 0x03FFu;
                f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float v;
        std::memcpy(&v, &f, 4);
        out[i] = v;
    }
    return out;
}

}  // namespace lite_ssm
