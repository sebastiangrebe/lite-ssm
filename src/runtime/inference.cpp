// src/runtime/inference.cpp — high-level prefill + decode driver.

#include "lite_ssm/inference.hpp"

#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"

#include <stdexcept>

namespace lite_ssm {

std::vector<uint32_t> generate(
    Mamba2Model& model,
    MetalOps& ops,
    Mamba2Workspace& ws,
    Mamba2State& state,
    const std::vector<uint32_t>& prompt_tokens,
    const GenerationConfig& cfg,
    std::function<void(uint32_t)> on_token
) {
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("generate: prompt must be non-empty");
    }

    // ---- Prefill ---------------------------------------------------------
    state.reset();
    const uint32_t L = static_cast<uint32_t>(prompt_tokens.size());
    for (uint32_t i = 0; i < L; ++i) {
        model.embed(ws, /*row*/ i, prompt_tokens[i]);
    }
    model.forward_prefill(ops, ws, state, L);

    // Logits at the last prompt position give us our first generated token.
    model.compute_logits(ops, ws, /*hidden_row*/ L - 1);
    const auto& logits = model.read_logits_fp32(ws);

    std::vector<uint32_t> generated;
    generated.reserve(cfg.max_new_tokens);

    uint64_t rng = cfg.rng_seed ? cfg.rng_seed : 1u;

    auto pick = [&]() {
        return cfg.greedy
             ? sample_greedy(logits)
             : sample_top_p(logits, cfg.top_p, cfg.temperature, rng);
    };

    uint32_t next = pick();
    generated.push_back(next);
    if (on_token) on_token(next);

    // ---- Decode ----------------------------------------------------------
    for (uint32_t step = 1; step < cfg.max_new_tokens; ++step) {
        if (next == cfg.eos_token_id) break;
        model.embed(ws, /*row*/ 0, next);
        model.forward_step(ops, ws, state);
        model.compute_logits(ops, ws, /*hidden_row*/ 0);
        // `logits` is a const reference to ws.logits_scratch(); the call
        // refills that buffer in place so the existing binding still sees
        // fresh data — no reassignment needed.
        model.read_logits_fp32(ws);
        next = pick();
        generated.push_back(next);
        if (on_token) on_token(next);
    }

    return generated;
}

}  // namespace lite_ssm
