#pragma once
// lite_ssm/inference.hpp — high-level prefill + decode driver.

#include <cstdint>
#include <functional>
#include <vector>

namespace lite_ssm {

class Mamba2Model;
class Mamba2Workspace;
class Mamba2State;
class MetalOps;

struct GenerationConfig {
    uint32_t max_new_tokens = 64;
    bool     greedy         = true;
    float    top_p          = 0.9f;
    float    temperature    = 1.0f;
    uint64_t rng_seed       = 0xCAFEBABEull;
    uint32_t eos_token_id   = 0xFFFFFFFFu;   // disabled by default
};

// Run prefill on `prompt_tokens`, then decode up to `cfg.max_new_tokens`.
// Returns the generated token ids (excluding the prompt). Optional
// `on_token` callback fires per generated token (useful for streaming).
std::vector<uint32_t> generate(
    Mamba2Model& model,
    MetalOps& ops,
    Mamba2Workspace& ws,
    Mamba2State& state,
    const std::vector<uint32_t>& prompt_tokens,
    const GenerationConfig& cfg = {},
    std::function<void(uint32_t)> on_token = {}
);

}  // namespace lite_ssm
