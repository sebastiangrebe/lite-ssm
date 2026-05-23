#pragma once
// lite_ssm/sampler.hpp — token sampling strategies.

#include <cstdint>
#include <vector>

namespace lite_ssm {

// Argmax over a fp32 logits vector. Smallest, fastest, deterministic.
std::uint32_t sample_greedy(const std::vector<float>& logits);

// Top-P (nucleus) sampling: sort by descending probability, keep the
// smallest set whose cumulative mass >= p, sample from it according to
// the renormalized probabilities. Deterministic given a seeded RNG.
std::uint32_t sample_top_p(const std::vector<float>& logits,
                           float top_p,
                           float temperature,
                           std::uint64_t& rng_state);

}  // namespace lite_ssm
