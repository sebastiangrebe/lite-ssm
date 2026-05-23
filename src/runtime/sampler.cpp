// src/runtime/sampler.cpp — greedy + top-P samplers.

#include "lite_ssm/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lite_ssm {

std::uint32_t sample_greedy(const std::vector<float>& logits) {
    if (logits.empty()) throw std::invalid_argument("sample_greedy: empty logits");
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return static_cast<std::uint32_t>(best);
}

namespace {

// Cheap xorshift64 — deterministic, no dep on <random>.
inline std::uint64_t xorshift64(std::uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}
inline float urandom01(std::uint64_t& s) {
    return float(xorshift64(s) >> 40) / float(1u << 24);   // [0, 1)
}

}  // namespace

std::uint32_t sample_top_p(const std::vector<float>& logits,
                           float top_p,
                           float temperature,
                           std::uint64_t& rng_state) {
    if (logits.empty()) throw std::invalid_argument("sample_top_p: empty logits");
    const float t = (temperature > 1e-6f) ? temperature : 1.0f;

    // Argsort indices by descending logit.
    std::vector<std::uint32_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(),
              [&](auto a, auto b) { return logits[a] > logits[b]; });

    // Softmax over the sorted logits (numerically stable).
    float max_l = logits[idx[0]] / t;
    std::vector<float> probs(idx.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        float p = std::exp(logits[idx[i]] / t - max_l);
        probs[i] = p;
        sum += p;
    }
    for (auto& p : probs) p /= sum;

    // Cut off at cumulative mass >= top_p.
    float cum = 0.0f;
    std::size_t cutoff = probs.size();
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    // Renormalize within nucleus.
    float nucleus_sum = 0.0f;
    for (std::size_t i = 0; i < cutoff; ++i) nucleus_sum += probs[i];
    float r = urandom01(rng_state) * nucleus_sum;
    float acc = 0.0f;
    for (std::size_t i = 0; i < cutoff; ++i) {
        acc += probs[i];
        if (r <= acc) return idx[i];
    }
    return idx[cutoff - 1];
}

}  // namespace lite_ssm
