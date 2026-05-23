// tests/stress/test_memory_bleed.cpp
//
// Endurance / leak probe.
//
// Runs an infinite_telemetry loop — prefill a synthetic log line, decode 5
// tokens, repeat. Samples the live process RSS via mach_task_basic_info()
// every 100 iterations. Fails if RSS climbs more than 1 MB above the
// post-warmup baseline after 5000 generated tokens.
//
// Exits:
//   0  RSS held flat (no leak) — pass
//   1  RSS grew beyond budget   — fatal leak warning, fail
//   2  configuration / setup error
//
// Not registered with CTest — long-running by design.
// Run manually: `build/test_memory_bleed [model.ssm] [iters_cap]`

#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"
#include "lite_ssm/tokenizer.hpp"

#include <mach/mach.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using clk = std::chrono::high_resolution_clock;

namespace {

const std::array<const char*, 6> LOG_LINES{
    "[INFO] Server heartbeat OK, uptime 12345s",
    "[WARN] Connection retry to db: timeout 5s",
    "[ERROR] Authentication failed for user admin",
    "[INFO] Cache hit ratio 0.943 over last 60s",
    "[WARN] Disk usage at 87% on /var, alerting",
    "[INFO] Request /api/v1/health 200 in 4ms",
};

constexpr uint32_t DECODE_TOKENS_PER_ITER = 5;
constexpr uint32_t TOKEN_BUDGET           = 5000;
constexpr uint32_t SAMPLE_EVERY_ITERS     = 100;
constexpr uint32_t BASELINE_AT_ITER       = 5;     // after warm-up + page-in
// Default 1 MB per spec; can be overridden via $LITE_SSM_BLEED_THRESHOLD_MB
// for diagnostic runs that want the full growth curve past the breach.
constexpr std::size_t LEAK_THRESHOLD_DEFAULT = 1ULL * 1024ULL * 1024ULL;

std::size_t leak_budget_default() {
    if (const char* env = std::getenv("LITE_SSM_BLEED_THRESHOLD_MB")) {
        return static_cast<std::size_t>(std::atoll(env)) * 1024ULL * 1024ULL;
    }
    return LEAK_THRESHOLD_DEFAULT;
}

std::size_t rss_bytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(),
                                 MACH_TASK_BASIC_INFO,
                                 reinterpret_cast<task_info_t>(&info),
                                 &count);
    if (kr != KERN_SUCCESS) return 0;
    return static_cast<std::size_t>(info.resident_size);
}

std::string format_mb(std::size_t b) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MB", b / 1024.0 / 1024.0);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "model.ssm";
    const uint32_t    cap_iters  = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 0;

    if (!std::filesystem::exists(model_path)) {
        std::fprintf(stderr, "test_memory_bleed: %s not found\n", model_path.c_str());
        return 2;
    }

    lite_ssm::SSMFile      weights(model_path);
    lite_ssm::Mamba2Model  model(weights);
    lite_ssm::MetalOps     ops;
    if (!ops.metallib_loaded()) {
        std::fprintf(stderr, "test_memory_bleed: metallib not built\n");
        return 2;
    }

    uint32_t max_L = model.config().chunk_size;
    for (auto* s : LOG_LINES) max_L = std::max<uint32_t>(max_L, std::strlen(s));
    lite_ssm::Mamba2Workspace ws(model.config(), max_L);
    lite_ssm::Mamba2State     state(model.config());

    // Tokenizer is optional — if absent we use byte-fallback so the test
    // runs even on a fresh checkout.
    lite_ssm::Tokenizer tok;
    bool have_bpe = std::filesystem::exists("tokenizer.model");
    if (have_bpe) tok.load("tokenizer.model");
    auto encode = [&](const char* text) {
        std::vector<uint32_t> out;
        if (have_bpe) return tok.encode(text);
        for (const char* p = text; *p; ++p) out.push_back(static_cast<uint8_t>(*p));
        return out;
    };

    std::printf("[bleed] model=%s  tokenizer=%s  decode_per_iter=%u  token_budget=%u\n",
                model_path.c_str(), have_bpe ? "bpe" : "byte-fallback",
                DECODE_TOKENS_PER_ITER, TOKEN_BUDGET);
    const std::size_t leak_budget = leak_budget_default();
    std::printf("[bleed] baseline taken at iter %u; leak threshold = %s above baseline\n\n",
                BASELINE_AT_ITER, format_mb(leak_budget).c_str());
    std::printf("%6s  %8s  %12s  %12s  %12s\n",
                "iter", "tokens", "rss", "Δ_baseline", "tok/s_recent");
    std::printf("%6s  %8s  %12s  %12s  %12s\n",
                "----", "------", "---", "----------", "------------");

    std::size_t baseline = 0;
    std::size_t peak     = 0;
    uint32_t    tokens_total = 0;
    uint32_t    iter         = 0;
    auto recent_t0           = clk::now();
    uint32_t    recent_tokens = 0;

    while (true) {
        const char* log = LOG_LINES[iter % LOG_LINES.size()];
        auto tokens = encode(log);
        const uint32_t L = static_cast<uint32_t>(tokens.size());

        state.reset();
        for (uint32_t k = 0; k < L; ++k) model.embed(ws, k, tokens[k]);

        model.forward_prefill(ops, ws, state, L);
        model.compute_logits(ops, ws, L - 1);
        const auto& logits = model.read_logits_fp32(ws);
        uint32_t next = lite_ssm::sample_greedy(logits);
        tokens_total += 1;
        recent_tokens += 1;

        for (uint32_t step = 1; step < DECODE_TOKENS_PER_ITER; ++step) {
            model.embed(ws, 0, next);
            model.forward_step(ops, ws, state);
            model.compute_logits(ops, ws, 0);
            model.read_logits_fp32(ws);
            next = lite_ssm::sample_greedy(logits);
            tokens_total += 1;
            recent_tokens += 1;
        }
        ++iter;

        if (iter == BASELINE_AT_ITER) {
            baseline = rss_bytes();
            peak     = baseline;
        }
        if (iter % SAMPLE_EVERY_ITERS == 0 || iter == BASELINE_AT_ITER) {
            std::size_t rss = rss_bytes();
            if (rss > peak) peak = rss;
            const std::ptrdiff_t delta = (baseline ? static_cast<std::ptrdiff_t>(rss) -
                                                     static_cast<std::ptrdiff_t>(baseline)
                                                   : 0);
            const double recent_ms = std::chrono::duration<double, std::milli>(
                                          clk::now() - recent_t0).count();
            const double recent_tps = recent_tokens * 1000.0 / recent_ms;
            std::printf("%6u  %8u  %12s  %+12.2f MB  %12.1f\n",
                        iter, tokens_total, format_mb(rss).c_str(),
                        delta / 1024.0 / 1024.0, recent_tps);
            recent_t0 = clk::now();
            recent_tokens = 0;

            if (baseline && delta > static_cast<std::ptrdiff_t>(leak_budget)) {
                std::printf("\n[bleed] FATAL: RSS grew %.2f MB above baseline (> %.0f MB budget)\n",
                            delta / 1024.0 / 1024.0,
                            leak_budget / 1024.0 / 1024.0);
                return 1;
            }
        }

        if (tokens_total >= TOKEN_BUDGET) break;
        if (cap_iters && iter >= cap_iters) break;
    }

    std::size_t final_rss = rss_bytes();
    std::printf("\n[bleed] SUMMARY\n");
    std::printf("  iterations           : %u\n", iter);
    std::printf("  tokens generated     : %u\n", tokens_total);
    std::printf("  baseline (iter %2u)  : %s\n", BASELINE_AT_ITER, format_mb(baseline).c_str());
    std::printf("  peak                 : %s\n", format_mb(peak).c_str());
    std::printf("  final                : %s\n", format_mb(final_rss).c_str());
    std::printf("  growth vs baseline   : %+.2f MB\n",
                (static_cast<double>(final_rss) - static_cast<double>(baseline)) / 1024.0 / 1024.0);

    const bool leaked = baseline && (final_rss > baseline + leak_budget);
    if (leaked) {
        std::printf("[bleed] FAIL: RSS leak detected.\n");
        return 1;
    }
    std::printf("[bleed] PASS: no RSS leak above 1 MB budget.\n");
    return 0;
}
