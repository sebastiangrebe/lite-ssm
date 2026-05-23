// examples/benchmark_litessm.cpp
//
// Head-to-head PoC: lite-ssm against the standard PyTorch / MPS path.
//
// Loops a synthetic log-ingestion workload. Each iteration:
//   1. Reset the recurrent state.
//   2. Prefill a system-log line as a byte-token prompt.
//   3. Generate a 5-token "summary" (greedy decode).
//
// Emits machine-readable BENCH_* lines on stdout (the harness shell script
// parses these). Boot time, time-to-first-token, decode tokens/sec, and the
// static Mamba2State size are reported.
//
// Usage:
//   build/benchmark_litessm [model.ssm] [iterations]
// Defaults: model.ssm in CWD, 12 iterations (first one discarded as warm-up).

#include "lite_ssm/inference.hpp"
#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"
#include "lite_ssm/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using clk = std::chrono::high_resolution_clock;

namespace {

double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// Small canon of fake log lines — chosen so each is realistic shell output
// and short enough that prefill latency stays well under a chunk worth.
const std::array<const char*, 6> LOG_LINES{
    "[INFO] Server heartbeat OK, uptime 12345s",
    "[WARN] Connection retry to db: timeout 5s",
    "[ERROR] Authentication failed for user admin",
    "[INFO] Cache hit ratio 0.943 over last 60s",
    "[WARN] Disk usage at 87% on /var, alerting",
    "[INFO] Request /api/v1/health 200 in 4ms",
};

constexpr uint32_t SUMMARY_TOKENS = 5;
constexpr uint32_t DEFAULT_ITERS  = 12;

struct IterStats {
    double ttft_ms;       // wall time from prefill-start to first sampled token
    double decode_ms;     // wall time for the remaining (SUMMARY_TOKENS - 1) tokens
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0
         : (v.size() % 2 ? v[v.size() / 2]
                         : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "model.ssm";
    const uint32_t    iters      = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2]))
                                              : DEFAULT_ITERS;

    // ---------------------------------------------------------------
    // BOOT: mmap + index parse + Metal device + workspace alloc
    // ---------------------------------------------------------------
    auto boot_t0 = clk::now();

    lite_ssm::SSMFile      weights(model_path);
    lite_ssm::Mamba2Model  model(weights);
    lite_ssm::MetalOps     ops;
    if (!ops.metallib_loaded()) {
        std::fprintf(stderr,
            "benchmark_litessm: metallib not built. Install Metal Toolchain:\n"
            "  xcodebuild -downloadComponent MetalToolchain\n");
        return 2;
    }

    // Workspace sized for the longest log line we'll feed (chunk-aligned).
    uint32_t max_L = 1;
    for (auto* s : LOG_LINES) max_L = std::max<uint32_t>(max_L, std::strlen(s));
    max_L = std::max<uint32_t>(max_L, model.config().chunk_size);
    lite_ssm::Mamba2Workspace workspace(model.config(), max_L);
    lite_ssm::Mamba2State     state(model.config());

    const double boot_ms = ms_since(boot_t0);

    // Load real BPE tokenizer if present; fall back to byte-per-token IDs.
    // The byte fallback keeps the benchmark runnable on a fresh checkout
    // before tools/export_tokenizer.py is invoked.
    lite_ssm::Tokenizer tok;
    bool have_bpe = std::filesystem::exists("tokenizer.model");
    if (have_bpe) tok.load("tokenizer.model");

    auto encode = [&](const char* text) -> std::vector<uint32_t> {
        if (have_bpe) return tok.encode(text);
        std::vector<uint32_t> out;
        for (const char* p = text; *p; ++p) out.push_back(static_cast<uint8_t>(*p));
        return out;
    };

    // ---------------------------------------------------------------
    // WORKLOAD: ingest one log, generate SUMMARY_TOKENS, repeat.
    // ---------------------------------------------------------------
    std::vector<IterStats> samples;
    samples.reserve(iters);

    for (uint32_t i = 0; i < iters; ++i) {
        const char* log = LOG_LINES[i % LOG_LINES.size()];
        auto tokens = encode(log);
        const uint32_t L = static_cast<uint32_t>(tokens.size());

        state.reset();
        for (uint32_t k = 0; k < L; ++k) {
            model.embed(workspace, k, tokens[k]);
        }

        // TTFT = prefill + final-position logits + first sample.
        auto t0 = clk::now();
        model.forward_prefill(ops, workspace, state, L);
        model.compute_logits(ops, workspace, L - 1);
        const auto& logits = model.read_logits_fp32(workspace);
        uint32_t next = lite_ssm::sample_greedy(logits);
        const double ttft_ms = ms_since(t0);

        // Decode rest.
        auto decode_t0 = clk::now();
        for (uint32_t step = 1; step < SUMMARY_TOKENS; ++step) {
            model.embed(workspace, 0, next);
            model.forward_step(ops, workspace, state);
            model.compute_logits(ops, workspace, 0);
            model.read_logits_fp32(workspace);
            next = lite_ssm::sample_greedy(logits);
        }
        const double decode_ms = ms_since(decode_t0);

        samples.push_back({ttft_ms, decode_ms});
    }

    // Drop the first iteration as warm-up (Metal pipeline / page-fault costs).
    if (samples.size() > 1) samples.erase(samples.begin());

    std::vector<double> ttfts, tps;
    for (auto& s : samples) {
        ttfts.push_back(s.ttft_ms);
        // Steady-state decode TPS: (SUMMARY_TOKENS - 1) tokens over decode_ms.
        tps.push_back((SUMMARY_TOKENS - 1) * 1000.0 / s.decode_ms);
    }

    const double ttft_med = median(ttfts);
    const double tps_med  = median(tps);

    // ---------------------------------------------------------------
    // Human-readable summary
    // ---------------------------------------------------------------
    std::printf("\n=== lite-ssm benchmark ===\n");
    std::printf("  model:          %s\n", model_path.c_str());
    std::printf("  iterations:     %u (first dropped as warm-up)\n", iters);
    std::printf("  boot:           %.1f ms  (mmap + index + Metal device + workspace)\n", boot_ms);
    std::printf("  TTFT median:    %.1f ms  (prefill + logits + sample)\n", ttft_med);
    std::printf("  decode TPS:     %.1f tok/s  (steady-state, %u tokens per loop)\n",
                tps_med, SUMMARY_TOKENS - 1);
    std::printf("  state size:     %.2f MB  (conv1d + SSD recurrent, fixed)\n",
                state.length() / 1.0e6);
    std::printf("  workspace size: %.2f MB  (activations scratch)\n",
                workspace.length() / 1.0e6);
    std::printf("  weights mmap:   %.2f MB  (file-backed, page-cached)\n",
                weights.mapping().file_size() / 1.0e6);

    // ---------------------------------------------------------------
    // Machine-readable lines (parsed by run_benchmark.sh)
    // ---------------------------------------------------------------
    std::printf("\nBENCH_ENGINE=lite-ssm\n");
    std::printf("BENCH_BOOT_MS=%.3f\n", boot_ms);
    std::printf("BENCH_TTFT_MS=%.3f\n", ttft_med);
    std::printf("BENCH_TPS=%.3f\n",    tps_med);
    std::printf("BENCH_STATE_BYTES=%zu\n",     state.length());
    std::printf("BENCH_WORKSPACE_BYTES=%zu\n", workspace.length());
    std::printf("BENCH_WEIGHTS_BYTES=%zu\n",   weights.mapping().file_size());
    std::printf("BENCH_ITERS=%u\n", iters - 1);

    return 0;
}
