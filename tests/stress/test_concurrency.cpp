// tests/stress/test_concurrency.cpp
//
// Swarm-contention probe.
//
// Loads model.ssm once → one shared SSMFile + UnifiedBuffer + Mamba2Model.
// Spawns N std::thread workers. Each thread builds its OWN MetalOps,
// Mamba2Workspace, and Mamba2State (recurrent state must not be shared).
// Each worker runs a fixed decode loop. We report:
//
//   * single-thread baseline TPS (for comparison)
//   * per-worker TPS
//   * aggregate TPS = sum of per-worker tokens / wall clock
//   * scaling factor   = aggregate_tps / single_tps
//
// The MTLCommandQueue is shared (singleton) and is documented as thread-
// safe — multiple threads may submit command buffers concurrently. The
// GPU itself is a single device though, so kernels will largely serialize
// on the SIMD pipes. Scaling below 8× is expected; the test is whether we
// CRASH / STALL / produce-garbage under contention, not whether we go
// faster.

#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"
#include "lite_ssm/tokenizer.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::high_resolution_clock;

namespace {

constexpr uint32_t DECODE_TOKENS_PER_WORKER = 100;

struct WorkerResult {
    int      id;
    uint32_t tokens;
    double   duration_ms;
};

double ms_diff(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// One worker's full prefill + decode pass. All allocations are local; the
// only shared object is the Mamba2Model (which is const after construction).
WorkerResult run_worker(int id,
                        const lite_ssm::Mamba2Model& model,
                        const std::vector<uint32_t>& prompt_ids,
                        std::atomic<bool>& go,
                        uint32_t decode_steps) {
    lite_ssm::MetalOps         ops;
    lite_ssm::Mamba2Workspace  ws(model.config(),
                                  std::max<uint32_t>(
                                      static_cast<uint32_t>(prompt_ids.size()),
                                      model.config().chunk_size));
    lite_ssm::Mamba2State      state(model.config());
    state.reset();

    // Embed prompt (CPU-side via unified memory).
    for (uint32_t i = 0; i < prompt_ids.size(); ++i) {
        model.embed(ws, i, prompt_ids[i]);
    }

    // Spin until released to keep workers as in-sync as possible.
    while (!go.load(std::memory_order_acquire)) { /* spin */ }

    auto t0 = clk::now();
    model.forward_prefill(ops, ws, state, static_cast<uint32_t>(prompt_ids.size()));
    model.compute_logits(ops, ws, static_cast<uint32_t>(prompt_ids.size() - 1));
    const auto& logits = model.read_logits_fp32(ws);
    uint32_t next = lite_ssm::sample_greedy(logits);
    uint32_t produced = 1;

    for (uint32_t step = 1; step < decode_steps; ++step) {
        model.embed(ws, 0, next);
        model.forward_step(ops, ws, state);
        model.compute_logits(ops, ws, 0);
        model.read_logits_fp32(ws);
        next = lite_ssm::sample_greedy(logits);
        ++produced;
    }
    auto t1 = clk::now();
    return {id, produced, ms_diff(t0, t1)};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "model.ssm";
    const int         n_threads  = (argc > 2) ? std::atoi(argv[2]) : 8;
    const uint32_t    decode_t   = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3]))
                                              : DECODE_TOKENS_PER_WORKER;

    if (!std::filesystem::exists(model_path)) {
        std::fprintf(stderr, "test_concurrency: %s not found\n", model_path.c_str());
        return 2;
    }

    lite_ssm::SSMFile     weights(model_path);
    lite_ssm::Mamba2Model model(weights);

    // Warm up the metallib + every pipeline state on the main thread so
    // worker threads don't race on first-compile. Touch each kernel via a
    // throwaway MetalOps.
    {
        lite_ssm::MetalOps warmup;
        if (!warmup.metallib_loaded()) {
            std::fprintf(stderr, "test_concurrency: metallib not built\n");
            return 2;
        }
        for (auto* k : {
                "rmsnorm_f16", "silu_gated_f16", "linear_f16_gemv",
                "linear_f16_gemm", "causal_conv1d_f16", "causal_conv1d_update_f16",
                "ssd_chunked_f16", "ssd_step_f16", "split_silu_xBC_f16",
                "add_inplace_f16"}) {
            (void)warmup.has_pipeline(k);
        }
    }

    // Build the prompt once. Use BPE if present; else byte-fallback.
    lite_ssm::Tokenizer tok;
    bool have_bpe = std::filesystem::exists("tokenizer.model");
    if (have_bpe) tok.load("tokenizer.model");
    const std::string prompt = "Mamba is a state space model. The key insight is that";
    std::vector<uint32_t> prompt_ids;
    if (have_bpe) prompt_ids = tok.encode(prompt);
    else for (unsigned char c : prompt) prompt_ids.push_back(c);

    std::printf("[concurrency] model=%s  threads=%d  decode/thread=%u  prompt=%zu ids\n\n",
                model_path.c_str(), n_threads, decode_t, prompt_ids.size());

    // ---- Single-thread baseline ----
    std::printf("[concurrency] single-thread baseline …\n");
    std::atomic<bool> solo{true};
    auto solo_res = run_worker(0, model, prompt_ids, solo, decode_t);
    const double solo_tps = solo_res.tokens * 1000.0 / solo_res.duration_ms;
    std::printf("[concurrency] solo: %u tokens in %.1f ms = %.1f tok/s\n\n",
                solo_res.tokens, solo_res.duration_ms, solo_tps);

    // ---- Concurrent run ----
    std::printf("[concurrency] launching %d threads concurrently…\n", n_threads);
    std::vector<std::thread>   threads;
    std::vector<WorkerResult>  results(n_threads);
    std::atomic<bool> go{false};

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i] {
            results[i] = run_worker(i, model, prompt_ids, go, decode_t);
        });
    }
    // Brief pause to let workers reach the spin barrier.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto t_wall0 = clk::now();
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    auto t_wall1 = clk::now();
    const double wall_ms = ms_diff(t_wall0, t_wall1);

    uint32_t total_tokens = 0;
    double   min_t = 1e18, max_t = 0;
    for (auto& r : results) {
        total_tokens += r.tokens;
        if (r.duration_ms < min_t) min_t = r.duration_ms;
        if (r.duration_ms > max_t) max_t = r.duration_ms;
    }
    const double aggregate_tps = total_tokens * 1000.0 / wall_ms;
    const double scaling       = aggregate_tps / solo_tps;

    std::printf("\n[concurrency] per-thread breakdown:\n");
    std::printf("%4s  %8s  %12s  %12s\n", "id", "tokens", "duration_ms", "tok/s");
    for (auto& r : results) {
        std::printf("%4d  %8u  %12.1f  %12.1f\n",
                    r.id, r.tokens, r.duration_ms,
                    r.tokens * 1000.0 / r.duration_ms);
    }

    std::printf("\n[concurrency] SUMMARY\n");
    std::printf("  wall clock           : %.1f ms\n", wall_ms);
    std::printf("  total tokens         : %u\n", total_tokens);
    std::printf("  aggregate TPS        : %.1f tok/s\n", aggregate_tps);
    std::printf("  single-thread TPS    : %.1f tok/s\n", solo_tps);
    std::printf("  scaling factor       : %.2fx  (ideal = %dx)\n", scaling, n_threads);
    std::printf("  fastest worker       : %.1f ms\n", min_t);
    std::printf("  slowest worker       : %.1f ms\n", max_t);
    std::printf("  spread (max/min)     : %.2fx\n", max_t / min_t);
    return 0;
}
