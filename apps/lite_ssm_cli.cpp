// apps/lite_ssm_cli.cpp
//
// Driver binary for lite-ssm. Two operating modes:
//
//   1. Native text mode (Phase 8 default):
//        lite-ssm "Mamba is a state space model"
//      Loads tokenizer.bin, encodes the prompt, runs prefill + decode,
//      streams DECODED TEXT to stdout as tokens are sampled.
//
//   2. Token-id passthrough mode (Phase 7 parity driver):
//        lite-ssm --input-ids 510,2403,629
//      Skips the tokenizer; emits a comma-separated id list on stdout.
//      Used by examples/quality_parity.py.
//
// Common flags:
//   --model PATH            .ssm weights (default: model.ssm)
//   --tokenizer PATH        tokenizer.bin (default: tokenizer.bin)
//   --tokens N              max new tokens (default: 200)
//   --greedy                argmax sampling (default)
//   --top-p P / --temperature T / --seed S    nucleus sampling
//   --dump-logits FILE      write final-step fp32 logits to FILE
//   --eos ID                stop on this id (default: model's eos)
//   --quiet                 suppress status lines on stderr (errors still print)
//   -h, --help              this help

#include "lite_ssm/inference.hpp"
#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"
#include "lite_ssm/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mach/mach.h>

namespace {

struct Args {
    std::string model        = "model.ssm";
    std::string tokenizer    = "tokenizer.model";
    uint32_t    max_tokens   = 200;
    bool        greedy       = true;
    float       top_p        = 0.9f;
    float       temperature  = 1.0f;
    uint64_t    seed         = 0xCAFEBABEull;
    bool        quiet        = false;
    bool        eos_set      = false;
    uint32_t    eos_id       = 0u;
    bool        probe_only   = false;   // load + page-in scan + RSS, then exit
    std::string dump_logits;
    // Exactly ONE of these is populated (unless probe_only is set).
    std::string           prompt_text;
    std::vector<uint32_t> input_ids;
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "lite-ssm: %s\n", msg.c_str());
    std::exit(2);
}

void usage() {
    std::fprintf(stderr,
        "lite-ssm — Mamba-2 inference driver\n\n"
        "Usage: lite-ssm [options] \"prompt text\"\n"
        "       lite-ssm [options] --input-ids ID,ID,...\n\n"
        "Options:\n"
        "  --model PATH         .ssm weights (default: model.ssm)\n"
        "  --tokenizer PATH     SentencePiece .model (default: tokenizer.model)\n"
        "  --tokens N           max new tokens (default: 200)\n"
        "  --greedy             argmax sampling (default)\n"
        "  --top-p P            top-P sampling\n"
        "  --temperature T      temperature (default: 1.0)\n"
        "  --seed S             RNG seed (default: 0xCAFEBABE)\n"
        "  --input-ids LIST     bypass tokenizer; emit ids on stdout\n"
        "  --dump-logits FILE   write final-step fp32 logits to FILE\n"
        "  --eos ID             stop on this id (default: tokenizer eos)\n"
        "  --quiet              suppress status lines on stderr\n"
        "  -h, --help           this help\n");
}

std::vector<uint32_t> parse_ids(std::string_view s) {
    std::vector<uint32_t> out;
    std::string buf;
    for (char c : s) {
        if (c == ',' || c == ' ') {
            if (!buf.empty()) { out.push_back(static_cast<uint32_t>(std::stoul(buf))); buf.clear(); }
        } else { buf.push_back(c); }
    }
    if (!buf.empty()) out.push_back(static_cast<uint32_t>(std::stoul(buf)));
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string_view k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) die(std::string(name) + " expects an argument");
            return std::string_view(argv[++i]);
        };
        if      (k == "--model")        a.model       = need("--model");
        else if (k == "--tokenizer")    a.tokenizer   = need("--tokenizer");
        else if (k == "--tokens")       a.max_tokens  = static_cast<uint32_t>(std::stoul(std::string(need("--tokens"))));
        else if (k == "--greedy")       a.greedy = true;
        else if (k == "--top-p")        { a.greedy = false; a.top_p = std::stof(std::string(need("--top-p"))); }
        else if (k == "--temperature")  a.temperature = std::stof(std::string(need("--temperature")));
        else if (k == "--seed")         a.seed        = std::stoull(std::string(need("--seed")));
        else if (k == "--input-ids")    a.input_ids   = parse_ids(need("--input-ids"));
        else if (k == "--dump-logits")  a.dump_logits = need("--dump-logits");
        else if (k == "--eos")          { a.eos_id = static_cast<uint32_t>(std::stoul(std::string(need("--eos")))); a.eos_set = true; }
        else if (k == "--quiet")        a.quiet = true;
        else if (k == "--probe-only")   a.probe_only = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else if (!k.empty() && k[0] == '-')  die("unknown flag: " + std::string(k));
        else                                  positionals.emplace_back(k);
    }
    if (a.input_ids.empty() && positionals.empty() && !a.probe_only) {
        usage();
        die("provide a prompt text positionally, --input-ids, or --probe-only");
    }
    if (!a.input_ids.empty() && !positionals.empty()) {
        die("can't combine --input-ids with a positional prompt");
    }
    if (!positionals.empty()) {
        // Join positional args with spaces so quoting at the shell is optional.
        a.prompt_text = positionals.front();
        for (std::size_t i = 1; i < positionals.size(); ++i) {
            a.prompt_text.push_back(' ');
            a.prompt_text += positionals[i];
        }
    }
    return a;
}

void say(bool quiet, const char* fmt, ...) {
    if (quiet) return;
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

}  // namespace

namespace {

std::size_t rss_bytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
    return static_cast<std::size_t>(info.resident_size);
}

}  // namespace

int main(int argc, char** argv) try {
    using clk = std::chrono::high_resolution_clock;
    Args args = parse_args(argc, argv);
    const bool text_mode = !args.prompt_text.empty();

    auto t_boot = clk::now();
    const std::size_t rss_pre_mmap = rss_bytes();
    lite_ssm::SSMFile      weights(args.model);
    const double           ms_after_mmap = std::chrono::duration<double, std::milli>(clk::now() - t_boot).count();
    const std::size_t      rss_after_mmap = rss_bytes();

    if (args.probe_only) {
        // Stress the unified-memory pager: touch one byte per page across the
        // entire weights mmap, time it, and report final RSS so the caller
        // can see how much of the file the OS chose to actually keep resident.
        const auto* base = static_cast<const std::uint8_t*>(weights.mapping().data());
        const std::size_t sz   = weights.mapping().file_size();
        const std::size_t step = 16384;   // 16 KB page on Apple Silicon
        auto t_scan = clk::now();
        volatile std::size_t sink = 0;
        for (std::size_t i = 0; i < sz; i += step) sink += base[i];
        (void)sink;
        const double ms_scan = std::chrono::duration<double, std::milli>(clk::now() - t_scan).count();
        const std::size_t rss_after_scan = rss_bytes();

        std::printf("== probe-only mmap stress ==\n");
        std::printf("  model file        : %s (%.2f GB)\n", args.model.c_str(), sz / 1e9);
        std::printf("  tensors indexed   : %zu\n", weights.num_tensors());
        std::printf("  rss pre-mmap      : %.1f MB\n", rss_pre_mmap  / 1024.0 / 1024.0);
        std::printf("  rss after mmap    : %.1f MB   (+%.1f MB, %.1f ms wall)\n",
                    rss_after_mmap / 1024.0 / 1024.0,
                    (rss_after_mmap - rss_pre_mmap) / 1024.0 / 1024.0,
                    ms_after_mmap);
        std::printf("  rss after scan    : %.1f MB   (page-touched every 16 KB, %.1f ms wall)\n",
                    rss_after_scan / 1024.0 / 1024.0, ms_scan);
        std::printf("  resident fraction : %.1f%% of file (the rest is paged out / cold)\n",
                    100.0 * rss_after_scan / static_cast<double>(sz));
        return 0;
    }

    lite_ssm::Mamba2Model  model(weights);
    lite_ssm::MetalOps     ops;
    if (!ops.metallib_loaded()) {
        die("lite_ssm.metallib not found; install Metal Toolchain via "
            "`xcodebuild -downloadComponent MetalToolchain` and rebuild");
    }

    // Tokenizer is required in text mode, optional in id-passthrough mode.
    lite_ssm::Tokenizer tok;
    if (text_mode) {
        if (!std::filesystem::exists(args.tokenizer)) {
            die("SentencePiece model not found at '" + args.tokenizer +
                "'. Generate with: tools/export_tokenizer.py");
        }
        tok.load(args.tokenizer);
    }

    // Resolve input ids.
    std::vector<uint32_t> input_ids = args.input_ids;
    if (text_mode) input_ids = tok.encode(args.prompt_text);
    if (input_ids.empty()) die("empty prompt after tokenization");

    const uint32_t prompt_len = static_cast<uint32_t>(input_ids.size());
    const uint32_t max_L      = std::max<uint32_t>(prompt_len, model.config().chunk_size);
    lite_ssm::Mamba2Workspace ws(model.config(), max_L);
    lite_ssm::Mamba2State     state(model.config());

    const double boot_ms = std::chrono::duration<double, std::milli>(clk::now() - t_boot).count();
    if (text_mode) {
        say(args.quiet, "[lite-ssm] boot %.1f ms | prompt=%u BPE ids | max_tokens=%u | mode=%s",
            boot_ms, prompt_len, args.max_tokens, args.greedy ? "greedy" : "top-p");
    } else {
        say(args.quiet, "[lite-ssm] boot %.1f ms | prompt=%u raw ids | max_tokens=%u | mode=%s",
            boot_ms, prompt_len, args.max_tokens, args.greedy ? "greedy" : "top-p");
    }

    // Determine EOS — explicit flag wins; else tokenizer's if loaded; else 0.
    const uint32_t eos = args.eos_set ? args.eos_id
                       : (tok.loaded() ? tok.eos_id() : 0xFFFFFFFFu);

    // ---- Prefill --------------------------------------------------------
    state.reset();
    for (uint32_t i = 0; i < prompt_len; ++i) {
        if (input_ids[i] >= model.config().vocab_size) {
            die("input id " + std::to_string(input_ids[i]) + " out of vocab range");
        }
        model.embed(ws, i, input_ids[i]);
    }
    auto t_prefill = clk::now();
    model.forward_prefill(ops, ws, state, prompt_len);
    model.compute_logits(ops, ws, prompt_len - 1);
    const auto& logits = model.read_logits_fp32(ws);
    const double prefill_ms = std::chrono::duration<double, std::milli>(clk::now() - t_prefill).count();

    uint64_t rng = args.seed ? args.seed : 1u;
    auto pick = [&]() {
        return args.greedy ? lite_ssm::sample_greedy(logits)
                           : lite_ssm::sample_top_p(logits, args.top_p, args.temperature, rng);
    };

    if (text_mode && !args.quiet) {
        // Echo prompt in dim brackets so the streaming continuation is visually
        // distinct from what the model is producing.
        std::fprintf(stderr, "[lite-ssm] prompt: ");
        std::fwrite(args.prompt_text.data(), 1, args.prompt_text.size(), stderr);
        std::fputc('\n', stderr);
        std::fprintf(stderr, "[lite-ssm] -- streaming --\n");
        // Re-echo prompt to stdout so users that pipe to less / grep keep context.
        std::fwrite(args.prompt_text.data(), 1, args.prompt_text.size(), stdout);
        std::fflush(stdout);
    }

    // SentencePiece decode is whole-sequence aware (handles ▁ word-boundary
    // markers + byte-fallback fragments correctly). We accumulate generated
    // ids, decode the full sequence each step, and print only the new tail.
    // This is the correct streaming pattern for SP — per-id IdToPiece would
    // drop or duplicate spaces on word boundaries.
    std::vector<uint32_t> generated;
    generated.reserve(args.max_tokens);
    std::string streamed;
    auto stream_after_push = [&]() {
        if (!text_mode) return;
        std::string full = tok.decode(generated);
        if (full.size() > streamed.size()) {
            std::string_view delta(full.data() + streamed.size(),
                                   full.size() - streamed.size());
            std::fwrite(delta.data(), 1, delta.size(), stdout);
            std::fflush(stdout);
            streamed = std::move(full);
        }
    };

    // ---- Sample first token + decode loop -------------------------------
    uint32_t next = pick();
    generated.push_back(next);
    stream_after_push();

    auto t_decode = clk::now();
    for (uint32_t step = 1; step < args.max_tokens; ++step) {
        if (next == eos) break;
        model.embed(ws, 0, next);
        model.forward_step(ops, ws, state);
        model.compute_logits(ops, ws, 0);
        model.read_logits_fp32(ws);   // refills `logits` (workspace scratch)
        next = pick();
        generated.push_back(next);
        stream_after_push();
    }
    const double decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t_decode).count();
    const double tps       = (generated.size() <= 1) ? 0.0
                            : (generated.size() - 1) * 1000.0 / decode_ms;

    if (text_mode) std::fputc('\n', stdout);
    std::fflush(stdout);

    say(args.quiet,
        "[lite-ssm] prefill %.1f ms | decode %.1f ms | %.1f tok/s | generated=%zu",
        prefill_ms, decode_ms, tps, generated.size());

    // ---- Dump final logits (Phase 7 KL parity hook) ---------------------
    if (!args.dump_logits.empty()) {
        std::ofstream f(args.dump_logits, std::ios::binary);
        if (!f) die("cannot open --dump-logits target: " + args.dump_logits);
        f.write(reinterpret_cast<const char*>(logits.data()),
                logits.size() * sizeof(float));
        if (!f.good()) die("write failed: " + args.dump_logits);
    }

    // ---- ID-passthrough mode also prints ids (matches Phase 7 contract) -
    if (!text_mode) {
        for (std::size_t i = 0; i < generated.size(); ++i) {
            if (i) std::fputc(',', stdout);
            std::fprintf(stdout, "%u", generated[i]);
        }
        std::fputc('\n', stdout);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "lite-ssm: %s\n", e.what());
    return 1;
}
