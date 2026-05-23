// tests/test_ssd_parity.cpp — end-to-end forward-pass parity vs PyTorch.
//
// Loads model.ssm + parity.bin (produced by tools/dump_parity.py), runs
// the Metal engine over the same byte-token prompt, then compares the
// last-token logits to PyTorch's fp16-path reference.
//
// Skips with code 77 (CTest SKIP) if either file is missing.

#include "lite_ssm/inference.hpp"
#include "lite_ssm/model.hpp"
#include "lite_ssm/ops.hpp"
#include "lite_ssm/sampler.hpp"
#include "lite_ssm/state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int      SKIP        = 77;
constexpr uint32_t PARITY_MAGIC = 0x50415254u;  // 'PART'

bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && st.st_size > 0;
}

#define REQUIRE(cond, msg) do {                                              \
    if (!(cond)) {                                                            \
        std::fprintf(stderr, "FAIL %s:%d: %s — %s\n", __FILE__, __LINE__,     \
                     #cond, msg);                                             \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

struct ParityDump {
    std::vector<uint32_t> tokens;
    std::vector<float>    last_logits;
};

ParityDump read_dump(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[parity] cannot open %s\n", path.c_str());
        std::exit(SKIP);
    }
    uint32_t header[4];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    REQUIRE(header[0] == PARITY_MAGIC, "parity.bin: bad magic");
    REQUIRE(header[1] == 1u,           "parity.bin: unsupported version");
    const uint32_t n_tokens   = header[2];
    const uint32_t vocab_size = header[3];

    ParityDump d;
    d.tokens.resize(n_tokens);
    d.last_logits.resize(vocab_size);
    f.read(reinterpret_cast<char*>(d.tokens.data()),     n_tokens * sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(d.last_logits.data()), vocab_size * sizeof(float));
    REQUIRE(f.good() || f.eof(), "parity.bin: read error");
    return d;
}

}  // namespace

int main() {
    const char* model_env = std::getenv("LITE_SSM_TEST_MODEL");
    const char* dump_env  = std::getenv("LITE_SSM_PARITY_DUMP");
    std::string model_path = model_env ? model_env : "model.ssm";
    std::string dump_path  = dump_env  ? dump_env  : "parity.bin";

    if (!file_exists(model_path) || !file_exists(dump_path)) {
        std::fprintf(stderr,
            "[parity] SKIP: missing artifacts\n"
            "  model.ssm  at %s : %s\n"
            "  parity.bin at %s : %s\n"
            "  Build artifacts with:\n"
            "    .venv/bin/python tools/export_mamba2.py --out model.ssm\n"
            "    .venv/bin/python tools/dump_parity.py    --out parity.bin\n",
            model_path.c_str(), file_exists(model_path) ? "ok" : "MISSING",
            dump_path.c_str(),  file_exists(dump_path)  ? "ok" : "MISSING");
        return SKIP;
    }

    auto dump = read_dump(dump_path);
    std::printf("[parity] dump: n_tokens=%zu vocab=%zu\n",
                dump.tokens.size(), dump.last_logits.size());

    // Load model + allocate workspace and state.
    lite_ssm::SSMFile        weights(model_path);
    lite_ssm::Mamba2Model    model(weights);
    lite_ssm::MetalOps       ops;
    REQUIRE(ops.metallib_loaded(), "metallib must be built (install Metal Toolchain)");

    const uint32_t max_L = std::max<uint32_t>(
        static_cast<uint32_t>(dump.tokens.size()),
        model.config().chunk_size);
    lite_ssm::Mamba2Workspace ws(model.config(), max_L);
    lite_ssm::Mamba2State     state(model.config());

    // Run prefill on the dumped tokens.
    state.reset();
    for (uint32_t i = 0; i < dump.tokens.size(); ++i) {
        model.embed(ws, i, dump.tokens[i]);
    }
    model.forward_prefill(ops, ws, state, static_cast<uint32_t>(dump.tokens.size()));
    model.compute_logits(ops, ws, /*hidden_row*/ static_cast<uint32_t>(dump.tokens.size() - 1));
    // Copy out because the parity test holds the values across additional
    // reads; the workspace scratch returned by read_logits_fp32 would be
    // overwritten by any subsequent call.
    auto cpp_logits = model.read_logits_fp32(ws);

    REQUIRE(cpp_logits.size() == dump.last_logits.size(), "logits length mismatch");

    // Stats: compare argmax, max abs error, mean abs error, RMS error.
    uint32_t py_argmax  = std::distance(dump.last_logits.begin(),
                                        std::max_element(dump.last_logits.begin(), dump.last_logits.end()));
    uint32_t cpp_argmax = std::distance(cpp_logits.begin(),
                                        std::max_element(cpp_logits.begin(), cpp_logits.end()));

    double sum_abs = 0.0, sum_sq = 0.0;
    float  max_abs = 0.0f;
    uint32_t max_at = 0;
    for (size_t i = 0; i < cpp_logits.size(); ++i) {
        float diff = std::abs(cpp_logits[i] - dump.last_logits[i]);
        sum_abs += diff;
        sum_sq  += double(diff) * diff;
        if (diff > max_abs) { max_abs = diff; max_at = static_cast<uint32_t>(i); }
    }
    double mean_abs = sum_abs / cpp_logits.size();
    double rms      = std::sqrt(sum_sq / cpp_logits.size());

    std::printf("[parity] py_argmax=%u  cpp_argmax=%u   %s\n",
                py_argmax, cpp_argmax,
                (py_argmax == cpp_argmax) ? "MATCH" : "MISMATCH");
    std::printf("[parity] max_abs_err=%.4f (at vocab id %u: py=%.4f cpp=%.4f)\n",
                max_abs, max_at,
                dump.last_logits[max_at], cpp_logits[max_at]);
    std::printf("[parity] mean_abs_err=%.4f  rms_err=%.4f\n", mean_abs, rms);

    // Tolerance: observed max_abs is ~0.06 (30-token chunked) to ~0.13
    // (1-token step) against PyTorch's fp16 reference, i.e. fundamentally
    // at the fp16 noise floor across a 24-layer forward + 50K-wide lm_head
    // gemv. 0.5 leaves 4× headroom for natural fp16 jitter on other prompts.
    constexpr float MAX_ABS_TOLERANCE = 0.5f;
    REQUIRE(max_abs < MAX_ABS_TOLERANCE, "max_abs_err exceeds tolerance");

    // Argmax can flip when the top-2 logits are within fp16 noise — log
    // the mismatch but don't fail the test on it alone.
    if (py_argmax != cpp_argmax) {
        std::fprintf(stderr,
            "[parity] WARN: argmax differs but max_abs within bound — "
            "likely top-2 logits within fp16 noise. Inspect logits[%u]=%.4f "
            "vs logits[%u]=%.4f (py).\n",
            py_argmax, dump.last_logits[py_argmax],
            cpp_argmax, dump.last_logits[cpp_argmax]);
    }

    std::printf("[parity] OK\n");
    return 0;
}
