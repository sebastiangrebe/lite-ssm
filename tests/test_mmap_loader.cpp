// tests/test_mmap_loader.cpp — Phase 3 plumbing check.
//
// Loads model.ssm via SSMFile, looks up tensors emitted by the Python
// exporter, asserts shape + dtype + offset + UnifiedBuffer bytes match.
// Exits 77 (CTest SKIP convention) if the .ssm file isn't present.

#include "lite_ssm/allocator.hpp"
#include "lite_ssm/model.hpp"
#include "lite_ssm/ssm_format.hpp"
#include "lite_ssm/tensor.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace {

constexpr int SKIP = 77;

bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && st.st_size > 0;
}

#define REQUIRE(cond, msg)                                              \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s — %s\n",               \
                         __FILE__, __LINE__, #cond, msg);               \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

}  // namespace

int main() {
    const char* env = std::getenv("LITE_SSM_TEST_MODEL");
    std::string path = env ? env : "model.ssm";

    if (!file_exists(path)) {
        std::fprintf(stderr,
                     "[test_mmap_loader] SKIP: model.ssm not found at '%s'.\n"
                     "  Generate with: .venv/bin/python tools/export_mamba2.py --out model.ssm\n",
                     path.c_str());
        return SKIP;
    }

    lite_ssm::SSMFile model(path);

    // --- header / config -------------------------------------------------
    const auto& cfg = model.config();
    std::printf("[test_mmap_loader] loaded %s  (%zu tensors)\n",
                path.c_str(), model.num_tensors());
    std::printf("[test_mmap_loader] config: d_model=%u n_layer=%u d_state=%u d_conv=%u "
                "expand=%u vocab_size=%u n_heads=%u d_head=%u chunk_size=%u\n",
                cfg.d_model, cfg.n_layer, cfg.d_state, cfg.d_conv,
                cfg.expand, cfg.vocab_size, cfg.n_heads, cfg.d_head, cfg.chunk_size);

    // Values pulled directly from the Python verifier output for the
    // mamba2-130m-hf export. If the exporter changes, update these.
    REQUIRE(cfg.d_model    == 768,   "d_model");
    REQUIRE(cfg.n_layer    == 24,    "n_layer");
    REQUIRE(cfg.d_state    == 128,   "d_state");
    REQUIRE(cfg.d_conv     == 4,     "d_conv");
    REQUIRE(cfg.expand     == 2,     "expand");
    REQUIRE(cfg.vocab_size == 50288, "vocab_size");
    REQUIRE(cfg.n_heads    == 24,    "n_heads");
    REQUIRE(cfg.d_head     == 64,    "d_head");
    REQUIRE(cfg.chunk_size == 256,   "chunk_size");
    REQUIRE(cfg.default_dtype == lite_ssm::DTYPE_F16, "default_dtype");

    // --- look up the embedding -------------------------------------------
    const lite_ssm::Tensor& emb = model.at("backbone.embeddings.weight");
    std::printf("[test_mmap_loader] embedding: dtype=%u rank=%u shape=(%llu, %llu) "
                "offset=%llu nbytes=%llu\n",
                static_cast<unsigned>(emb.dtype), emb.rank,
                (unsigned long long)emb.shape[0], (unsigned long long)emb.shape[1],
                (unsigned long long)emb.offset_bytes, (unsigned long long)emb.nbytes);
    REQUIRE(emb.dtype        == lite_ssm::DTYPE_F16, "embed dtype");
    REQUIRE(emb.rank         == 2,                   "embed rank");
    REQUIRE(emb.shape[0]     == 50288,               "embed shape[0]");
    REQUIRE(emb.shape[1]     == 768,                 "embed shape[1]");
    REQUIRE(emb.offset_bytes == 17600,               "embed offset (vs Python verifier)");
    REQUIRE(emb.nbytes       == 50288ULL * 768ULL * 2ULL, "embed nbytes");

    // --- spot-check a mid-layer tensor too -------------------------------
    const lite_ssm::Tensor& ip = model.at("backbone.layers.12.mixer.in_proj.weight");
    REQUIRE(ip.rank     == 2,    "in_proj rank");
    REQUIRE(ip.shape[0] == 3352, "in_proj shape[0]");  // 2*d_inner + 2*d_state + n_heads
    REQUIRE(ip.shape[1] == 768,  "in_proj shape[1]");
    REQUIRE(ip.offset_bytes % lite_ssm::TENSOR_DATA_ALIGN == 0,
            "in_proj offset must be 64-byte aligned");

    // --- UnifiedBuffer: contents pointer must match the mmap base --------
    const auto& buf = model.buffer();
    REQUIRE(buf.valid(),                                  "UnifiedBuffer valid");
    REQUIRE(buf.contents() == model.mapping().data(),     "UnifiedBuffer contents == mmap base");
    REQUIRE(buf.length()   == model.mapping().mapped_size(),
            "UnifiedBuffer length == mmap mapped_size (page-aligned)");
    REQUIRE(buf.metal_buffer() != nullptr,                "UnifiedBuffer has an MTLBuffer handle");

    // --- read the first few embedding fp16 values via the buffer ---------
    // Confirms (offset, nbytes) point at real data through the same memory
    // that the GPU will see.
    const auto* base = static_cast<const std::uint8_t*>(buf.contents());
    const auto* fp16 = reinterpret_cast<const std::uint16_t*>(base + emb.offset_bytes);
    std::printf("[test_mmap_loader] embedding row 0 first 4 fp16 bits: "
                "0x%04x 0x%04x 0x%04x 0x%04x\n", fp16[0], fp16[1], fp16[2], fp16[3]);

    std::printf("[test_mmap_loader] OK\n");
    return 0;
}
