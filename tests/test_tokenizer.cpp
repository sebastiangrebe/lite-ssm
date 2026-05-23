// tests/test_tokenizer.cpp
//
// Native SentencePiece parity test (Phase 12). Loads:
//   * tokenizer.model     — Google SentencePiece protobuf
//   * tokenizer_refs.bin  — a fixed corpus of strings + the IDs HF produced
// then asserts our C++ encode is bit-identical to HF's, and that decode
// round-trips back to the original UTF-8 bytes.
//
// Exits 77 (CTest SKIP) if either artifact is missing.

#include "lite_ssm/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int      SKIP      = 77;
constexpr uint32_t REF_MAGIC = 0x4645524Cu; // "LREF" little-endian

bool exists(const std::string& p) {
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

struct RefCase {
    std::string           text;
    std::vector<uint32_t> ids;
};

std::vector<RefCase> read_refs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(static_cast<bool>(f), "refs: cannot open");

    auto u32 = [&]() { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; };

    char magic[4]; f.read(magic, 4);
    REQUIRE(std::memcmp(magic, "LREF", 4) == 0, "refs: bad magic");
    uint32_t version = u32();
    REQUIRE(version == 1, "refs: bad version");
    uint32_t n = u32();
    (void)u32(); // pad

    std::vector<RefCase> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t text_len = u32();
        uint32_t ids_len  = u32();
        RefCase r;
        r.text.resize(text_len);
        if (text_len) f.read(r.text.data(), text_len);
        r.ids.resize(ids_len);
        if (ids_len) f.read(reinterpret_cast<char*>(r.ids.data()), ids_len * 4);
        std::size_t consumed = 8 + text_len + ids_len * 4;
        std::size_t pad = (4 - (consumed & 3)) & 3;
        if (pad) f.ignore(pad);
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace

int main() {
    const std::string tok_path  = "tokenizer.model";
    const std::string refs_path = "tokenizer_refs.bin";
    if (!exists(tok_path) || !exists(refs_path)) {
        std::fprintf(stderr,
            "[test_tokenizer] SKIP: missing %s and/or %s\n"
            "  Generate with: .venv/bin/python tools/export_tokenizer.py\n",
            tok_path.c_str(), refs_path.c_str());
        return SKIP;
    }

    lite_ssm::Tokenizer tok;
    tok.load(tok_path);
    std::printf("[test_tokenizer] vocab=%u  eos=%u  bos=%u\n",
                tok.vocab_size(), tok.eos_id(), tok.bos_id());

    auto refs = read_refs(refs_path);
    std::printf("[test_tokenizer] %zu reference cases loaded\n", refs.size());

    int failures = 0;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        const auto& r = refs[i];
        auto cpp_ids = tok.encode(r.text);

        bool encode_match = (cpp_ids == r.ids);
        std::string round_trip = tok.decode(cpp_ids);
        bool decode_match = (round_trip == r.text);

        const char* ok = (encode_match && decode_match) ? "ok" : "FAIL";
        std::printf("[test_tokenizer]  case %zu  enc=%s  dec=%s  text=%.60s%s\n",
                    i, encode_match ? "match" : "MISMATCH",
                    decode_match  ? "match" : "MISMATCH",
                    r.text.c_str(),
                    r.text.size() > 60 ? "…" : "");

        if (!encode_match) {
            std::fprintf(stderr, "  hf ids  ("); for (auto x : r.ids)     std::fprintf(stderr, " %u", x); std::fprintf(stderr, " )\n");
            std::fprintf(stderr, "  cpp ids ("); for (auto x : cpp_ids)  std::fprintf(stderr, " %u", x); std::fprintf(stderr, " )\n");
        }
        if (!decode_match) {
            std::fprintf(stderr, "  decoded: %s\n", round_trip.c_str());
        }
        if (!encode_match || !decode_match) ++failures;
        (void)ok;
    }

    if (failures) {
        std::fprintf(stderr, "[test_tokenizer] FAIL: %d/%zu cases failed\n",
                     failures, refs.size());
        return 1;
    }
    std::printf("[test_tokenizer] OK: all %zu cases match HF bit-for-bit\n", refs.size());
    return 0;
}
