#pragma once
// lite_ssm/tokenizer.hpp — thin wrapper around Google SentencePiece.
//
// Phase 12 replaced the hand-rolled GPT-NeoX byte-level BPE with the
// industry-standard SentencePiece processor. The wrapper hides the SP
// types from this public header so callers don't need SP includes.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lite_ssm {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(const Tokenizer&)            = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Load a SentencePiece model file (the .model protobuf produced by
    // HuggingFace's `tokenizer.model` artifact, or by `spm_train`).
    void load(const std::string& path);

    bool      loaded()     const;
    uint32_t  vocab_size() const;
    uint32_t  eos_id()     const;
    uint32_t  bos_id()     const;

    // Encode a UTF-8 string into token ids using the loaded SentencePiece
    // model. Handles SP's word-boundary marker (▁), byte-fallback tokens
    // (`<0xNN>`), and special tokens transparently.
    std::vector<uint32_t> encode(std::string_view text) const;

    // Decode a sequence of ids back into a UTF-8 string. SP joins pieces
    // correctly across word boundaries and reconstructs raw bytes from
    // byte-fallback tokens.
    std::string decode(const std::vector<uint32_t>& ids) const;

    // Convenience: decode a single id. NOTE — for SentencePiece this may
    // produce surprising results on partial multibyte sequences (the byte-
    // fallback splits a multibyte UTF-8 codepoint across multiple ids).
    // Streaming callers should prefer decode(history) and print the delta
    // rather than concatenating per-token outputs.
    std::string decode_one(uint32_t id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lite_ssm
