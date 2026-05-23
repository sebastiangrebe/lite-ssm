// src/core/tokenizer.cpp — SentencePiece adapter.
//
// Phase 12: replaced the in-tree byte-level BPE (vocab.bin + merges hash
// + GPT-NeoX pre-tokenizer) with Google SentencePiece. The .model file
// we load is the SP protobuf format (the same `tokenizer.model` HuggingFace
// ships for Mistral / Llama / T5 / Codestral / etc.).

#include "lite_ssm/tokenizer.hpp"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_ssm {

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
    bool                                  loaded = false;
};

Tokenizer::Tokenizer()  : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept            = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void Tokenizer::load(const std::string& path) {
    auto status = impl_->sp.Load(path);
    if (!status.ok()) {
        throw std::runtime_error("Tokenizer::load(\"" + path + "\"): " +
                                 status.ToString());
    }
    impl_->loaded = true;
}

bool     Tokenizer::loaded()     const { return impl_->loaded; }
uint32_t Tokenizer::vocab_size() const { return static_cast<uint32_t>(impl_->sp.GetPieceSize()); }

uint32_t Tokenizer::eos_id() const {
    int e = impl_->sp.eos_id();
    return (e < 0) ? 0xFFFFFFFFu : static_cast<uint32_t>(e);
}

uint32_t Tokenizer::bos_id() const {
    int b = impl_->sp.bos_id();
    return (b < 0) ? 0xFFFFFFFFu : static_cast<uint32_t>(b);
}

std::vector<uint32_t> Tokenizer::encode(std::string_view text) const {
    if (!impl_->loaded) throw std::runtime_error("Tokenizer::encode: not loaded");
    std::vector<int> ids;
    auto status = impl_->sp.Encode(std::string(text), &ids);
    if (!status.ok()) {
        throw std::runtime_error("Tokenizer::encode: " + status.ToString());
    }
    std::vector<uint32_t> out;
    out.reserve(ids.size());
    for (int i : ids) out.push_back(static_cast<uint32_t>(i));
    return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    if (!impl_->loaded) throw std::runtime_error("Tokenizer::decode: not loaded");
    std::vector<int> int_ids;
    int_ids.reserve(ids.size());
    for (uint32_t i : ids) int_ids.push_back(static_cast<int>(i));
    std::string out;
    auto status = impl_->sp.Decode(int_ids, &out);
    if (!status.ok()) {
        throw std::runtime_error("Tokenizer::decode: " + status.ToString());
    }
    return out;
}

std::string Tokenizer::decode_one(uint32_t id) const {
    return decode({id});
}

}  // namespace lite_ssm
