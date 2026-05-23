// src/core/ssm_loader.cpp — parse .ssm header + index, build name -> Tensor map.

#include "lite_ssm/model.hpp"

#include "lite_ssm/allocator.hpp"
#include "lite_ssm/ssm_format.hpp"
#include "lite_ssm/tensor.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace lite_ssm {

namespace {

void check_in_bounds(const void* base, std::size_t size, const void* p, std::size_t n,
                     const char* what) {
    const auto* b = static_cast<const std::uint8_t*>(base);
    const auto* q = static_cast<const std::uint8_t*>(p);
    if (q < b || q + n > b + size) {
        throw std::runtime_error(std::string("SSMFile: out-of-bounds read while parsing ") + what);
    }
}

Mamba2Config config_from_header(const FileHeader& h) {
    Mamba2Config c;
    c.d_model       = h.d_model;
    c.n_layer       = h.n_layer;
    c.d_state       = h.d_state;
    c.d_conv        = h.d_conv;
    c.expand        = h.expand;
    c.vocab_size    = h.vocab_size;
    c.n_heads       = h.n_heads;
    c.d_head        = h.d_head;
    c.chunk_size       = h.chunk_size;
    c.n_groups         = (h.n_groups == 0) ? 1u : h.n_groups;  // pre-Phase-11 files
    c.norm_before_gate = (h.norm_before_gate != 0);            // pre-Phase-16 → false
    c.default_dtype    = static_cast<DType>(h.default_dtype);
    return c;
}

}  // namespace

SSMFile::SSMFile(const std::string& path)
    : mapping_(path),
      buffer_(mapping_) {

    const std::uint8_t* base = static_cast<const std::uint8_t*>(mapping_.data());
    const std::size_t   size = mapping_.file_size();

    // --- header ----------------------------------------------------------
    if (size < sizeof(FileHeader)) {
        throw std::runtime_error("SSMFile: file too small to contain a header (" +
                                 std::to_string(size) + " bytes)");
    }
    FileHeader hdr{};
    std::memcpy(&hdr, base, sizeof(hdr));

    if (std::memcmp(hdr.magic, SSM_MAGIC, sizeof(SSM_MAGIC)) != 0) {
        throw std::runtime_error("SSMFile: bad magic; not a .ssm file");
    }
    if (hdr.version != SSM_VERSION) {
        throw std::runtime_error("SSMFile: unsupported version " + std::to_string(hdr.version) +
                                 " (expected " + std::to_string(SSM_VERSION) + ")");
    }
    if (hdr.header_size != sizeof(FileHeader)) {
        throw std::runtime_error("SSMFile: header_size mismatch — file claims " +
                                 std::to_string(hdr.header_size) +
                                 " bytes, build expects " + std::to_string(sizeof(FileHeader)));
    }
    if (hdr.index_offset < sizeof(FileHeader) || hdr.index_offset > size ||
        hdr.data_offset  < hdr.index_offset   || hdr.data_offset  > size) {
        throw std::runtime_error("SSMFile: index/data offsets out of range");
    }

    config_ = config_from_header(hdr);

    // --- index walk ------------------------------------------------------
    tensors_.reserve(hdr.n_tensors);

    const std::uint8_t* cursor = base + hdr.index_offset;
    const std::uint8_t* limit  = base + hdr.data_offset;

    for (std::uint32_t i = 0; i < hdr.n_tensors; ++i) {
        check_in_bounds(base, size, cursor, sizeof(TensorIndexPrefix), "tensor index prefix");

        TensorIndexPrefix prefix{};
        std::memcpy(&prefix, cursor, sizeof(prefix));
        cursor += sizeof(prefix);

        if (prefix.rank == 0 || prefix.rank > TENSOR_MAX_RANK) {
            throw std::runtime_error("SSMFile: tensor " + std::to_string(i) +
                                     " has unsupported rank " + std::to_string(prefix.rank));
        }

        const std::size_t shape_nbytes = sizeof(std::uint64_t) * prefix.rank;
        check_in_bounds(base, size, cursor, shape_nbytes, "tensor shape");

        Tensor t{};
        t.dtype        = static_cast<DType>(prefix.dtype);
        t.rank         = prefix.rank;
        t.offset_bytes = prefix.offset;
        t.nbytes       = prefix.nbytes;
        std::memcpy(t.shape.data(), cursor, shape_nbytes);
        cursor += shape_nbytes;

        check_in_bounds(base, size, cursor, prefix.name_len, "tensor name");
        std::string name(reinterpret_cast<const char*>(cursor), prefix.name_len);
        cursor += prefix.name_len;

        // entry padding: round entry size up to 8 bytes
        const std::size_t entry_raw = sizeof(prefix) + shape_nbytes + prefix.name_len;
        const std::size_t pad       = align_up(entry_raw, 8) - entry_raw;
        cursor += pad;

        // sanity: payload lies inside the file and matches the declared size
        if (t.offset_bytes + t.nbytes > size) {
            throw std::runtime_error("SSMFile: payload for '" + name + "' overflows file");
        }
        const std::size_t expected = dtype_packed_nbytes(t.dtype, t.numel());
        if (expected != t.nbytes) {
            throw std::runtime_error("SSMFile: '" + name + "' nbytes=" +
                                     std::to_string(t.nbytes) + " but expected packed=" +
                                     std::to_string(expected));
        }

        if (!tensors_.emplace(std::move(name), t).second) {
            throw std::runtime_error("SSMFile: duplicate tensor name in index");
        }

        if (cursor > limit) {
            throw std::runtime_error("SSMFile: index walk overran data_offset");
        }
    }
}

const Tensor* SSMFile::find(std::string_view name) const {
    // unordered_map keyed by std::string; .find with string_view needs a
    // transparent hash/equal in C++20. Keep it simple: construct a temporary.
    auto it = tensors_.find(std::string(name));
    return (it == tensors_.end()) ? nullptr : &it->second;
}

const Tensor& SSMFile::at(std::string_view name) const {
    const Tensor* t = find(name);
    if (!t) {
        throw std::out_of_range("SSMFile: no tensor named '" + std::string(name) + "'");
    }
    return *t;
}

}  // namespace lite_ssm
