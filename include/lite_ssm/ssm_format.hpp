#pragma once
// lite_ssm/ssm_format.hpp — .ssm binary on-disk spec.
//
// Single source of truth for the file format. tools/ssm_format.py mirrors
// the constants and struct layout exactly.
//
// Layout (see docs/ssm_format.md for the full diagram):
//
//   [ FileHeader (96 B) ]
//   [ TensorIndexEntry * n_tensors (variable, 8-byte aligned each) ]
//   [ pad to TENSOR_DATA_ALIGN ]
//   [ Tensor payloads, each TENSOR_DATA_ALIGN-aligned ]
//
// All multi-byte integers are little-endian. The file is intended to be
// mmap'd whole; tensors are then (offset, nbytes) views into that mapping.

#include <cstddef>
#include <cstdint>

namespace lite_ssm {

inline constexpr char     SSM_MAGIC[4]      = {'L', 'S', 'S', 'M'};
inline constexpr uint32_t SSM_VERSION       = 1;

// 64 B covers Apple Silicon cache lines and Metal half8 / float4 vector
// loads. Keep in sync with tools/ssm_format.py:TENSOR_DATA_ALIGN.
inline constexpr size_t   TENSOR_DATA_ALIGN = 64;

enum DType : uint32_t {
    DTYPE_F16        = 0,
    DTYPE_BF16       = 1,
    DTYPE_F32        = 2,
    DTYPE_I8         = 3,
    DTYPE_U8         = 4,
    DTYPE_I32        = 5,
    // Phase 14: weight-only quantization. 32-element block, signed 4-bit
    // nibbles (range [-8, 7]) + one fp16 scale per block. Layout per tensor:
    //   [packed nibbles: numel/2 bytes][per-block scales: numel/32 * 2 bytes]
    // Compression: 9 bits per fp16 weight (vs 16). ~3.55x smaller.
    DTYPE_INT4_B32   = 6,
};

inline constexpr size_t INT4_BLOCK_SIZE = 32;

// Per-element size in bytes for "regular" dtypes. Returns 0 for block-quantized
// dtypes — callers should use `dtype_packed_nbytes` instead, which is layout-
// aware.
inline constexpr size_t dtype_itemsize(DType dt) {
    switch (dt) {
        case DTYPE_F16:  return 2;
        case DTYPE_BF16: return 2;
        case DTYPE_F32:  return 4;
        case DTYPE_I8:   return 1;
        case DTYPE_U8:   return 1;
        case DTYPE_I32:  return 4;
        case DTYPE_INT4_B32: return 0;   // block-packed, not a single itemsize
    }
    return 0;
}

// On-disk packed size for a tensor with `numel` logical elements. Handles
// the INT4_B32 block layout: numel/2 bytes of nibbles + numel/32 * 2 bytes
// of fp16 scales.
inline constexpr size_t dtype_packed_nbytes(DType dt, size_t numel) {
    if (dt == DTYPE_INT4_B32) {
        return (numel / 2) + (numel / INT4_BLOCK_SIZE) * 2;
    }
    return numel * dtype_itemsize(dt);
}

#pragma pack(push, 1)

// Fixed 96-byte header at file offset 0.
struct FileHeader {
    char     magic[4];          // "LSSM"
    uint32_t version;           // SSM_VERSION
    uint32_t header_size;       // sizeof(FileHeader) == 96
    uint32_t n_tensors;
    uint64_t index_offset;      // byte offset of first index entry
    uint64_t data_offset;       // byte offset of first tensor payload

    // Mamba-2 hyperparameters
    uint32_t d_model;
    uint32_t n_layer;
    uint32_t d_state;
    uint32_t d_conv;
    uint32_t expand;
    uint32_t vocab_size;
    uint32_t n_heads;
    uint32_t d_head;
    uint32_t chunk_size;        // SSD chunk size
    uint32_t default_dtype;     // DType code; per-tensor may differ

    // Phase 11: repurposed the first reserved word for n_groups. Old files
    // (pre-Phase 11) wrote zero here; the loader treats zero as 1 so the
    // .ssm format remains backward-compatible at version=1.
    uint32_t n_groups;
    // Phase 16: norm_before_gate. 0 = standard Mamba-2 (gate-then-norm),
    // 1 = Codestral's inverted order (norm-then-gate). Old files default
    // to 0 → backward-compatible.
    uint32_t norm_before_gate;
    uint32_t reserved[4];       // zeroed; pads header to 96 B so the first
                                // index entry's u64 fields land 8-aligned
};

static_assert(sizeof(FileHeader) == 96,
              "FileHeader must be exactly 96 bytes — keep in lockstep with tools/ssm_format.py");

// Fixed prefix of each TensorIndexEntry. The variable-length shape (u64[rank])
// and name (char[name_len]) follow immediately, then the entry is padded out
// to 8 bytes so the next entry's u64 fields stay aligned.
struct TensorIndexPrefix {
    uint32_t name_len;          // bytes, no NUL
    uint32_t rank;
    uint32_t dtype;             // DType code
    uint32_t pad0;              // keeps the next u64 8-aligned
    uint64_t offset;            // absolute byte offset in the file
    uint64_t nbytes;            // payload size in bytes
};

static_assert(sizeof(TensorIndexPrefix) == 32,
              "TensorIndexPrefix must be exactly 32 bytes — keep in lockstep with tools/ssm_format.py");

#pragma pack(pop)

inline constexpr size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

}  // namespace lite_ssm
