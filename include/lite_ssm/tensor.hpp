#pragma once
// lite_ssm/tensor.hpp — non-owning Tensor view into a single shared MTLBuffer.
//
// A Tensor is a description, not a container: it never allocates or frees
// memory. The bytes live inside the mmap'd .ssm file, which is wrapped by
// one big MTLBuffer (see UnifiedBuffer). To bind a tensor to a Metal
// kernel, hand the encoder (buffer, offset_bytes) and the kernel reads
// `nbytes` worth of `dtype`.

#include <array>
#include <cstddef>
#include <cstdint>

#include "lite_ssm/ssm_format.hpp"

namespace lite_ssm {

// Max rank we ever expect from a Mamba-2 checkpoint.
//   conv1d.weight is 3D, in_proj.weight is 2D, biases/scalars are 1D.
// Fixed array avoids any heap allocation per Tensor.
inline constexpr uint32_t TENSOR_MAX_RANK = 4;

struct Tensor {
    DType                                   dtype        = DTYPE_F16;
    uint32_t                                rank         = 0;
    std::array<uint64_t, TENSOR_MAX_RANK>   shape        = {0, 0, 0, 0};
    uint64_t                                offset_bytes = 0;   // into shared MTLBuffer
    uint64_t                                nbytes       = 0;

    uint64_t numel() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < rank; ++i) n *= shape[i];
        return n;
    }

    uint64_t dim(uint32_t i) const { return shape[i]; }

    bool empty() const { return nbytes == 0; }
};

}  // namespace lite_ssm
