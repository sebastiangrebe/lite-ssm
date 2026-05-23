# `.ssm` Binary Format (v1)

Single source of truth: [`include/lite_ssm/ssm_format.hpp`](../include/lite_ssm/ssm_format.hpp).
Python mirror: [`tools/ssm_format.py`](../tools/ssm_format.py).

## Goals

- mmap-friendly: contiguous file, fixed-position header at offset 0.
- Self-describing: every tensor names itself, dtype, shape, byte offset.
- Aligned: each tensor payload starts at a `TENSOR_DATA_ALIGN` (64 B) boundary
  so GPU vector loads (`half8`, `float4`) and simdgroup operations don't
  hit unaligned faults.
- One mmap, one MTLBuffer. Tensors are `(offset, nbytes)` views into the
  shared mapping — no per-tensor allocations.

## Layout

```
+----------------------------------+ offset 0
| FileHeader               (96 B)  |
+----------------------------------+ index_offset
| TensorIndexEntry * n_tensors     |  variable, each 8-byte aligned
+----------------------------------+
| pad to TENSOR_DATA_ALIGN (64 B)  |
+----------------------------------+ data_offset
| Tensor 0 payload                 |
| pad to 64 B                      |
| Tensor 1 payload                 |
| pad to 64 B                      |
| ...                              |
+----------------------------------+
```

## FileHeader (96 bytes, little-endian)

| Field          | Type    | Notes                                          |
|----------------|---------|------------------------------------------------|
| magic          | char[4] | `"LSSM"`                                       |
| version        | u32     | `SSM_VERSION = 1`                              |
| header_size    | u32     | `sizeof(FileHeader)` (96); guards future growth |
| n_tensors      | u32     |                                                |
| index_offset   | u64     | byte offset of first index entry               |
| data_offset    | u64     | byte offset of first tensor payload            |
| d_model        | u32     | Mamba-2 hyperparameter                         |
| n_layer        | u32     |                                                |
| d_state        | u32     | SSM state dim                                  |
| d_conv         | u32     | short conv kernel width                        |
| expand         | u32     | inner expansion factor                         |
| vocab_size     | u32     |                                                |
| n_heads        | u32     | Mamba-2 SSD heads                              |
| d_head         | u32     |                                                |
| chunk_size     | u32     | SSD chunk size for prefill                     |
| default_dtype  | u32     | DType code; per-tensor may override            |
| reserved[6]    | u32x6   | zeroed; pads header to 96 B (8-byte boundary)  |

## TensorIndexEntry

Variable size. Each entry padded to 8 bytes so the next entry's u64s stay aligned.

```
+----+ +----+ +----+ +----+ +--------+ +--------+ +----+...+ +----+...+ +----+
|nlen| |rank| |dty.| |pad0| | offset | | nbytes | | shape   | | name    | | pad |
| u32| | u32| | u32| | u32| |  u64   | |  u64   | | u64*rank| |char*nl  | |0..7B|
+----+ +----+ +----+ +----+ +--------+ +--------+ +----+...+ +----+...+ +----+
```

## dtype codes

| Code | Name | Bytes |
|------|------|-------|
| 0    | f16  | 2     |
| 1    | bf16 | 2     |
| 2    | f32  | 4     |
| 3    | i8   | 1     |
| 4    | u8   | 1     |
| 5    | i32  | 4     |

## C++ load sketch

```cpp
int fd = ::open(path, O_RDONLY);
struct stat st;  ::fstat(fd, &st);
auto* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

auto* hdr = reinterpret_cast<const lite_ssm::FileHeader*>(base);
// validate magic/version/header_size

auto* cursor = static_cast<const uint8_t*>(base) + hdr->index_offset;
for (uint32_t i = 0; i < hdr->n_tensors; ++i) {
    auto* prefix = reinterpret_cast<const lite_ssm::TensorIndexPrefix*>(cursor);
    const uint64_t* shape = reinterpret_cast<const uint64_t*>(prefix + 1);
    const char*     name  = reinterpret_cast<const char*>(shape + prefix->rank);
    const uint8_t*  data  = static_cast<const uint8_t*>(base) + prefix->offset;
    // ... register (name, dtype, shape, data) into the model
    size_t raw  = sizeof(*prefix) + 8 * prefix->rank + prefix->name_len;
    cursor += lite_ssm::align_up(raw, 8);
}
```
