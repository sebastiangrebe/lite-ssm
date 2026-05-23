"""tools/ssm_format.py

On-disk specification for the lite-ssm `.ssm` weight file.

Mirror of `include/lite_ssm/ssm_format.hpp`. Keep the two files in lockstep.

Goals
-----
* Trivial to mmap from C++ (single contiguous file, fixed-position header,
  64-byte aligned tensor payloads so the GPU can load via vector / simdgroup
  instructions without unaligned faults).
* Self-describing: every tensor names itself, its dtype, its shape, and its
  byte offset, so the C++ loader builds a name -> view map in one pass.
* Forward compatible: a version field plus a reserved word in the header
  let us extend the hyperparameter struct without rewriting old loaders.

Layout
------
    +----------------------------------+
    | FileHeader              (96 B)   |
    +----------------------------------+
    | TensorIndexEntry * n_tensors     |   variable
    +----------------------------------+
    | pad to TENSOR_DATA_ALIGN (64 B)  |
    +----------------------------------+
    | Tensor 0 payload (64 B aligned)  |
    | pad to 64 B                      |
    | Tensor 1 payload (64 B aligned)  |
    | ...                              |
    +----------------------------------+

FileHeader
----------
Fixed 96 bytes. Always at file offset 0. Little-endian.

    Field            Type        Bytes
    -------------    --------    -----
    magic            char[4]     4      = "LSSM"
    version          u32         4      = SSM_VERSION
    header_size      u32         4      sizeof(FileHeader), redundant but
                                        lets future versions grow safely
    n_tensors        u32         4
    index_offset     u64         8      byte offset of tensor index
    data_offset      u64         8      byte offset of first tensor payload
    d_model          u32         4
    n_layer          u32         4
    d_state          u32         4
    d_conv           u32         4
    expand           u32         4
    vocab_size       u32         4
    n_heads          u32         4
    d_head           u32         4
    chunk_size       u32         4      Mamba-2 SSD chunk size
    default_dtype    u32         4      DTYPE_* code; per-tensor may differ
    n_groups         u32         4      Mamba-2 SSD group count for B/C
                                        broadcasting. 0 in old files is
                                        backward-compat decoded as 1.
    norm_before_gate u32         4      0 = standard Mamba-2 gate-then-norm
                                        1 = Codestral norm-then-gate. Default
                                        for older files: 0.
    reserved         u32[4]      16     zeroed; pads header to 96 B

TensorIndexEntry
----------------
Variable size because of the name. Encoded contiguously.

    Field            Type        Bytes
    -------------    --------    -----
    name_len         u32         4      length of name in bytes (no NUL)
    rank             u32         4
    dtype            u32         4      DTYPE_* code
    pad0             u32         4      keep next u64 8-aligned
    offset           u64         8      absolute byte offset in the file
    nbytes           u64         8      payload size in bytes
    shape            u64[rank]   8*rank
    name             char[name_len]
    pad              align entry to 8 bytes

Tensor payload
--------------
Each tensor starts at an offset aligned to `TENSOR_DATA_ALIGN`. Bytes are
the raw flat buffer in C-order (row-major), dtype-encoded.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Constants — must match include/lite_ssm/ssm_format.hpp
# -----------------------------------------------------------------------------

SSM_MAGIC: bytes = b"LSSM"
SSM_VERSION: int = 1

# Alignment for each tensor payload. 64 bytes covers Apple Silicon cache
# lines and Metal vector loads (float4/half8). Increase to page size if we
# ever want per-tensor MTLBuffers (we don't — one buffer over the whole
# mmap region is the plan).
TENSOR_DATA_ALIGN: int = 64

# dtype codes — keep in lockstep with include/lite_ssm/ssm_format.hpp
DTYPE_F16: int      = 0
DTYPE_BF16: int     = 1
DTYPE_F32: int      = 2
DTYPE_I8: int       = 3
DTYPE_U8: int       = 4
DTYPE_I32: int      = 5
DTYPE_INT4_B32: int = 6   # Phase 14: 32-element blocks, signed 4-bit + fp16 scale

INT4_BLOCK_SIZE = 32

DTYPE_NAMES = {
    DTYPE_F16:      "f16",
    DTYPE_BF16:     "bf16",
    DTYPE_F32:      "f32",
    DTYPE_I8:       "i8",
    DTYPE_U8:       "u8",
    DTYPE_I32:      "i32",
    DTYPE_INT4_B32: "int4_b32",
}

DTYPE_ITEMSIZE = {
    DTYPE_F16: 2,
    DTYPE_BF16: 2,
    DTYPE_F32: 4,
    DTYPE_I8: 1,
    DTYPE_U8: 1,
    DTYPE_I32: 4,
}


def packed_nbytes(dtype: int, numel: int) -> int:
    """On-disk packed size in bytes. Layout-aware for block-quantized dtypes."""
    if dtype == DTYPE_INT4_B32:
        return (numel // 2) + (numel // INT4_BLOCK_SIZE) * 2
    return numel * DTYPE_ITEMSIZE[dtype]

# struct format strings (little-endian)
HEADER_STRUCT = "<4s I I I Q Q I I I I I I I I I I I 5I"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)
assert HEADER_SIZE == 96, f"FileHeader must be 96 bytes, got {HEADER_SIZE}"

# Fixed-size prefix of each index entry (before shape and name).
INDEX_PREFIX_STRUCT = "<I I I I Q Q"
INDEX_PREFIX_SIZE = struct.calcsize(INDEX_PREFIX_STRUCT)
assert INDEX_PREFIX_SIZE == 32

# -----------------------------------------------------------------------------
# Hyperparameter container
# -----------------------------------------------------------------------------

@dataclass
class Mamba2Hparams:
    d_model: int
    n_layer: int
    d_state: int
    d_conv: int
    expand: int
    vocab_size: int
    n_heads: int
    d_head: int
    chunk_size: int = 256
    n_groups: int = 1
    norm_before_gate: int = 0     # 0 = gate-then-norm, 1 = norm-then-gate (Codestral)
    default_dtype: int = DTYPE_F16


@dataclass
class TensorEntry:
    name: str
    dtype: int
    shape: Tuple[int, ...]
    offset: int = 0    # filled in by the writer
    nbytes: int = 0    # filled in by the writer

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def expected_nbytes(self) -> int:
        return packed_nbytes(self.dtype, self.numel())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def pack_header(h: Mamba2Hparams, n_tensors: int, index_offset: int, data_offset: int) -> bytes:
    return struct.pack(
        HEADER_STRUCT,
        SSM_MAGIC,
        SSM_VERSION,
        HEADER_SIZE,
        n_tensors,
        index_offset,
        data_offset,
        h.d_model,
        h.n_layer,
        h.d_state,
        h.d_conv,
        h.expand,
        h.vocab_size,
        h.n_heads,
        h.d_head,
        h.chunk_size,
        h.default_dtype,
        h.n_groups,             # repurposed reserved[0] in Phase 11
        h.norm_before_gate,     # repurposed reserved[1] in Phase 16
        0, 0, 0, 0,
    )


def unpack_header(buf: bytes) -> Tuple[Mamba2Hparams, dict]:
    """Returns (hparams, meta) where meta has n_tensors / index_offset / data_offset / version."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"Header truncated: {len(buf)} < {HEADER_SIZE}")
    fields = struct.unpack(HEADER_STRUCT, buf[:HEADER_SIZE])
    (magic, version, hsize, n_tensors, idx_off, data_off,
     d_model, n_layer, d_state, d_conv, expand, vocab_size,
     n_heads, d_head, chunk_size, default_dtype,
     n_groups, norm_before_gate, _r2, _r3, _r4, _r5) = fields
    if n_groups == 0:
        # Backward compat: pre-Phase-11 exporters wrote zero here.
        n_groups = 1
    if magic != SSM_MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {SSM_MAGIC!r})")
    if version != SSM_VERSION:
        raise ValueError(f"Unsupported version {version} (expected {SSM_VERSION})")
    if hsize != HEADER_SIZE:
        raise ValueError(f"Header size mismatch: file={hsize} code={HEADER_SIZE}")
    hp = Mamba2Hparams(
        d_model=d_model, n_layer=n_layer, d_state=d_state, d_conv=d_conv,
        expand=expand, vocab_size=vocab_size, n_heads=n_heads, d_head=d_head,
        chunk_size=chunk_size, n_groups=n_groups,
        norm_before_gate=norm_before_gate,
        default_dtype=default_dtype,
    )
    meta = dict(version=version, n_tensors=n_tensors,
                index_offset=idx_off, data_offset=data_off)
    return hp, meta


def pack_index_entry(entry: TensorEntry) -> bytes:
    name_bytes = entry.name.encode("utf-8")
    rank = len(entry.shape)
    prefix = struct.pack(
        INDEX_PREFIX_STRUCT,
        len(name_bytes),       # name_len
        rank,                  # rank
        entry.dtype,           # dtype
        0,                     # pad0
        entry.offset,          # offset
        entry.nbytes,          # nbytes
    )
    shape_blob = struct.pack(f"<{rank}Q", *[int(d) for d in entry.shape])
    body = prefix + shape_blob + name_bytes
    # Pad each entry to 8 bytes so the next entry's u32 / u64 fields stay aligned.
    pad_len = align_up(len(body), 8) - len(body)
    return body + b"\x00" * pad_len


def index_entry_size(name: str, rank: int) -> int:
    name_len = len(name.encode("utf-8"))
    raw = INDEX_PREFIX_SIZE + 8 * rank + name_len
    return align_up(raw, 8)


def total_index_size(entries: List[TensorEntry]) -> int:
    return sum(index_entry_size(e.name, len(e.shape)) for e in entries)


def parse_index(buf: bytes, n_tensors: int) -> List[TensorEntry]:
    out: List[TensorEntry] = []
    pos = 0
    for _ in range(n_tensors):
        if pos + INDEX_PREFIX_SIZE > len(buf):
            raise ValueError("Index truncated reading prefix")
        (name_len, rank, dtype, _pad0, offset, nbytes) = struct.unpack(
            INDEX_PREFIX_STRUCT, buf[pos:pos + INDEX_PREFIX_SIZE]
        )
        pos += INDEX_PREFIX_SIZE
        shape = struct.unpack(f"<{rank}Q", buf[pos:pos + 8 * rank])
        pos += 8 * rank
        name = buf[pos:pos + name_len].decode("utf-8")
        pos += name_len
        # consume entry padding
        entry_raw = INDEX_PREFIX_SIZE + 8 * rank + name_len
        pos += align_up(entry_raw, 8) - entry_raw
        out.append(TensorEntry(name=name, dtype=dtype, shape=tuple(shape),
                               offset=offset, nbytes=nbytes))
    return out


# -----------------------------------------------------------------------------
# High-level writer
# -----------------------------------------------------------------------------

@dataclass
class FilePlan:
    """Layout decision: where each tensor lands in the file."""
    header: bytes
    index_blob: bytes
    data_offset: int
    entries: List[TensorEntry] = field(default_factory=list)


def plan_layout(hparams: Mamba2Hparams, entries: List[TensorEntry]) -> FilePlan:
    """Compute byte offsets and pack the header + index.

    Mutates `entries` in place: sets `offset` and `nbytes` for each one.
    """
    # Pass 1: compute index size.
    n = len(entries)
    index_size = total_index_size(entries)
    index_offset = HEADER_SIZE
    data_offset = align_up(index_offset + index_size, TENSOR_DATA_ALIGN)

    # Pass 2: assign aligned offsets in payload region.
    cursor = data_offset
    for e in entries:
        e.nbytes = e.expected_nbytes()
        e.offset = cursor
        cursor = align_up(cursor + e.nbytes, TENSOR_DATA_ALIGN)

    # Pass 3: serialize header + index now that offsets are final.
    header = pack_header(hparams, n, index_offset, data_offset)
    index_blob = b"".join(pack_index_entry(e) for e in entries)
    assert len(index_blob) == index_size

    return FilePlan(header=header, index_blob=index_blob,
                    data_offset=data_offset, entries=entries)
