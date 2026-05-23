"""tools/verify_export.py

Sanity-check a `.ssm` file: parse the header + index, walk each tensor's
byte offset, confirm the payload is reachable and 64-byte aligned.

Usage:
    python tools/verify_export.py model.ssm
    python tools/verify_export.py model.ssm --show 20  # print first N tensors
"""

from __future__ import annotations

import argparse
import mmap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssm_format as fmt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a .ssm file")
    ap.add_argument("path", help="Path to .ssm")
    ap.add_argument("--show", type=int, default=10,
                    help="How many tensors to print (default 10, 0 = none, -1 = all)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[verify] file not found: {path}", file=sys.stderr)
        return 2

    file_size = path.stat().st_size
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        # Header
        hparams, meta = fmt.unpack_header(bytes(mm[:fmt.HEADER_SIZE]))
        print(f"[verify] {path} ({file_size / 1e6:.2f} MB)")
        print(f"[verify] magic OK, version={meta['version']}, n_tensors={meta['n_tensors']}")
        print(f"[verify] index_offset={meta['index_offset']} data_offset={meta['data_offset']}")
        print(f"[verify] hparams: d_model={hparams.d_model} n_layer={hparams.n_layer} "
              f"d_state={hparams.d_state} d_conv={hparams.d_conv} expand={hparams.expand} "
              f"vocab_size={hparams.vocab_size} n_heads={hparams.n_heads} d_head={hparams.d_head} "
              f"chunk_size={hparams.chunk_size} default_dtype={fmt.DTYPE_NAMES.get(hparams.default_dtype)}")

        # Index — read from index_offset to data_offset (minus any alignment padding).
        idx_blob = bytes(mm[meta["index_offset"]:meta["data_offset"]])
        entries = fmt.parse_index(idx_blob, meta["n_tensors"])

        # Walk each entry: confirm alignment, in-bounds, expected size.
        errors = 0
        total_payload = 0
        for e in entries:
            total_payload += e.nbytes
            if e.offset % fmt.TENSOR_DATA_ALIGN != 0:
                print(f"[verify] !! {e.name}: offset {e.offset} not {fmt.TENSOR_DATA_ALIGN}-byte aligned")
                errors += 1
            if e.offset + e.nbytes > file_size:
                print(f"[verify] !! {e.name}: payload exceeds file (offset={e.offset} nbytes={e.nbytes} file={file_size})")
                errors += 1
            expected = e.numel() * fmt.DTYPE_ITEMSIZE[e.dtype]
            if expected != e.nbytes:
                print(f"[verify] !! {e.name}: nbytes={e.nbytes} but shape*itemsize={expected}")
                errors += 1
            # Probe the first and last byte of the payload to confirm reachability.
            _ = mm[e.offset]
            _ = mm[e.offset + e.nbytes - 1]

        if errors:
            print(f"[verify] FAIL: {errors} errors across {len(entries)} tensors")
            return 1

        print(f"[verify] OK: {len(entries)} tensors, {total_payload / 1e6:.2f} MB payload, all offsets aligned + in bounds")

        # Print a sample
        if args.show != 0:
            show = entries if args.show < 0 else entries[:args.show]
            print(f"[verify] first {len(show)} tensors:")
            name_w = max(len(e.name) for e in show)
            for e in show:
                shape_s = "(" + ", ".join(str(d) for d in e.shape) + ")"
                print(f"  {e.name:<{name_w}}  dtype={fmt.DTYPE_NAMES.get(e.dtype, '?'):<4}  "
                      f"shape={shape_s:<24}  offset={e.offset:>10}  nbytes={e.nbytes:>10}")
        return 0
    finally:
        mm.close()


if __name__ == "__main__":
    raise SystemExit(main())
