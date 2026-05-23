"""tools/export_tokenizer.py

Phase 12 — Native SentencePiece edition.

We no longer compile vocab+merges into a custom `.bin`. Google SentencePiece
ships its own protobuf format (`tokenizer.model`) and our C++ runtime now
loads that directly. This script's job is to:

  1. Download the requested HF model.
  2. Locate its `tokenizer.model` artifact in the snapshot cache.
  3. Copy / symlink it to the requested output path (default `tokenizer.model`).
  4. Optionally produce `tokenizer_refs.bin` — a small parity corpus that
     the C++ unit test (`tests/test_tokenizer.cpp`) cross-checks against.

Default target: mistralai/Mamba-Codestral-7B-v0.1 (Codestral 7B Mamba —
the model that motivated this phase). Pass `--model <hub-id>` to point at
any other HF model that publishes a SentencePiece `tokenizer.model`.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download   # noqa: E402
from transformers import AutoTokenizer        # noqa: E402


MAGIC_REF = b"LREF"
VERSION   = 1


def fetch_tokenizer_model(repo_id: str) -> Path:
    """Resolve the path of `tokenizer.model` inside the HF snapshot cache."""
    path = Path(hf_hub_download(repo_id=repo_id, filename="tokenizer.model"))
    if not path.exists():
        raise SystemExit(f"tokenizer.model not present in {repo_id} snapshot")
    return path


def pad4(buf: bytearray) -> None:
    while len(buf) % 4:
        buf.append(0)


def write_refs(out_path: Path, tok) -> None:
    cases = [
        "Hello, world!",
        "Hello, world! 🚀 This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Don't stop believing — it's only the beginning.",
        "Mamba-2 is a state-space model. The price is $1,234.56.",
        "1 + 2 = 3, but π ≈ 3.14159.",
        "Mixed: ascii + 中文 + русский + emoji 🎉🔥",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n",
    ]
    buf = bytearray()
    buf += struct.pack("<4s I I I", MAGIC_REF, VERSION, len(cases), 0)
    for text in cases:
        ids = tok.encode(text, add_special_tokens=False)
        text_b = text.encode("utf-8")
        buf += struct.pack("<I I", len(text_b), len(ids))
        buf += text_b
        for i in ids:
            buf += struct.pack("<I", i)
        pad4(buf)
        print(f"[export_tokenizer] ref {text!r:60.60} -> {len(ids):3d} ids", flush=True)
    out_path.write_bytes(buf)
    print(f"[export_tokenizer] wrote {out_path} ({len(buf)} bytes)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mamba-Codestral-7B-v0.1",
                    help="HF repo id that publishes a SentencePiece tokenizer.model.")
    ap.add_argument("--out",   default="tokenizer.model",
                    help="Where to drop the .model file (default: ./tokenizer.model)")
    ap.add_argument("--refs",  default="tokenizer_refs.bin",
                    help="Where to drop the parity refs for the C++ unit test.")
    args = ap.parse_args()

    print(f"[export_tokenizer] target = {args.model}", flush=True)
    src = fetch_tokenizer_model(args.model)
    out = Path(args.out)
    if out.resolve() != src.resolve():
        shutil.copyfile(src, out)
    print(f"[export_tokenizer] tokenizer.model -> {out} ({out.stat().st_size} bytes)", flush=True)

    # Refs are produced with HF's tokenizer (which wraps SP for these models),
    # giving us ground-truth ids the C++ side must match exactly.
    print(f"[export_tokenizer] loading HF tokenizer for parity refs", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    write_refs(Path(args.refs), tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
