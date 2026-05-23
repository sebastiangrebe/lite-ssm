"""tools/dump_parity.py

Run AntonV/mamba2-130m-hf in fp16 on a fixed token sequence and dump the
last-position logits (fp32) so the C++ engine can be compared against
PyTorch's reference numerics.

Output file format (parity.bin, little-endian):

  magic       u32   = 0x50415254 ('PART' for Parity)
  version     u32   = 1
  n_tokens    u32
  vocab_size  u32
  tokens      u32 * n_tokens
  logits      f32 * vocab_size       # logits at position n_tokens - 1
"""

from __future__ import annotations

import argparse
import struct
import sys
import warnings
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM

# Silence harmless HF/urllib3 chatter.
warnings.filterwarnings("ignore")


MAGIC   = 0x50415254
VERSION = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AntonV/mamba2-130m-hf")
    ap.add_argument("--out",   default="parity.bin")
    ap.add_argument("--text",  default="The quick brown fox jumps over",
                    help="Prompt text; encoded as byte tokens (matches the C++ side's "
                         "byte-fallback path).")
    args = ap.parse_args()

    print(f"[parity] loading {args.model} in fp16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.eval()

    # Byte-fallback token ids — matches lite_ssm::Tokenizer::encode_bytes.
    tokens: List[int] = [b for b in args.text.encode("utf-8")]
    n_tokens = len(tokens)
    print(f"[parity] prompt {args.text!r} -> {n_tokens} byte tokens: {tokens}", flush=True)

    vocab_size = int(model.config.vocab_size)

    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=ids)
        # logits: (1, n_tokens, vocab). We want the LAST position.
        last_logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
    assert last_logits.shape == (vocab_size,), last_logits.shape
    print(f"[parity] last-token logits: min={last_logits.min():.4f} "
          f"max={last_logits.max():.4f} argmax={int(last_logits.argmax())}", flush=True)

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, n_tokens, vocab_size))
        f.write(struct.pack(f"<{n_tokens}I", *tokens))
        f.write(last_logits.astype("float32").tobytes())
    print(f"[parity] wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
