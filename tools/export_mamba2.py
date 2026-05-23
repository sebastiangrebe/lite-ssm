"""tools/export_mamba2.py

Export a HuggingFace Mamba-2 checkpoint to the lite-ssm `.ssm` binary
format. Default target: state-spaces/mamba2-130m (small, iterates fast).

All tensors are cast to float16 — Metal's fast path on Apple Silicon GPUs.

Usage:
    python tools/export_mamba2.py                          # 130m default
    python tools/export_mamba2.py --model state-spaces/mamba2-370m
    python tools/export_mamba2.py --out model.ssm --dtype f16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Local module: keep the format spec in lockstep with the C++ header.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssm_format as fmt  # noqa: E402


DTYPE_TORCH_TO_CODE = {
    torch.float16:  fmt.DTYPE_F16,
    torch.bfloat16: fmt.DTYPE_BF16,
    torch.float32:  fmt.DTYPE_F32,
    torch.int8:     fmt.DTYPE_I8,
    torch.uint8:    fmt.DTYPE_U8,
    torch.int32:    fmt.DTYPE_I32,
}

CAST_DTYPE = {
    "f16":  torch.float16,
    "bf16": torch.bfloat16,
    "f32":  torch.float32,
}

# Phase 14: which tensors to quantize when --quantize int4.
#
# We quantize the big 2D Linear weights — `mixer.in_proj.weight`,
# `mixer.out_proj.weight`, and `lm_head.weight` (when present and untied).
# Embeddings stay fp16 because the C++ engine reads single rows via a CPU
# memcpy through unified memory; on-the-fly dequant during that path would
# complicate the embed_lookup. 1D tensors (norms, biases, A_log, D, dt_bias)
# are kept fp16 per the spec — they're tiny and quantization noise on them
# hurts disproportionately.
def is_int4_quantizable(name: str, shape) -> bool:
    if len(shape) != 2:
        return False
    if shape[-1] % fmt.INT4_BLOCK_SIZE != 0:
        return False
    # Phase 21 — strictly only the two big projection matrices. lm_head must
    # stay fp16: a 4-bit final vocab projection blows up logit KL (>10 nats).
    return (
        name.endswith(".in_proj.weight") or
        name.endswith(".out_proj.weight")
    )


def quantize_int4_block32(t_fp16: torch.Tensor) -> bytes:
    """Block-wise symmetric int4 quantization along the last dim.

    Layout returned (matches `dtype_packed_nbytes(INT4_B32, numel)`):
      [packed nibbles: numel/2 bytes]
      [per-block scales: (numel/32) fp16 = numel/16 bytes]

    Per block of 32 fp16 values:
      scale = max(|w|) / 7.0       (symmetric, range ±7)
      q     = round(w / scale), clipped to [-8, 7]
      pack two int4 nibbles per byte (low = q[2k], high = q[2k+1])
    """
    assert t_fp16.dim() == 2, "int4 quantizer expects 2D weight"
    N, K = t_fp16.shape
    assert K % fmt.INT4_BLOCK_SIZE == 0, f"K={K} not divisible by block 32"

    # Work in fp32 for numerical stability; final scale is downcast to fp16.
    w = t_fp16.detach().to(torch.float32).contiguous()
    blocks = w.view(N, K // fmt.INT4_BLOCK_SIZE, fmt.INT4_BLOCK_SIZE)

    # Per-block absmax + symmetric scale. eps prevents div-by-zero on a
    # block of literal zeros (which can happen near init).
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-12)
    scale  = absmax / 7.0
    scale_fp16 = scale.squeeze(-1).to(torch.float16)   # (N, K/32)

    # Quantize and clip to signed int4 [-8, 7].
    q = torch.round(blocks / scale).clamp_(min=-8, max=7).to(torch.int8)
    q_pairs = q.view(N, K // 2, 2)               # adjacent (low, high) nibbles

    # Two's-complement nibble packing: take low 4 bits of each int8.
    low  = (q_pairs[..., 0] & 0x0F).to(torch.uint8)
    high = (q_pairs[..., 1] & 0x0F).to(torch.uint8)
    packed = (high << 4) | low                   # (N, K/2)

    return packed.cpu().numpy().tobytes() + scale_fp16.cpu().numpy().tobytes()


def _get(cfg, *names, default=None):
    """Pull the first attribute that exists on the HF config."""
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


def hparams_from_config(cfg) -> fmt.Mamba2Hparams:
    """Map a HuggingFace Mamba2Config -> our hparam struct.

    Field names have varied across transformers versions and across the
    state-spaces vs HF ports, so we probe a few aliases for each.
    """
    d_model    = _get(cfg, "hidden_size", "d_model")
    n_layer    = _get(cfg, "num_hidden_layers", "n_layer")
    vocab_size = _get(cfg, "vocab_size")

    d_state    = _get(cfg, "state_size", "ssm_state_size", "d_state", default=128)
    d_conv     = _get(cfg, "conv_kernel", "d_conv", default=4)
    expand     = _get(cfg, "expand", default=2)
    n_heads    = _get(cfg, "num_heads", "n_heads")
    d_head     = _get(cfg, "head_dim", "d_head")
    chunk_size = _get(cfg, "chunk_size", default=256)
    n_groups   = _get(cfg, "n_groups", "num_groups", "ssm_n_groups", default=1)
    norm_before_gate = bool(_get(cfg, "norm_before_gate", default=False))

    # Derive heads/d_head if absent: in Mamba-2, d_inner = expand * d_model = n_heads * d_head.
    if n_heads is None and d_head is not None:
        n_heads = (expand * d_model) // d_head
    if d_head is None and n_heads is not None:
        d_head = (expand * d_model) // n_heads
    if n_heads is None and d_head is None:
        d_head = 64
        n_heads = (expand * d_model) // d_head

    missing = [k for k, v in dict(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size).items() if v is None]
    if missing:
        raise ValueError(f"Could not infer required hparams: {missing}. Config: {cfg}")

    return fmt.Mamba2Hparams(
        d_model=int(d_model),
        n_layer=int(n_layer),
        d_state=int(d_state),
        d_conv=int(d_conv),
        expand=int(expand),
        vocab_size=int(vocab_size),
        n_heads=int(n_heads),
        d_head=int(d_head),
        chunk_size=int(chunk_size),
        n_groups=int(n_groups),
        norm_before_gate=1 if norm_before_gate else 0,
        default_dtype=fmt.DTYPE_F16,
    )


def collect_tensors(state_dict: Dict[str, torch.Tensor],
                    cast_to: torch.dtype,
                    quantize: str = "fp16") -> List[fmt.TensorEntry]:
    """Build the .ssm tensor index. `quantize` controls per-tensor compression:
      * "fp16" — every float tensor stays fp16 (no compression).
      * "int4" — large 2D Linear weights get block-wise int4 quantization;
                 everything else stays fp16.
    """
    entries: List[fmt.TensorEntry] = []
    for name, t in state_dict.items():
        # Floats get cast to fp16 first; integers passthrough.
        if t.is_floating_point():
            t_fp16 = t.detach().to(torch.float16).contiguous()
            if quantize == "int4" and is_int4_quantizable(name, t_fp16.shape):
                e = fmt.TensorEntry(name=name,
                                    dtype=fmt.DTYPE_INT4_B32,
                                    shape=tuple(t_fp16.shape))
                e._packed_blob = quantize_int4_block32(t_fp16)  # type: ignore[attr-defined]
                entries.append(e)
                continue
            t_cast = t.detach().to(cast_to).contiguous()
        else:
            t_cast = t.detach().contiguous()
        code = DTYPE_TORCH_TO_CODE.get(t_cast.dtype)
        if code is None:
            raise ValueError(f"Tensor {name!r} has unsupported dtype {t_cast.dtype}")
        entries.append(fmt.TensorEntry(
            name=name,
            dtype=code,
            shape=tuple(t_cast.shape),
        ))
        # Stash the cast tensor on the entry so the writer can grab the bytes.
        entries[-1]._tensor = t_cast  # type: ignore[attr-defined]
    return entries


def write_ssm(out_path: Path,
              hparams: fmt.Mamba2Hparams,
              entries: List[fmt.TensorEntry]) -> None:
    plan = fmt.plan_layout(hparams, entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(plan.header)
        f.write(plan.index_blob)
        cursor = fmt.HEADER_SIZE + len(plan.index_blob)
        # pad up to data_offset
        f.write(b"\x00" * (plan.data_offset - cursor))
        cursor = plan.data_offset
        for e in entries:
            assert cursor == e.offset, f"layout drift for {e.name}: cursor={cursor} offset={e.offset}"
            # Two paths: pre-packed quantized blob, or a raw torch tensor.
            packed = getattr(e, "_packed_blob", None)
            if packed is not None:
                raw = packed
            else:
                tensor = e._tensor  # type: ignore[attr-defined]
                if tensor.dtype == torch.bfloat16:
                    raw = tensor.contiguous().view(torch.float16).numpy().tobytes()
                else:
                    raw = tensor.cpu().numpy().tobytes()
            assert len(raw) == e.nbytes, f"size mismatch for {e.name}: bytes={len(raw)} expected={e.nbytes}"
            f.write(raw)
            cursor += e.nbytes
            pad = fmt.align_up(cursor, fmt.TENSOR_DATA_ALIGN) - cursor
            if pad:
                f.write(b"\x00" * pad)
                cursor += pad


def main() -> int:
    ap = argparse.ArgumentParser(description="Export HuggingFace Mamba-2 -> .ssm")
    ap.add_argument("--model", default="AntonV/mamba2-130m-hf",
                    help="HF model id or local path (default: AntonV/mamba2-130m-hf — "
                         "the state-spaces/mamba2-130m checkpoint repacked for HF's "
                         "Mamba2Config layout; the native state-spaces repo uses a "
                         "custom config that AutoConfig can't parse)")
    ap.add_argument("--out", default="model.ssm", help="Output .ssm path")
    ap.add_argument("--dtype", default="f16", choices=list(CAST_DTYPE),
                    help="Cast float weights to this dtype (default: f16)")
    ap.add_argument("--quantize", default="fp16", choices=["fp16", "int4"],
                    help="Weight-only quantization for large 2D Linears "
                         "(default: fp16 = no compression). int4 = block-32 "
                         "signed 4-bit + fp16 scale per block (~3.5x smaller).")
    ap.add_argument("--revision", default=None, help="Optional HF revision pin")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[export] loading config for {args.model}", flush=True)
    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    hparams = hparams_from_config(cfg)
    hparams.default_dtype = {"f16": fmt.DTYPE_F16, "bf16": fmt.DTYPE_BF16, "f32": fmt.DTYPE_F32}[args.dtype]
    print(f"[export] hparams: {hparams}", flush=True)

    print(f"[export] loading weights (this can take a while on first run)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32, trust_remote_code=True,
    )
    sd = model.state_dict()
    print(f"[export] state_dict: {len(sd)} tensors", flush=True)

    entries = collect_tensors(sd, CAST_DTYPE[args.dtype], quantize=args.quantize)
    total_payload = sum(e.expected_nbytes() for e in entries)
    n_quant = sum(1 for e in entries if e.dtype == fmt.DTYPE_INT4_B32)
    print(f"[export] quantize mode: {args.quantize}  ({n_quant} tensors int4-packed)", flush=True)
    print(f"[export] total payload: {total_payload / 1e6:.1f} MB", flush=True)

    out_path = Path(args.out)
    write_ssm(out_path, hparams, entries)
    sz = out_path.stat().st_size
    print(f"[export] wrote {out_path} ({sz / 1e6:.1f} MB) in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
