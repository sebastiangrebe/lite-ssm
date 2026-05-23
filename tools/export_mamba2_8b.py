"""tools/export_mamba2_8b.py

Stress-test variant of tools/export_mamba2.py. Where the 130M exporter
loads a single safetensors file in memory, this one is designed for the
multi-shard checkpoints used by larger Mamba-2 ports.

Workflow:
  1. Resolve every safetensors shard for the target model via the HF
     download API (this materializes them in ~/.cache/huggingface).
  2. Iterate shard-by-shard, opening each via `safe_open` (mmap, lazy
     tensor loads). Collect (name, dtype, shape) without loading payloads.
  3. Compute the .ssm layout: header + index + 64-byte-aligned payload
     stripes. Critically, every tensor offset must remain 64-byte aligned
     even when its preceding tensor lives in a different shard.
  4. Walk the layout again, loading each tensor on demand, casting to fp16,
     and streaming bytes straight to disk. Never holds more than one
     tensor in Python memory at a time, so this can produce a ~16 GB file
     on a 16 GB Mac without paging out.

Target spec: `state-spaces/mamba-2-8b`. Note: at the time of writing no
public 8B Mamba-2 checkpoint exists on Hugging Face under that id — the
state-spaces org tops out at mamba2-2.7b, and Mistral's 7B Codestral Mamba
uses a different architecture. Run with `--model <hub-id>` to point at
another sharded checkpoint; we keep the requested id as the default so
the failure mode (404) is visible in the stress report.

Usage:
  python tools/export_mamba2_8b.py                           # default 8b id
  python tools/export_mamba2_8b.py --model AntonV/mamba2-2.7b-hf
  python tools/export_mamba2_8b.py --probe                   # don't download, just report
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssm_format as fmt                                             # noqa: E402
from export_mamba2 import (                                          # noqa: E402
    quantize_int4_block32,
    is_int4_quantizable,
)


DEFAULT_MODEL = "state-spaces/mamba-2-8b"


# -------------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------------
def probe_model(model_id: str) -> Dict[str, object]:
    """Returns metadata about the requested checkpoint without downloading
    any weight tensors. Used by --probe."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:
        return {"available": False, "error": repr(e)}

    shards: List[Tuple[str, int]] = []
    total = 0
    for sib in info.siblings:
        name = sib.rfilename
        sz   = getattr(sib, "size", None) or 0
        if name.endswith(".safetensors") or name.endswith(".bin"):
            shards.append((name, sz))
            total += sz
    cfg = None
    try:
        cfg_path = hf_hub_download(model_id, "config.json")
        cfg = json.loads(Path(cfg_path).read_text())
    except Exception as e:  # noqa: BLE001
        cfg = {"_error": repr(e)}
    return {
        "available": True,
        "id": model_id,
        "n_shards": len(shards),
        "total_bytes": total,
        "shards": shards,
        "config": cfg,
    }


# -------------------------------------------------------------------------
# Two-pass exporter
# -------------------------------------------------------------------------
def _hparams_from_config(cfg) -> fmt.Mamba2Hparams:
    """Same probe-many-aliases dance as tools/export_mamba2.py — needed
    because Mamba-2 ports use slightly different config field names."""
    def g(*names, default=None):
        for n in names:
            if n in cfg: return cfg[n]
        return default
    d_model    = g("hidden_size", "d_model")
    n_layer    = g("num_hidden_layers", "n_layer")
    vocab      = g("vocab_size")
    d_state    = g("state_size", "ssm_state_size", "d_state", default=128)
    d_conv     = g("conv_kernel", "d_conv",                     default=4)
    expand     = g("expand",                                    default=2)
    n_heads    = g("num_heads", "n_heads")
    d_head     = g("head_dim", "d_head")
    chunk_size = g("chunk_size",                                default=256)
    n_groups   = g("n_groups", "num_groups", "ssm_n_groups",   default=1)
    norm_before_gate = bool(g("norm_before_gate",              default=False))

    if n_heads is None and d_head is not None:
        n_heads = (expand * d_model) // d_head
    if d_head is None and n_heads is not None:
        d_head = (expand * d_model) // n_heads
    if d_head is None and n_heads is None:
        d_head = 64
        n_heads = (expand * d_model) // d_head

    return fmt.Mamba2Hparams(
        d_model=int(d_model), n_layer=int(n_layer), d_state=int(d_state),
        d_conv=int(d_conv), expand=int(expand), vocab_size=int(vocab),
        n_heads=int(n_heads), d_head=int(d_head), chunk_size=int(chunk_size),
        n_groups=int(n_groups),
        norm_before_gate=1 if norm_before_gate else 0,
        default_dtype=fmt.DTYPE_F16,
    )


def _scan_shard(shard_path: Path) -> List[Tuple[str, str, Tuple[int, ...], str]]:
    """Return [(tensor_name, dtype_str, shape, shard_path), ...]. Lazy —
    safetensors.safe_open mmaps the shard and exposes metadata cheaply."""
    from safetensors import safe_open
    out = []
    with safe_open(str(shard_path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)        # this still mmaps; cheap
            out.append((k, str(t.dtype).split(".")[-1], tuple(t.shape), str(shard_path)))
            del t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--out",   default="model_8b.ssm")
    ap.add_argument("--probe", action="store_true",
                    help="Don't download weights; print shard inventory + estimated size and exit.")
    ap.add_argument("--quantize", default="fp16", choices=["fp16", "int4"],
                    help="Weight-only quantization for the big 2-D Linears "
                         "(default: fp16). int4 = block-32 signed nibbles + "
                         "fp16 scale (~3.5x smaller per quantized tensor).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[export_8b] target = {args.model}", flush=True)

    # ---- Probe: cheap availability check ---------------------------------
    info = probe_model(args.model)
    if not info["available"]:
        print(f"[export_8b] checkpoint not accessible: {info.get('error')}", flush=True)
        print("[export_8b] hint: the public `state-spaces/mamba-2-8b` id does not exist; "
              "the largest published state-spaces Mamba-2 is `state-spaces/mamba2-2.7b`. "
              "Re-run with `--model <hub-id>` to point at another sharded checkpoint.",
              flush=True)
        return 2

    total_gb = info["total_bytes"] / 1e9
    print(f"[export_8b] shards : {info['n_shards']}", flush=True)
    print(f"[export_8b] size   : {total_gb:.2f} GB on disk", flush=True)
    for name, sz in info["shards"][:8]:
        print(f"[export_8b]    - {name}  ({sz/1e9:.2f} GB)", flush=True)
    if info["n_shards"] > 8:
        print(f"[export_8b]    … and {info['n_shards'] - 8} more", flush=True)

    if args.probe:
        print(f"[export_8b] probe complete in {time.time() - t0:.1f}s; --probe set, exiting.",
              flush=True)
        return 0

    # ---- Full download + scan -------------------------------------------
    from huggingface_hub import snapshot_download
    print(f"[export_8b] downloading shards (cache: ~/.cache/huggingface)…", flush=True)
    snap = Path(snapshot_download(
        args.model,
        allow_patterns=["*.safetensors", "*.bin", "config.json", "model.safetensors.index.json"],
    ))
    print(f"[export_8b] snapshot at {snap}", flush=True)

    cfg_path = snap / "config.json"
    cfg = json.loads(cfg_path.read_text())
    hparams = _hparams_from_config(cfg)
    print(f"[export_8b] hparams: {hparams}", flush=True)

    # Enumerate tensors across all shards (deterministic order: by name).
    shard_files = sorted(snap.glob("*.safetensors")) or sorted(snap.glob("*.bin"))
    if not shard_files:
        print("[export_8b] no safetensors / bin shards found in snapshot", flush=True)
        return 2

    # When a repo ships BOTH a `consolidated.safetensors` (single-file flatten)
    # and the canonical `model-NNNNN-of-NNNNN.safetensors` shards, every tensor
    # appears twice. Prefer the sharded form — that's exactly the multi-shard
    # boundary case we want to exercise.
    if any("of-" in p.name for p in shard_files) and \
       any("consolidated" in p.name for p in shard_files):
        shard_files = [p for p in shard_files if "consolidated" not in p.name]
        print(f"[export_8b] both consolidated + sharded files present; "
              f"using sharded form only ({len(shard_files)} files)", flush=True)
    print(f"[export_8b] {len(shard_files)} shard file(s) on disk", flush=True)

    # Build the global tensor index (pass 1: scan only metadata).
    entries: List[fmt.TensorEntry] = []
    entry_shards: List[str] = []                 # parallel array of source shard paths
    n_quantized = 0
    for shard in shard_files:
        for name, dtype_str, shape, src in _scan_shard(shard):
            # Phase 15: route big 2-D Linears to int4 block-32 when requested.
            if (args.quantize == "int4"
                    and dtype_str in ("float16", "float32", "bfloat16", "float64")
                    and is_int4_quantizable(name, shape)):
                code = fmt.DTYPE_INT4_B32
                n_quantized += 1
            elif dtype_str in ("float16", "float32", "bfloat16", "float64"):
                code = fmt.DTYPE_F16
            elif dtype_str == "int8":   code = fmt.DTYPE_I8
            elif dtype_str == "uint8":  code = fmt.DTYPE_U8
            elif dtype_str == "int32":  code = fmt.DTYPE_I32
            else:
                raise SystemExit(f"unsupported dtype {dtype_str} for {name}")
            entries.append(fmt.TensorEntry(name=name, dtype=code, shape=shape))
            entry_shards.append(src)
    print(f"[export_8b] quantize mode: {args.quantize}  "
          f"({n_quantized} tensors int4-packed)", flush=True)

    plan = fmt.plan_layout(hparams, entries)
    total_out = plan.data_offset + sum(
        fmt.align_up(e.nbytes, fmt.TENSOR_DATA_ALIGN) for e in entries)
    print(f"[export_8b] planned output size = {total_out / 1e9:.2f} GB", flush=True)
    print(f"[export_8b] header={fmt.HEADER_SIZE} B  index={len(plan.index_blob)} B  "
          f"data_offset={plan.data_offset} B  tensors={len(entries)}", flush=True)

    # ---- Pass 2: stream each tensor to the output file ------------------
    from safetensors import safe_open
    out_path = Path(args.out)
    n_boundary_align_checks = 0
    with open(out_path, "wb") as f:
        f.write(plan.header)
        f.write(plan.index_blob)
        cursor = fmt.HEADER_SIZE + len(plan.index_blob)
        if cursor < plan.data_offset:
            f.write(b"\x00" * (plan.data_offset - cursor))
            cursor = plan.data_offset

        last_shard = None
        shard_handle = None

        try:
            for idx, (e, src) in enumerate(zip(entries, entry_shards)):
                if src != last_shard:
                    if shard_handle is not None:
                        shard_handle.__exit__(None, None, None)
                    last_shard   = src
                    shard_handle = safe_open(src, framework="pt").__enter__()
                    n_boundary_align_checks += 1
                    if cursor % fmt.TENSOR_DATA_ALIGN != 0:
                        raise SystemExit(
                            f"alignment broken at shard boundary {src}: cursor={cursor}")

                if cursor != e.offset:
                    raise SystemExit(
                        f"plan/cursor drift at {e.name}: cursor={cursor} planned={e.offset}")

                t = shard_handle.get_tensor(e.name)
                if e.dtype == fmt.DTYPE_INT4_B32:
                    t_fp16 = t.detach().to(torch.float16).contiguous()
                    raw = quantize_int4_block32(t_fp16)
                elif t.is_floating_point():
                    raw = t.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()
                else:
                    raw = t.detach().contiguous().cpu().numpy().tobytes()
                if len(raw) != e.nbytes:
                    raise SystemExit(
                        f"size mismatch for {e.name}: bytes={len(raw)} expected={e.nbytes}")
                f.write(raw)
                cursor += len(raw)
                pad = fmt.align_up(cursor, fmt.TENSOR_DATA_ALIGN) - cursor
                if pad:
                    f.write(b"\x00" * pad)
                    cursor += pad

                if (idx + 1) % 50 == 0 or idx == len(entries) - 1:
                    print(f"[export_8b]   wrote {idx + 1}/{len(entries)} tensors  "
                          f"({cursor / 1e9:.2f} GB)", flush=True)
        finally:
            if shard_handle is not None:
                shard_handle.__exit__(None, None, None)

    sz = out_path.stat().st_size
    print(f"[export_8b] wrote {out_path} ({sz / 1e9:.2f} GB) in {time.time() - t0:.1f}s", flush=True)
    print(f"[export_8b] alignment-at-shard-boundary checks passed: {n_boundary_align_checks}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
