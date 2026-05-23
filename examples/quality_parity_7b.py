"""examples/quality_parity_7b.py — heavyweight INT4 quality test.

Compares three engines on the SAME prompt + tokenization:

  PyTorch (mps/fp16) reference   — `mistralai/Mamba-Codestral-7B-v0.1`
  lite-ssm (Metal/fp16)          — `model_7b.ssm`
  lite-ssm (Metal/int4_b32)      — `model_7b.int4.ssm`

Reports:
  * Generated text from each engine (eye test).
  * Per-token exact-match rate (int4 + fp16 each vs PyTorch).
  * KL divergence on the FINAL decode-step logits for each engine vs PyTorch
    — measures how much int4 compression actually shifts the distribution.

Hardware constraints honored:
  * PyTorch loads in fp16 (NOT fp32) — ~14 GB resident, fits comfortably
    on 16 GB+ Macs.
  * Each `--dump-logits` produces a single fp32 vocab vector (~128 KB) so
    cross-process KL math is cheap.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer            # noqa: E402
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache       # noqa: E402


DEFAULT_PROMPT = "Write a Python function to calculate the Fibonacci sequence"


# --------------------------------------------------------------------------
_TTY = sys.stdout.isatty()
def c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _TTY else s
BOLD, DIM, GREEN, RED, YELLOW, MAG = "1", "2", "32", "31", "33", "35"


def device_sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
def run_pytorch(model_id: str, device: str, prompt_ids, n_tokens: int):
    """Greedy decode `n_tokens` from `prompt_ids` via HF Mamba2 + Mamba2Cache.
    Returns (generated_ids, final_step_logits_fp32, wall_ms)."""
    print(c(BOLD, "[pytorch] ") +
          f"loading {model_id} on {device} (fp16)…", flush=True)
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    device_sync(device)
    t_load1 = time.perf_counter()
    print(c(BOLD, "[pytorch] ") +
          f"loaded in {(t_load1 - t_load0):.1f}s", flush=True)

    cache = Mamba2Cache(config=model.config, batch_size=1,
                        dtype=torch.float16, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    last_logits: np.ndarray | None = None

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, cache_params=cache, use_cache=True,
                    cache_position=torch.arange(ids.shape[1], device=device))
        logits = out.logits[:, -1, :].float()
        next_id = int(logits.argmax(dim=-1).item())
        last_logits = logits.squeeze(0).cpu().numpy()
        generated.append(next_id)

        pos = ids.shape[1]
        for _ in range(n_tokens - 1):
            inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(input_ids=inp, cache_params=cache, use_cache=True,
                        cache_position=torch.tensor([pos], device=device))
            logits = out.logits[:, -1, :].float()
            next_id = int(logits.argmax(dim=-1).item())
            last_logits = logits.squeeze(0).cpu().numpy()
            generated.append(next_id)
            pos += 1
        device_sync(device)
    t1 = time.perf_counter()
    print(c(BOLD, "[pytorch] ") +
          f"{n_tokens} tokens in {(t1 - t0) * 1000:.1f} ms "
          f"({n_tokens / (t1 - t0):.1f} tok/s)", flush=True)
    # Drop the model to free its 14 GB of weights before we spawn lite-ssm.
    del model, cache
    if device == "mps": torch.mps.empty_cache()
    return generated, last_logits, (t1 - t0) * 1000.0


def run_litessm(label: str, bin_path: Path, weights: Path,
                prompt_ids: list[int], n_tokens: int):
    print(c(BOLD, f"[{label}] ") + f"spawning {bin_path.name}…", flush=True)
    dump = Path(f"litessm_{label}.logits.bin").resolve()
    if dump.exists(): dump.unlink()
    cmd = [
        str(bin_path),
        "--model",       str(weights),
        "--tokens",      str(n_tokens),
        "--greedy",
        "--input-ids",   ",".join(str(i) for i in prompt_ids),
        "--dump-logits", str(dump),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    t1 = time.perf_counter()
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"lite-ssm({label}) exited rc={res.returncode}")
    if res.stderr.strip():
        for line in res.stderr.strip().splitlines():
            print(c(DIM, f"[{label}·stderr] {line}"))
    ids = [int(x) for x in res.stdout.strip().split(",") if x]
    if not dump.exists():
        raise RuntimeError(f"lite-ssm({label}) did not write {dump}")
    final_logits = np.fromfile(dump, dtype=np.float32)
    print(c(BOLD, f"[{label}] ") +
          f"{n_tokens} tokens, subprocess wall {(t1 - t0) * 1000:.1f} ms", flush=True)
    return ids, final_logits, (t1 - t0) * 1000.0


# --------------------------------------------------------------------------
def kl_div(p_logits: np.ndarray, q_logits: np.ndarray):
    """KL(P||Q), KL(Q||P) on softmaxed logits — fp64 to avoid underflow."""
    p = F.softmax(torch.from_numpy(p_logits).double(), dim=-1)
    q = F.softmax(torch.from_numpy(q_logits).double(), dim=-1)
    eps = 1e-12
    kl_pq = float((p * (torch.log(p + eps) - torch.log(q + eps))).sum().item())
    kl_qp = float((q * (torch.log(q + eps) - torch.log(p + eps))).sum().item())
    return kl_pq, kl_qp


def exact_match(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]), n


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--bin",         default="build/lite-ssm")
    ap.add_argument("--weights-fp16",default="model_7b.ssm")
    ap.add_argument("--weights-int4",default="model_7b.int4.ssm")
    ap.add_argument("--tokens",      type=int, default=20)
    ap.add_argument("--device",      default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--prompt",      default=DEFAULT_PROMPT)
    args = ap.parse_args()

    bin_path = Path(args.bin).resolve()
    int4_path = Path(args.weights_int4).resolve()
    fp16_path = Path(args.weights_fp16).resolve()

    for label, p in [("lite-ssm bin", bin_path), ("int4 weights", int4_path)]:
        if not p.exists():
            print(c(RED, f"ERROR: {label} not found at {p}"), file=sys.stderr)
            return 2

    print(c(BOLD, "[setup] ") + f"loading HF tokenizer for {args.model}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    print(c(BOLD, "[setup] ") + f"prompt = {len(prompt_ids)} BPE ids", flush=True)
    print(c(BOLD, "[setup] ") + f"target tokens = {args.tokens} (greedy)", flush=True)

    # PyTorch first — releases ~14 GB of weights before we touch lite-ssm,
    # so the two engines don't compete for the unified-memory page cache.
    py_ids, py_logits, py_ms = run_pytorch(args.model, args.device,
                                           prompt_ids, args.tokens)

    int4_ids, int4_logits, int4_ms = run_litessm(
        "int4", bin_path, int4_path, prompt_ids, args.tokens)

    fp16_ids = fp16_logits = fp16_ms = None
    if fp16_path.exists():
        fp16_ids, fp16_logits, fp16_ms = run_litessm(
            "fp16", bin_path, fp16_path, prompt_ids, args.tokens)

    # ----- report ----------------------------------------------------------
    print()
    print(c(BOLD, "═" * 78))
    print(c(BOLD, " lite-ssm (int4) vs PyTorch — Codestral 7B Heavyweight Parity "))
    print(c(BOLD, "═" * 78))

    print()
    print(c(BOLD, "Prompt:"))
    print("  " + args.prompt)

    def show_text(label, ids, color):
        text = tok.decode(ids, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False)
        print()
        print(c(BOLD, f"{label} — {len(ids)} tokens:"))
        print("  " + c(color, text))

    show_text("PyTorch (mps/fp16)",       py_ids,  GREEN)
    if fp16_ids:
        show_text("lite-ssm (Metal/fp16)", fp16_ids, GREEN)
    show_text("lite-ssm (Metal/int4_b32)", int4_ids, YELLOW)

    print()
    print(c(BOLD, "Metrics:"))

    def report_leg(label, ids, logits):
        matched, total = exact_match(py_ids, ids)
        rate = (matched / total * 100.0) if total else 0.0
        rate_col = GREEN if rate >= 90 else (YELLOW if rate >= 60 else RED)
        kl_pq, kl_qp = kl_div(py_logits, logits)
        print(f"  {label:<28} exact-match vs PyTorch: "
              f"{c(rate_col + ';1', f'{matched}/{total} = {rate:.1f}%')}")
        print(f"  {' ' * 28} KL(PyTorch || engine):  {c(MAG, f'{kl_pq:.3e}')} nats")
        print(f"  {' ' * 28} KL(engine || PyTorch):  {c(MAG, f'{kl_qp:.3e}')} nats")

    if fp16_ids is not None:
        report_leg("lite-ssm (fp16)", fp16_ids, fp16_logits)
    report_leg("lite-ssm (int4_b32)", int4_ids, int4_logits)

    # Isolate the quantization effect by comparing the two lite-ssm legs to
    # each other — same engine, same kernels, only the weight precision differs.
    if fp16_ids is not None:
        matched, total = exact_match(fp16_ids, int4_ids)
        rate = (matched / total * 100.0) if total else 0.0
        rate_col = GREEN if rate >= 90 else (YELLOW if rate >= 60 else RED)
        kl_pq, kl_qp = kl_div(fp16_logits, int4_logits)
        print()
        print(c(BOLD, "Pure int4 quantization drift (lite-ssm fp16 ↔ int4):"))
        print(f"  exact-match              : {c(rate_col + ';1', f'{matched}/{total} = {rate:.1f}%')}")
        print(f"  KL(fp16 || int4)         : {c(MAG, f'{kl_pq:.3e}')} nats")
        print(f"  KL(int4 || fp16)         : {c(MAG, f'{kl_qp:.3e}')} nats")

    print()
    print(c(BOLD, "Wall time per leg:"))
    print(f"  PyTorch (mps/fp16)        : {py_ms:8.1f} ms")
    if fp16_ms is not None:
        print(f"  lite-ssm (fp16, in proc)  : {fp16_ms:8.1f} ms")
    print(f"  lite-ssm (int4_b32, in proc): {int4_ms:8.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
