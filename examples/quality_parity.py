"""examples/quality_parity.py

Phase 7 — head-to-head generation-quality + drift benchmark.

For a non-trivial multi-sentence prompt, drive BOTH engines with the SAME
HuggingFace BPE tokenization, force greedy decoding for `--tokens` steps,
and report:

  * exact-match rate token-by-token
  * the position of first divergence
  * KL divergence on the FINAL-step logits distributions
  * side-by-side decoded text

This proves that switching from PyTorch/MPS fp16 to lite-ssm's hand-rolled
Metal fp16 path doesn't degrade generation quality.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

warnings.filterwarnings("ignore")


DEFAULT_PROMPT = (
    "Mamba is a new class of state-space models that scale linearly with "
    "sequence length. Unlike traditional transformers, they maintain a "
    "fixed-size hidden state. The key insight is that"
)


# --------------------------------------------------------------------------
# ANSI colours (cheap, no extra dep). Disabled if stdout isn't a tty.
# --------------------------------------------------------------------------
_TTY = sys.stdout.isatty()
def c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _TTY else s

BOLD   = "1"
DIM    = "2"
GREEN  = "32"
RED    = "31"
YELLOW = "33"
CYAN   = "36"
MAG    = "35"


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------
def device_sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def run_pytorch(model_id: str, device: str, dtype, prompt_ids: list[int],
                n_tokens: int) -> tuple[list[int], np.ndarray]:
    """Greedy-decode `n_tokens` from `prompt_ids`. Returns (generated_ids,
    final_step_logits_fp32) where logits are the distribution OVER the
    next-token slot AFTER the very last generated token (matches what
    lite-ssm's --dump-logits writes)."""
    print(c(BOLD, "[pytorch] ") + f"loading {model_id} on {device}…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    cache = Mamba2Cache(config=model.config, batch_size=1, dtype=dtype, device=device)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    last_logits = None

    print(c(BOLD, "[pytorch] ") + f"generating {n_tokens} tokens (greedy)…", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        # Prefill
        out = model(
            input_ids=ids,
            cache_params=cache,
            use_cache=True,
            cache_position=torch.arange(ids.shape[1], device=device),
        )
        logits = out.logits[:, -1, :].float()
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        last_logits = logits.squeeze(0).cpu().numpy()

        pos = ids.shape[1]
        for _ in range(n_tokens - 1):
            inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(
                input_ids=inp,
                cache_params=cache,
                use_cache=True,
                cache_position=torch.tensor([pos], device=device),
            )
            logits = out.logits[:, -1, :].float()
            next_id = int(logits.argmax(dim=-1).item())
            generated.append(next_id)
            last_logits = logits.squeeze(0).cpu().numpy()
            pos += 1
        device_sync(device)
    t1 = time.perf_counter()
    print(c(BOLD, "[pytorch] ") +
          f"done in {(t1 - t0) * 1000:.1f} ms "
          f"({n_tokens / (t1 - t0):.1f} tok/s)", flush=True)
    return generated, last_logits


def run_litessm(bin_path: Path, model_path: Path, prompt_ids: list[int],
                n_tokens: int) -> tuple[list[int], np.ndarray]:
    """Spawn the C++ CLI in greedy mode, get back generated IDs + final
    logits (via --dump-logits)."""
    print(c(BOLD, "[lite-ssm] ") + f"spawning {bin_path}…", flush=True)
    dump_path = Path("lite_logits.bin").resolve()
    if dump_path.exists():
        dump_path.unlink()

    cmd = [
        str(bin_path),
        "--model",        str(model_path),
        "--tokens",       str(n_tokens),
        "--greedy",
        "--input-ids",    ",".join(str(i) for i in prompt_ids),
        "--dump-logits",  str(dump_path),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    t1 = time.perf_counter()
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"lite-ssm exited with {res.returncode}")
    if res.stderr.strip():
        for line in res.stderr.strip().splitlines():
            print(c(DIM, f"[lite-ssm·stderr] {line}"))

    ids = [int(x) for x in res.stdout.strip().split(",") if x]
    if not dump_path.exists():
        raise RuntimeError("lite-ssm did not write --dump-logits target")
    final_logits = np.fromfile(dump_path, dtype=np.float32)
    print(c(BOLD, "[lite-ssm] ") +
          f"done in {(t1 - t0) * 1000:.1f} ms "
          f"({n_tokens / (t1 - t0):.1f} tok/s)", flush=True)
    return ids, final_logits


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def exact_match(a: list[int], b: list[int]) -> tuple[int, int]:
    """Returns (matched, divergence_index_or_-1)."""
    n = min(len(a), len(b))
    matched = 0
    div = -1
    for i in range(n):
        if a[i] == b[i]:
            matched += 1
        else:
            if div < 0:
                div = i
    if len(a) != len(b) and div < 0:
        div = n
    return matched, div


def kl_div(p_logits: np.ndarray, q_logits: np.ndarray) -> tuple[float, float]:
    """KL(P || Q) and KL(Q || P) in nats, computed from logits via stable softmax."""
    p = F.softmax(torch.from_numpy(p_logits).double(), dim=-1)
    q = F.softmax(torch.from_numpy(q_logits).double(), dim=-1)
    eps = 1e-12
    kl_pq = float((p * (torch.log(p + eps) - torch.log(q + eps))).sum().item())
    kl_qp = float((q * (torch.log(q + eps) - torch.log(p + eps))).sum().item())
    return kl_pq, kl_qp


# --------------------------------------------------------------------------
# Pretty print
# --------------------------------------------------------------------------
def render_text_diff(tok, py_ids: list[int], lite_ids: list[int],
                     divergence_at: int) -> tuple[str, str]:
    """Decode both sequences. Highlight the first divergent token with
    colour so the report makes the diff visually obvious."""
    def decode(ids):
        return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if divergence_at < 0:
        return decode(py_ids), decode(lite_ids)

    pre = decode(py_ids[:divergence_at])
    py_div_tok  = decode([py_ids[divergence_at]])  if divergence_at < len(py_ids)  else ""
    lite_div_tok= decode([lite_ids[divergence_at]]) if divergence_at < len(lite_ids) else ""
    py_post    = decode(py_ids[divergence_at + 1:])    if divergence_at + 1 < len(py_ids)   else ""
    lite_post  = decode(lite_ids[divergence_at + 1:])  if divergence_at + 1 < len(lite_ids) else ""

    py_render   = pre + c(RED  + ";1", py_div_tok)   + c(DIM, py_post)
    lite_render = pre + c(GREEN+ ";1", lite_div_tok) + c(DIM, lite_post)
    return py_render, lite_render


def report(prompt: str, n_tokens: int,
           py_ids: list[int], lite_ids: list[int],
           py_logits: np.ndarray, lite_logits: np.ndarray,
           tok,
           int4_ids: list[int] | None = None,
           int4_logits: np.ndarray | None = None) -> int:

    matched, div_at = exact_match(py_ids, lite_ids)
    total = max(len(py_ids), len(lite_ids))
    rate = matched / total * 100.0

    kl_pq, kl_qp = kl_div(py_logits, lite_logits)

    py_text, lite_text = render_text_diff(tok, py_ids, lite_ids, div_at)
    prompt_disp = prompt.strip()

    int4_matched = int4_div_at = int4_rate = None
    int4_kl_pq = int4_kl_qp = None
    int4_text = None
    if int4_ids is not None and int4_logits is not None:
        int4_matched, int4_div_at = exact_match(py_ids, int4_ids)
        int4_rate = int4_matched / max(len(py_ids), len(int4_ids)) * 100.0
        int4_kl_pq, int4_kl_qp = kl_div(py_logits, int4_logits)
        # Reuse the diff renderer to highlight where int4 diverges from py.
        _, int4_text = render_text_diff(tok, py_ids, int4_ids, int4_div_at)

    print()
    print(c(BOLD, "═" * 78))
    print(c(BOLD, " lite-ssm vs PyTorch / MPS  —  Generation Quality & Drift "))
    print(c(BOLD, "═" * 78))

    print()
    print(c(BOLD, "Prompt:"))
    print(c(CYAN, "  " + prompt_disp))

    print()
    print(c(BOLD, f"PyTorch (mps/fp16) — {len(py_ids)} tokens:"))
    print("  " + py_text)

    print()
    print(c(BOLD, f"lite-ssm (Metal/fp16) — {len(lite_ids)} tokens:"))
    print("  " + lite_text)

    print()
    print(c(BOLD, "Metrics — lite-ssm (fp16) vs PyTorch:"))
    rate_str = f"{matched}/{total} = {rate:.1f}%"
    rate_col = GREEN if rate == 100.0 else (YELLOW if rate >= 90.0 else RED)
    print(f"  Exact-match rate     : {c(rate_col + ';1', rate_str)}")
    div_str = (f"position {div_at}" if div_at >= 0 else "none (identical sequences)")
    div_col = RED if div_at >= 0 else GREEN
    print(f"  First divergence     : {c(div_col, div_str)}")
    print(f"  KL(PyTorch || lite)  : {c(MAG, f'{kl_pq:.3e}')} nats  (final-step distribution)")
    print(f"  KL(lite || PyTorch)  : {c(MAG, f'{kl_qp:.3e}')} nats")

    if int4_rate is not None:
        print()
        print(c(BOLD, "Metrics — lite-ssm (int4_b32) vs PyTorch:"))
        int4_total = max(len(py_ids), len(int4_ids))
        rate_str = f"{int4_matched}/{int4_total} = {int4_rate:.1f}%"
        rate_col = GREEN if int4_rate >= 90.0 else (YELLOW if int4_rate >= 60.0 else RED)
        print(f"  Exact-match rate     : {c(rate_col + ';1', rate_str)}")
        div_str = (f"position {int4_div_at}" if int4_div_at >= 0 else "none (identical sequences)")
        div_col = RED if int4_div_at >= 0 else GREEN
        print(f"  First divergence     : {c(div_col, div_str)}")
        print(f"  KL(PyTorch || int4)  : {c(MAG, f'{int4_kl_pq:.3e}')} nats  <- quantization drift")
        print(f"  KL(int4 || PyTorch)  : {c(MAG, f'{int4_kl_qp:.3e}')} nats")
        print()
        print(c(BOLD, f"lite-ssm (int4_b32) — {len(int4_ids)} tokens:"))
        print("  " + int4_text)

    print()
    if rate == 100.0:
        print(c(GREEN + ";1", "✔ PASS: greedy sequences are bit-for-bit identical."))
    elif kl_pq < 1e-3:
        print(c(YELLOW + ";1",
                "⚠ NEAR-MISS: sequences differ but KL is at fp16 noise floor "
                "— the engines agree on the distribution; greedy argmax flipped "
                "on near-tied logits and the recurrent state then walked off."))
    else:
        print(c(RED + ";1", "✘ FAIL: distributions diverge beyond fp16 noise."))

    return 0 if rate == 100.0 or kl_pq < 1e-3 else 1


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",   default="AntonV/mamba2-130m-hf")
    ap.add_argument("--bin",     default="build/lite-ssm")
    ap.add_argument("--weights", default="model.ssm")
    ap.add_argument("--weights-int4", default="model.int4.ssm",
                    help="Optional Phase 14 int4 weights — when present, runs "
                         "a third leg and reports drift vs PyTorch.")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    bin_path     = Path(args.bin).resolve()
    weights_path = Path(args.weights).resolve()
    if not bin_path.exists():
        print(c(RED, f"ERROR: lite-ssm binary missing at {bin_path}. "
                     f"Build with: cmake --build build"), file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(c(RED, f"ERROR: model.ssm missing at {weights_path}. "
                     f"Run tools/export_mamba2.py."), file=sys.stderr)
        return 2

    print(c(BOLD, "[setup] ") + f"loading HF tokenizer for {args.model}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    print(c(BOLD, "[setup] ") + f"prompt = {len(prompt_ids)} BPE tokens", flush=True)

    py_ids, py_logits = run_pytorch(args.model, args.device, torch.float16,
                                    prompt_ids, args.tokens)
    lite_ids, lite_logits = run_litessm(bin_path, weights_path,
                                        prompt_ids, args.tokens)

    int4_weights = Path(args.weights_int4)
    int4_ids = int4_logits = None
    if int4_weights.exists():
        print(c(BOLD, "[setup] ") +
              f"int4 weights present at {int4_weights} — running third leg",
              flush=True)
        int4_ids, int4_logits = run_litessm(bin_path, int4_weights.resolve(),
                                            prompt_ids, args.tokens)

    return report(args.prompt, args.tokens, py_ids, lite_ids,
                  py_logits, lite_logits, tok,
                  int4_ids=int4_ids, int4_logits=int4_logits)


if __name__ == "__main__":
    raise SystemExit(main())
