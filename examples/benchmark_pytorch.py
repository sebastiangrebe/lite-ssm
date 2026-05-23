"""examples/benchmark_pytorch.py

Head-to-head PoC baseline. Runs the *standard* PyTorch / HuggingFace path
on Apple Silicon's MPS device — no custom kernels, no hand-tuned setup.
This is what an average developer reaches for when they say "run Mamba-2
on my Mac".

Loop matches examples/benchmark_litessm.cpp:
  1. Reset the recurrent cache.
  2. Prefill a system-log line as byte tokens.
  3. Greedy-decode a 5-token summary.

Emits BENCH_* lines on stdout for run_benchmark.sh.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings

import torch
from transformers import AutoModelForCausalLM
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

warnings.filterwarnings("ignore")


LOG_LINES = [
    "[INFO] Server heartbeat OK, uptime 12345s",
    "[WARN] Connection retry to db: timeout 5s",
    "[ERROR] Authentication failed for user admin",
    "[INFO] Cache hit ratio 0.943 over last 60s",
    "[WARN] Disk usage at 87% on /var, alerting",
    "[INFO] Request /api/v1/health 200 in 4ms",
]
SUMMARY_TOKENS = 5
DEFAULT_ITERS  = 12


def device_sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def fresh_cache(model, dtype, device):
    return Mamba2Cache(
        config=model.config,
        batch_size=1,
        dtype=dtype,
        device=device,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="AntonV/mamba2-130m-hf")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--dtype",  default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--iters",  type=int, default=DEFAULT_ITERS)
    args = ap.parse_args()

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    # ---------------------------------------------------------------
    # BOOT: from_pretrained + .to(device). Includes weight download
    # only on first run; HF caches them after that.
    # ---------------------------------------------------------------
    boot_t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype)
    model.to(args.device).eval()
    device_sync(args.device)
    boot_ms = (time.perf_counter() - boot_t0) * 1000.0

    # ---------------------------------------------------------------
    # WORKLOAD
    # ---------------------------------------------------------------
    ttfts: list[float] = []
    decode_mss: list[float] = []

    for i in range(args.iters):
        line = LOG_LINES[i % len(LOG_LINES)]
        token_ids = list(line.encode("utf-8"))            # byte-fallback to match C++
        ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)

        cache = fresh_cache(model, torch_dtype, args.device)

        # --- TTFT: prefill + first sample
        device_sync(args.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=ids,
                cache_params=cache,
                use_cache=True,
                cache_position=torch.arange(ids.shape[1], device=args.device),
            )
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            device_sync(args.device)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        # --- Decode the remaining tokens, one step at a time
        device_sync(args.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            pos = ids.shape[1]
            for _ in range(SUMMARY_TOKENS - 1):
                out = model(
                    input_ids=next_id,
                    cache_params=cache,
                    use_cache=True,
                    cache_position=torch.tensor([pos], device=args.device),
                )
                logits = out.logits[:, -1, :]
                next_id = logits.argmax(dim=-1, keepdim=True)
                pos += 1
            device_sync(args.device)
        decode_ms = (time.perf_counter() - t0) * 1000.0

        ttfts.append(ttft_ms)
        decode_mss.append(decode_ms)

    # Drop first iter as warm-up.
    ttfts = ttfts[1:]
    decode_mss = decode_mss[1:]
    tps = [(SUMMARY_TOKENS - 1) * 1000.0 / d for d in decode_mss]

    ttft_med = statistics.median(ttfts)
    tps_med  = statistics.median(tps)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print()
    print("=== pytorch / mps benchmark ===")
    print(f"  model:          {args.model}")
    print(f"  device / dtype: {args.device} / {args.dtype}")
    print(f"  iterations:     {args.iters} (first dropped as warm-up)")
    print(f"  boot:           {boot_ms:.1f} ms  (from_pretrained + .to({args.device}))")
    print(f"  TTFT median:    {ttft_med:.1f} ms  (prefill + sample)")
    print(f"  decode TPS:     {tps_med:.1f} tok/s  (steady-state, {SUMMARY_TOKENS - 1} tokens per loop)")

    print()
    print("BENCH_ENGINE=pytorch-mps")
    print(f"BENCH_BOOT_MS={boot_ms:.3f}")
    print(f"BENCH_TTFT_MS={ttft_med:.3f}")
    print(f"BENCH_TPS={tps_med:.3f}")
    print(f"BENCH_DEVICE={args.device}")
    print(f"BENCH_DTYPE={args.dtype}")
    print(f"BENCH_ITERS={args.iters - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
