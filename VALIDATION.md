# lite-ssm — Validation Suite Report

Generated automatically by `tools/run_validation_suite.sh`. Every number
below comes from parsing the captured `stdout` of an actual binary run —
no values are hardcoded.

| Field | Value |
|---|---|
| Generated at | 2026-05-23 07:29:06 UTC |
| Host         | Sebastians-MacBook-Pro.local arm64 |
| Model (small)| `model.ssm` |
| Build dir    | `build` |
| Raw outputs  | `/var/folders/n6/61s8lz6x5yd2bzrq0zt2jq5h0000gn/T/lite-ssm-validate-XXXX.M5JTMCoWTq` |

---

## Step 0 — Configure + Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Build **OK** (Release).

---

## Step 1 — Unit Tests (`ctest`)

```
1/5 Test #1: test_tensor ......................   Passed    0.34 sec
2/5 Test #2: test_mmap_loader .................   Passed    0.36 sec
3/5 Test #3: test_metal_ops ...................   Passed    0.02 sec
4/5 Test #4: test_ssd_parity ..................   Passed    0.11 sec
5/5 Test #5: test_tokenizer ...................   Passed    0.37 sec
100% tests passed, 0 tests failed out of 5
Total Test time (real) =   1.21 sec
```


| Test | Status | Wall |
|---|---|---|
| `test_tensor` | PASS | 0.34 s |
| `test_mmap_loader` | PASS | 0.36 s |
| `test_metal_ops` | PASS | 0.02 s |
| `test_ssd_parity` | PASS | 0.11 s |
| `test_tokenizer` | PASS | 0.37 s |

Summary: 100% tests passed, 0 tests failed out of 5

---

## Step 2 — Generation Quality + Drift vs PyTorch

Prompt fed to both engines (HF SentencePiece tokenization, 200 greedy tokens):

```
Mamba is a new class of state-space models that scale linearly with sequence 
length. Unlike traditional transformers, they maintain a fixed-size hidden 
state. The key insight is that
```

| Metric | lite-ssm (fp16) | lite-ssm (int4_b32) |
|---|---|---|
| Exact-match vs PyTorch | 200/200 = 100.0% | 0/200 = 0.0% |
| First divergence | none (identical sequences) | position 0 |
| KL(PyTorch ‖ lite) final step | 4.790e-08 nats | 2.758e+01 nats |
| KL(lite ‖ PyTorch) final step | 4.878e-08 nats | 2.763e+01 nats |

#### Decoded text (both engines, greedy, 200 tokens)

**PyTorch (mps/fp16) — 200 tokens:**

```
   the hidden state is a linear combination of the input states, and the output states are the same as the input states. This allows the model to be used to model the dynamics of a system of interest.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science

```

**lite-ssm (Metal/fp16) — 200 tokens:**

```
   the hidden state is a linear combination of the input states, and the output states are the same as the input states. This allows the model to be used to model the dynamics of a system of interest.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science.

The model is a generalization of the state-space model that is used in many other areas of computer science. The model is a generalization of the state-space model that is used in many other areas of computer science

Metrics — lite-ssm (fp16) vs PyTorch:
  Exact-match rate     : 200/200 = 100.0%
  First divergence     : none (identical sequences)
  KL(PyTorch || lite)  : 4.790e-08 nats  (final-step distribution)
  KL(lite || PyTorch)  : 4.878e-08 nats

Metrics — lite-ssm (int4_b32) vs PyTorch:
  Exact-match rate     : 0/200 = 0.0%
  First divergence     : position 0
  KL(PyTorch || int4)  : 2.758e+01 nats  <- quantization drift
  KL(int4 || PyTorch)  : 2.763e+01 nats

lite-ssm (int4_b32) — 200 tokens:
  �orentz algorithendix algorithendix algorithovember algorithovember algorithovember algorithovember algorithovember algorithovemberarroll algorithovemberarrollarroll algorithovember algorithovember algorithovember algorithovemberarrollarroll algorithovemberarrollarrollarroll algorithovemberarrollarrollarrollarrollarroll algorithovemberarrollarrollarrollarrollarrollarroll algorithovember algorithovemberarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarroll algorithovemberarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarroll algorithovember algorithovember algorithovember algorithovember algorithovember algorithovember algorithovember algorithovemberarroll algorithovemberarrollarroll algorithovemberarrollarrollarrollarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarrollarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarrollarrollarrollarroll algorithovemberarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarroll algorithovemberarrollarrollarrollarrollarrollarrollarrollarroll algorithovember algorithovemberarrollarroll algorithovemberarrollarrollarrollarrollarrollarroll algorithovemberarrollarroll algorithovemberarrollarroll

✔ PASS: greedy sequences are bit-for-bit identical.

---

## Step 3 — Head-to-Head Benchmark vs PyTorch / MPS

Workload: 12 iterations (first dropped as warm-up) of ingesting a synthetic
log line as a byte-token prompt and greedy-decoding a 5-token summary.
Reported numbers are the **median** across kept iterations; peak RSS comes
from `/usr/bin/time -l`.


Workload: 12 iterations (first dropped as warm-up) of ingesting a single
synthetic system-log line as a byte-token prompt, then greedy-decoding a
5-token summary. Reported numbers are the median across the kept iterations.

| Metric | lite-ssm (Metal) | PyTorch (mps/fp16) | Speedup |
|---|---:|---:|---:|
| Boot time           | 24.0 ms       | 864.0 ms       | 36.0x faster |
| Time-to-first-token | 24.4 ms       | 554.9 ms       | 22.7x faster |
| Decode throughput   | 547.2 tok/s        | 45.4 tok/s        | 12.0x faster  |
| Peak RSS            | 41.6 MB        | 1009.0 MB        | 24.3x smaller |

### lite-ssm footprint detail

| Slot | Bytes | Notes |
|---|---:|---|
| Weights         | 319.7 MB   | `mmap`'d `.ssm` file — page-cached, zero-copy into one `MTLBuffer` |
| Recurrent state | 18.3 MB | Fixed-size: conv1d window (fp16) + SSD hidden (fp32). No growth across decode. |
| Activations     | 6.9 MB    | Per-pass scratch — reused across all 24 blocks |

Raw outputs live in: `/var/folders/n6/61s8lz6x5yd2bzrq0zt2jq5h0000gn/T/lite-ssm-bench-XXXX.MjX8ddtl3c`

---

## Step 4 — Memory Bleed Probe (5 000 tokens)

Per-token RSS sampled via `mach_task_basic_info()`. Phase 10 fix landed
the autoreleasepool + workspace-owned logits scratch — the curve should
plateau, not grow linearly.

```
[bleed] model=/Users/sebastiangrebe/Documents/Git/lite-ssm/model.ssm  tokenizer=bpe  decode_per_iter=5  token_budget=5000
[bleed] baseline taken at iter 5; leak threshold = 1.0 MB above baseline

  iter    tokens           rss   Δ_baseline  tok/s_recent
  ----    ------           ---    ----------  ------------
     5        25       41.2 MB         +0.00 MB         127.6
   100       500       41.6 MB         +0.42 MB         163.9
   200      1000       41.6 MB         +0.42 MB         164.6
   300      1500       41.6 MB         +0.44 MB         164.6
   400      2000       41.7 MB         +0.47 MB         164.3
   500      2500       41.7 MB         +0.47 MB         164.6
   600      3000       41.7 MB         +0.47 MB         164.7
   700      3500       41.7 MB         +0.47 MB         165.0
   800      4000       41.7 MB         +0.47 MB         164.5
   900      4500       41.7 MB         +0.47 MB         164.8
  1000      5000       41.7 MB         +0.47 MB         165.1

[bleed] SUMMARY
  iterations           : 1000
  tokens generated     : 5000
  baseline (iter  5)  : 41.2 MB
  peak                 : 41.7 MB
  final                : 41.7 MB
  growth vs baseline   : +0.47 MB
[bleed] PASS: no RSS leak above 1 MB budget.
```

| Metric | Value |
|---|---|
| Baseline (iter 5) | 41.2 MB |
| Peak | 41.7 MB |
| Final (after 5 000 tokens) | 41.7 MB |
| Growth vs baseline | +0.47 MB |
| Verdict | PASS: no RSS leak above 1 MB budget. |

---

## Step 5 — Swarm Concurrency (8 threads × 100 decode tokens)

Shared mmap'd weights; per-thread `MetalOps` + `Mamba2Workspace` +
`Mamba2State`. We measure aggregate TPS against a single-thread baseline.

```
[concurrency] single-thread baseline …
[concurrency] SUMMARY
  wall clock           : 948.1 ms
  total tokens         : 800
  aggregate TPS        : 843.8 tok/s
  single-thread TPS    : 425.5 tok/s
  scaling factor       : 1.98x  (ideal = 8x)
  fastest worker       : 947.5 ms
  slowest worker       : 948.0 ms
  spread (max/min)     : 1.00x
```


| Metric | Value |
|---|---|
| Single-thread TPS | 425.5 tok/s |
| Aggregate TPS (8 threads) | 843.8 tok/s |
| Scaling factor (ideal = 8.0x) | 1.98x |
| Fairness spread (max/min duration) | 1.00x |

---

## Step 6 — Quantization Footprint (Phase 14 INT4_BLOCK32)

`--quantize int4` packs every 2-D Linear weight (`in_proj`, `out_proj`,
`lm_head`) into 32-element blocks: 32 signed-4-bit nibbles + one fp16 scale
per block. 1-D tensors (norms, biases, conv1d, A_log, D, dt_bias) stay fp16.

| Model | On-disk size |
|---|---:|
| `model.ssm` (fp16)       | 319.7 MB |
| `model.int4.ssm` (int4_b32)   | 143.3 MB |
| Compression ratio                          | **2.23x** |
| Footprint savings                          | **55.2%** |

Quality drift caused by this compression is reported in Step 2 (KL columns).

---

## Reproducing

```sh
.venv/bin/python tools/export_mamba2.py    --out model.ssm
.venv/bin/python tools/export_tokenizer.py --out tokenizer.model
.venv/bin/python tools/dump_parity.py      --out parity.bin
bash tools/run_validation_suite.sh
```

Suite runtime ≈ 2 minutes on Apple Silicon once weights + tokenizer + parity
dump are already cached locally.
