#!/usr/bin/env bash
# examples/run_benchmark.sh
#
# Head-to-head PoC orchestrator:
#   1. Build the C++ benchmark.
#   2. Run lite-ssm under /usr/bin/time -l to capture peak RSS.
#   3. Run the PyTorch / MPS baseline under /usr/bin/time -l.
#   4. Parse BENCH_* metrics + max RSS, emit a Markdown comparison table.
#
# Both processes use the same workload (synthetic log ingestion + 5-token
# summary, repeated 12 times, first iteration dropped as warm-up).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${ROOT}/build"
CMAKE="${CMAKE:-/opt/homebrew/bin/cmake}"
PY="${PY:-${ROOT}/.venv/bin/python}"
MODEL_SSM="${MODEL_SSM:-${ROOT}/model.ssm}"
LITE_BIN="${BUILD_DIR}/benchmark_litessm"
PY_SCRIPT="${ROOT}/examples/benchmark_pytorch.py"
ITERS="${ITERS:-12}"

OUT_DIR="$(mktemp -d -t lite-ssm-bench-XXXX)"
LITE_STDOUT="${OUT_DIR}/lite.stdout"
LITE_STDERR="${OUT_DIR}/lite.stderr"
PY_STDOUT="${OUT_DIR}/py.stdout"
PY_STDERR="${OUT_DIR}/py.stderr"

cleanup() { :; }   # keep tmp on disk for post-mortem; cheap.
trap cleanup EXIT

# --------------------------------------------------------------------------
log() { printf "[bench] %s\n" "$*" >&2; }
require() {
    local what="$1" path="$2"
    if [[ ! -e "$path" ]]; then
        printf "ERROR: %s not found at %s\n" "$what" "$path" >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------
# 1. Build
# --------------------------------------------------------------------------
require "model.ssm" "$MODEL_SSM"
require "Python venv" "$PY"
require "PyTorch baseline script" "$PY_SCRIPT"
require "cmake" "$CMAKE"

log "configure + build ($BUILD_DIR)"
"$CMAKE" -S . -B "$BUILD_DIR" -G Ninja >/dev/null
"$CMAKE" --build "$BUILD_DIR" --target benchmark_litessm >/dev/null
require "lite-ssm benchmark binary" "$LITE_BIN"

# --------------------------------------------------------------------------
# 2. Run lite-ssm with /usr/bin/time -l
# --------------------------------------------------------------------------
log "running lite-ssm benchmark ($ITERS iters)"
/usr/bin/time -l "$LITE_BIN" "$MODEL_SSM" "$ITERS" \
    >"$LITE_STDOUT" 2>"$LITE_STDERR" || {
        printf "lite-ssm benchmark failed; stderr below:\n" >&2
        cat "$LITE_STDERR" >&2
        exit 1
    }

# --------------------------------------------------------------------------
# 3. Run PyTorch / MPS baseline with /usr/bin/time -l
# --------------------------------------------------------------------------
log "running pytorch / mps benchmark ($ITERS iters)"
/usr/bin/time -l "$PY" "$PY_SCRIPT" --iters "$ITERS" \
    >"$PY_STDOUT" 2>"$PY_STDERR" || {
        printf "pytorch benchmark failed; stderr below:\n" >&2
        cat "$PY_STDERR" >&2
        exit 1
    }

# --------------------------------------------------------------------------
# 4. Parse BENCH_* + peak RSS
# --------------------------------------------------------------------------

# Pull KEY=value from BENCH_KEY=value lines in a stdout file.
get_bench() {
    local file="$1" key="$2"
    awk -F= -v k="BENCH_${key}" '$1 == k { print $2 }' "$file" | tail -n1
}

# /usr/bin/time -l writes the resource summary to stderr in a fixed format.
# The "maximum resident set size" line is bytes on modern macOS.
get_rss_bytes() {
    local file="$1"
    awk '/maximum resident set size/ { print $1; exit }' "$file"
}

LITE_BOOT=$(get_bench "$LITE_STDOUT" BOOT_MS)
LITE_TTFT=$(get_bench "$LITE_STDOUT" TTFT_MS)
LITE_TPS=$(get_bench  "$LITE_STDOUT" TPS)
LITE_STATE=$(get_bench "$LITE_STDOUT" STATE_BYTES)
LITE_WS=$(get_bench    "$LITE_STDOUT" WORKSPACE_BYTES)
LITE_WTS=$(get_bench   "$LITE_STDOUT" WEIGHTS_BYTES)
LITE_RSS=$(get_rss_bytes "$LITE_STDERR")

PY_BOOT=$(get_bench "$PY_STDOUT" BOOT_MS)
PY_TTFT=$(get_bench "$PY_STDOUT" TTFT_MS)
PY_TPS=$(get_bench  "$PY_STDOUT" TPS)
PY_DTYPE=$(get_bench "$PY_STDOUT" DTYPE)
PY_DEVICE=$(get_bench "$PY_STDOUT" DEVICE)
PY_RSS=$(get_rss_bytes "$PY_STDERR")

# Helpers for table formatting.
mb()  { awk -v b="$1" 'BEGIN { printf "%.1f MB", b/1024/1024 }'; }
ms()  { awk -v v="$1" 'BEGIN { printf "%.1f ms", v }'; }
tps() { awk -v v="$1" 'BEGIN { printf "%.1f tok/s", v }'; }

ratio_div() {
    # ratio of a / b, printed as "X.Yx"
    awk -v a="$1" -v b="$2" 'BEGIN { if (b == 0) print "—"; else printf "%.1fx", a/b }'
}

# Speedup is C++ winning -> baseline / lite for latency, lite / baseline for tps
SPEED_BOOT=$(ratio_div "$PY_BOOT" "$LITE_BOOT")
SPEED_TTFT=$(ratio_div "$PY_TTFT" "$LITE_TTFT")
SPEED_TPS=$(ratio_div  "$LITE_TPS" "$PY_TPS")
SPEED_RSS=$(ratio_div  "$PY_RSS"  "$LITE_RSS")

# --------------------------------------------------------------------------
# 5. Emit Markdown comparison
# --------------------------------------------------------------------------
cat <<MD

# lite-ssm vs PyTorch / MPS — Head-to-Head Benchmark

Workload: 12 iterations (first dropped as warm-up) of ingesting a single
synthetic system-log line as a byte-token prompt, then greedy-decoding a
5-token summary. Reported numbers are the median across the kept iterations.

| Metric | lite-ssm (Metal) | PyTorch (${PY_DEVICE}/${PY_DTYPE}) | Speedup |
|---|---:|---:|---:|
| Boot time           | $(ms  "$LITE_BOOT")       | $(ms  "$PY_BOOT")       | ${SPEED_BOOT} faster |
| Time-to-first-token | $(ms  "$LITE_TTFT")       | $(ms  "$PY_TTFT")       | ${SPEED_TTFT} faster |
| Decode throughput   | $(tps "$LITE_TPS")        | $(tps "$PY_TPS")        | ${SPEED_TPS} faster  |
| Peak RSS            | $(mb  "$LITE_RSS")        | $(mb  "$PY_RSS")        | ${SPEED_RSS} smaller |

### lite-ssm footprint detail

| Slot | Bytes | Notes |
|---|---:|---|
| Weights         | $(mb "$LITE_WTS")   | \`mmap\`'d \`.ssm\` file — page-cached, zero-copy into one \`MTLBuffer\` |
| Recurrent state | $(mb "$LITE_STATE") | Fixed-size: conv1d window (fp16) + SSD hidden (fp32). No growth across decode. |
| Activations     | $(mb "$LITE_WS")    | Per-pass scratch — reused across all 24 blocks |

Raw outputs live in: \`$OUT_DIR\`
MD
