#!/usr/bin/env bash
# tools/run_validation_suite.sh
#
# Master validation orchestrator. Compiles the project, runs every existing
# probe (ctest + quality parity + head-to-head benchmark + memory bleed +
# concurrency stress), parses each one's real stdout, and renders the
# combined result into VALIDATION.md.
#
# Nothing here invents numbers. If a step fails or its artifact is missing
# the markdown reports that honestly rather than substituting a placeholder.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CMAKE="${CMAKE:-/opt/homebrew/bin/cmake}"
CTEST="${CTEST:-/opt/homebrew/bin/ctest}"
PY="${PY:-${ROOT}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
MODEL_SSM="${MODEL_SSM:-${ROOT}/model.ssm}"
PARITY_DUMP="${PARITY_DUMP:-${ROOT}/parity.bin}"
OUT_MD="${OUT_MD:-${ROOT}/VALIDATION.md}"
WORK="$(mktemp -d -t lite-ssm-validate-XXXX)"
trap 'echo "[validate] raw outputs preserved under $WORK"' EXIT

log() { printf "[validate] %s\n" "$*" >&2; }
strip_ansi() { sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g'; }

# Helpers used while composing the markdown ---------------------------------
fmt_size_mb() { awk -v b="$1" 'BEGIN { if (b == "" ) print "—"; else printf "%.1f MB", b/1024/1024 }'; }
escape_md()   { sed -E 's/`/\\`/g'; }

# --------------------------------------------------------------------------
# STEP 0: Configure + build (Release).
# --------------------------------------------------------------------------
log "configure + build (Release)"
"$CMAKE" -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    >"$WORK/cmake.configure" 2>&1
BUILD_RC=0
"$CMAKE" --build "$BUILD_DIR" --config Release \
    >"$WORK/cmake.build" 2>&1 || BUILD_RC=$?
if [[ $BUILD_RC -ne 0 ]]; then
    log "BUILD FAILED (rc=$BUILD_RC). Inspect $WORK/cmake.build"
fi

# --------------------------------------------------------------------------
# STEP 1: ctest.
# --------------------------------------------------------------------------
log "step 1 — ctest"
CTEST_RC=0
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure \
    >"$WORK/ctest.out" 2>&1 || CTEST_RC=$?

# --------------------------------------------------------------------------
# STEP 2: generation quality + drift.
# --------------------------------------------------------------------------
log "step 2 — quality_parity.py"
QP_RC=0
QP_PROMPT='Mamba is a new class of state-space models that scale linearly with sequence length. Unlike traditional transformers, they maintain a fixed-size hidden state. The key insight is that'
"$PY" examples/quality_parity.py --tokens 200 --prompt "$QP_PROMPT" \
    >"$WORK/quality.out.raw" 2>&1 || QP_RC=$?
strip_ansi <"$WORK/quality.out.raw" >"$WORK/quality.out"

# --------------------------------------------------------------------------
# STEP 3: head-to-head benchmark vs PyTorch / MPS.
# --------------------------------------------------------------------------
log "step 3 — run_benchmark.sh"
BENCH_RC=0
ITERS=12 bash examples/run_benchmark.sh \
    >"$WORK/benchmark.out" 2>>"$WORK/benchmark.stderr" || BENCH_RC=$?

# --------------------------------------------------------------------------
# STEP 4: memory bleed.
# --------------------------------------------------------------------------
log "step 4 — test_memory_bleed"
BLEED_RC=0
"$BUILD_DIR/test_memory_bleed" "$MODEL_SSM" \
    >"$WORK/bleed.out" 2>&1 || BLEED_RC=$?

# --------------------------------------------------------------------------
# STEP 5: concurrency.
# --------------------------------------------------------------------------
log "step 5 — test_concurrency (8 threads x 100 decode tokens)"
CONC_RC=0
"$BUILD_DIR/test_concurrency" "$MODEL_SSM" 8 100 \
    >"$WORK/concurrency.out" 2>&1 || CONC_RC=$?

# ==========================================================================
# Parsers — each emits markdown fragments to stdout.
# ==========================================================================

emit_header() {
    local now
    now=$(date -u "+%Y-%m-%d %H:%M:%S UTC")
    local host
    host=$(uname -mn)
    cat <<EOF
# lite-ssm — Validation Suite Report

Generated automatically by \`tools/run_validation_suite.sh\`. Every number
below comes from parsing the captured \`stdout\` of an actual binary run —
no values are hardcoded.

| Field | Value |
|---|---|
| Generated at | $now |
| Host         | $host |
| Model (small)| \`$(basename "$MODEL_SSM")\` |
| Build dir    | \`$(basename "$BUILD_DIR")\` |
| Raw outputs  | \`$WORK\` |

---
EOF
}

emit_build() {
    cat <<EOF

## Step 0 — Configure + Build

\`\`\`
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
\`\`\`

EOF
    if [[ $BUILD_RC -eq 0 ]]; then
        echo "Build **OK** (Release)."
    else
        echo "Build **FAILED** with rc=$BUILD_RC. See \`$WORK/cmake.build\`."
    fi
}

emit_ctest() {
    cat <<EOF

---

## Step 1 — Unit Tests (\`ctest\`)

\`\`\`
$(grep -E '^(Test #|[0-9]+/[0-9]+ Test|100% tests|Total Test time|The following tests)' "$WORK/ctest.out" | head -40)
\`\`\`

EOF
    # CTest line shape: "1/5 Test #1: test_tensor ............. Passed 0.34 sec"
    # Portable POSIX awk (BSD awk on macOS lacks gawk's 3-arg match).
    awk '
        /Test #[0-9]+:/ {
            # Extract the test name: token right after "Test #N:"
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^Test$/ && $(i+1) ~ /^#[0-9]+:$/) {
                    name = $(i+2)
                    break
                }
            }
            status = "?"
            if (index($0, "Passed"))   status = "PASS"
            else if (index($0, "Failed")) status = "FAIL"
            else if (index($0, "Skipped")) status = "SKIP"
            wall = ""
            for (i = NF; i > 0; i--) {
                if ($i == "sec") { wall = $(i-1) " s"; break }
            }
            printf "| `%s` | %s | %s |\n", name, status, wall
            name = ""
        }
    ' "$WORK/ctest.out" > "$WORK/ctest.rows"

    if [[ -s "$WORK/ctest.rows" ]]; then
        cat <<EOF

| Test | Status | Wall |
|---|---|---|
EOF
        cat "$WORK/ctest.rows"
    fi

    local summary
    summary=$(grep -E '^[0-9]+% tests passed' "$WORK/ctest.out" | head -1)
    if [[ -n "$summary" ]]; then
        echo
        echo "Summary: $summary"
    fi
    if [[ $CTEST_RC -ne 0 ]]; then
        echo
        echo "**ctest exited with rc=$CTEST_RC.**"
    fi
}

emit_quality() {
    cat <<EOF

---

## Step 2 — Generation Quality + Drift vs PyTorch

Prompt fed to both engines (HF SentencePiece tokenization, 200 greedy tokens):

\`\`\`
$(printf "%s\n" "$QP_PROMPT" | fold -s -w 78)
\`\`\`

EOF

    # Pull the metrics block (already stripped of ANSI by strip_ansi).
    local rate firstdiv klpq klqp
    rate=$(grep -E 'Exact-match rate'    "$WORK/quality.out" | head -1 | sed -E 's/.*: //; s/  +\(.*\)$//; s/  +<-.*$//')
    firstdiv=$(grep -E 'First divergence' "$WORK/quality.out" | head -1 | sed -E 's/.*: //; s/  +\(.*\)$//')
    klpq=$(grep -E 'KL\(PyTorch \|\| lite\)' "$WORK/quality.out" | head -1 | sed -E 's/.*: //; s/  +\(.*\)$//')
    klqp=$(grep -E 'KL\(lite \|\| PyTorch\)' "$WORK/quality.out" | head -1 | sed -E 's/.*: //; s/  +\(.*\)$//')

    # Phase 14 — optional int4 leg
    local int4_rate int4_div int4_klpq int4_klqp
    int4_rate=$(grep -E 'Exact-match rate' "$WORK/quality.out" | sed -n '2p' | sed -E 's/.*: //; s/  +\(.*\)$//; s/  +<-.*$//')
    int4_div=$(grep -E 'First divergence' "$WORK/quality.out" | sed -n '2p' | sed -E 's/.*: //; s/  +\(.*\)$//')
    int4_klpq=$(grep -E 'KL\(PyTorch \|\| int4\)' "$WORK/quality.out" | head -1 | sed -E 's/.*: //; s/  +<-.*$//')
    int4_klqp=$(grep -E 'KL\(int4 \|\| PyTorch\)' "$WORK/quality.out" | head -1 | sed -E 's/.*: //')

    cat <<EOF
| Metric | lite-ssm (fp16) | lite-ssm (int4_b32) |
|---|---|---|
| Exact-match vs PyTorch | ${rate:-—} | ${int4_rate:-—} |
| First divergence | ${firstdiv:-—} | ${int4_div:-—} |
| KL(PyTorch ‖ lite) final step | ${klpq:-—} | ${int4_klpq:-—} |
| KL(lite ‖ PyTorch) final step | ${klqp:-—} | ${int4_klqp:-—} |
EOF

    # Side-by-side text excerpts.
    cat <<'EOF'

#### Decoded text (both engines, greedy, 200 tokens)

EOF
    awk '
        /^PyTorch \(mps/ { in_py=1; in_lite=0; print "**" $0 "**"; print ""; print "```"; next }
        /^lite-ssm \(Metal/ { in_lite=1; in_py=0; print "```"; print ""; print "**" $0 "**"; print ""; print "```"; next }
        /^Metrics:/ { if (in_py || in_lite) print "```"; in_py=0; in_lite=0 }
        in_py || in_lite { print }
    ' "$WORK/quality.out"

    if [[ $QP_RC -ne 0 ]]; then
        echo
        echo "**quality_parity.py exited with rc=$QP_RC** — partial data above."
    fi
}

emit_benchmark() {
    cat <<EOF

---

## Step 3 — Head-to-Head Benchmark vs PyTorch / MPS

Workload: 12 iterations (first dropped as warm-up) of ingesting a synthetic
log line as a byte-token prompt and greedy-decoding a 5-token summary.
Reported numbers are the **median** across kept iterations; peak RSS comes
from \`/usr/bin/time -l\`.

EOF
    if [[ $BENCH_RC -ne 0 ]]; then
        echo "**run_benchmark.sh exited with rc=$BENCH_RC.**"
        echo
    fi
    # Embed the markdown the benchmark script emits, but skip its own
    # top-of-doc heading + the leading workload paragraph (we already wrote
    # our own h2 + paragraph just above).
    awk '
        /^# lite-ssm vs PyTorch/      { skip = 1; next }
        skip && /^Workload:/          { next }
        skip && /^synthetic system-log/ { next }
        skip && /^5-token summary\./   { next }
        skip && /^$/                   { skip = 0; next }
        skip                           { next }
        { print }
    ' "$WORK/benchmark.out"
}

emit_bleed() {
    cat <<EOF

---

## Step 4 — Memory Bleed Probe (5 000 tokens)

Per-token RSS sampled via \`mach_task_basic_info()\`. Phase 10 fix landed
the autoreleasepool + workspace-owned logits scratch — the curve should
plateau, not grow linearly.

\`\`\`
$(cat "$WORK/bleed.out")
\`\`\`

EOF
    # Pull the summary block into a tight table. Grab the trailing value
    # together with its unit by taking everything after the ":" delimiter.
    local baseline peak finalrss growth verdict
    baseline=$(grep 'baseline (iter' "$WORK/bleed.out" | head -1 | sed -E 's/.*: //')
    peak=$(grep    'peak ' "$WORK/bleed.out" | grep ' : '   | head -1 | sed -E 's/.*: //')
    finalrss=$(grep 'final '       "$WORK/bleed.out" | grep ' : ' | head -1 | sed -E 's/.*: //')
    growth=$(grep  'growth vs baseline' "$WORK/bleed.out" | head -1 | sed -E 's/.*: //')
    verdict=$(grep -E '\[bleed\] (PASS|FAIL):' "$WORK/bleed.out" | sed -E 's/^\[bleed\] //' | head -1)

    cat <<EOF
| Metric | Value |
|---|---|
| Baseline (iter 5) | ${baseline:-—} |
| Peak | ${peak:-—} |
| Final (after 5 000 tokens) | ${finalrss:-—} |
| Growth vs baseline | ${growth:-—} |
| Verdict | ${verdict:-—} |
EOF

    if [[ $BLEED_RC -ne 0 ]]; then
        echo
        echo "**test_memory_bleed exited with rc=$BLEED_RC.**"
    fi
}

emit_concurrency() {
    cat <<EOF

---

## Step 5 — Swarm Concurrency (8 threads × 100 decode tokens)

Shared mmap'd weights; per-thread \`MetalOps\` + \`Mamba2Workspace\` +
\`Mamba2State\`. We measure aggregate TPS against a single-thread baseline.

\`\`\`
$(awk '/SUMMARY|wall clock|total tokens|aggregate|single-thread|scaling|fastest|slowest|spread/ { print }' "$WORK/concurrency.out")
\`\`\`

EOF

    local solo agg scaling spread
    solo=$(awk    '/single-thread TPS/ { print $(NF-1) " " $NF }' "$WORK/concurrency.out" | head -1)
    agg=$(awk     '/aggregate TPS/     { print $(NF-1) " " $NF }' "$WORK/concurrency.out" | head -1)
    scaling=$(awk '/scaling factor/    { print $4 }'             "$WORK/concurrency.out" | head -1)
    spread=$(awk  '/spread \(max\/min\)/ { print $NF }'          "$WORK/concurrency.out" | head -1)

    cat <<EOF

| Metric | Value |
|---|---|
| Single-thread TPS | ${solo:-—} |
| Aggregate TPS (8 threads) | ${agg:-—} |
| Scaling factor (ideal = 8.0x) | ${scaling:-—} |
| Fairness spread (max/min duration) | ${spread:-—} |
EOF

    if [[ $CONC_RC -ne 0 ]]; then
        echo
        echo "**test_concurrency exited with rc=$CONC_RC.**"
    fi
}

emit_quant_footprint() {
    local fp16_path="${MODEL_SSM}"
    local int4_path="${ROOT}/model.int4.ssm"
    if [[ ! -f "$fp16_path" || ! -f "$int4_path" ]]; then
        return
    fi
    local fp16_bytes int4_bytes ratio savings
    fp16_bytes=$(stat -f %z "$fp16_path")
    int4_bytes=$(stat -f %z "$int4_path")
    ratio=$(awk -v a="$fp16_bytes" -v b="$int4_bytes" 'BEGIN { printf "%.2fx", a/b }')
    savings=$(awk -v a="$fp16_bytes" -v b="$int4_bytes" 'BEGIN { printf "%.1f%%", 100*(a-b)/a }')

    cat <<EOF

---

## Step 6 — Quantization Footprint (Phase 14 INT4_BLOCK32)

\`--quantize int4\` packs every 2-D Linear weight (\`in_proj\`, \`out_proj\`,
\`lm_head\`) into 32-element blocks: 32 signed-4-bit nibbles + one fp16 scale
per block. 1-D tensors (norms, biases, conv1d, A_log, D, dt_bias) stay fp16.

| Model | On-disk size |
|---|---:|
| \`$(basename "$fp16_path")\` (fp16)       | $(fmt_size_mb "$fp16_bytes") |
| \`$(basename "$int4_path")\` (int4_b32)   | $(fmt_size_mb "$int4_bytes") |
| Compression ratio                          | **${ratio}** |
| Footprint savings                          | **${savings}** |

Quality drift caused by this compression is reported in Step 2 (KL columns).
EOF
}

emit_footer() {
    cat <<'EOF'

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
EOF
}

# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
{
    emit_header
    emit_build
    emit_ctest
    emit_quality
    emit_benchmark
    emit_bleed
    emit_concurrency
    emit_quant_footprint
    emit_footer
} > "$OUT_MD"

log "wrote $OUT_MD"
log "raw outputs under $WORK"

# Exit non-zero only if a critical step failed.
if (( BUILD_RC || CTEST_RC )); then
    exit 1
fi
exit 0
