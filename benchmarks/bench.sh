#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Build release binary, start server, run one fixed LLMPerf benchmark, shut down.
#
# Usage:
#   ./benchmarks/bench.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLMPERF_DIR="$SCRIPT_DIR/llmperf"
RESULTS_ROOT="$SCRIPT_DIR/results"
HISTORY_FILE="$SCRIPT_DIR/history.csv"
PORT=3000
SERVER_PID=""

# Fixed benchmark contract. Change workload inputs only when intentionally
# starting a new historical benchmark series.
MODEL="meta-llama/Llama-3.2-3B"       # Change when benchmarking a different model.
WORKLOAD="single_stream_fixed"        # Change only if request shape changes: concurrency, token lengths, prompt distribution, sampling, or API mode.
TRIAL_ID="t2"                         # Increment for every recorded run; must be globally unique in history.csv.
PLATFORM="m3_metal"                     # Short hardware/backend key used in charts, e.g. m3_cpu, m3_metal, rtx_5090.
CHANGE=""                             # Optional short implementation change, e.g. flash_attention, paged_kv, cuda_graphs.
PRECISION="BF16"                      # Change when weights/compute precision changes, e.g. FP32, FP16, BF16, INT8.
RUN_LABEL="${TRIAL_ID}-${PLATFORM}"   # Chart label; usually derived from trial + platform + optional change.
if [[ -n "$CHANGE" ]]; then
    RUN_LABEL="${RUN_LABEL}-${CHANGE}"
fi

# The LLMPerf workload literals are intentionally inlined below. Changing them
# invalidates direct comparisons with prior rows and should come with a new
# WORKLOAD name plus a new history series.

# ── preflight ────────────────────────────────────────────────────────────────
echo "==> checking benchmark setup..."
if [[ ! -d "$LLMPERF_DIR" ]]; then
    echo "error: llmperf checkout not found at $LLMPERF_DIR"
    exit 1
fi

uv run --project "$LLMPERF_DIR" --python 3.10 \
    python "$SCRIPT_DIR/report.py" \
    --history "$HISTORY_FILE" \
    --check-trial-id "$TRIAL_ID"

# ── build ────────────────────────────────────────────────────────────────────
echo "==> building..."
cd "$PROJECT_DIR"
cargo build --release 2>&1

# ── start server ─────────────────────────────────────────────────────────────
echo "==> starting server on port $PORT..."
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

./target/release/infer --serve --port "$PORT" &
SERVER_PID=$!

echo -n "    waiting"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo " ready"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo " died"
        exit 1
    fi
    echo -n "."
    sleep 1
done

if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "error: server did not become ready within 120s"
    exit 1
fi

# ── benchmark ────────────────────────────────────────────────────────────────
echo "==> running fixed LLMPerf benchmark..."
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$RESULTS_ROOT/${TIMESTAMP}_${COMMIT}"
mkdir -p "$RESULTS_DIR"

export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="local-llmperf"

uv run --project "$LLMPERF_DIR" --python 3.10 \
    python "$LLMPERF_DIR/token_benchmark_ray.py" \
    --model "$MODEL" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 0 \
    --mean-output-tokens 256 \
    --stddev-output-tokens 0 \
    --max-num-completed-requests 5 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "$RESULTS_DIR" \
    --llm-api openai \
    --additional-sampling-params '{"temperature":0.0,"seed":42}' \
    --metadata "git_commit=${COMMIT},benchmark=${WORKLOAD},trial_id=${TRIAL_ID},platform=${PLATFORM},change=${CHANGE},run_label=${RUN_LABEL},precision=${PRECISION}"

rm -f "$RESULTS_ROOT/latest"
ln -s "$(basename "$RESULTS_DIR")" "$RESULTS_ROOT/latest"
echo "    results saved to $RESULTS_DIR"
echo "    latest symlink: $RESULTS_ROOT/latest"
echo
echo "==> metrics to save in README optimization log..."
SUMMARY_FILE="$(ls "$RESULTS_DIR"/*_summary.json)"
uv run --project "$LLMPERF_DIR" --python 3.10 python "$SCRIPT_DIR/report.py" "$SUMMARY_FILE" \
    --trial-id "$TRIAL_ID" \
    --platform "$PLATFORM" \
    --change "$CHANGE" \
    --run-label "$RUN_LABEL" \
    --precision "$PRECISION"

# ── cleanup ──────────────────────────────────────────────────────────────────
echo "==> stopping server..."
cleanup
trap - EXIT
echo "    done"
