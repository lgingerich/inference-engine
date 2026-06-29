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
ENV_FILE="$PROJECT_DIR/.env"
VAST_SETUP="$SCRIPT_DIR/vast_setup.sh"
PORT="${PORT:-3000}"
SERVER_PID=""

# The LLMPerf workload literals are intentionally inlined below. Changing them
# invalidates direct comparisons with prior rows and should come with a new
# WORKLOAD name plus a new history series.

load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
}

ensure_hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN
        export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
        return
    fi

    if [[ -t 0 ]]; then
        echo "==> Hugging Face token not found in .env"
        read -rsp "    HF_TOKEN: " HF_TOKEN
        echo
        if [[ -n "$HF_TOKEN" ]]; then
            umask 077
            if [[ -f "$ENV_FILE" ]]; then
                printf '\nHF_TOKEN=%q\n' "$HF_TOKEN" >> "$ENV_FILE"
            else
                printf 'HF_TOKEN=%q\n' "$HF_TOKEN" > "$ENV_FILE"
            fi
            export HF_TOKEN
            export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
            echo "    saved token to $ENV_FILE"
            return
        fi
    fi

    echo "error: HF_TOKEN is required. Add it to $ENV_FILE or export it before running."
    exit 1
}

sanitize_slug() {
    echo "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -cs '[:alnum:]' '_' \
        | tr -s '_' \
        | sed 's/^_//; s/_$//'
}

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

apply_tool_paths() {
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
}

detect_backend() {
    local os arch gpu_name gpu_slug

    if [[ -n "${BENCH_BACKEND:-}" ]]; then
        case "$BENCH_BACKEND" in
            cuda)
                BACKEND="cuda"
                if has_nvidia_gpu; then
                    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
                    gpu_slug="$(sanitize_slug "$gpu_name")"
                    DETECTED_PLATFORM="${gpu_slug}_cuda"
                else
                    DETECTED_PLATFORM="cuda"
                fi
                return
                ;;
            metal)
                BACKEND="metal"
                DETECTED_PLATFORM="m3_metal"
                return
                ;;
            cpu)
                BACKEND="cpu"
                os="$(uname -s)"
                arch="$(uname -m)"
                DETECTED_PLATFORM="$(sanitize_slug "${os}_${arch}")_cpu"
                return
                ;;
            *)
                echo "error: BENCH_BACKEND must be one of: cpu, metal, cuda"
                exit 1
                ;;
        esac
    fi

    os="$(uname -s)"
    arch="$(uname -m)"

    if [[ "$os" == "Linux" ]] && has_nvidia_gpu; then
        BACKEND="cuda"
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
        gpu_slug="$(sanitize_slug "$gpu_name")"
        DETECTED_PLATFORM="${gpu_slug}_cuda"
        return
    fi

    if [[ "$os" == "Darwin" ]] && [[ "$arch" == "arm64" ]]; then
        BACKEND="metal"
        DETECTED_PLATFORM="m3_metal"
        return
    fi

    BACKEND="cpu"
    DETECTED_PLATFORM="$(sanitize_slug "${os}_${arch}")_cpu"
}

configure_cargo() {
    CARGO_FEATURE_ARGS=()
    case "$BACKEND" in
        cuda)
            CARGO_FEATURE_ARGS=(--no-default-features --features cuda)
            ;;
        metal)
            CARGO_FEATURE_ARGS=(--no-default-features --features metal)
            ;;
        cpu)
            CARGO_FEATURE_ARGS=(--no-default-features)
            ;;
        *)
            echo "error: unknown backend $BACKEND"
            exit 1
            ;;
    esac
}

load_env
apply_tool_paths
detect_backend

if [[ "$BACKEND" == "cuda" && "${SKIP_VAST_SETUP:-0}" != "1" ]]; then
    if [[ ! -x "$VAST_SETUP" ]]; then
        echo "error: Vast setup script is missing or not executable: $VAST_SETUP"
        exit 1
    fi
    "$VAST_SETUP"
    apply_tool_paths
fi

configure_cargo
ensure_hf_token

# Fixed benchmark contract. Change workload inputs only when intentionally
# starting a new historical benchmark series.
MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"       # Change when benchmarking a different model.
WORKLOAD="${WORKLOAD:-single_stream_fixed}"     # Change only if request shape changes: concurrency, token lengths, prompt distribution, sampling, or API mode.
TRIAL_ID="${TRIAL_ID:-t1}"                      # Increment for every recorded run; must be globally unique in history.csv.
PLATFORM="${PLATFORM:-$DETECTED_PLATFORM}"      # Short hardware/backend key used in charts, e.g. m3_cpu, m3_metal, rtx_5090_cuda.
CHANGE="${CHANGE:-}"                            # Optional short implementation change, e.g. flash_attention, paged_kv, cuda_graphs.
DEFAULT_PRECISION="BF16"
if [[ "$BACKEND" == "cpu" ]]; then
    DEFAULT_PRECISION="F32"
fi
PRECISION="${PRECISION:-$DEFAULT_PRECISION}"    # Change when weights/compute precision changes, e.g. F32, FP16, BF16, INT8.
MEAN_INPUT_TOKENS="${MEAN_INPUT_TOKENS:-550}"   # Fixed prompt token target for the benchmark series.
MEAN_OUTPUT_TOKENS="${MEAN_OUTPUT_TOKENS:-256}" # Fixed requested output token target for the benchmark series.
MIN_OUTPUT_TOKENS="${MIN_OUTPUT_TOKENS:-200}"   # Guardrail: reject runs that stop before meaningful decode work.
COMPLETED_REQUESTS="${COMPLETED_REQUESTS:-5}"   # Fixed completed request count for the benchmark series.
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-1}" # Fixed concurrency for the benchmark series.
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"         # Unrecorded requests to warm model/cache/kernel paths.
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-16}"    # Keep warmup cheap while exercising decode.
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-180}"
if [[ "$BACKEND" == "cuda" ]]; then
    BENCH_SYNC_DEST="${BENCH_SYNC_DEST:-lgingerich@localhost:/Users/lgingerich/Documents/Code/inference-engine/benchmarks}"
    BENCH_SYNC_SSH_PORT="${BENCH_SYNC_SSH_PORT:-2222}"
else
    BENCH_SYNC_DEST="${BENCH_SYNC_DEST:-}"
    BENCH_SYNC_SSH_PORT="${BENCH_SYNC_SSH_PORT:-}"
fi
BENCH_SYNC_SSH_OPTS="${BENCH_SYNC_SSH_OPTS:-}"  # Optional extra ssh args, e.g. "-i ~/.ssh/id_ed25519".
BENCH_SYNC_REQUIRED="${BENCH_SYNC_REQUIRED:-0}" # Set to 1 to fail the benchmark if artifact sync fails.
RUN_LABEL="${RUN_LABEL:-${TRIAL_ID}-${PLATFORM}}" # Chart label; usually derived from trial + platform + optional change.
if [[ -n "$CHANGE" && "$RUN_LABEL" == "${TRIAL_ID}-${PLATFORM}" ]]; then
    RUN_LABEL="${RUN_LABEL}-${CHANGE}"
fi

run_warmup() {
    if [[ "$WARMUP_REQUESTS" -le 0 ]]; then
        echo "==> skipping warmup requests"
        return
    fi

    echo "==> running $WARMUP_REQUESTS unrecorded warmup request(s)..."
    for i in $(seq 1 "$WARMUP_REQUESTS"); do
        echo "    warmup $i/$WARMUP_REQUESTS"
        curl -sfN --max-time "$WARMUP_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one short warmup sentence about inference benchmarking.\"}],\"stream\":true,\"max_tokens\":${WARMUP_MAX_TOKENS},\"temperature\":0.0,\"seed\":42}" \
            "http://localhost:${PORT}/v1/chat/completions" > /dev/null
    done
}

sync_results() {
    if [[ -z "$BENCH_SYNC_DEST" ]]; then
        return
    fi

    if [[ "$BENCH_SYNC_DEST" != *:* ]]; then
        echo "warning: BENCH_SYNC_DEST must look like user@host:/path/to/inference-engine/benchmarks"
        if [[ "$BENCH_SYNC_REQUIRED" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    local ssh_cmd=(ssh)
    if [[ -n "$BENCH_SYNC_SSH_PORT" ]]; then
        ssh_cmd+=(-p "$BENCH_SYNC_SSH_PORT")
    fi
    if [[ -n "$BENCH_SYNC_SSH_OPTS" ]]; then
        # shellcheck disable=SC2206
        local extra_ssh_opts=($BENCH_SYNC_SSH_OPTS)
        ssh_cmd+=("${extra_ssh_opts[@]}")
    fi

    echo "==> syncing benchmark artifacts..."
    echo "    destination: $BENCH_SYNC_DEST"

    local sync_status=0
    "${ssh_cmd[@]}" "${BENCH_SYNC_DEST%%:*}" "mkdir -p '${BENCH_SYNC_DEST#*:}/results' '${BENCH_SYNC_DEST#*:}/charts'" || sync_status=$?
    if [[ "$sync_status" -eq 0 ]]; then
        rsync -az --partial -e "${ssh_cmd[*]}" "$RESULTS_DIR/" "$BENCH_SYNC_DEST/results/$(basename "$RESULTS_DIR")/" || sync_status=$?
        rsync -az --partial -e "${ssh_cmd[*]}" "$HISTORY_FILE" "$BENCH_SYNC_DEST/history.csv" || sync_status=$?
        rsync -az --partial -e "${ssh_cmd[*]}" "$SCRIPT_DIR/charts/" "$BENCH_SYNC_DEST/charts/" || sync_status=$?
    fi

    if [[ "$sync_status" -ne 0 ]]; then
        echo "warning: benchmark artifact sync failed with exit code $sync_status"
        if [[ "$BENCH_SYNC_REQUIRED" == "1" ]]; then
            return "$sync_status"
        fi
        return 0
    fi

    echo "    sync complete"
}

# ── preflight ────────────────────────────────────────────────────────────────
echo "==> checking benchmark setup..."
echo "    backend: $BACKEND"
echo "    platform: $PLATFORM"
echo "    cargo flags: ${CARGO_FEATURE_ARGS[*]:-(default)}"

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
cargo build --release "${CARGO_FEATURE_ARGS[@]}" 2>&1

# ── benchmark output paths ───────────────────────────────────────────────────
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$RESULTS_ROOT/${TIMESTAMP}_${COMMIT}"
ENGINE_METRICS_FILE="$RESULTS_DIR/engine_metrics.jsonl"
mkdir -p "$RESULTS_DIR"
rm -f "$ENGINE_METRICS_FILE"
export ENGINE_METRICS_FILE

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
export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="local-llmperf"

run_warmup

echo "==> running fixed LLMPerf benchmark..."
uv run --project "$LLMPERF_DIR" --python 3.10 \
    python "$LLMPERF_DIR/token_benchmark_ray.py" \
    --model "$MODEL" \
    --mean-input-tokens "$MEAN_INPUT_TOKENS" \
    --stddev-input-tokens 0 \
    --mean-output-tokens "$MEAN_OUTPUT_TOKENS" \
    --stddev-output-tokens 0 \
    --max-num-completed-requests "$COMPLETED_REQUESTS" \
    --timeout 600 \
    --num-concurrent-requests "$CONCURRENT_REQUESTS" \
    --results-dir "$RESULTS_DIR" \
    --llm-api openai \
    --additional-sampling-params '{"temperature":0.0,"seed":42}' \
    --metadata "git_commit=${COMMIT},benchmark=${WORKLOAD},trial_id=${TRIAL_ID},platform=${PLATFORM},change=${CHANGE},run_label=${RUN_LABEL},precision=${PRECISION},min_output_tokens=${MIN_OUTPUT_TOKENS},warmup_requests=${WARMUP_REQUESTS}"

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
    --precision "$PRECISION" \
    --min-output-tokens "$MIN_OUTPUT_TOKENS" \
    --engine-metrics "$ENGINE_METRICS_FILE"

sync_results

# ── cleanup ──────────────────────────────────────────────────────────────────
echo "==> stopping server..."
cleanup
trap - EXIT
echo "    done"
