# infer

A small Rust inference engine for learning how LLM serving works from the inside out.

Right now it targets Llama-style decoder-only models, starting with `meta-llama/Llama-3.2-3B`, and is being tuned toward a single RTX 5090.

## Project layout

```
src/
  main.rs       — autoregressive loop, sampling, output
  api.rs        — OpenAI-compatible HTTP API (chat completions with SSE streaming)
benchmarks/
  bench.sh      — build + serve + benchmark + teardown (one command)
  vast_setup.sh — idempotent Vast.ai CUDA host setup used by bench.sh
  report.py     — records LLMPerf summaries and generates charts
  history.csv   — durable benchmark history used by generated charts
  llmperf/      — LLMPerf load-test runner used by the benchmark
course/         — Python educational materials (transformers + inference deep dives)
```

## Benchmarking

```bash
./benchmarks/bench.sh
```

`bench.sh` builds the release binary, starts the OpenAI-compatible server, runs one fixed LLMPerf workload, records the run in `benchmarks/history.csv`, and regenerates the chart below. Raw LLMPerf output is saved under `benchmarks/results/`.

The script detects the host backend automatically:

- Apple Silicon uses the Candle Metal build.
- Linux with an NVIDIA GPU uses the Candle CUDA build and first runs `benchmarks/vast_setup.sh` to install/verify CUDA, Rust, uv, and build tools.
- Other hosts use the CPU build.

Put your Hugging Face token in a local `.env` file before running, or enter it when prompted:

```bash
HF_TOKEN=hf_...
```

Useful overrides:

```bash
TRIAL_ID=t3 CHANGE=cuda_baseline ./benchmarks/bench.sh
BENCH_BACKEND=cpu TRIAL_ID=t1 PLATFORM=m3_cpu ./benchmarks/bench.sh
```

### Vast.ai

Use the local working tree as the source of truth for Vast runs. Do not clone on the instance unless you intentionally want committed code only.

1. Use the SSH key already registered with Vast.ai. On this machine that is `~/.ssh/id_ed25519.pub`:

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

2. Rent an Ubuntu/Debian RTX 5090 instance with SSH enabled, preferably direct SSH, and at least 40-60 GB of disk.

3. Copy the current local checkout to the instance. Replace `PORT` and `HOST` with the direct SSH values from the Vast instance card:

```bash
rsync -az --delete --partial --progress --stats \
  --exclude target \
  --exclude benchmarks/results \
  --exclude benchmarks/llmperf/.venv \
  -e "ssh -p PORT" \
  ./ \
  root@HOST:/root/inference-engine/
```

4. SSH in and run the benchmark. If you want benchmark artifacts to sync back to this local checkout automatically, enable **System Settings → General → Sharing → Remote Login** on macOS, then open the Vast SSH session with a reverse tunnel:

```bash
ssh -p PORT root@HOST \
  -L 8080:localhost:3000 \
  -R 2222:localhost:22
```

The `-R 2222:localhost:22` tunnel gives the remote benchmark script a temporary SSH route back to your Mac. CUDA runs default to syncing artifacts to this local checkout through that tunnel.

Then run the benchmark on the remote:

```bash
cd ~/inference-engine
./benchmarks/bench.sh
```

This syncs the run directory, `history.csv`, and generated charts back to the local checkout after a successful benchmark. To make sync failure fail the whole benchmark, run with `BENCH_SYNC_REQUIRED=1`.

If local port `8080` is already in use, pick another local port for forwarding, for example:

```bash
ssh -p PORT root@HOST -L 8081:localhost:3000 -R 2222:localhost:22
```

The benchmark is intentionally fixed:

- Model: Llama 3.2 3B
- Requests: 5 completed, 1 concurrent
- Tokens: 550 input, 256 requested output
- Sampling: `temperature=0.0`, `seed=42`

![LLMPerf performance history](benchmarks/charts/performance.png)

### Results

`TPS` is output tokens per second. New runs use engine-native model-token metrics for TPS/TPOT; legacy rows below predate that fix and should be rerun before comparing throughput. `TTFT` is time to first token. Lower latency is better; higher TPS is better.

| Run | Date | Platform | Precision | Model | Optimization | tok/s | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | TPOT p95 (s) | E2E p50 (s) | E2E p95 (s) | Requests | Errors | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| t1-darwin_arm64_cpu | 2026-06-29 | darwin_arm64_cpu | F32 | Llama 3.2 3B | CPU baseline | 2.09 | 4.131 | 4.547 | 0.4792 | 0.4842 | 126.319 | 127.671 | 5 | 0 | Candle CPU path, F32 weights, engine-native token accounting |

## License

MIT
