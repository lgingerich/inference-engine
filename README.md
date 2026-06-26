# infer

A minimal inference engine built from scratch in Rust. Designed for a single RTX 5090. No frameworks, no black boxes — every tensor shape and operation is explicit.

The goal is to understand what goes into LLM inference by building it, then systematically close the performance gap against production engines.

## Project layout

```
src/
  main.rs       — autoregressive loop, sampling, output
  model.rs      — transformer architecture (attention, MLP, forward pass)
course/         — Python educational materials (transformers + inference deep dives)
```

## Benchmarking

The standard way to compare inference engines is to measure under consistent conditions and report the metrics that users experience directly.

### Metrics that matter

| Metric | Definition | Why it matters |
|---|---|---|
| TTFT | Time to First Token — latency from request to first output byte | Perceived responsiveness |
| TPOT | Time Per Output Token — median inter-token latency during decode | Smoothness of streaming |
| tok/s (decode) | Tokens per second during the decode phase | Raw throughput for a single stream |
| E2E latency | End-to-end time for a fixed output length | User-facing completion time |
| req/s | Concurrent requests served per second under load | Multi-user throughput |

### Industry baseline: Artificial Analysis

[artificialanalysis.ai](https://artificialanalysis.ai) continuously benchmarks every major inference API provider (OpenAI, Anthropic, Together AI, Fireworks, Groq, DeepInfra) against these metrics, per model. For Llama 3.1 8B — the primary target of this engine — they publish per-provider numbers for TTFT, output tok/s, and total latency. These are the numbers to compare against once the engine is running on a 5090 with real weights.

### Benchmarking your engine against the standard

Once the engine exposes an OpenAI-compatible API (`/v1/chat/completions` with SSE streaming), use these tools to produce comparable numbers:

**[llmperf](https://github.com/ray-project/llmperf)** (recommended)

The most widely adopted open-source LLM performance benchmarking tool. Supports the OpenAI chat completions format and SS

- Measures TTFT, inter-token latency, total throughput, and latency percentiles (p50, p95, p99)
- Configurable prompt lengths, output token counts, and concurrency levels
- Produces standardized output suitable for direct comparison with published numbers

```bash
# Benchmark a local engine (once the OpenAI-compatible API is running)
python token_benchmark_ray.py \
    --model "llama-3.1-8b" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 150 \
    --mean-output-tokens 256 \
    --stddev-output-tokens 32 \
    --num-concurrent-requests 1 \
    --results-dir results/
```

**[GenAI-Perf](https://github.com/triton-inference-server/genai-perf)** (NVIDIA)

More detailed profiling built by NVIDIA for their inference stack. Useful when you want GPU telemetry alongside latency numbers — memory bandwidth, SM utilization, and kernel timing. Works with any OpenAI-compatible endpoint.

### Expected baseline (single GPU)

For a single-stream decode workload on Llama 3.1 8B (no batching), expected numbers by engine tier:

| Engine tier | Decode tok/s | Notes |
|---|---|---|
| Together AI / Fireworks | 80–130 | Heavily customized vLLM forks, A100/H100 |
| Stock vLLM (A100) | 60–90 | Out of the box |
| Stock llama.cpp (consumer GPU) | 40–60 | Quantized, CPU offloading for layers |
| **This engine (target, RTX 5090)** | **TBD** | 5090 > A100 in raw compute |

The 5090's raw FP16 throughput and memory bandwidth exceed the A100. A well-optimized single-stream engine on a 5090 should land between stock vLLM and the production forks. Multi-stream throughput (continuous batching) is a separate optimization target and typically lags production engines by a larger margin until batching is implemented.

### Tracking your own progress

The most useful leaderboard is internal. After each optimization, remeasure and track the delta:

| Optimization | tok/s before | tok/s after | Delta |
|---|---|---|---|
| KV cache | — | — | — |
| CUDA backend | — | — | — |
| FlashAttention | — | — | — |
| PagedAttention | — | — | — |
| FP8 quantization | — | — | — |
| Speculative decoding | — | — | — |

Each row represents a concrete speedup. The cumulative effect is your engine's story.

## License

MIT
