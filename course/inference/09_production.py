"""
INFERENCE MODULE 9: PRODUCTION ENGINES & THE STATE OF THE ART
===============================================================

We've built every component of an inference engine from scratch:
  - KV cache (Module 2)
  - Continuous batching (Module 3)
  - Quantization (Module 4)
  - FlashAttention & PagedAttention (Module 5)
  - Speculative decoding (Module 6)
  - Distributed inference (Module 7)
  - HTTP API serving (Module 8)

This final module surveys how production engines combine these at scale,
what tradeoffs they make, and where the field is heading. Think of it as
"here's what you'd build next if you scaled up each module by 1000×."

WHAT YOU'LL LEARN:
   1. The architecture of vLLM, TGI, SGLang, and llama.cpp
   2. How production engines differ from our educational implementations
   3. The current state of the art (2025-2026)
   4. Where to go next: running real models, contributing, researching

AFTER THIS MODULE:
   You'll understand the full landscape of LLM serving and know exactly
   where to start for real-world projects.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# THE MAJOR PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PRODUCTION INFERENCE ENGINES — Architecture Deep-Dive")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1. vLLM — The throughput king (and the most popular)           │
│  https://github.com/vllm-project/vllm                          │
└─────────────────────────────────────────────────────────────────┘

ARCHITECTURE:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ REST API │──→│ Scheduler│──→│  Worker  │──→│  Worker  │
  │ (FastAPI)│   │(continu- │   │ (GPU 0)  │   │ (GPU 1)  │
  └──────────┘   │ ous batch│   │ w/ TP    │   │ w/ TP    │
                 └──────────┘   └──────────┘   └──────────┘
                 ↕ prefix cache, priority queue, preemption

KEY INNOVATIONS (each corresponding to a course module):
  ● PagedAttention (Module 5): virtual memory for KV cache
  ● Continuous batching (Module 3): dynamic request management
  ● Prefix caching: share KV cache for common system prompts
  ● CUDA graph capture: pre-record decode step → 30% speedup
  ● Tensor + pipeline parallelism (Module 7)
  ● Chunked prefill: split long prefills to avoid blocking decodes
  ● Speculative decoding (Module 6): built-in EAGLE/Medusa support
  ● FP8 quantization (Module 4): hardware-accelerated on H100

PERFORMANCE (LLaMA-7B, A100, continuous batching):
  - 15-25× higher throughput than raw HuggingFace transformers
  - < 50ms TTFT for typical prompts
  - Supports up to 128K context with FlashAttention-3
  - Used by: LMSYS (Chatbot Arena), Anyscale, most startups

BY FAR the most popular inference engine as of 2026.
34K+ GitHub stars, 500+ contributors, backed by UC Berkeley.

┌─────────────────────────────────────────────────────────────────┐
│ 2. TGI — HuggingFace's native engine                           │
│  https://github.com/huggingface/text-generation-inference      │
└─────────────────────────────────────────────────────────────────┘

KEY DIFFERENCES FROM vLLM:
  ● Written in Rust (core) + Python (API) — the Rust core gives
    predictable latency with no garbage collection pauses
  ● Watermarking: detect AI-generated text via token selection bias
  ● Grammar-constrained generation: enforce JSON/XML schema,
    regex patterns, valid Python syntax — directly in the sampler
  ● Supports ALL HuggingFace model architectures natively
    (vLLM supports most but not all)
  ● HuggingFace ecosystem integration: safetensors, transformers
    pipeline compatibility, hub model loading

WHEN TO USE TGI: deep in HF ecosystem, need watermarks/grammars
WHEN TO USE vLLM: maximum throughput, broadest community support

┌─────────────────────────────────────────────────────────────────┐
│ 3. SGLang — Agent-centric serving                              │
│  https://github.com/sgl-project/sglang                         │
└─────────────────────────────────────────────────────────────────┘

KEY INNOVATIONS:
  ● RADIX ATTENTION: automatic prefix caching that works for ANY
    shared prefix — even partial overlaps between requests.
    No manual "system prompt" tagging required. Uses a radix tree
    (prefix tree) to detect and cache matching KV cache blocks.

  ● STRUCTURED GENERATION LANGUAGE: write Python programs that
    compose LLM calls — branching, loops, conditionals. The engine
    optimizes across the entire program, not per-call.

  ● Agent-native: designed for the "think → act → observe → think"
    pattern. Prefix sharing across repeated calls from the same
    session is automatic and efficient.

  Performance is competitive with vLLM; unique in its programming
  model. Growing rapidly in agent/AI workflow use cases.

┌─────────────────────────────────────────────────────────────────┐
│ 4. llama.cpp — Local, CPU, everywhere                          │
│  https://github.com/ggerganov/llama.cpp                        │
└─────────────────────────────────────────────────────────────────┘

THE LOCAL INFERENCE ENGINE. Runs on CPU, GPU (CUDA/Metal/Vulkan),
and hybrid CPU+GPU. Designed for CONSUMER hardware.

KEY INNOVATIONS:
  ● GGUF format: self-contained model files with quantization
    metadata. Single file → everything needed to run.
  ● MMQ (Matrix Multiply Quantized): hand-optimized INT4/INT8
    matmul kernels for x86 (AVX2), ARM (NEON), Apple Silicon.
  ● KV cache quantization: INT8 cache, not just INT8 weights.
    For long contexts, this is as important as weight quantization.
  ● Multi-backend: CUDA, Metal, Vulkan, SYCL, ROCm.
  ● Apple Silicon: uses Metal Performance Shaders + ANE (Neural
    Engine) on M1/M2/M3/M4 — runs 7B models at interactive speeds
    on a laptop with no fan noise.

WHEN TO USE: you want to run models LOCALLY, no server needed.
WHEN NOT TO: maximum throughput (use GPU-native engines instead).

┌─────────────────────────────────────────────────────────────────┐
│ 5. OTHER NOTABLE PROJECTS                                      │
└─────────────────────────────────────────────────────────────────┘

MLC-LLM: compiles models to run ANYWHERE — iOS, Android, WebGPU,
  game consoles. Uses Apache TVM. Most impressive mobile perf.

Ollama: Docker-like UX: "ollama run llama3". Wraps llama.cpp.
  REST API built in. The easiest way to run LLMs locally.

ExLlamaV2: custom CUDA kernels for LLaMA/Mistral. Often FASTER
  than vLLM for single-GPU inference. Expert-crafted CUDA.

Mistral.rs: Rust-native inference. Supports GGUF, ISQ quantization.
  Minimal dependencies. Good for embedding in Rust applications.

Candle (HuggingFace): Rust ML framework with LLM inference.
  Pure Rust — no Python, no CUDA toolkit. For embedding LLMs in
  native applications.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCTION vs EDUCATIONAL — What's Different
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PRODUCTION vs EDUCATIONAL — What We Simplified")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ FROM NUMPY TO CUDA — The implementation gap                    │
└─────────────────────────────────────────────────────────────────┘

WHAT WE BUILT IN THIS COURSE:
  MiniGPT:       NumPy, single-threaded, ~23K params
  KV cache:      NumPy arrays, manual concatenation
  Batching:      Sequential loop over requests
  Quantization:  Group-wise INT4, on-the-fly dequant
  Attention:     Tiled (NumPy), PagedAttention concept
  Serving:       http.server, blocking I/O

WHAT PRODUCTION ENGINES DO:

  1. CUDA CUSTOM KERNELS: NumPy is CPU-only. Production engines
     write GPU kernels that run directly on the silicon. FlashAttention
     IS a CUDA kernel. Custom GEMM for quantized matmul (Marlin,
     CUTLASS). These are hand-optimized for specific GPU generations.

  2. GPU MEMORY POOLS: Production engines pre-allocate GPU VRAM and
     manage it with custom allocators. No malloc/free during inference.
     Memory is pinned, page-locked, and managed with CUDA streams
     for overlapping compute and data transfer.

  3. KERNEL FUSION: Multiple operations fused into one kernel launch:
     LayerNorm + QKV projection + RoPE → single kernel.
     This eliminates intermediate HBM reads/writes entirely.

  4. CUDA GRAPHS: The entire decode step is pre-recorded as a CUDA
     graph. Then replayed for each new token. This eliminates CPU-side
     kernel launch overhead (~30% of decode latency!).

  5. ASYNCHRONOUS PIPELINING: While GPU computes, CPU prepares the
     next batch. While one CUDA stream transfers data, another computes.
     Maximize GPU utilization by hiding all CPU latency.

  6. AUTO-TUNING: Production engines profile and tune kernel parameters
     (block sizes, warps, shared memory) for the specific model AND GPU.
     What's optimal for A100 isn't optimal for H100.

  BUT: The PRINCIPLES are identical. KV cache, continuous batching,
  quantization, FlashAttention — the ALGORITHMS are what we built.
  The production difference is engineering maturity, not fundamental
  complexity.
""")


# ──────────────────────────────────────────────────────────────────────────────
# STATE OF THE ART — 2025-2026
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("STATE OF THE ART — 2025-2026 Trends")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1. FP8 & FP4 (Hardware Native)                                 │
└─────────────────────────────────────────────────────────────────┘
  H100: FP8 tensor cores → 2× faster matmuls.
  B200: FP4 tensor cores → 4× faster with minimal quality loss.
  Software: vLLM FP8, TensorRT-LLM FP8 are production-ready.

┌─────────────────────────────────────────────────────────────────┐
│ 2. LONG CONTEXT (128K-2M tokens)                               │
└─────────────────────────────────────────────────────────────────┘
  FlashAttention-3 + Ring Attention → 1M+ contexts feasible.
  Gemini 1.5 Pro: 2M token context. GPT-4: 128K. Claude: 200K.
  KV cache compression: MLA (DeepSeek-V2), Quest, InfiniGen.

┌─────────────────────────────────────────────────────────────────┐
│ 3. SPECULATIVE DECODING GOES MAINSTREAM                        │
└─────────────────────────────────────────────────────────────────┘
  vLLM has built-in EAGLE and Medusa draft models.
  Draft models: typically 70-150M params → 2-4× throughput.
  Tree attention: explore multiple draft paths speculatively.

┌─────────────────────────────────────────────────────────────────┐
│ 4. PREFIX CACHING IS AUTOMATIC                                 │
└─────────────────────────────────────────────────────────────────┘
  SGLang's RadixAttention: tree-based automatic shared prefix
  detection. vLLM's APC (Automatic Prefix Caching): hash-based.
  No manual configuration needed — the engine figures it out.

┌─────────────────────────────────────────────────────────────────┐
│ 5. DISAGGREGATED PREFILL AND DECODE                            │
└─────────────────────────────────────────────────────────────────┘
  Separate GPU pools: prefill GPUs (compute, high FLOPS) and
  decode GPUs (high memory bandwidth). SplitBrain, Mooncake,
  DistServe. Better hardware matching for each phase.

┌─────────────────────────────────────────────────────────────────┐
│ 6. KV CACHE OFFLOADING                                         │
└─────────────────────────────────────────────────────────────────┘
  Split KV cache: hot tokens in GPU VRAM, cold tokens in CPU RAM
  or NVMe SSD. InfiniGen, FlexGen. Enables serving with limited
  VRAM — run 70B models on 24 GB GPUs with CPU offloading.

┌─────────────────────────────────────────────────────────────────┐
│ 7. MULTI-MODAL INFERENCE                                       │
└─────────────────────────────────────────────────────────────────┘
  Vision (LLaVA, GPT-4V, Claude 3), audio (Whisper, Gemini),
  video. Prefill now includes image encoding, audio processing.
  Serving engines need to handle heterogeneous input types.

┌─────────────────────────────────────────────────────────────────┐
│ 8. ON-DEVICE INFERENCE                                         │
└─────────────────────────────────────────────────────────────────┘
  Apple Intelligence: models on iPhone. Google Gemini Nano: on
  Android. llama.cpp, MLC-LLM, ExecuTorch. Different constraints:
  power, privacy, no network latency. The fastest-growing segment.

┌─────────────────────────────────────────────────────────────────┐
│ 9. OPEN-SOURCE CONSOLIDATION                                   │
└─────────────────────────────────────────────────────────────────┘
  vLLM: dominates throughput/scale. llama.cpp: dominates local.
  Ollama: dominates ease-of-use. SGLang: rising in agent workflows.
  TGI: strong in HuggingFace ecosystem. The field is stabilizing.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# COURSE WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("COURSE COMPLETE: What You Now Understand About LLM Inference")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ FROM RAW LOOP TO PRODUCTION ENGINE — The full arc              │
└─────────────────────────────────────────────────────────────────┘

MODULE 0: WHY INFERENCE IS HARD
  → The physics: memory-bandwidth-bound, roofline model, GPU hierarchy.
    Training is FLOPS-bound; inference waits for memory.

MODULE 1: THE RAW AUTOREGRESSIVE LOOP
  → The naive baseline. O(n³) total cost. 99% of computation is
    redundant — we recompute K and V for every past token.

MODULE 2: KV CACHE
  → Store K and V from previous steps. Don't recompute. The single
    most important optimization — 10-100× speed improvement.
    Every production engine is built on this.

MODULE 3: BATCHING & SCHEDULING
  → Process multiple requests together. Continuous batching lets
    requests join and leave fluidly — no waiting for slowest.
    Throughput-latency tradeoff is the central tension.

MODULE 4: QUANTIZATION
  → Make weights 4× smaller (INT4 vs FP16). Group-wise quantization
    fixes the outlier problem. GPTQ/AWQ/GGUF are the formats.
    Smaller = less HBM traffic = faster = fits on more GPUs.

MODULE 5: ATTENTION OPTIMIZATIONS
  → FlashAttention: O(n) memory by tiling with online softmax.
    PagedAttention: zero KV cache fragmentation.
    Together: the foundation of vLLM's throughput.

MODULE 6: SPECULATIVE DECODING
  → Draft model guesses K tokens cheaply. Large model verifies all
    at once. Mathematically lossless. 2-5× token throughput.

MODULE 7: DISTRIBUTED INFERENCE
  → Tensor parallelism: split weight matrices across GPUs.
    Pipeline parallelism: split layers across GPUs.
    Combined: serve 70B-175B+ models.

MODULE 8: THE SERVING STACK
  → OpenAI API format, streaming (SSE), metrics.
    The bridge between your engine and real users.

MODULE 9: PRODUCTION ENGINES
  → vLLM, TGI, SGLang, llama.cpp: architectures, tradeoffs, state
    of the art. Production adds CUDA kernels, memory pools, kernel
    fusion, auto-tuning — but the ALGORITHMS are what we built.

┌─────────────────────────────────────────────────────────────────┐
│ YOU NOW UNDERSTAND:                                            │
│                                                                 │
│ ✓ What happens when you type a prompt into ChatGPT              │
│ ✓ How vLLM serves thousands of requests/min on a single GPU    │
│ ✓ Why an 8× A100 server costs $200-320/hr and what it's doing  │
│ ✓ How to run LLMs on your laptop (quantization + llama.cpp)    │
│ ✓ How to make generation faster (KV cache + speculative decode)│
│ ✓ How to serve larger models (tensor + pipeline parallelism)   │
│ ✓ What metrics matter in production (TTFT, TPOT, throughput)   │
│ ✓ Where the field is going (disaggregated, agentic, on-device) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ WHAT'S NEXT — The real journey begins here                      │
│                                                                 │
│ 1. pip install vllm → serve a real model (Llama-3, Mistral)    │
│ 2. Install llama.cpp → run GGUF models on your laptop           │
│ 3. Read the vLLM source code: see PagedAttention in action      │
│ 4. Read the FlashAttention paper: it's surprisingly readable    │
│ 5. Try speculative decoding with a drafter model                │
│ 6. Run 8 GPUs in parallel: experience distributed inference     │
│ 7. Build an agent that calls YOUR inference engine              │
│ 8. Contribute to vLLM or llama.cpp (they have good-first-issues)│
│                                                                 │
│ The journey from here to production:                            │
│   NumPy → PyTorch → CUDA kernels → production engine.           │
│   Every concept you implemented IS what engines use.            │
│   The difference is engineering maturity and GPU hardware.      │
└─────────────────────────────────────────────────────────────────┘

THIS IS THE END OF THE INFERENCE COURSE.

To revisit any module:
    uv run course/inference/run.py [module_number]
    uv run course/inference/run.py list

To run the transformers course:
    uv run course/transformer/run.py
""")

if __name__ == "__main__":
    print("\nInference course complete!")
    print("Run 'uv run course/inference/run.py list' for module listing.")
    print("Run 'uv run course/inference/run.py' to re-run the full course.")
