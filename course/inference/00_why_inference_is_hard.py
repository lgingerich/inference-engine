"""
INFERENCE MODULE 0: WHY INFERENCE IS HARD
==========================================

You've built a transformer and (conceptually) trained it. Now you want to
serve it to users. How hard can it be? Just call model.generate(), right?

Wrong. Inference is a fundamentally different engineering discipline from
training. The naive approach is catastrophically slow, memory-hungry, and
expensive. This module explains WHY — because understanding the physics of
the problem is essential before we can optimize it.

WHAT YOU'LL LEARN:
   1. Why training and inference have OPPOSITE performance bottlenecks
   2. Prefill vs decode: the two phases of every request
   3. GPU memory hierarchy: why bandwidth matters more than FLOPS
   4. The roofline model: how to predict if you're compute or memory bound
   5. The core metrics: TTFT, TPOT, throughput, latency

AFTER THIS MODULE:
   You'll understand exactly what constraints every inference optimization
   in the rest of this course is working against — and why "just use a
   bigger GPU" doesn't solve the fundamental problem.
"""

import time
import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY INFERENCE IS A SEPARATE DISCIPLINE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY INFERENCE DESERVES ITS OWN COURSE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING ≠ INFERENCE — They're opposites, not cousins           │
└─────────────────────────────────────────────────────────────────┘

Training a transformer is ONE giant problem:
  "Given 500 billion tokens, minimize the cross-entropy loss."
  → One forward+backward pass processes an ENTIRE batch at once.
  → The GPU is always busy — it's compute-bound (FLOPS are the limit).
  → Everything is batched, fused, and optimized for throughput.

Inference is THOUSANDS of tiny problems:
  "User A wants 50 tokens about quantum physics."
  "User B wants 5 tokens about pizza recipes."
  → Each user gets one token at a time.
  → The GPU spends 99% of its time waiting for weights to load from
    memory — it's memory-bandwidth-bound.
  → Latency matters as much as throughput.

┌─────────────────────────────────────────────────────────────────┐
│ WHY YOU CAN'T JUST USE TRAINING CODE FOR INFERENCE              │
└─────────────────────────────────────────────────────────────────┘

If you took the training code from Module 8 and ran it to generate
text, it would:
  1. Process the ENTIRE batch in one shot (not autoregressively)
  2. Compute forward AND backward passes (unnecessary gradients)
  3. Store optimizer states (2× the model size in memory!)

Training code is optimized for a DIFFERENT optimization target.
This course is about making the FORWARD pass alone as fast as
possible under the sequential constraint of autoregression.

Think of it as the difference between:
  - Building a factory (training): optimize for MAXIMUM OUTPUT
  - Running a restaurant (inference): optimize for PER-CUSTOMER SPEED
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: TRAINING vs INFERENCE — The Fundamental Difference
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: TRAINING VS INFERENCE — WHY THEY'RE OPPOSITES")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 1.1  The batch size difference
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THE BATCH SIZE DIFFERENCE — The root of everything        │
└─────────────────────────────────────────────────────────────────┘

Training: processes MANY tokens simultaneously
  → batch_size × seq_len tokens in one forward pass
  → Typical: 64 sequences × 2048 tokens = 131,072 tokens
  → The matmul is HUGE: (131K, 4096) @ (4096, 4096)
  → GPU utilization: 90%+ (lots of work per weight load)

Inference (decode phase): processes ONE token per request
  → 1 × 1 × d_model dimensions
  → The matmul is TINY: (1, 4096) @ (4096, 4096)
  → GPU utilization: <1% (barely any work per weight load)
  → You still have to LOAD all 14 GB of weights for that one token!

     Training batch              Inference (single token decode)
  ┌─────────────────────┐    ┌─────────────────────┐
  │ ████████████████████ │    │ ░░░░░░░░░░░░░░░░░░░ │
  │ ████████████████████ │    │ ░░░░░░░░░░░░░░░░░░░ │  ░ = idle GPU cores
  │ ████████████████████ │    │ ░░░░░░█░░░░░░░░░░░░ │  █ = active GPU cores
  │ ████████████████████ │    │ ░░░░░░░░░░░░░░░░░░░ │
  │ ████████████████████ │    │ ░░░░░░░░░░░░░░░░░░░ │
  └─────────────────────┘    └─────────────────────┘
   All cores compute         Weight loading dominates
   FLOPS-bound               MEMORY-BANDWIDTH-bound
""")

# Numerical demonstration
print("Concrete numbers — LLaMA-7B on an A100 GPU (312 TFLOPS, 2 TB/s HBM):\n")

print("TRAINING (batch=64, seq_len=2048, d_model=4096):")
matmul_flops = 2 * 64 * 2048 * 4096 * 4096
print(f"  Matmul FLOPS: {matmul_flops:.2e} ≈ {matmul_flops/1e12:.1f} TFLOPs")
print(f"  Time if compute-bound: {matmul_flops/312e12*1000:.1f} ms")
print(f"  GPU utilization: ~90%+ (most cores are computing)")
print()

print("INFERENCE DECODE (batch=1, seq_len=1, d_model=4096):")
decode_flops = 2 * 1 * 1 * 4096 * 4096
bytes_to_load = 4096 * 4096 * 2  # FP16
print(f"  Matmul FLOPS: {decode_flops:.2e} ≈ {decode_flops/1e6:.0f} MFLOPs (tiny!)")
print(f"  Bytes to load (one weight matrix): {bytes_to_load:,} bytes ≈ {bytes_to_load/1e6:.1f} MB")
print(f"  Time if compute-bound: {decode_flops/312e12*1e6:.2f} μs (instant)")
print(f"  Time if memory-bound:  {bytes_to_load/2e12*1e9:.0f} ns ({bytes_to_load/2e12*1e6:.1f} μs)")
print(f"  → Compute is {bytes_to_load/2e12*1e9 / (decode_flops/312e12*1e6):.0f}× faster than data loading!")
print(f"  → GPU utilization: <1% (99% of time waiting for weights from HBM)")


# ═══════════════════════════════════════════════════════════════════
# 1.2  The roofline model — A formal way to think about bottlenecks
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  THE ROOFLINE MODEL — Is your operation compute or memory bound?│
└─────────────────────────────────────────────────────────────────┘

Every operation on a GPU has an "arithmetic intensity" — the ratio
of FLOPS to bytes of data moved:

  arithmetic_intensity = FLOPs / bytes_moved

The roofline model says:
  - If intensity < GPU's FLOPS/bandwidth ratio → MEMORY-BOUND
  - If intensity > GPU's FLOPS/bandwidth ratio → COMPUTE-BOUND

For an A100: ratio = 312 TFLOPS / 2 TB/s = 156 FLOPs/byte

A matmul (b×m×k) @ (k×n) has:
  FLOPS = 2 × b × m × k × n
  Bytes = (m×k + k×n + m×n) × bytes_per_element (if no reuse)
  Intensity = FLOPS / Bytes

Training matmul (b=64, m=2048, k=4096, n=4096):
  FLOPS = 2 × 64 × 2048 × 4096 × 4096 ≈ 4.4 × 10^12
  Bytes ≈ (2048×4096 + 4096×4096 + 2048×4096) × 2 ≈ 67 MB
  Intensity = 4.4e12 / 67e6 ≈ 65,000 FLOPs/byte
  → FAR above 156 → COMPUTE-BOUND (good! GPU is working hard)

Inference decode matmul (b=1, m=1, k=4096, n=4096):
  FLOPS = 2 × 1 × 1 × 4096 × 4096 ≈ 33.5 × 10^6
  Bytes ≈ (1×4096 + 4096×4096 + 1×4096) × 2 ≈ 33.5 MB
  Intensity = 33.5e6 / 33.5e6 = 1.0 FLOP/byte
  → FAR below 156 → MEMORY-BOUND (GPU is waiting for data!)

This is NOT an implementation issue. It's a FUNDAMENTAL property of
processing one token at a time. The only way to increase intensity
is to batch MORE tokens together — which is what continuous batching
(Module 3) does.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: PREFILL vs DECODE — Two Phases, Two Bottlenecks
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: PREFILL VS DECODE — THE TWO PHASES OF EVERY REQUEST")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 2.1  Why there are two phases
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY TWO PHASES? — Because attention needs history          │
└─────────────────────────────────────────────────────────────────┘

Every autoregressive generation has two distinct phases because
attention is a two-step process:

  1. BUILD THE CONTEXT: The model needs to "understand" the prompt
     before it can generate. This is like reading a question before
     answering it — you process ALL the words at once.

  2. GENERATE: Then you answer one word at a time, but each new word
     must "see" all previous words (prompt + generated so far).

  ┌──────────────────────────────────────────────────┐
  │  User sends: "Explain quantum entanglement"       │
  │              [token_ids: 4 tokens]                │
  └────────────────────┬─────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  PHASE 1:       │
              │  PREFILL        │
              │  ─────────────  │
              │  "Encode the    │
              │   prompt"       │
              │                 │
              │  • Process ALL  │
              │    prompt tokens│  ← Compute-bound
              │    in one pass  │     (batch all
              │  • Build KV     │      prompt tokens)
              │    cache for    │
              │    ALL tokens   │  ← Creates the cache
              │                 │     for reuse
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  PHASE 2:       │
              │  DECODE         │
              │  ─────────────  │
              │  "Generate one  │
              │   token at a    │
              │   time"          │
              │                 │
              │  • Process ONE  │  ← Memory-bound
              │    new token    │     (tiny matmul,
              │  • Use KV cache │      huge weight loads)
              │    for history  │
              │  • Repeat until │
              │    EOS / limit  │  ← Sequential!
              └─────────────────┘

KEY INSIGHT: These phases have COMPLETELY different hardware demands.
A good inference engine treats them DIFFERENTLY — it batches prefills
aggressively (they're compute-bound, like training) but handles decodes
delicately (they're memory-bound, and latency matters).
""")


# ═══════════════════════════════════════════════════════════════════
# 2.2  Why you can't parallelize the decode phase
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  THE SEQUENTIAL CONSTRAINT — Why decode can't be parallel  │
└─────────────────────────────────────────────────────────────────┘

Token N+1 depends on token N. This is a SERIAL DEPENDENCY:

  token_1 = f(prompt)
  token_2 = f(prompt, token_1)      ← depends on token_1
  token_3 = f(prompt, token_1, token_2)  ← depends on token_1 AND token_2

You CANNOT compute token_3 before token_2 because:
  - token_3's embedding depends on knowing token_2
  - The attention at position 3 must see token_2's K and V
  - The sampling decision for position 3 depends on position 2's output

This is not a GPU limitation — it's a mathematical dependency. No
amount of hardware, parallelism, or clever engineering can break it.

What we CAN do:
  ✓ Make each individual decode step FASTER (KV cache, quantization)
  ✓ Overlap MULTIPLE requests (batching — Module 3)
  ✓ Guess ahead and verify (speculative decoding — Module 6)

What we CANNOT do:
  ✗ Generate multiple tokens in parallel for a single request
  ✗ Skip the sequential dependency chain

This constraint is the ENTIRE REASON the inference engineering
discipline exists. If decode were parallelizable, we'd just run
everything in one big batch and be done.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: GPU MEMORY HIERARCHY — The Physical Bottleneck
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 3: GPU MEMORY HIERARCHY — THE PHYSICAL BOTTLENECK")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 3.1  The two-level memory system
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  THE GPU'S TWO MEMORIES — Why latency ≠ bandwidth           │
└─────────────────────────────────────────────────────────────────┘

Modern GPUs have a two-level memory hierarchy designed for throughput:

  ┌──────────────────────────────────────────────────┐
  │  HBM (High-Bandwidth Memory) — GPU "RAM"         │
  │  • Size: 40–80 GB (A100) or 24 GB (RTX 4090)    │
  │  • Bandwidth: 1.5–2.0 TB/s (yes, TERABYTES/sec) │
  │  • WHERE MODEL WEIGHTS LIVE                     │
  │  • Every weight load during inference comes      │
  │    from here — and you must load ALL weights for │
  │    EVERY token (the roofline bottleneck!)        │
  │  • Latency: ~hundreds of nanoseconds per access  │
  └────────────────────┬─────────────────────────────┘
                       │  ~2 TB/s bandwidth
  ┌────────────────────▼─────────────────────────────┐
  │  SRAM (on-chip shared memory) — GPU "L1 cache"   │
  │  • Size: ~20 MB per SM (tiny!)                  │
  │  • Bandwidth: ~20 TB/s (10× faster than HBM)     │
  │  • WHERE COMPUTATION HAPPENS                    │
  │  • Too small to hold model weights, but you can │
  │    TILE computation to use it effectively        │
  │  • FlashAttention (Module 5) exploits this       │
  └──────────────────────────────────────────────────┘

WHY THIS MATTERS FOR INFERENCE:
  Every single matmul in your model must:
    1. Load weights from HBM → SRAM (the SLOW part)
    2. Load inputs from HBM → SRAM (small, so fast)
    3. Compute in SRAM (blazing fast)
    4. Write outputs from SRAM → HBM (small, so fast)

  Step 1 is your bottleneck. For a (1, 4096) @ (4096, 4096) matmul:
    - Step 1: load ~33.5 MB of weights → ~17 μs at 2 TB/s
    - Step 3: compute 33.5M FLOPS → ~0.1 μs at 312 TFLOPS
    - The GPU spends ~17 μs waiting, 0.1 μs computing → 99.4% idle!

  This is the SINGLE MOST IMPORTANT FACT about LLM inference.
  Every optimization in this course is either:
    a) Reduce data moved (KV cache in Module 2, quantization in Module 4)
    b) Increase effective bandwidth (FlashAttention in Module 5)
    c) Reuse data across requests (batching in Module 3)
    d) Skip computation entirely (speculative decoding in Module 6)
""")


# ═══════════════════════════════════════════════════════════════════
# 3.2  Why GPU memory bandwidth hasn't kept up with compute
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.2  THE MEMORY WALL — Why bandwidth lags behind                │
└─────────────────────────────────────────────────────────────────┘

GPU compute has improved faster than memory bandwidth:

  Year   GPU       TFLOPS (FP16)    HBM Bandwidth    Compute/BW ratio
  ─────  ────────  ──────────────   ──────────────   ────────────────
  2017   V100      125              900 GB/s         139 FLOPs/byte
  2020   A100      312              2.0 TB/s         156 FLOPs/byte
  2022   H100      990              3.35 TB/s        296 FLOPs/byte
  2024   B200      2250             8.0 TB/s         281 FLOPs/byte

The Compute/BW ratio has DOUBLED since 2017. This means:
  - Newer GPUs are MORE bottlenecked by memory bandwidth!
  - Inference (memory-bound) benefits LESS from newer GPUs than
    training (compute-bound) does.
  - The gap between "what the GPU CAN compute" and "what the GPU
    CAN access per second" is WIDENING.

This is why inference optimization is an increasingly important
field, and why techniques that reduce memory traffic (quantization,
smaller KV caches, kernel fusion) are MORE valuable with each
GPU generation, not less.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: KEY METRICS — How We Judge Inference Performance
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: THE FOUR METRICS THAT MATTER")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 4.1  The four critical metrics
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE FOUR CRITICAL METRICS                                 │
└─────────────────────────────────────────────────────────────────┘

1. TTFT (Time-To-First-Token) — "How long until I see the first word?"
   This is the PREFILL time + first decode step.
   Users perceive this as "responsiveness." A 2-second TTFT feels
   slow; 200ms feels instant.
   Target: < 200ms for chat, < 500ms acceptable.

   WHY IT MATTERS: If TTFT is slow, users think "is it broken?"
   and abandon the request. This is the most important UX metric.

2. TPOT (Time-Per-Output-Token) — "How fast do new tokens appear?"
   This is the average DECODE step time. Determines "reading speed."
   Target: < 50ms per token (20+ tokens/sec) for chat.
   At 20 tok/s, a 100-token response takes 5 seconds of generation.

   WHY IT MATTERS: Users read the response as it streams. If tokens
   appear slower than reading speed (~5-10 tok/s for humans), the
   experience feels laggy, like a slow typist.

3. THROUGHPUT — "How many requests can I serve per second?"
   = total tokens generated / wall-clock time.
   This determines COST — higher throughput = fewer GPUs needed.
   Target: as high as possible, but not at the expense of latency.

   WHY IT MATTERS: At $2-4/GPU-hour for cloud A100s, doubling
   throughput halves your serving cost.

4. LATENCY (P50, P95, P99) — "What's the worst-case experience?"
   P95 = "95% of requests finish within X seconds."
   The AVERAGE can be fast while some users wait forever.
   Users remember the slow requests, not the average speed.

   WHY IT MATTERS: A single slow request in a batch of 100 can ruin
   ONE user's experience. P99 latency directly correlates with
   user complaints.
""")


# ═══════════════════════════════════════════════════════════════════
# 4.2  The fundamental tradeoff
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.2  THE FUNDAMENTAL TRADEOFF — Throughput vs Latency          │
└─────────────────────────────────────────────────────────────────┘

You cannot simultaneously maximize throughput and minimize latency
for individual requests. This is the central tension of inference
serving:

  HIGH THROUGHPUT (large batch):
    + GPU is busy → more tokens per dollar
    + Better hardware utilization
    - Each request waits longer (other requests in same batch)
    - Higher TTFT and TPOT

  LOW LATENCY (small batch):
    + Each request gets fast responses
    + Good for interactive chat
    - GPU is mostly idle → fewer tokens per dollar
    - Lower throughput

Continuous batching (Module 3) partially resolves this by letting
requests join and leave the batch fluidly, rather than waiting
for the slowest batchmate. But the tradeoff exists even with
continuous batching — a GPU can only do so much work per second,
and that work must be divided among all active requests.

                     Throughput
                         ↑
                    ┌────┼────┐
                    │    │    │  ← Large batch: high throughput,
                    │    │    │     but each request waits
                    │  ╱ ╲  │
                    │ ╱   ╲ │  ← The "sweet spot" depends on
                    │╱     ╲│     your use case
                    │       │
                    └───────┘──→ Latency

  Chat applications:      prioritize latency (smaller batches)
  Batch processing (API): prioritize throughput (larger batches)
  Mixed workloads:        continuous batching to balance both
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: SETTING UP OUR TEST MODEL
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: OUR TEST MODEL — MiniGPT Ready for Optimization")
print("=" * 70)

np.random.seed(42)

# Create the same tiny model we'll use throughout the inference course
model = MiniGPT(
    vocab_size=100,
    d_model=32,
    num_heads=4,
    num_layers=2,
    max_seq_len=64,
    ffn_expansion=4,
)

# Count total parameters for reference
n_params = 0
n_params += model.vocab_size * model.d_model  # token embed
n_params += model.max_seq_len * model.d_model  # pos embed
for _block in model.blocks:
    n_params += 4 * model.d_model * model.d_model  # attention projections
    n_params += 2 * model.d_model * model.d_model * 4  # FFN (expansion=4)
n_params += model.d_model * model.vocab_size  # lm_head

print(f"\nMiniGPT parameters:")
print(f"  Vocab: {model.vocab_size}, d_model: {model.d_model}")
print(f"  Heads: {model.blocks[0].attention.num_heads}, "
      f"Layers: {model.num_layers}")
print(f"  Max seq len: {model.max_seq_len}")
print(f"  Total parameters: ~{n_params:,}")
print(f"  Memory (FP64): ~{n_params * 8 / 1024:.0f} KB")

# Quick generation test to establish baseline
print(f"\nQuick generation test (prompt=[1,2,3], 10 tokens):")
start = time.perf_counter()
result = model.generate([1, 2, 3], max_new_tokens=10, temperature=0.8)
elapsed = time.perf_counter() - start

tokens_generated = len(result) - 3  # subtract prompt length
print(f"  Generated: {result}")
print(f"  New tokens: {tokens_generated}")
print(f"  Time: {elapsed:.4f}s")
print(f"  Tokens/sec: {tokens_generated / elapsed:.1f}")
print(f"  (untrained model → random tokens, this is expected)")

# Show the speed difference vs a real model
print(f"\n  Our tiny model: ~{tokens_generated / elapsed:.0f} tok/s on CPU")
print(f"  LLaMA-7B:       ~20-50 tok/s on A100 GPU")
print(f"  → Our model is {tokens_generated / elapsed / 30 * 100:.0f}x faster because it's")
print(f"    {7000000000/n_params:.0f}x smaller. The principles scale identically.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 6: WHAT THIS MEANS FOR A REAL MODEL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 6: SCALING UP — What Changes at Production Scale")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ FROM 23K TO 7B PARAMETERS — The exponential cost curve         │
└─────────────────────────────────────────────────────────────────┘

Our MiniGPT runs instantly on CPU because it's tiny. But scale up:

  Metric            MiniGPT (23K)    LLaMA-7B (7B)     Factor
  ───────────────── ───────────────  ────────────────   ─────────
  Model memory       ~92 KB          14 GB (FP16)       160,000×
  KV cache (4K ctx)  ~8 KB           2 GB               250,000×
  Tokens/sec         >10,000         20-50              0.005×
  Cost to serve      $0              $2-4/GPU-hr        ∞

The PHYSICS are the same. A matmul is a matmul. The memory hierarchy
is identical. The bottleneck (loading weights from HBM) is the same.

What changes is the SCALE:
  - A 7B model loads 14 GB of weights per forward pass.
  - At 2 TB/s HBM bandwidth, that's 7 ms just for weight loading.
  - Even with zero compute time, 7 ms per token = 143 tok/s max.
  - With batching, overhead, and KV cache, real-world is 20-50 tok/s.

Every optimization in this course multiplies against that 7 ms base.
KV cache doubles effective speed. Quantization halves weight loading.
Speculative decoding triples tokens per step. Together, they can take
you from 20 tok/s to 200+ tok/s.

READY? → Module 1: The Raw Autoregressive Loop
""")


print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE PHYSICS OF LLM INFERENCE                                    │
│                                                                 │
│ 1. TRAINING = FLOPS-bound (big batches, compute-heavy)         │
│    INFERENCE = MEMORY-BANDWIDTH-bound (tiny ops, data-heavy)   │
│                                                                 │
│ 2. PREFILL = process prompt at once (compute-bound)            │
│    DECODE = generate one token at a time (memory-bound)        │
│                                                                 │
│ 3. GPU HBM (big, slow) holds weights;                           │
│    GPU SRAM (tiny, fast) does computation.                      │
│    Weights must cross HBM→SRAM for every token.                │
│                                                                 │
│ 4. ROOFLINE: decode has ~1 FLOP/byte intensity.                 │
│    A100 needs >156 FLOPs/byte for compute-bound.                │
│    → Decode is 156× below the compute-bound threshold.         │
│                                                                 │
│ 5. KEY METRICS: TTFT (first token), TPOT (per token),          │
│    throughput (total output), latency (P95/P99).                │
│                                                                 │
│ 6. FUNDAMENTAL TRADEOFF: throughput vs latency.                 │
│    Your use case determines where you sit on the curve.         │
│                                                                 │
│ 7. The serial dependency of decode CANNOT be parallelized.      │
│    All optimizations work within this constraint.               │
└─────────────────────────────────────────────────────────────────┘

You now understand WHY inference is hard. Next, we'll see just how
bad the naive approach is — and start fixing it.
""")

if __name__ == "__main__":
    print("\nModule 0 complete! Next: i01_autoregressive_loop.py")
    print("Run with: uv run python course/inference/i01_autoregressive_loop.py")
