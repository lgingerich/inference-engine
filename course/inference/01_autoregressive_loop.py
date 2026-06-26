"""
INFERENCE MODULE 1: THE RAW AUTOREGRESSIVE LOOP
=================================================

Module 0 told you that inference is memory-bandwidth-bound. Now you'll
experience it firsthand. We implement the SIMPLEST possible text
generator — the raw autoregressive loop — and measure exactly how bad
it is. Understanding the waste is the first step toward eliminating it.

The raw loop works like this:
    for each new token:
        run the ENTIRE model forward pass on ALL tokens so far
        sample the next token
        append it and repeat

It's correct. It's simple. And it's catastrophically inefficient.
By the end of this module, you'll know exactly WHY and by HOW MUCH.

WHAT YOU'LL LEARN:
   1. The autoregressive loop, step by step, with every tensor shape
   2. Why it's O(n²) — the hidden recomputation trap
   3. How to measure inference speed (tokens/sec, time per token)
   4. What "wasted computation" means numerically
   5. Why this first module is the UNOPTIMIZED BASELINE for the course

AFTER THIS MODULE:
   You'll be able to trace through an entire generation loop and
   identify every redundant computation your GPU is doing. You'll
   know exactly what the next module (KV cache) eliminates.
"""

import time
import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY MEASURE BEFORE OPTIMIZING?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: THE UNOPTIMIZED BASELINE — Why Start Here?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE FIRST RULE OF OPTIMIZATION: Measure, don't guess            │
└─────────────────────────────────────────────────────────────────┘

Every optimization in this course is defined by what it eliminates
from THIS loop. Before we can add caching, batching, quantization,
or speculative decoding, we need to know:

  1. Exactly what operations happen per generated token
  2. Which of those operations are REDUNDANT (computed multiple times
     with the same inputs)
  3. How the cost GROWS as the sequence gets longer
  4. What the raw performance is (tokens per second baseline)

This module is the CONTROL EXPERIMENT for the entire course.
Every module after this will claim "X% improvement" — those claims
are relative to the performance we measure RIGHT HERE.

The raw autoregressive loop is also what most people write when they
first try to generate text with a transformer. It's the intuitive
approach — and it's precisely what we need to move beyond.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE RAW GENERATION LOOP — Step by Step
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: THE RAW AUTOREGRESSIVE LOOP — LITERALLY WHAT HAPPENS")
print("=" * 70)

np.random.seed(42)
model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                num_layers=2, max_seq_len=64)

prompt = [1, 2, 3, 4, 5]  # 5 prompt tokens
max_new = 6  # generate 6 new tokens


# ═══════════════════════════════════════════════════════════════════
# 1.1  The generation loop, explained at the token level
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THE LOOP — What happens when you call generate()           │
└─────────────────────────────────────────────────────────────────┘

At its core, text generation is:

  generated = prompt_tokens
  for step in range(max_new_tokens):
      1. Run model.forward(ALL generated tokens)
         → produces logits at each position
      2. Take logits from the LAST position only
         → this predicts what comes next
      3. Convert logits to probabilities (softmax / temperature)
      4. Sample one token from this distribution
      5. Append it to "generated" and repeat

Notice step 1: we pass ALL generated tokens through the model.
This includes tokens we've already processed on PREVIOUS steps.
That's the hidden waste — and the focus of Module 2 (KV cache).
""")

print(f"\nPrompt: {prompt}")
print(f"Max new tokens: {max_new}")
print("\nGenerating, one token at a time:\n")

generated = list(prompt)

for step in range(max_new):
    # ── STEP 1: Package the entire context into a batch ──
    context = generated[-model.max_seq_len:]  # truncate if too long
    x = np.array([context])  # shape: (1, current_length)

    # ── STEP 2: Run the FULL forward pass ──
    # This is the expensive part! For step 0: processes 5 tokens.
    # For step 3: processes 8 tokens — INCLUDING recomputing
    # attention for the first 5 tokens we already did 3 times before.
    logits = model.forward(x)  # (1, current_length, vocab_size)

    # ── STEP 3: Get prediction from LAST position only ──
    # Why last position? Next-token prediction: position i predicts
    # token i+1. The final output position predicts what comes after
    # the last known token.
    next_logits = logits[0, -1, :]  # (vocab_size,)

    # ── STEP 4: Convert to probabilities and sample ──
    probs = softmax(next_logits / 0.8)  # temperature=0.8
    next_token = int(np.random.choice(model.vocab_size, p=probs))

    # ── STEP 5: Append and repeat ──
    generated.append(next_token)

    current_len = len(generated)
    print(f"  Step {step}: context_len={current_len}, "
          f"new_token={next_token}, logits_shape={logits.shape}")

print(f"\nFinal generated sequence: {generated}")
print(f"Prompt length: {len(prompt)}, Generated: {len(generated) - len(prompt)}")


# ═══════════════════════════════════════════════════════════════════
# 1.2  What model.forward() actually does inside
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  INSIDE model.forward() — What's actually being computed    │
└─────────────────────────────────────────────────────────────────┘

When you call model.forward(x) with a (1, seq_len) tensor, here's the
full computation pipeline (simplified for our 2-layer MiniGPT):

  1. EMBEDDING: token_embed[token_ids]  → (1, seq, d_model)
     + positional_embed[positions]      → (1, seq, d_model)

  2. LAYER 0:
     a. LayerNorm(x)                    → (1, seq, d_model)
     b. Q0 = x @ W_Q0  (recompute ALL seq tokens!)
        K0 = x @ W_K0  (ditto — will change next)
        V0 = x @ W_V0  (ditto)
     c. attn_scores = Q0 @ K0^T / √d_k  → (1, h, seq, seq)
        attn_weights = softmax(attn_scores + mask)
        attn_out = attn_weights @ V0    → (1, seq, d_model)
     d. x = x + attn_out @ W_O0         → residual

     e. LayerNorm(x)                    → (1, seq, d_model)
     f. hidden = GELU(x @ W1_0 + b1)   → (1, seq, 4×d_model)
        ffn_out = hidden @ W2_0 + b2   → (1, seq, d_model)
     g. x = x + ffn_out                 → residual

  3. LAYER 1: (same as Layer 0, with different weights)

  4. OUTPUT:
     a. FinalLayerNorm(x)               → (1, seq, d_model)
     b. logits = x @ lm_head            → (1, seq, vocab_size)

NOTICE: In Step 2b, Q/K/V for ALL seq_len tokens are recomputed.
This is the waste. The Key and Value vectors for a token at position
i depend ONLY on:
  - embedding[token_at_position_i]
  - the weight matrices W_K and W_V (fixed after training)

Neither of these change during generation! Yet we recompute K_i and
V_i at EVERY generation step for EVERY previous token. This is the
O(n²) trap we'll eliminate in Module 2.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: THE HIDDEN O(n²) COST — Measuring the Waste
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: THE HIDDEN O(n²) COST — HOW MUCH WORK ARE WE DOING?")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 2.1  Counting operations in attention
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  COUNTING ATTENTION OPERATIONS — The O(n²) trap            │
└─────────────────────────────────────────────────────────────────┘

The dominant cost in each forward pass is ATTENTION. For one layer,
one head, with seq_len tokens:

    Q @ K^T:  (seq_len, d_k) @ (d_k, seq_len) → O(seq_len² × d_k)
    The output has seq_len² entries, each computed from d_k products.

For a generation of the form: prompt (P tokens) + N new tokens:

  Step 0: process P tokens → P² attention ops
  Step 1: process P+1 tokens → (P+1)² attention ops
  Step 2: process P+2 tokens → (P+2)² attention ops
  ...
  Step N-1: process P+N tokens → (P+N)² attention ops

  TOTAL attention ops = Σ_{i=0}^{N-1} (P + i)²
""")

prompt_len = 5
new_tokens = 6

print(f"\nFor prompt={prompt_len} tokens, generating {new_tokens} new tokens:\n")
print(f"  Step  |  Context len  |  Attention ops (n²)  |  Cumulative")
print(f"  ──────┼───────────────┼───────────────────────┼────────────")

attn_cumulative = 0
for i in range(new_tokens):
    n = prompt_len + i
    ops = n * n
    attn_cumulative += ops
    print(f"  {i:>4}  │  {n:>12}  │  {ops:>20,}  │  {attn_cumulative:>11,}")

print()


# ═══════════════════════════════════════════════════════════════════
# 2.2  What if we didn't recompute? (KV cache preview)
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  WITH A CACHE — What the optimized version looks like       │
└─────────────────────────────────────────────────────────────────┘

If we stored K and V from previous steps (the KV Cache from Module 2),
each decode step would only compute Q for the NEW token against ALL
cached K/V tokens. The cost would be:

  Step 0 (prefill): process P tokens → P² ops (unavoidable)
  Step 1 (decode):  Q_new @ K_cached → 1 × (P+1) ops
  Step 2 (decode):  Q_new @ K_cached → 1 × (P+2) ops
  ...

  TOTAL with cache = P² + Σ_{i=0}^{N-1} (P + i)
""")

print(f"With KV cache:\n")
print(f"  Step  |  Context len  |  Attention ops       |  Cumulative")
print(f"  ──────┼───────────────┼───────────────────────┼────────────")

# Prefill: still need P² for the first pass
cache_cumulative = prompt_len * prompt_len
print(f"  prefill│  {prompt_len:>12}  │  {prompt_len**2:>20,}  │  {cache_cumulative:>11,}")

for i in range(new_tokens):
    n = prompt_len + i + 1
    ops = n  # one query against all keys
    cache_cumulative += ops
    print(f"  {i:>4}  │  {n:>12}  │  {ops:>20,}  │  {cache_cumulative:>11,}")

print(f"\n  Without cache: {attn_cumulative:,} attention ops")
print(f"  With cache:    {cache_cumulative:,} attention ops")
print(f"  Savings:       {attn_cumulative - cache_cumulative:,} ops "
      f"({100*(1-cache_cumulative/attn_cumulative):.1f}% reduction!)")


# ═══════════════════════════════════════════════════════════════════
# 2.3  Scaling up — what this means for a real conversation
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.3  SCALING UP — What this means for a real chat conversation  │
└─────────────────────────────────────────────────────────────────┘

A typical ChatGPT interaction: 100 token prompt, 500 token response.

  Without cache:
    Total = Σ_{i=0}^{499} (100 + i)²
          ≈ Σ n² from n=100 to n=599
          ≈ 599³/3 - 99³/3  (using Σn² = n(n+1)(2n+1)/6 approximation)
          ≈ 71,500,000 attention operations

  With cache:
    Total = 100² + Σ_{i=0}^{499} (100 + i)
          = 10,000 + 500×100 + 499×500/2
          ≈ 185,000 attention operations

  Reduction: 71.5M → 185K → ~386× fewer attention computations!

For longer responses, the ratio gets even better:
  - 1000 token response: ~569× reduction
  - 2000 token response: ~733× reduction

This is NOT a minor optimization. Without caching, a 500-token
response does ~386× more work than necessary. With a 7B model,
that's the difference between 3 tokens/second (unusable) and
30 tokens/second (perfectly fine).

This is why the KV Cache (Module 2) is the SINGLE most important
optimization in LLM inference. Everything else in this course
builds ON TOP of it.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: MEASURING ACTUAL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 3: MEASURING INFERENCE SPEED — Tokens Per Second")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 3.1  Benchmarking the raw loop
# ═══════════════════════════════════════════════════════════════════

prompt_long = [1, 2, 3, 4, 5]
num_new = 25
num_trials = 5

print(f"\nBenchmark: {num_trials} runs of {num_new} token generation")
print(f"Model: MiniGPT (vocab=100, d_model=32, num_layers=2)")
print(f"Note: Our model is tiny (< 1ms per forward pass). Real models")
print(f"      are 1000× slower, making the measurements proportionally")
print(f"      meaningful. The TREND (longer = slower) is what matters.\n")

times = []
for trial in range(num_trials):
    start = time.perf_counter()
    result = model.generate(prompt_long, max_new_tokens=num_new, temperature=0.8)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

avg_time = np.mean(times)
tokens_per_sec = num_new / avg_time

print(f"  Raw times: {[f'{t:.4f}s' for t in times]}")
print(f"  Average:   {avg_time:.4f}s")
print(f"  Tokens/sec: {tokens_per_sec:.1f}")
print(f"  Time per token: {avg_time / num_new * 1000:.2f} ms")


# ═══════════════════════════════════════════════════════════════════
# 3.2  How generation time grows with sequence length
# ═══════════════════════════════════════════════════════════════════

print(f"\nHow generation time grows with context length:")
print(f"  (Each forward pass gets more expensive as context grows)\n")
print(f"  Prompt len  |  Time for 10 tokens  |  Cost-per-token trend")
print(f"  ────────────┼──────────────────────┼──────────────────────")

prev_time = None
for prompt_len_test in [5, 15, 30, 50]:
    prompt_test = list(range(prompt_len_test))
    start = time.perf_counter()
    model.generate(prompt_test, max_new_tokens=10, temperature=0.8)
    elapsed = time.perf_counter() - start
    per_token = elapsed / 10 * 1000

    trend = ""
    if prev_time is not None:
        ratio = elapsed / prev_time
        trend = f"{ratio:.1f}× longer"
    else:
        trend = "baseline"
    prev_time = elapsed

    print(f"  {prompt_len_test:>10}  |  {elapsed:.4f}s ({per_token:.2f}ms/tok)  |  {trend}")

print(f"\n  → As context doubles, generation roughly doubles in cost")
print(f"  → This is the O(n²) cost growing in the attention computation")
print(f"  → The KV cache (Module 2) makes this cost nearly CONSTANT!")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: WHY THIS DOESN'T SCALE — The Formal Math
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: THE PRECISE COST OF RECOMPUTATION — The Math")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 4.1  The formal complexity analysis
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  FORMAL COMPLEXITY — Proving O(n²) vs O(n)                  │
└─────────────────────────────────────────────────────────────────┘

For a single attention layer with d_k dimensions:

WITHOUT CACHE (our current loop):
  Generation step i (context = P+i tokens):
    - Q, K, V projection: (P+i) × d_k² operations (per matrix)
    - Attention scores: (P+i)² × d_k operations
    - Weighted sum: (P+i)² × d_k operations

  Total over N steps: Θ(N × P² × d_k) for QKV + Θ(N³ × d_k) for attention
  → Dominant term when N >> P: O(N³ × d_k)
  → Quadratic in CONTEXT, cubic in GENERATION LENGTH!

WITH KV CACHE:
  Prefill: P² × d_k (once)
  Decode step i:
    - Q, K, V for ONE token: 1 × d_k² operations
    - Attention: 1 × (P+i) × d_k operations
    - Weighted sum: 1 × (P+i) × d_k operations

  Total over N steps: Θ(N × d_k²) for QKV + Θ(P×N + N²) × d_k for attention
  → Dominant term: O(N × d_k² + N × P × d_k)
  → LINEAR in both context AND generation length!

COMPARISON for N >> P:
  Without cache: O(N³)
  With cache:    O(N)
  Ratio: N³/N = N²

For N=500 (a typical chat response): 250,000× fewer attention
computations with caching. The math is unambiguous.
""")


# ═══════════════════════════════════════════════════════════════════
# 4.2  Visualizing the waste — Token by token trace
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.2  TOKEN-BY-TOKEN TRACE — What's being recomputed and when    │
└─────────────────────────────────────────────────────────────────┘
""")

prompt_trace = [10, 20, 30]  # 3 prompt tokens
print(f"Trace for prompt={prompt_trace}, generating 3 new tokens:\n")

for step in range(3):
    # Simulate growing context
    sim_tokens = prompt_trace + [40 + step] if step > 0 else prompt_trace
    current_len = len(prompt_trace) + step

    print(f"  ── Generation step {step} (generating token at position {current_len}) ──")

    if step == 0:
        print(f"    Context: {prompt_trace}")
        print(f"    Compute Q,K,V for positions 0,1,2 (prompt)")
        print(f"      ← First time for prompt tokens. Necessary.")
        print(f"    Attention: 3 queries × 3 keys = 9 dot products")
        print(f"    Output: token for position 3")
    elif step == 1:
        print(f"    Context: {prompt_trace + [40]}")
        print(f"    Compute Q,K,V for positions 0,1,2 ← AGAIN! (waste!)")
        print(f"    Compute Q,K,V for position 3      ← First time. Necessary.")
        print(f"    Attention: 4 queries × 4 keys = 16 dot products")
        print(f"    Output: token for position 4")
        print(f"    Waste: recomputed K,V for positions 0,1,2 (3 of 4 tokens)")
    else:
        print(f"    Context: {prompt_trace + [40, 41]}")
        print(f"    Compute Q,K,V for positions 0,1,2 ← THIRD TIME! (waste!)")
        print(f"    Compute Q,K,V for positions 3,4   ← Second time! (waste!)")
        print(f"    Attention: 5 queries × 5 keys = 25 dot products")
        print(f"    Output: token for position 5")
        print(f"    Waste: recomputed K,V for positions 0-4 (5 of 5 tokens, minus the new one)")

    print()

total_work = 9 + 16 + 25  # 3² + 4² + 5²
necessary_prefill = 9  # the initial prompt processing
necessary_decode = (3+1) + (3+2)  # one Q against K_cached per step
necessary_total = necessary_prefill + necessary_decode

print(f"  Total attention work:    {total_work} dot products")
print(f"  Necessary work if cached: {necessary_total} dot products")
print(f"  Wasted:                  {total_work - necessary_total} "
      f"({100*(1-necessary_total/total_work):.0f}% wasted!)")

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ THE WASTE COMPOUNDS                                            │
│                                                                 │
│ For a 500-token response from a REAL model (7B params):        │
│                                                                 │
│   Without cache: each forward pass loads 14 GB of weights.     │
│    500 passes × 14 GB = 7 TB of data moved through HBM.        │
│    At 2 TB/s: ~3.5 seconds just for weight loading.            │
│                                                                 │
│   Plus: >99% of attention computation is redundant.             │
│   K and V for the FIRST prompt token are recomputed 500 times.  │
│   Each recomputation loads W_K and W_V from HBM.                │
│                                                                 │
│   With cache: weights loaded ONCE per layer.                    │
│    Only the NEW token's K and V are computed each step.        │
│    Total weight loading drops from 7 TB to ~14 GB.             │
│    → 500× reduction in weight traffic.                          │
│                                                                 │
│ This is why you CANNOT serve LLMs without KV caching.           │
└─────────────────────────────────────────────────────────────────┘
""")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE RAW AUTOREGRESSIVE LOOP — Why we need to move beyond it    │
│                                                                 │
│ 1. The loop: forward pass on ALL tokens, sample one, append,   │
│    repeat. Correct but catastrophic.                            │
│                                                                 │
│ 2. Each step recomputes K and V for ALL previous tokens.       │
│    K_i depends only on embedding[token_i] and W_K — both fixed. │
│    → K_i is identical every time we compute it.                 │
│    → Computing it more than once is pure waste.                │
│                                                                 │
│ 3. Attention cost without cache: O(N³) for N generated tokens.  │
│    Attention cost with cache:    O(N²) for N generated tokens.  │
│    → For N=500, this is a ~500× reduction.                      │
│                                                                 │
│ 4. In a real model, recomputation means:                       │
│    - Loading 14 GB of weights 500 times instead of once         │
│    - ~7 TB of HBM traffic for a single chat response           │
│    - ~99% of GPU time spent waiting for redundant data          │
│                                                                 │
│ 5. This module is the BASELINE. Every optimization in the       │
│    remaining modules makes this loop faster by eliminating      │
│    waste.                                                       │
└─────────────────────────────────────────────────────────────────┘

Pipeline at this stage:
    prompt → model.forward(ALL tokens) → sample → append → repeat
    Each step processes ALL previous tokens → O(n³) total cost
    This is correct, but 99% of the work is redundant.

Next: Module 2 — The KV Cache, the most important inference
optimization. We'll eliminate ALL the redundant K/V computation.
""")

if __name__ == "__main__":
    print("\nModule 1 complete! Next: i02_kv_cache.py")
    print("Run with: uv run python course/inference/i02_kv_cache.py")
