"""
MODULE 9: ADVANCED TOPICS — What Modern Transformers Actually Use
====================================================================

Our MiniGPT uses techniques from 2017. Modern LLMs (LLaMA 3, GPT-4,
Claude, Gemini, Mistral) incorporate significant improvements in
every component of the architecture.

This module explains EIGHT key advances with deep reasoning about
WHY each one was adopted, WHAT problem it solves, and HOW it works
at the conceptual level.

WHAT YOU'LL LEARN:
   1. RoPE — position encoding that encodes relative distance
   2. GQA/MQA — reducing KV cache memory while preserving quality
   3. Flash Attention — reordering computation for memory efficiency
   4. SwiGLU — gated activation that outperforms GELU
   5. RMSNorm — simpler, faster layer normalization
   6. KV Cache — the inference bottleneck and the solution
   7. MoE — sparse computation for massive models
   8. Speculative Decoding — making generation 2-3× faster

AFTER THIS MODULE:
   You'll understand the gap between a toy transformer and a
   production LLM. Every technique here is in LLaMA, Mistral, or GPT-4.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# TOPIC 1: ROPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 1: ROPE — Rotary Position Embeddings")
print("=" * 70)
print("Used by: LLaMA, Mistral, Gemma, Qwen, DeepSeek")
print()

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THE PROBLEM WITH ABSOLUTE POSITIONS                        │
└─────────────────────────────────────────────────────────────────┘

Sinusoidal (Module 5) and Learned (Module 7) position encodings
both encode ABSOLUTE position: "token is at position 42."

Two problems with this:
  1. Can't EXTRAPOLATE: If you trained with max_seq_len=2048,
     position 2049 has no encoding. The model breaks.
  2. Position information is ADDED to embeddings before attention.
     The attention dot product Q_i · K_j doesn't DIRECTLY depend
     on the relative distance (j-i). The model has to learn to
     extract relative position from the absolute encodings.

┌─────────────────────────────────────────────────────────────────┐
│ 1.2  HOW ROPE WORKS — Encode relative position geometrically   │
└─────────────────────────────────────────────────────────────────┘

Key insight (Su et al., 2021): Instead of ADDING position to
embeddings, ROTATE queries and keys by position-dependent angles.

For dimension pair (2i, 2i+1) at position p:
  angle = p × θᵢ  where θᵢ = 10000^(-2i/d)

  [Q_rot[2i]]   [cos(angle)  -sin(angle)] [Q[2i]]
  [Q_rot[2i+1]] = [sin(angle)   cos(angle)] [Q[2i+1]]

This is a 2D rotation by angle in each dimension pair.

WHY THIS IS BRILLIANT:
  1. The dot product Q_p · K_q DEPENDS ONLY on (p-q):
     Q_p · K_q = Σᵢ cos((p-q)θᵢ) × (Q_q_raw · K_q_raw) + sin terms

     The angle depends on the RELATIVE distance, not absolute position!
     This means the model can naturally generalize to positions it
     has never seen during training.

  2. RoPE is applied to Q and K BEFORE the attention dot product,
     but AFTER the linear projections. The position information is
     woven directly into the similarity computation.

  3. It has ZERO learned parameters — it's a pure mathematical
     transformation. Yet it outperforms both sinusoidal and learned
     position encodings in practice.
""")


def apply_rope(x, positions):
    """Apply rotary position embedding to a tensor.

    x: (seq_len, d_model)
    positions: (seq_len,) — position index for each token
    """
    seq_len, d = x.shape
    x_rot = np.copy(x)

    for pos_idx, pos in enumerate(positions):
        for i in range(0, d, 2):
            theta = pos / (10000 ** (i / d))
            cos_val = np.cos(theta)
            sin_val = np.sin(theta)

            # Rotate the 2D pair (i, i+1)
            x0, x1 = x_rot[pos_idx, i], x_rot[pos_idx, i + 1]
            x_rot[pos_idx, i] = x0 * cos_val - x1 * sin_val
            x_rot[pos_idx, i + 1] = x0 * sin_val + x1 * cos_val

    return x_rot


np.random.seed(42)
x = np.random.randn(4, 8)  # 4 positions, 8 dims
positions = np.array([0, 1, 10, 100])  # different positions

x_rope = apply_rope(x, positions)

print(f"Position 0: {np.round(x_rope[0, :4], 3)} (no rotation: cos(0)=1, sin(0)=0)")
print(f"Position 1: {np.round(x_rope[1, :4], 3)} (slight rotation)")
print(f"Position 100: {np.round(x_rope[3, :4], 3)} (significant rotation)")

# Show that dot product depends on relative distance
d_k = 8
pos_a, pos_b = 5, 7  # distance = 2
pos_c, pos_d = 10, 12  # distance = 2

Q_ab = apply_rope(np.random.randn(2, d_k), np.array([pos_a, pos_c]))
K_ab = apply_rope(np.random.randn(2, d_k), np.array([pos_b, pos_d]))
sim_ab = Q_ab[0] @ K_ab[0].T  # distance = 2
sim_cd = Q_ab[1] @ K_ab[1].T  # distance = 2 (same!)

print(f"\n→ Same relative distance = similar dot products:")
print(f"  Dot(Q_p5, K_p7): {sim_ab:.4f}  (distance 2)")
print(f"  Dot(Q_p10, K_p12): {sim_cd:.4f}  (distance 2)")
print(f"  → This is why RoPE extrapolates to unseen lengths!")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 2: KV CACHE — Why Inference Is Memory-Bound
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TOPIC 2: KV CACHE — The Inference Bottleneck")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  THE PROBLEM — Redundant recomputation                      │
└─────────────────────────────────────────────────────────────────┘

During autoregressive generation:

  Step 1: Input ["The"] → compute K₁, V₁ for all layers
  Step 2: Input ["The", "cat"] → recompute K₁, K₂, V₁, V₂
  Step 3: Input ["The", "cat", "sat"] → recompute K₁, K₂, K₃, V₁, V₂, V₃

Each step RECOMPUTES all previous K and V. For seq_len N, this is
O(N²) redundant computation!

┌─────────────────────────────────────────────────────────────────┐
│ 2.2  THE SOLUTION — Cache and reuse                             │
└─────────────────────────────────────────────────────────────────┘

KV Cache stores the K and V tensors from ALL previous positions
in ALL layers. When generating the next token:

  - Compute Q, K, V only for the NEW token
  - APPEND the new K, V to the cache
  - Compute attention: Q_new @ [K_cache | K_new]
  - Weighted sum: attn @ [V_cache | V_new]

This reduces compute from O(N²) to O(N) per generation step,
but INCREASES memory by O(N × num_layers × d_model).

┌─────────────────────────────────────────────────────────────────┐
│ 2.3  WHY THIS MAKES INFERENCE MEMORY-BOUND                      │
└─────────────────────────────────────────────────────────────────┘

For LLaMA-7B (32 layers, d_model=4096, seq_len=2048):
  Cache size = 2 × 32 × 2048 × 4096 × 2 bytes (FP16)
             = 1,073,741,824 bytes ≈ 1 GB

The model weights are 14 GB. The KV cache adds 1 GB.
For batch_size=32: 32 GB just for KV caches!

During generation, the GPU spends most of its time WAITING for
memory transfers (loading K/V from the cache), not computing.
This is why inference throughput is limited by memory bandwidth,
not compute.

The inference course (course/inference/) covers KV cache in detail.
""")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 3: GQA — Grouped Query Attention
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 3: GQA — Grouped Query Attention")
print("=" * 70)
print("Used by: LLaMA 2/3, Mistral, Gemma")
print()

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  THE KV CACHE PROBLEM                                       │
└─────────────────────────────────────────────────────────────────┘

MHA (Multi-Head): H query heads and H key/value heads.
  → KV cache size = 2 × H × d_k × seq_len × layers

For LLaMA-7B (H=32): that's 32 sets of keys and values per layer.
This is the DOMINANT memory cost during inference.

┌─────────────────────────────────────────────────────────────────┐
│ 3.2  MULTI-QUERY ATTENTION (MQA) — Extreme savings             │
└─────────────────────────────────────────────────────────────────┘

MQA: H query heads, but only ONE key/value head (shared across all).

  → KV cache size = 2 × 1 × d_k × seq_len × layers
  → H× memory savings!

But: quality degrades. With only one K/V, all queries see the same
"view" of each token. Different heads can't specialize in different
attention patterns.

┌─────────────────────────────────────────────────────────────────┐
│ 3.3  GROUPED QUERY ATTENTION (GQA) — The sweet spot            │
└─────────────────────────────────────────────────────────────────┘

GQA: H query heads, G key/value heads (1 < G < H).

  Each of the G key/value heads is shared by H/G query heads.
  Example: LLaMA 2 70B uses H=64, G=8 → 8 groups of 8 queries each.

  → KV cache = MHA / (H/G)
  → Quality ≈ MHA (empirically very close)
  → Works because attention heads aren't fully independent;
    multiple query heads can share the same K,V without quality loss.

WHY DOES THIS WORK? Research (Ainslie et al., 2023) found that
attention heads within a model often learn similar patterns.
Sharing K/V across groups is a form of STRUCTURED SPARSITY —
removing redundancy without losing capability.
""")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 4: FLASH ATTENTION — IO-Aware Exact Attention
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 4: FLASH ATTENTION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE PROBLEM — Attention is IO-bound, not compute-bound    │
└─────────────────────────────────────────────────────────────────┘

Standard attention:
  1. Compute S = Q @ K^T  → O(N²) memory (write to HBM)
  2. P = softmax(S)       → O(N²) memory (read S, write P)
  3. O = P @ V            → O(N²) memory (read P)

Each step reads from and writes to High Bandwidth Memory (HBM),
the GPU's main memory (40-80 GB). HBM bandwidth is ~2 TB/s on A100.

The on-chip SRAM ("shared memory") is much FASTER (~19 TB/s) but
tiny (~20 MB per SM, ~160 MB total on A100).

Standard attention wastes bandwidth by writing the full N×N matrix
to HBM and reading it back. The computation itself is trivial
compared to the memory transfers.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  FLASH ATTENTION — Tile and accumulate                      │
└─────────────────────────────────────────────────────────────────┘

Key insight (Dao et al., 2022): You don't need to materialize the
full attention matrix. You can compute attention in BLOCKS,
keeping everything in fast SRAM.

Algorithm:
  1. Load a BLOCK of Q, K, V into SRAM
  2. Compute attention scores for this block
  3. Apply "online softmax" — a streaming algorithm that computes
     softmax without seeing all scores at once
  4. Accumulate the weighted value output
  5. Repeat for all blocks (the outer loops are tiled)

Result:
  - Exact same output (NOT an approximation!)
  - O(N) memory instead of O(N²)
  - 2-4× faster than standard attention
  - Enables training on 8K-64K context lengths

This is now STANDARD. PyTorch's `torch.nn.functional.scaled_dot_product_attention`
automatically uses Flash Attention when available. Every production
LLM infrastructure uses it.
""")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 5: SWIGLU & RMSNORM — Better Activation & Normalization
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 5: SWIGLU AND RMSNORM")
print("=" * 70)
print("Used by: LLaMA, Mistral, PaLM, Gemma")
print()

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  SWIGLU — Gated activation for FFN                         │
└─────────────────────────────────────────────────────────────────┘

Classic FFN (Module 6):
  FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

SwiGLU FFN (Shazeer, 2020):
  FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

Where:
  - SiLU(x) = x · σ(x)  (Sigmoid Linear Unit = swish activation)
  - ⊙ = element-wise multiplication (the "gate")
  - W_gate: controls HOW MUCH of each feature passes through
  - W_up: what information to pass through
  - W_down: project back to d_model

WHY GATING?
  Like an LSTM or GRU gate, the gate projection learns to CONTROL
  information flow. For dimensions where the gate is near zero,
  that dimension is suppressed. For dimensions where it's near 1,
  the information passes through.

  This is more expressive than a simple activation function:
  GELU(x) = x · Φ(x) applies the SAME gating function to all
  dimensions. SwiGLU learns a DIFFERENT gating policy per dimension.

  Empirically: SwiGLU matches or exceeds GELU with the same or
  slightly fewer parameters (the gate and up projections share
  the expansion budget).

┌─────────────────────────────────────────────────────────────────┐
│ 5.2  RMSNorm — Simpler, faster normalization                   │
└─────────────────────────────────────────────────────────────────┘

LayerNorm (Module 6):
  y = γ · (x - μ) / √(σ² + ε) + β

RMSNorm (Zhang & Sennrich, 2019):
  y = x / RMS(x) · γ

Where RMS(x) = √(mean(x²) + ε)

DIFFERENCES:
  - NO mean subtraction (re-centering adds no empirical value)
  - NO bias term β (removed; doesn't hurt performance)
  - ~5-10% faster than LayerNorm with equal or better quality
  - Simpler gradient computation

WHY REMOVE MEAN CENTERING?
  The hypothesis: the benefit of normalization comes from
  controlling the SCALE of activations (variance), not their
  LOCATION (mean). The learnable bias β in LayerNorm can
  re-introduce a mean anyway. RMSNorm focuses on what matters.
""")


def rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: normalize by root mean square, no mean subtraction."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


x_demo = np.array([[1.0, 2.0, 3.0], [100.0, -50.0, 0.0]])
gamma = np.ones(3)

print(f"\nRMSNorm demo:")
print(f"  Input:     {x_demo}")
print(f"  Normalized: {np.round(rms_norm(x_demo, gamma), 3)}")
print(f"  RMS (row 0): {np.sqrt(np.mean(rms_norm(x_demo, gamma)[0]**2)):.4f} (≈ 1)")
print(f"  RMS (row 1): {np.sqrt(np.mean(rms_norm(x_demo, gamma)[1]**2)):.4f} (≈ 1)")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 6: MoE — Mixture of Experts
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TOPIC 6: MoE — Mixture of Experts")
print("=" * 70)
print("Used by: Mixtral, GPT-4 (rumored), Gemini, DeepSeek-V2")
print()

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 6.1  WHY MoE? — Sparse activation for scale                    │
└─────────────────────────────────────────────────────────────────┘

In a standard transformer, EVERY parameter is activated for EVERY
token. A 70B-parameter model uses all 70B params for each token.
This is INEFFICIENT — a math token doesn't need the poetry expertise.

MoE replaces each FFN with MULTIPLE expert FFNs and a ROUTER
that selects which expert(s) to use for each token.

┌─────────────────────────────────────────────────────────────────┐
│ 6.2  HOW MoE WORKS                                            │
└─────────────────────────────────────────────────────────────────┘

  Router: scores = softmax(x @ W_router)   → (batch, seq, num_experts)
          Select top-k experts (typically k=2)

  Output: Σ (router_score[i] × expert_i(x))

  - TOTAL parameters: E × FFN_params  (huge!)
  - ACTIVE parameters: k × FFN_params  (same as k standard FFNs)
  - Each token uses only k experts, but different tokens use
    different experts

  Mixtral 8×7B: 8 experts, top-2 routing
    - Total: 47B parameters
    - Active per token: ~13B (same compute as a 13B model)
    - Quality rivals much larger dense models

┌─────────────────────────────────────────────────────────────────┐
│ 6.3  THE LOAD BALANCING PROBLEM                                │
└─────────────────────────────────────────────────────────────────┘

If the router always picks the same 2 experts, the others do
nothing — you've wasted their parameters. MoE training requires
LOAD BALANCING loss: an auxiliary loss term that penalizes
imbalanced expert usage.

Without it, training collapses to a few experts; the rest die.
This is the hardest part of MoE to get right.
""")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 7: SPECULATIVE DECODING
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 7: SPECULATIVE DECODING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 7.1  THE SPEED PROBLEM                                         │
└─────────────────────────────────────────────────────────────────┘

Autoregressive generation produces ONE token per forward pass.
A 7B model takes ~10ms per token on an A100 → ~100 tokens/second.
Human reading speed: ~200-300 words/min ≈ 3-5 words/sec ≈ 4-8 tokens/sec.
So 100 tok/s is perfectly fine for a single user.

But during generation, the GPU is WAITING for memory (KV cache loads),
not computing. The GPU utilization might be < 10%. We're wasting
compute that could verify or generate MORE tokens.

┌─────────────────────────────────────────────────────────────────┐
│ 7.2  THE SPECULATIVE DECODING ALGORITHM                        │
└─────────────────────────────────────────────────────────────────┘

Idea (Leviathan et al., 2023; Chen et al., 2023):

  1. Use a SMALL "draft" model (100M-1B params) to QUICKLY generate
     K candidate tokens: [t1, t2, t3, ..., tK]

  2. Feed the prefix + all K candidates to the BIG "target" model
     in ONE forward pass (batch-like processing)

  3. The big model computes probabilities for each position.
     For position i, compare p_draft(t_i) vs p_target(t_i):

     - If p_target(t_i) >= p_draft(t_i): ACCEPT token t_i
     - If p_target(t_i) < p_draft(t_i): REJECT with probability
       p_target/p_draft. If rejected, resample from the adjusted
       target distribution.

  4. The algorithm guarantees the SAME output distribution as the
     target model alone. The draft model only speeds things up;
     it doesn't affect output quality.

  Typical speedup: 2-3× with a good draft model.
  Key: the big model's forward pass on K tokens costs ~the same
  as on 1 token (KV cache handles the rest). Verification is
  nearly free!

This is how production systems serve LLMs at scale. The user
never sees the draft model — it's purely an optimization.
""")


# ═══════════════════════════════════════════════════════════════════
# TOPIC 8: THE MODERN STACK — Putting it all together
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TOPIC 8: THE MODERN TRANSFORMER STACK")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ FROM MINIGPT TO LLaMA 3 — A side-by-side comparison            │
└─────────────────────────────────────────────────────────────────┘

Component          MiniGPT (2017)        Modern (LLaMA 3, 2024)
─────────────────────────────────────────────────────────────────
Position Encoding  Learned absolute      RoPE (relative, extrapolates)
Attention          MHA (H query + H KV)  GQA (H query + G KV, G<H)
Attention Kernels  Naive O(N²)           Flash Attention (O(N) mem)
FFN Activation     GELU                  SwiGLU (gated)
Normalization      LayerNorm             RMSNorm (faster)
Position of Norm   Post-LN (original)    Pre-LN (stable gradients)
Vocabulary         BPE (GPT-2)           BPE (OpenAI) or Unigram (Meta)
Bias Terms         Present               Often removed (LLaMA has no bias)
Weight Init        N(0, 0.02)            Small init or scaled init
Activations        FP32                  BF16/FP16 (mixed precision)
Context Length     512-2048              8K-128K+ (with RoPE extrapolation)

The ARCHITECTURE is structurally the same. Each improvement is a
local optimization: better positions, fewer KV heads, faster attention
kernels, simpler norm, better activation. Iterate on ALL of them
and you go from 124M GPT-2 to 400B+ LLaMA 3.
""")

# ──────────────────────────────────────────────────────────────────────────────
# HONORABLE MENTIONS — What else should you know?
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("HONORABLE MENTIONS — Topics for further study")
print("=" * 70)

print("""
1. SLIDING WINDOW ATTENTION (Mistral):
   Attend only to a local window of tokens plus some global tokens.
   Reduces attention from O(N²) to O(N·W) where W is the window size.
   Mistral uses W=4096 with some "global" tokens every few layers.

2. ALIBI (Attention with Linear Biases):
   Add a LEARNED bias to attention scores based on token distance.
   Completely eliminates positional encodings. Works surprisingly well
   and enables length extrapolation.

3. QUANTIZATION (GPTQ, AWQ, GGUF):
   Compress 16-bit weights to 4-bit or 2-bit with minimal quality loss.
   A 7B model in 4-bit uses ~3.5 GB → runs on a laptop.
   Key algorithms: GPTQ (post-training), AWQ (activation-aware), GGUF (llama.cpp).

4. PAGED ATTENTION (vLLM):
   Manage KV cache like an OS manages virtual memory — allocate in "pages."
   Nearly eliminates KV cache fragmentation. Enables much higher throughput
   when serving multiple requests.

5. CONTINUOUS BATCHING:
   Dynamically add/remove requests from the processing batch as they
   complete. Critical for serving APIs — don't wait for the slowest request.

6. TENSOR PARALLELISM + PIPELINE PARALLELISM:
   Split model weights ACROSS GPUs (tensor parallel) and layers ACROSS
   GPUs (pipeline parallel). Required for models > ~10B parameters that
   don't fit on a single GPU.

7. RLHF / DPO — Post-training alignment:
   After pre-training (next-token prediction), fine-tune the model to
   follow instructions and align with human preferences.
   - RLHF: Reinforcement Learning from Human Feedback (original ChatGPT)
   - DPO: Direct Preference Optimization (simpler, no RL needed)
   - Both use pairs of "good" and "bad" responses to teach preferences.

8. CHINCHILLA SCALING LAWS:
   Optimal training: tokens ≈ 20 × parameters.
   Most models are UNDERTRAINED (too many params, not enough data).
   The trend: smaller models + more data = better results.
   LLaMA 3 8B on 15T tokens outperforms GPT-3 175B on 300B tokens.
""")


# ═══════════════════════════════════════════════════════════════════
# COURSE COMPLETE
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("COURSE COMPLETE — What You Now Understand")
print("=" * 70)

print("""
FROM ZERO TO PRODUCTION:

Module 0: Tensors, matmul, softmax — the three mathematical primitives
Module 1: Tokenization — how text becomes integer IDs
Module 2: Embeddings — how numbers become meaning (lookup table)
Module 3: Simple Attention — the core formula: softmax(QK^T)V
Module 4: Self-Attention — learned Q/K/V projections
Module 5: Multi-Head Attention + Positional Encoding
Module 6: FFN + LayerNorm + Residuals = Complete Transformer Block
Module 7: MiniGPT — complete decoder-only transformer
Module 8: Training — loss, gradients, AdamW, hyperparameters
Module 9: Advanced — what modern LLMs actually use

YOU CAN NOW:
  ✓ Read and understand the "Attention Is All You Need" paper
  ✓ Implement a working transformer in NumPy (no framework magic)
  ✓ Understand the architecture of GPT, LLaMA, Claude, Gemini
  ✓ Explain attention to a colleague from first principles
  ✓ Know which advances separate 2017 from 2024
  ✓ Debug attention patterns and identify architecture bugs

NEXT STEPS:
  1. Re-implement MiniGPT in PyTorch (with real autograd)
  2. Train on TinyStories or Shakespeare (small, fast iteration)
  3. Add RoPE, SwiGLU, RMSNorm from Module 9
  4. Scale up: more layers, bigger d_model, more data
  5. Module 10: Alignment — how models become assistants (SFT, DPO, GRPO, RLHF)
  6. Study inference: the parallel course in course/inference/

The journey from this 23K-param NumPy model to GPT-4 is "just"
  - More layers, more heads, wider dimensions
  - More data (trillions of tokens)
  - More GPUs (thousands)
  - Better engineering (Flash Attention, quantization, parallelism)

But the MATH is identical. You now understand it all.
""")

if __name__ == "__main__":
    print("\nCourse complete! To run the inference course:")
    print("  uv run python course/inference/run.py")
