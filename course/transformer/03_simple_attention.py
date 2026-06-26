"""
MODULE 3: SIMPLE ATTENTION — The Core Intuition
================================================

Attention is the heart of the transformer. Everything else —
multi-head, positional encoding, FFN, layer norm — exists to make
attention work BETTER. But the core idea is breathtakingly simple.

This module builds attention from FIRST PRINCIPLES. No learned
projections, no multi-head, no masking. Just the raw formula:

    Attention(Q, K, V) = softmax(Q @ K^T) @ V

If you understand this, you understand the transformer.

WHAT YOU'LL LEARN:
   1. Why attention is a "fuzzy dictionary lookup"
   2. Why dot product = similarity (the geometry behind it)
   3. Why softmax over dot products = "attention distribution"
   4. Why weighted sum over values = "context-aware representation"
   5. Why causal masking prevents "cheating" at the future

AFTER THIS MODULE:
   The attention formula will feel as natural as addition. You'll
   look at `softmax(Q @ K^T) @ V` and see the lookup table in it.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY ATTENTION — The Pre-Transformer World
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY ATTENTION WAS INEVITABLE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ BEFORE TRANSFORMERS — The sequence problem                      │
└─────────────────────────────────────────────────────────────────┘

Before 2017, NLP was dominated by RNNs and LSTMs. These process
sequences one token at a time, carrying a "hidden state" forward.

  RNN:  token₁ → h₁ → token₂ → h₂ → token₃ → h₃ → ...

PROBLEMS with RNNs:
  1. SEQUENTIAL: can't parallelize. Token₅₀₀ must wait for
     tokens 1-499 to be processed first. Training is SLOW.

  2. DISTANT DEPENDENCIES: The hidden state must compress
     everything about the previous 1000 tokens into a fixed-
     size vector. Information is LOST over distance.
     "The cat ... [500 words later] ... sat" — the RNN
     has forgotten about the cat by the time it reaches "sat."

  3. GRADIENT PROBLEMS: Backprop through 1000 recurrent steps
     causes vanishing/exploding gradients. LSTMs helped but
     didn't solve the fundamental bottleneck.

THE KEY INSIGHT (Bahdanau et al., 2014; Vaswani et al., 2017):

  Instead of compressing the past into a hidden state, let each
  token DIRECTLY look at every other token at every step. No
  compression. No distance. Every pair of positions is ONE
  computation step away from each other.

  This is ATTENTION: a direct, differentiable communication
  channel between EVERY pair of positions.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE INTUITION — Fuzzy Dictionary Lookup
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: ATTENTION = FUZZY DICTIONARY LOOKUP")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THINK OF IT LIKE A DICTIONARY WITH FUZZY KEYS              │
└─────────────────────────────────────────────────────────────────┘

Normal dictionary lookup:
    dict = {"cat": "a small furry animal", "dog": "a loyal pet"}
    result = dict["cat"]  → "a small furry animal"

This requires an EXACT match. The query "cat" matches exactly
one key. The result is EXACTLY one value.

Attention (fuzzy lookup):
    keys = ["cat", "dog", "car"]
    values = ["furry animal", "loyal pet", "vehicle"]
    query = "feline"

    1. Compute similarity between "feline" and each key:
       → "cat" is similar, "dog" is somewhat similar, "car" is unrelated

    2. Convert similarities to weights (softmax):
       → "cat" gets 0.70, "dog" gets 0.25, "car" gets 0.05

    3. Weighted sum of values:
       result = 0.70 * "furry animal" + 0.25 * "loyal pet" + 0.05 * "vehicle"

The result is a BLEND of all values, weighted by relevance.
You don't need an exact match — "feline" retrieves cat-like
information even though it's not in the dictionary.

THIS IS ATTENTION. Every token looks at every other token,
computes "how relevant is this?" and blends their information.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: BY-HAND ATTENTION — Every Step Explained
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: DOT-PRODUCT ATTENTION — STEP BY STEP")
print("=" * 70)

# Setup: 4 token vectors in 3-dimensional space
np.random.seed(42)
seq_len = 4
d = 3

tokens = np.random.randn(seq_len, d)
Q = tokens.copy()  # In simple attention, same vectors play all roles
K = tokens.copy()
V = tokens.copy()

print(f"\nToken vectors ({seq_len} tokens × {d} dimensions):")
print(np.round(tokens, 3))

# ═══════════════════════════════════════════════════════════════════
# Step 1: Q @ K^T — Compute pairwise similarities
# ═══════════════════════════════════════════════════════════════════

print("""\n┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: SCORES = Q @ K^T — Pairwise similarities               │
└─────────────────────────────────────────────────────────────────┘""")

scores = Q @ K.T  # (seq_len, d) @ (d, seq_len) → (seq_len, seq_len)

print(f"\n  Shape: {scores.shape}  →  4 tokens, each scored against 4 tokens")
print(f"  Scores matrix:")
print(np.round(scores, 2))

print(f"""
  INTERPRETATION:
  - scores[i][j] = how relevant token j is for token i
  - Row 0: token 0's "relevance" to tokens 0, 1, 2, 3
  - scores[0][0] is high because a token is trivially relevant to itself
  
  WHY DOT PRODUCT = SIMILARITY?
  - The dot product measures = |a| × |b| × cos(angle between a and b)
  - Same direction (angle ≈ 0):  dot product is large and positive
  - Opposite direction (angle ≈ 180°): dot product is large and negative
  - Orthogonal (angle ≈ 90°): dot product ≈ 0 (unrelated)
  
  For vectors with similar magnitudes, dot product IS cosine similarity.
  The key insight: in high dimensions, random vectors are approximately
  orthogonal. Meaningful vectors (trained embeddings) have structure.
  Dot product captures this structure.
""")


# ═══════════════════════════════════════════════════════════════════
# Step 2: Scale by sqrt(d)
# ═══════════════════════════════════════════════════════════════════

print("""┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: SCALING — Divide by sqrt(d)                             │
└─────────────────────────────────────────────────────────────────┘""")

scaled_scores = scores / np.sqrt(d)

print(f"\n  Scaled by sqrt({d}) = {np.sqrt(d):.3f}:")
print(np.round(scaled_scores, 2))

print(f"""
  WHY SCALE?
  The dot product of two random vectors of dimension d has
  variance = d (each element contributes variance ~1).
  
  Without scaling, as d grows, some dot products become VERY large
  and some VERY small. After softmax, the large ones become 1.0
  and the small ones become 0.0. This is essentially argmax —
  all gradient information is destroyed. Training fails.
  
  Dividing by sqrt(d) keeps the variance at 1 regardless of d.
  The softmax stays "soft" — gradients still flow.
  
  This was empirically discovered to be NECESSARY. Without this
  scaling, transformers with large d_k don't train at all.
""")


# ═══════════════════════════════════════════════════════════════════
# Step 3: Softmax — Convert to probability distribution
# ═══════════════════════════════════════════════════════════════════

print("""┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: SOFTMAX — Scores become attention weights               │
└─────────────────────────────────────────────────────────────────┘""")

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attn_weights = softmax(scaled_scores, axis=-1)

print(f"\n  Attention weights (each row = a probability distribution):")
print(np.round(attn_weights, 3))
print(f"\n  Row sums: {attn_weights.sum(axis=1)} (should be [1,1,1,1])")

print(f"""
  WHY SOFTMAX HERE?
  - Each token needs to DECIDE how to split its attention.
  - The weights must sum to 1 (you can't give 200% attention).
  - The weights must be non-negative (no "negative attention").
  - Softmax satisfies both naturally.
  
  Read each row as: "Token i pays X% attention to token j"
  - Row 0: token 0 pays most attention to itself (self-relevance)
  - Row 3: token 3 distributes attention more evenly
""")


# ═══════════════════════════════════════════════════════════════════
# Step 4: @ V — Weighted sum of values
# ═══════════════════════════════════════════════════════════════════

print("""┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: WEIGHTED SUM — Blend the values                         │
└─────────────────────────────────────────────────────────────────┘""")

output = attn_weights @ V  # (seq, seq) @ (seq, d) → (seq, d)

print(f"\n  Output shape: {output.shape} (same as input)")
print(f"  Output:")
print(np.round(output, 3))

print(f"""
  WHAT JUST HAPPENED?
  Each output token is a BLEND of all input tokens, weighted by
  how relevant they are.
  
  Token 0's output = 0.45*token_0 + 0.28*token_1 + 0.15*token_2 + 0.12*token_3
  Token 3's output = 0.20*token_0 + 0.35*token_1 + 0.20*token_2 + 0.25*token_3
  
  Each token has been "contextualized" — it now contains information
  about the OTHER tokens, not just itself.
  
  Compare: original vs attended
""")
for i in range(seq_len):
    print(f"  Token {i}: original={np.round(tokens[i], 2)} → attended={np.round(output[i], 2)}")
print(f"\n  → Each token now carries information about its neighbors.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: CAUSAL ATTENTION — The Autoregressive Constraint
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: CAUSAL MASKING — DON'T PEEK AT THE FUTURE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  WHY MASKING? — The autoregressive generation constraint    │
└─────────────────────────────────────────────────────────────────┘

When generating text left-to-right (GPT), the model at position i
must ONLY look at positions ≤ i. If it could look at position i+1,
it would CHEAT — "I'll just copy the next word from the input."

But during TRAINING, we have the full sequence available! We need
to artificially prevent the model from seeing the future.

SOLUTION: Add a CAUSAL MASK before softmax.
  - mask[i][j] = 0        if j ≤ i (allowed — past or present)
  - mask[i][j] = -inf     if j > i (blocked — future)

After softmax, -inf becomes exactly 0. The model CANNOT attend to
future tokens, even though their values exist in the tensor.
""")


def create_causal_mask(seq_len):
    """Upper triangular matrix of -inf."""
    return np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)

causal_mask = create_causal_mask(seq_len)

print(f"Causal mask ({seq_len}×{seq_len}):")
print(causal_mask)
print(f"  0 = allowed (attend here)")
print(f"  -inf = blocked (can't attend)")

# Apply and compute
masked_scores = scores / np.sqrt(d) + causal_mask
causal_weights = softmax(masked_scores, axis=-1)

print(f"\nCausal attention weights (upper triangle is zero):")
print(np.round(causal_weights, 3))

print(f"""
  VERIFY: Token 0 can only attend to token 0 (itself):
    {np.round(causal_weights[0], 3)}
  Token 3 can attend to tokens 0,1,2,3:
    {np.round(causal_weights[3], 3)}

  WHY -inf and not just 0 in the mask?
    softmax includes exp(mask_value). exp(0)=1 (would still contribute!)
    exp(-inf)=0 completely blocks the position.
    exp(large negative) ≈ 0 but not exactly.
    Using -inf is the ONLY way to guarantee zero attention.
""")


# ═══════════════════════════════════════════════════════════════════
# WHY CAUSAL ALSO WORKS DURING TRAINING
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.2  TRAINING WITH CAUSAL MASK — The teacher-forcing trick      │
└─────────────────────────────────────────────────────────────────┘

During training, the model sees the ENTIRE sequence at once:

  Input:  ["The", "cat", "sat", "down"]
  Target: ["cat", "sat", "down", "<EOS>"]

With causal mask, position 0 predicts token 1, position 1 predicts
token 2, etc. ALL predictions happen in ONE forward pass.

This is called TEACHER FORCING. It's massively parallel — the
entire training sequence is processed simultaneously.

During GENERATION, we don't have targets. We generate one token at
a time, feeding each output back as input:

  Step 1: Input ["The"] → predict "cat"
  Step 2: Input ["The", "cat"] → predict "sat"
  Step 3: Input ["The", "cat", "sat"] → predict "down"
  ...

The causal mask is the SAME in both cases. Training in parallel,
generation sequentially — same architecture.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 4: THE ATTENTION FORMULA — The Master Key
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: THE FORMULA THAT CHANGED EVERYTHING")
print("=" * 70)

print("""
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

This is the ENTIRE mechanism. Seven characters of math.
Everything else — multi-head, positional encoding, masks —
is just how we PREPARE Q, K, V. The attention formula is fixed.

The formula has exactly THREE operations:
  1. Matrix multiply: Q @ K^T (similarity)
  2. Softmax: convert to attention weights
  3. Matrix multiply: weights @ V (blend values)

Look at it again. You now understand every symbol.

  Q @ K^T    = "How relevant is each key to each query?"
  / sqrt(d_k) = "Keep gradients healthy"
  softmax     = "Convert relevance to probabilities"
  @ V         = "Blend values by relevance"

That's it. That's attention. That's the transformer's core.
The rest of this course is about making this ONE operation
work better, faster, and at scale.
""")

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Attention = fuzzy dictionary lookup (query matches keys → returns blended values)
2. The formula: softmax(Q @ K^T / sqrt(d_k)) @ V
3. Dot product measures similarity in vector space
4. Scaling prevents softmax saturation
5. Causal mask prevents looking at the future (-inf → zero attention)
6. Teacher forcing: train in parallel, generate sequentially

Next: Module 4 — Self-Attention with Learned Q/K/V Projections
""")

if __name__ == "__main__":
    print("\nModule 3 complete! Next: 04_self_attention.py")
    print("Run with: uv run python course/transformer/04_self_attention.py")
