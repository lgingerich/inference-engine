"""
MODULE 4: SELF-ATTENTION — Learned Q/K/V Projections
======================================================

In Module 3, we used the same token vectors for Query, Key, and Value.
That works for building intuition, but it's not what transformers use.

Real self-attention learns SEPARATE linear projections for Q, K, and V.
Each token now has THREE different views of itself: what it's LOOKING FOR,
what it CONTAINS, and what it CONTRIBUTES.

WHAT YOU'LL LEARN:
   1. Why a single vector can't serve as Q, K, and V simultaneously
   2. How learned projections create "roles" for each token
   3. The full scaled dot-product attention as a single function
   4. Why these projections have no bias terms (important detail!)
   5. How parameter scale compares to the rest of the model

AFTER THIS MODULE:
   You'll have implemented the EXACT attention mechanism from the
   "Attention Is All You Need" paper. The rest is building around it.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# BACKGROUND: Why raw embeddings aren't enough
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY RAW EMBEDDINGS CAN'T DO THREE JOBS AT ONCE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE THREE ROLE PROBLEM                                          │
│                                                                 │
│ In the attention formula:                                       │
│   softmax(Q @ K^T / sqrt(d_k)) @ V                              │
│                                                                 │
│ Each token must simultaneously serve as:                        │
│                                                                 │
│   QUERY: "What am I looking for in other tokens?"              │
│          → A noun might look for its adjective.                 │
│          → A verb might look for its subject.                   │
│                                                                 │
│   KEY: "How should other tokens find me?"                      │
│          → An adjective wants to be found by nouns.             │
│          → A verb wants to be found by the next token.          │
│                                                                 │
│   VALUE: "What information do I contribute?"                   │
│          → The adjective's MEANING, not just its position.      │
│          → The verb's tense, mood, and semantics.               │
│                                                                 │
│ These are THREE DIFFERENT JOBS. Expecting one vector to do      │
│ all three with no transformation is like expecting one person   │
│ to simultaneously be the asker, the answerer, and the content.  │
└─────────────────────────────────────────────────────────────────┘

With learned projections:
  x → x @ W_Q → Q (query: "what I'm asking")
  x → x @ W_K → K (key: "how I describe myself to others")
  x → x @ W_V → V (value: "what I contribute to the conversation")

Each weight matrix is LEARNED — the model discovers which features
are useful for each role.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: LEARNED Q/K/V PROJECTIONS — Implementation
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: LEARNED PROJECTIONS")
print("=" * 70)

np.random.seed(42)
seq_len, d_model, d_k = 4, 16, 16

x = np.random.randn(seq_len, d_model)

# Initialize weight matrices
# WHY small initialization (×0.02)? 
#   - Random normal has variance 1. If d_model=768, x @ W has
#     variance 768 → huge values → softmax saturates.
#   - Scaling by 0.02 (or 1/sqrt(d_model)) keeps initial outputs small.
#   - This prevents training from starting in the saturated regime.
W_Q = np.random.randn(d_model, d_k) * 0.02
W_K = np.random.randn(d_model, d_k) * 0.02
W_V = np.random.randn(d_model, d_k) * 0.02

# WHY no bias in Q/K/V projections?
#   - Bias would add a CONSTANT to every token's query/key/value.
#   - For K and V: this constant would be the same for every position,
#     and attention is invariant to adding a constant to keys (the
#     softmax normalizes it out). Adding it wastes parameters.
#   - For Q: a constant query would add the same direction to every
#     token's attention preference — arguably useful, but empirically
#     unnecessary. Most implementations omit bias.

Q = x @ W_Q  # (seq, d_k) — each token's "question"
K = x @ W_K  # (seq, d_k) — each token's "description"
V = x @ W_V  # (seq, d_k) — each token's "content"

print(f"Input:  x  = {x.shape} ({seq_len} tokens, {d_model} dimensions)")
print(f"Query:  Q  = {Q.shape}  ({seq_len} tokens, {d_k} query dimensions)")
print(f"Key:    K  = {K.shape}")
print(f"Value:  V  = {V.shape}")

# ═══════════════════════════════════════════════════════════════════
# PART 1A: What the projections actually DO
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ WHAT THE PROJECTIONS DO — A geometric view                      │
└─────────────────────────────────────────────────────────────────┘

Each projection x @ W is a linear transformation that maps each
token from "embedding space" to a "role space":

  - Embedding space: the raw meaning of the token
  - Query space: optimized for ASKING questions
  - Key space: optimized for BEING matched against queries
  - Value space: optimized for CONTRIBUTING information

The dot product Q[i] @ K[j] now measures: "How well does token i's
question match token j's description?" With learned projections,
this is MUCH more expressive than raw embedding similarity.

Example of what a trained head might learn:
  - Q projects tokens into a "seeking nouns" direction
  - K projects tokens into a "being an adjective" direction
  - Result: adjectives get high attention from nouns

Without projections, the noun would need to have a raw embedding
that simultaneously encodes "I am a noun AND I seek adjectives."
With projections, the noun's QUERY encodes "I seek adjectives"
and its KEY encodes "I am a noun" — separate, clean, learnable.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: SCALED DOT-PRODUCT ATTENTION — The Complete Function
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: THE COMPLETE ATTENTION FUNCTION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY sqrt(d_k) REVISITED — The mathematics                  │
└─────────────────────────────────────────────────────────────────┘

Let's prove why sqrt(d_k) is the right scaling factor.

Assume Q and K vectors have independent components with mean 0
and variance 1 (valid after LayerNorm).

  The dot product q·k = Σᵢ qᵢ·kᵢ for i in 1..d_k

  Each term qᵢ·kᵢ has:
    - mean: E[qᵢ]·E[kᵢ] = 0·0 = 0
    - variance: Var[qᵢ·kᵢ] = Var[qᵢ]·Var[kᵢ] = 1 (since independent)

  The sum of d_k independent terms has:
    - mean: 0
    - variance: d_k  (additivity of variance for independent terms)

  So q·k has standard deviation sqrt(d_k).

  Dividing by sqrt(d_k) makes the variance = 1 again.

  Without this: std dev grows with sqrt(d_k). For d_k=128 (LLaMA-7B),
  dot products have std dev ≈ 11.3. Scores of 0 vs 11 are effectively
  one-hot after softmax → gradients vanish → training fails.

  This is NOT an optimization detail. It is a MATHEMATICAL NECESSITY
  for training stability.
""")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    The canonical attention function from "Attention Is All You Need."

    Args:
        Q: Query, shape (..., seq_len, d_k)
        K: Key,   shape (..., seq_len, d_k)
        V: Value, shape (..., seq_len, d_v)  — d_v can differ from d_k
        mask: Additive mask, shape (..., seq_len, seq_len) or broadcastable

    Returns:
        output: shape (..., seq_len, d_v)
        attention_weights: shape (..., seq_len, seq_len)

    WHY is mask ADDITIVE not multiplicative?
      - Additive: scores + mask works naturally with -inf for blocking.
        softmax(-inf + score) = softmax(-inf) = 0 exactly.
      - Multiplicative: scores * mask would require mask values of 0 or 1.
        But 0 * large score = 0, and softmax(0) ≈ 1/vocab_size (NOT zero!).
        You'd still get small attention to masked positions.

    WHY swapaxes instead of .T?
      - .T reverses ALL axes: (a,b,c,d) → (d,c,b,a). We only want to
        swap the LAST TWO axes: (a,b,c,d) → (a,b,d,c).
      - swapaxes(-2, -1) is explicit about WHICH axes to swap.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask  # additive: -inf → zero after softmax

    # Numerically stable softmax on last dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = attention_weights @ V
    return output, attention_weights


# Test it
mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)
output, weights = scaled_dot_product_attention(Q, K, V, mask)

print(f"\n  Output shape: {output.shape}")
print(f"  Attention weights shape: {weights.shape}")
print(f"\n  Causal attention pattern:")
print(f"  Token 0 attends: {np.round(weights[0], 3)}  (only itself)")
print(f"  Token 3 attends: {np.round(weights[3], 3)}  (all previous)")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: WHERE THE PARAMETERS ARE — And Why Attention Isn't Heavy
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: PARAMETER ANALYSIS — Attention Is Surprisingly Light")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  PARAMETERS PER ATTENTION LAYER                             │
└─────────────────────────────────────────────────────────────────┘

W_Q, W_K, W_V: each (d_model × d_k) where d_k = d_model typically
W_O: (d_model × d_model) — the output projection

Total attention params = 4 × d_model × d_model = 4d²

For d_model=768 (GPT-2 Small):
  Attention:  4 × 768 × 768  = 2,359,296 params

For reference, the FFN (Module 6):
  FFN:  2 × d_model × 4d_model = 8d² = 4,718,592 params

The FFN has TWICE the parameters of attention. Attention is NOT
the parameter-heavy part of a transformer. It's the COMPUTE-heavy
part because it requires O(n²) operations per sequence.

Memory is the real cost: the attention matrix is O(n²) before
softmax. For seq_len=2048 with batch=8, d_k=64:
  8 × 12 heads × 2048 × 2048 × 4 bytes (float32) = 1.6 GB
  Just for ONE attention layer's intermediate scores!
""")

d = 768
attn_params = 4 * d * d
ffn_params = 2 * d * 4 * d
print(f"d_model={d}:")
print(f"  Attention: {attn_params:,} params")
print(f"  FFN:       {ffn_params:,} params")
print(f"  Ratio: FFN is {ffn_params/attn_params:.1f}× larger than attention")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Learned projections separate Q/K/V roles for each token
2. The formula: softmax(Q @ K^T / sqrt(d_k) + mask) @ V
3. sqrt(d_k) is mathematically necessary — controls variance
4. No bias in projections (adds nothing useful)
5. Attention has fewer params than FFN, but more compute

Pipeline:
  Embeddings → W_Q, W_K, W_V → Q, K, V → Attention → Output vectors

Next: Module 5 — Multi-Head Attention (running this in parallel H times)
""")

if __name__ == "__main__":
    print("\nModule 4 complete! Next: 05_multi_head_attention.py")
