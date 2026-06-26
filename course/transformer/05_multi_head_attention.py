"""
MODULE 5: MULTI-HEAD ATTENTION — Parallel Attention Patterns
==============================================================

A single attention head can only learn ONE type of relationship.
But language has many: syntax, semantics, coreference, topic,
tense agreement, negation scope...

Multi-head attention runs MULTIPLE attention mechanisms in parallel,
each with its own Q/K/V projections. Different heads can specialize
in different linguistic relationships.

WHAT YOU'LL LEARN:
   1. Why multiple heads exist (the specialization argument)
   2. How heads split d_model across H independent subspaces
   3. The reshape/transpose trick for efficient batching
   4. Positional encoding — how tokens know WHERE they are
   5. Why attention is otherwise order-invariant

AFTER THIS MODULE:
   You'll have implemented the entire multi-head attention block
   from the original paper, plus positional encodings.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# BACKGROUND: Why one attention pattern isn't enough
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY MULTIPLE HEADS?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE SPECIALIZATION ARGUMENT                                     │
│                                                                 │
│ A single attention head applies ONE softmax to ONE QK^T matrix. │
│ The resulting weights are a 2D distribution: for each token,    │
│ ONE set of attention weights across all other tokens.           │
│                                                                 │
│ But a token might need to attend to DIFFERENT tokens for        │
│ DIFFERENT reasons simultaneously:                               │
│                                                                 │
│   "The cat that chased the mouse sat on the mat"                │
│                                                                 │
│ - "sat" needs to find its subject ("cat")     → syntax head     │
│ - "cat" needs to reference "that" clause      → coreference     │
│ - Every token needs positional context        → position head   │
│ - "chased" and "mouse" share semantics        → semantic head   │
│                                                                 │
│ One head = one type of relationship at a time.                   │
│ Multiple heads = multiple relationship types learned in parallel.│
└─────────────────────────────────────────────────────────────────┘

HOW IT WORKS:
  Instead of one set of (W_Q, W_K, W_V), we have H sets.
  Each head operates on d_k = d_model // H dimensions.
  All heads process the same input in parallel.
  Outputs are CONCATENATED and projected back to d_model.

  For GPT-2 Small: H=12, d_model=768 → d_k=64 per head.
  For LLaMA-7B:    H=32, d_model=4096 → d_k=128 per head.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: MULTI-HEAD ATTENTION — Implementation
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: MULTI-HEAD ATTENTION IMPLEMENTATION")
print("=" * 70)


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """Multi-head self-attention.

    IMPLEMENTATION NOTE: Instead of H separate weight matrices
    (which would require a Python loop), we use ONE big weight
    matrix of shape (d_model, d_model) and reshape to separate
    heads. This is the "fused" implementation used in practice.

    W_Q shape: (d_model, d_model) — all H heads' Q weights packed
        together. After the matmul x@W_Q, we get (batch, seq, d_model),
        which we reshape to (batch, num_heads, seq, d_k).

    WHY fused? GPU matmul is fastest on large matrices. H small
    matmuls of size (seq, d_k) would underutilize the GPU. One big
    matmul of size (seq, d_model) fully saturates the tensor cores.
    """

    def __init__(self, d_model=64, num_heads=4):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Fused weight matrices: one big matmul, then reshape
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

        print(f"  MHA: d_model={d_model}, heads={num_heads}, d_k={self.d_k}")

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: broadcastable to (batch, num_heads, seq, seq)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Step 1: Fused linear projections
        # WHY one big matmul? GPU efficiency.
        # H separate (seq, d_k) matmuls would waste compute.
        Q = x @ self.W_Q  # (batch, seq, d_model)
        K = x @ self.W_K
        V = x @ self.W_V

        # Step 2: Split into heads
        # From (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        #
        # WHY reshape + transpose instead of a loop?
        #   Loops in Python over batches/heads would be 100-1000× slower.
        #   The reshape just CHANGES THE VIEW of the same memory — zero cost.
        #   After reshape, the tensor is in the right shape for batched matmul.
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(0, 2, 1, 3)   # (batch, heads, seq, d_k)

        K = K.reshape(batch, seq_len, self.num_heads, self.d_k)
        K = K.transpose(0, 2, 1, 3)

        V = V.reshape(batch, seq_len, self.num_heads, self.d_k)
        V = V.transpose(0, 2, 1, 3)

        # Step 3: Scaled dot-product attention per head
        # Q: (B, H, S, d_k)   K^T: (B, H, d_k, S)   → scores: (B, H, S, S)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)

        # Step 4: Weighted sum of values
        # attn: (B, H, S, S) @ V: (B, H, S, d_k) → (B, H, S, d_k)
        head_outputs = attn_weights @ V

        # Step 5: Concatenate heads
        # (B, H, S, d_k) → (B, S, H, d_k) → (B, S, d_model)
        head_outputs = head_outputs.transpose(0, 2, 1, 3)
        concat = head_outputs.reshape(batch, seq_len, self.d_model)

        # Step 6: Output projection — mix information across heads
        output = concat @ self.W_O

        return output, attn_weights


# Test
np.random.seed(42)
mha = MultiHeadAttention(d_model=64, num_heads=4)

batch_size, seq_len = 2, 8
x = np.random.randn(batch_size, seq_len, 64)
mask = np.triu(np.ones((1, 1, seq_len, seq_len)) * float('-inf'), k=1)

output, attn_weights = mha.forward(x, mask)

print(f"\n  Forward pass:")
print(f"    Input:    {x.shape}")
print(f"    Output:   {output.shape}")
print(f"    Attn:     {attn_weights.shape}  (batch, heads, seq, seq)")

# Show that different heads produce different attention
print(f"\n  Head 0, batch 0 attention (first 4 tokens):")
print(np.round(attn_weights[0, 0, :4, :4], 3))
print(f"\n  Head 1, batch 0 attention (first 4 tokens):")
print(np.round(attn_weights[0, 1, :4, :4], 3))
print(f"\n  → Different heads = different attention patterns!")
print(f"  → After training, heads specialize in different relationships.")


# ═══════════════════════════════════════════════════════════════════
# Why heads actually specialize (and sometimes don't)
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ DO HEADS ACTUALLY SPECIALIZE? — Not always                     │
└─────────────────────────────────────────────────────────────────┘

Research (Voita et al., 2019; Clark et al., 2019) analyzing BERT's
attention heads found:

  ~30% of heads learn interpretable patterns:
    - Attending to previous/next token (positional)
    - Attending to syntactic dependents
    - Attending to [SEP] tokens (delimiter-information)

  ~50% of heads are "diffuse" — they don't specialize clearly
    - They might be doing something useful we can't interpret
    - Or they might be redundant and could be pruned

  ~20% of heads can be REMOVED with no performance loss
    - This is the basis for head pruning (Michel et al., 2019)
    - At test time, you can often drop 30-50% of heads
    - Training benefits from the extra capacity (regularization)

The lesson: more heads = more capacity, but not all of it is
necessary. This is why model compression often starts with
head pruning and why GQA (Module 9) deliberately uses fewer
key/value heads — it's a structured form of pruning.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: POSITIONAL ENCODING — Tokens Need To Know WHERE They Are
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: POSITIONAL ENCODING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY ATTENTION IS POSITION-BLIND                           │
└─────────────────────────────────────────────────────────────────┘

Attention computes: softmax(Q @ K^T) @ V

This is a SET operation. If you shuffle the tokens, you get the
SAME output, just shuffled. "The cat sat" and "sat The cat" produce
identical attention patterns. Attention has NO notion of order.

We must INJECT position information. Three approaches:

  1. SINUSOIDAL (Vaswani et al., 2017): Fixed sine/cosine waves
     added to embeddings. No learned parameters. Enables length
     extrapolation but quality degrades.

  2. LEARNED (GPT-2, BERT): A learnable embedding per position.
     Simple, effective, but can't extrapolate beyond training length.
     "Position 1025" has no embedding if trained on max 1024.

  3. ROPE (LLaMA, Mistral, 2023+): Rotary position embeddings.
     Rotates Q and K by position-dependent angles. Encodes RELATIVE
     position naturally. Extrapolates well. Module 9 covers this.
""")


# SINUSOIDAL POSITIONAL ENCODING — the original approach
def sinusoidal_positional_encoding(seq_len, d_model):
    """Create positional encodings using sine and cosine.

    WHY sine and cosine? Two properties make them ideal:

    1. UNIQUE: Each position gets a unique vector (no collisions).
    2. RELATIVE: The encoding at position p+k can be expressed as
       a linear function of the encoding at position p. This means
       the model can learn to attend to "the token 5 positions ago"
       as a simple linear operation.

    WHY multiple frequencies (10000^(2i/d)):
      - Low frequencies (large wavelength): distinguish far-apart
        positions — like knowing if you're in the first half or
        second half of the document.
      - High frequencies (small wavelength): distinguish nearby
        positions — like knowing if you're at position 5 or 6.
      - The geometric progression of frequencies gives the model
        access to position at multiple scales simultaneously.
    """
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pos_enc[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(angle)
    return pos_enc


pos_enc = sinusoidal_positional_encoding(seq_len=10, d_model=8)

print(f"\n  Sinusoidal positional encoding (10 positions × 8 dims):")
print(np.round(pos_enc, 3))

# Demonstrate the relative position property
print(f"\n  Position 0 vs 1 dot product: {(pos_enc[0] @ pos_enc[1]):.3f}")
print(f"  Position 0 vs 5 dot product: {(pos_enc[0] @ pos_enc[5]):.3f}")
print(f"  → Nearby positions have HIGHER similarity!")
print(f"  → The encoding naturally captures relative distance.")


# LEARNED POSITION EMBEDDINGS — what GPT actually uses
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  LEARNED POSITION EMBEDDINGS — GPT's approach               │
└─────────────────────────────────────────────────────────────────┘

Most practical transformers (GPT-2, BERT) use LEARNED position
embeddings: just another embedding table, but indexed by position
rather than token ID.

  pos_embed = nn.Embedding(max_seq_len, d_model)
  x = token_embed[token_ids] + pos_embed[position_ids]

WHY learned?
  - Simpler to implement and train
  - Empirically works as well or better than sinusoidal
  - The model can learn position representations optimized for
    the specific training data distribution

WHY NOT learned?
  - Can't extrapolate: position 2049 is unknown if trained on 0-2047
  - This is a real problem — GPT-3 has a hard limit at 2048 tokens
  - RoPE (Module 9) fixes this while keeping the benefits
""")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Multi-head attention = H parallel attention heads, concatenated
2. Reshape trick = one big matmul instead of H small ones (GPU efficiency)
3. Heads CAN specialize (syntax, semantics, position) but often don't
4. Positional encoding fixes attention's position-blindness
5. Sine/cosine (original) vs learned (GPT) vs RoPE (modern)

Pipeline:
  Embeddings + Positions → Multi-Head Attention → Context-Aware Vectors

Next: Module 6 — The Full Transformer Block (adding FFN + LayerNorm)
""")

if __name__ == "__main__":
    print("\nModule 5 complete! Next: 06_transformer_block.py")
