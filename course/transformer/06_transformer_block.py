"""
MODULE 6: THE FULL TRANSFORMER BLOCK — Completing the Building Block
======================================================================

Attention alone is "just" a weighted average. It can mix information
across tokens, but it can't TRANSFORM information within a token.

The complete transformer block adds three essential components:
  1. Feed-Forward Network (FFN) — per-token non-linear transformation
  2. Layer Normalization — training stability
  3. Residual Connections — gradient highways for deep networks

This module assembles ALL of these into a single TransformerBlock
that you can stack N times to create a GPT-scale model.

WHAT YOU'LL LEARN:
   1. Why attention is just "mixing" and FFN is "thinking"
   2. Why the FFN uses 4× expansion (and why it matters)
   3. Why LayerNorm normalizes across features (not batch)
   4. Why residuals make 100-layer networks trainable
   5. Pre-LN vs Post-LN: a historical design evolution
   6. The exact parameter count per block

AFTER THIS MODULE:
   You have every piece needed to build a GPT. Stacking N copies
   of TransformerBlock is the entire model (Module 7).
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# BACKGROUND: Why attention isn't enough
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: ATTENTION ≠ INTELLIGENCE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ ATTENTION IS LINEAR COMBINATION — That's it                    │
└─────────────────────────────────────────────────────────────────┘

The attention output for token i is:

  output_i = Σⱼ α_ij · V_j

  where α_ij are attention weights (sum to 1, all non-negative)

This is a CONVEX COMBINATION. In geometry, the result is INSIDE the
convex hull of the value vectors. You can't create anything outside
the set of values you already have.

What attention CAN'T do:
  - Can't multiply features: no interaction BETWEEN dimensions
    within a single token.
  - Can't threshold or gate: no "if feature > 0 then use else ignore"
  - Can't map to completely new regions of vector space

What the FFN CAN do:
  - Two linear layers with a non-linearity = any continuous function
    (universal approximation theorem in each token's subspace).
  - The FFN is where the model "thinks" — per-token computation
    that doesn't depend on other tokens.

Together:
  Attention = communication ("what do other tokens say?")
  FFN = computation ("what do I do with this information?")
""")


# Reuse from previous modules
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit.

    WHY GELU and not ReLU?
      ReLU(x) = max(0, x) is non-differentiable at x=0 and has zero
      gradient for all negative values. During training, neurons can
      "die" — if a neuron's output is always negative, it never updates.

      GELU(x) = x · Φ(x) where Φ is the standard Gaussian CDF.
      - Smooth and differentiable everywhere
      - Non-zero gradient for negative values (small but nonzero)
      - Empirically outperforms ReLU in transformers
      - Approximated as: x · sigmoid(1.702x) or the tanh version below

      Modern models (LLaMA) use SiLU/SwiGLU instead (Module 9).
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: FEED-FORWARD NETWORK — The "Thinking" Part
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: FEED-FORWARD NETWORK (FFN)")
print("=" * 70)


class FeedForward:
    """Position-wise feed-forward network.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Applied independently to each token position with the SAME weights.
    Think of it as a tiny 2-layer MLP that every token passes through.

    WHY 4× expansion?
      d_ff = 4 × d_model is the standard ratio (from Vaswani et al.).
      - The first layer EXPANDS the representation, giving it more
        "room" to compute non-linear interactions.
      - The second layer COMPRESSES back to d_model.
      - Empirically, 4× works well. 2× loses quality. 8× doesn't
        improve much but doubles parameters.
      - The expansion is where most of the model's PARAMETERS live.
    """

    def __init__(self, d_model=64, expansion=4):
        self.d_model = d_model
        self.d_ff = d_model * expansion

        # First layer: expand d_model → d_ff
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.02
        self.b1 = np.zeros(self.d_ff)  # bias helps shift activation range

        # Second layer: compress d_ff → d_model
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

        param_count = (d_model * self.d_ff + self.d_ff) + (self.d_ff * d_model + d_model)
        print(f"  FFN: d_model={d_model} → d_ff={self.d_ff} → d_model")
        print(f"       {param_count:,} params")

    def forward(self, x):
        """
        x: (batch, seq, d_model)
        returns: (batch, seq, d_model)

        WHY same weights for all positions?
          The FFN learns transformations that are useful for ANY token
          position. "If a token has feature X > 0, amplify feature Y."
          This is position-independent logic. Sharing weights across
          positions is a strong inductive bias: the same "computation"
          applies everywhere, just with different input values.
        """
        hidden = gelu(x @ self.W1 + self.b1)  # (B, S, d_ff)
        output = hidden @ self.W2 + self.b2     # (B, S, d_model)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: LAYER NORMALIZATION — Keeping Activations Stable
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: LAYER NORMALIZATION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY NORMALIZE? — The internal covariate shift problem     │
└─────────────────────────────────────────────────────────────────┘

As data flows through layers, the distribution of activations
changes. Early layers see small values; later layers see larger
ones. This is "internal covariate shift" — each layer has to
adapt to a moving target of input distributions.

Without normalization:
  - Weights must be carefully initialized (Xavier/Glorot, He init)
  - Learning rate must be small (to handle varying scales)
  - Deep networks (>10 layers) barely train at all

With normalization:
  - Each layer always sees inputs with mean≈0, std≈1
  - Training is stable even with 100+ layers
  - Learning rates can be much larger (faster training)

┌─────────────────────────────────────────────────────────────────┐
│ 2.2  WHY LAYER NORM, NOT BATCH NORM?                           │
└─────────────────────────────────────────────────────────────────┘

Batch Normalization normalizes across the BATCH dimension:
  - For each feature, compute mean/std across all samples in batch
  - Problem for transformers: sequence lengths vary. Padding tokens
    would corrupt the statistics.
  - Problem for small batches: unreliable statistics.
  - Problem for inference: batch size = 1, no batch to normalize over

Layer Normalization normalizes across the FEATURE dimension:
  - For each sample (token), compute mean/std across its features
  - Independent of batch size — works for batch=1
  - Independent of sequence length
  - Small constant overhead but perfectly parallel
""")


class LayerNorm:
    """Normalize across features (last axis) per sample independently.

    y = γ · (x - μ) / √(σ² + ε) + β

    γ (gamma): learnable scale — lets the network undo normalization
               if it turns out to be harmful for some dimensions.
    β (beta): learnable shift — same idea, lets the network restore
              a useful bias if needed.
    ε (epsilon): small constant to avoid division by zero.
                  Typical: 1e-5 for float32, 1e-3 for float16.

    WHY per-feature statistics?
       Each token is an independent sample. Normalizing across its
       d_model features means every token gets the same treatment
       regardless of sequence position or batch membership.
    """

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# Demonstrate
x = np.array([[1.0, 100.0], [-5.0, 5.0], [1000.0, -1000.0]])
ln = LayerNorm(2)
normalized = ln.forward(x)

print(f"\n  Before LayerNorm:")
print(f"    {x}")
print(f"  After LayerNorm:")
print(f"    {np.round(normalized, 3)}")
print(f"  Row means: {np.round(normalized.mean(axis=1), 6)}  (≈ 0)")
print(f"  Row stds:  {np.round(normalized.std(axis=1), 3)}   (≈ 1)")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: RESIDUAL CONNECTIONS — The Gradient Highways
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: RESIDUAL CONNECTIONS")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  WHY RESIDUALS — The vanishing gradient problem             │
└─────────────────────────────────────────────────────────────────┘

Without residual:  x → F(x)   (output = F(x))
With residual:     x → x + F(x)   (output = x + F(x))

The gradient of the residual version is:
  d/dx (x + F(x)) = I + dF/dx

The IDENTITY matrix I means the gradient can ALWAYS flow directly
back through the skip connection. Even if dF/dx is near zero
(vanishing gradient), the I term ensures SOME gradient reaches
earlier layers.

This is why we can train 100-layer transformers. Without residuals,
gradients would decay exponentially and early layers would learn
nothing.

┌─────────────────────────────────────────────────────────────────┐
│ 3.2  PRE-LN VS POST-LN — A critical design choice               │
└─────────────────────────────────────────────────────────────────┘

Original paper (Vaswani et al., 2017): POST-LN
  x = LayerNorm(x + Sublayer(x))

Modern practice (GPT-2, LLaMA, etc.): PRE-LN
  x = x + Sublayer(LayerNorm(x))

WHY PRE-LN WON:
  Post-LN: the gradient must flow through LayerNorm AFTER the
    residual addition. LayerNorm can attenuate gradients.

  Pre-LN: the gradient flows through the residual connection
    BEFORE normalization. The "identity highway" is cleaner.

  Empirically: Pre-LN trains MORE STABLY, especially early in
  training. The warmup phase can be shorter. For very deep models,
  Pre-LN is essentially required.

  The trade-off: Pre-LN slightly limits the expressiveness of
  each sublayer (because its input is normalized before use).
  This actually acts as regularization — the model learns more
  robust features.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: THE COMPLETE TRANSFORMER BLOCK
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: THE COMPLETE TRANSFORMER BLOCK")
print("=" * 70)


class MultiHeadAttention:
    """Minimal MHA for the block (same logic as Module 5)."""
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = softmax(scores, axis=-1)
        out = attn @ V
        return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model) @ self.W_O


class TransformerBlock:
    """One complete transformer block.

    Architecture (Pre-LN):
        x = x + Attention(LayerNorm(x), mask)
        x = x + FFN(LayerNorm(x))

    WHY Pre-LN order:
      1. Cleaner gradient flow through the residual ("identity highway")
      2. More stable training, especially early
      3. Allows shorter/no learning rate warmup
      4. Acts as implicit regularization
    """

    def __init__(self, d_model=64, num_heads=4, ffn_expansion=4):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ffn_expansion)
        self.ln1 = LayerNorm(d_model)  # for attention sublayer
        self.ln2 = LayerNorm(d_model)  # for FFN sublayer

    def forward(self, x, mask=None):
        # Sublayer 1: Self-attention with residual
        x = x + self.attention.forward(self.ln1.forward(x), mask)

        # Sublayer 2: FFN with residual
        x = x + self.ffn.forward(self.ln2.forward(x))

        return x


# Test
np.random.seed(42)
batch_size, seq_len, d_model = 2, 8, 64

block = TransformerBlock(d_model, num_heads=4)
x = np.random.randn(batch_size, seq_len, d_model)
mask = np.triu(np.ones((1, 1, seq_len, seq_len)) * float('-inf'), k=1)

output = block.forward(x, mask)

print(f"\n  Transformer block test:")
print(f"    Input:  {x.shape}")
print(f"    Output: {output.shape}")
diff = np.abs(output - x).mean()
print(f"    Mean change: {diff:.4f} (should be > 0 — block did something)")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: PARAMETER BREAKDOWN — Where the Numbers Live
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 5: PARAMETER DISTRIBUTION IN A TRANSFORMER BLOCK")
print("=" * 70)

d = 768
h = 12
mha = 4 * d * d            # W_Q, W_K, W_V, W_O
ffn = 2 * d * 4 * d       # W1 (d→4d) + W2 (4d→d)
ffn_bias = 4*d + d         # b1 + b2
ln = 2 * d + 2 * d         # 2 LayerNorms (γ + β each)
total = mha + ffn + ffn_bias + ln

print(f"""
  Component              Parameters       Percentage
  ────────────────────────────────────────────────
  Multi-Head Attention   {mha:>10,}         {mha*100/total:.1f}%
  Feed-Forward Network   {ffn:>10,}         {ffn*100/total:.1f}%
  FFN Biases             {ffn_bias:>10,}        {ffn_bias*100/total:.1f}%
  Layer Norms             {ln:>10,}          {ln*100/total:.1f}%
  ────────────────────────────────────────────────
  Total per block        {total:>10,}         100.0%

  Per block: attention is {mha*100/total:.0f}% of params, FFN is {ffn*100/total:.0f}%.
  The FFN is {(ffn/mha):.1f}× larger than attention in parameter count.

  For 12 blocks (GPT-2 Small): {total*12:,} params (core only)
  Plus embeddings + LM head: ~{total*12 + 2*50257*d:,} params total
""")

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. FFN = per-token non-linear computation (the "thinking" part)
2. 4× expansion ratio is empirically optimal (from the original paper)
3. LayerNorm = normalize across features, not batch (works for batch=1)
4. Residuals = gradient highways (I + dF/dx prevents vanishing gradients)
5. Pre-LN = modern standard (cleaner gradients, more stable training)

Block architecture:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Stack N of these → GPT!

Next: Module 7 — MiniGPT: A Complete Decoder-Only Transformer
""")

if __name__ == "__main__":
    print("\nModule 6 complete! Next: 07_mini_gpt.py")
