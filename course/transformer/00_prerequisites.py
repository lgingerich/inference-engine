"""
MODULE 0: PREREQUISITES — Tensors, Matrix Multiplication, and Softmax
=====================================================================

Before we build a transformer, you need to understand three mathematical
operations — not just what they do, but WHY they're used in neural
networks, WHERE they appear in the architecture, and WHAT happens if
you get them wrong.

But before even THAT, you need to understand what a neural network IS.
Not hand-waving about "neurons" and "brains" — the actual computation.
This module starts from absolute zero and builds up.

WHAT YOU'LL LEARN:
   0. What a neural network actually computes (input × weight + bias, repeat)
   1. Tensors — why shape matters, how broadcasting works, why GPUs love them
   2. Matrix multiplication — why it's the universal mixing operation,
      how it makes attention work, why it's so fast on GPUs
   3. Softmax — why we need exp(), why we subtract the max, what
      temperature really does, how it creates differentiable "selection"

AFTER THIS MODULE:
   You'll understand why a neural network is just a chain of matrix
   multiplications and non-linearities, and you'll be able to mentally
   parse every operation in:

       Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: WHAT IS A NEURAL NETWORK, ACTUALLY?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 0: WHAT IS A NEURAL NETWORK? (skip if you know this)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 0.1  The simplest "network" possible: one linear layer
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 0.1  THE SIMPLEST NEURAL NETWORK: y = x @ W + b                 │
└─────────────────────────────────────────────────────────────────┘

Forget neurons, synapses, and brain analogies. A neural network is a
FUNCTION that maps inputs to outputs. The simplest version is:

    y = x @ W + b

where:
  x = input vector (what you feed in, e.g. a token embedding)
  W = weight matrix (LEARNED parameters — the "knowledge")
  b = bias vector (LEARNED offset — shifts the output)
  @ = matrix multiplication
  y = output vector (the "prediction")

x @ W means: "multiply each input feature by a learned weight and sum"
  + b means: "add a learned constant"

This is called a LINEAR LAYER. It can ONLY learn straight-line
relationships. If your data forms a curve, a single linear layer
can't fit it — no matter how good the weights are.
""")

# Demonstrate a linear layer
np.random.seed(42)
x_input = np.array([0.5, -0.3, 0.8])   # 3 input features
W_linear = np.random.randn(3, 2) * 0.5  # 3 inputs → 2 outputs
b_linear = np.array([0.1, -0.2])        # bias for each output

y_output = x_input @ W_linear + b_linear

print(f"  Input x:      {x_input}")
print(f"  Weight W:\n{W_linear}")
print(f"  Bias b:       {b_linear}")
print(f"  Output y = x@W + b: {np.round(y_output, 3)}")
print(f"")
print(f"  Each output is a weighted sum of all inputs:")
print(f"  y[0] = {x_input[0]:.1f}*{W_linear[0,0]:.2f} + "
      f"{x_input[1]:.1f}*{W_linear[1,0]:.2f} + "
      f"{x_input[2]:.1f}*{W_linear[2,0]:.2f} + {b_linear[0]:.2f}")
print(f"       = {(x_input @ W_linear + b_linear)[0]:.3f}")


# ═══════════════════════════════════════════════════════════════════
# 0.2  Stacking layers: why one isn't enough
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 0.2  STACKING LAYERS — Deep means more than one                 │
└─────────────────────────────────────────────────────────────────┘

Stack two linear layers with nothing in between:

    h = x @ W1 + b1
    y = h @ W2 + b2

But this is STILL just a linear function! Because:

    y = (x @ W1 + b1) @ W2 + b2
      = x @ (W1 @ W2) + (b1 @ W2 + b2)
      = x @ W_combined + b_combined

Two linear layers in a row = one equivalent linear layer. No matter
how many you stack, you never get non-linearity. You need an
ACTIVATION FUNCTION between them.
""")

# Prove: two linear layers = one linear layer
W1 = np.random.randn(3, 4) * 0.5
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2) * 0.5
b2 = np.random.randn(2)

# Two-layer path
h = x_input @ W1 + b1
y_two_layer = h @ W2 + b2

# Equivalent single layer
W_combined = W1 @ W2
b_combined = b1 @ W2 + b2
y_single = x_input @ W_combined + b_combined

print(f"  Two linear layers:  {np.round(y_two_layer, 5)}")
print(f"  One combined layer: {np.round(y_single, 5)}")
print(f"  → Identical! Stacking linear layers without activation does nothing.")


# ═══════════════════════════════════════════════════════════════════
# 0.3  Activation functions: the secret sauce
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 0.3  ACTIVATION FUNCTIONS — Breaking linearity                  │
└─────────────────────────────────────────────────────────────────┘

An activation function is a simple NON-LINEAR operation applied
element-wise between layers:

    h = activation(x @ W1 + b1)    ← non-linearity here!
    y = h @ W2 + b2

Common activations:
  ReLU(x)  = max(0, x)       → zero out negatives
  GELU(x)  = x · Φ(x)        → smooth version of ReLU (used in GPT)
  SiLU(x)  = x · sigmoid(x)  → "swish" (used in LLaMA)
  tanh(x)                    → squash to [-1, 1]

The key insight: with activation functions, stacking layers CAN
approximate ANY continuous function (universal approximation theorem).
Without them, you're stuck with linear. With them, you can learn curves,
boundaries, and complex patterns.
""")

# Show what ReLU does
relu = lambda x: np.maximum(0, x)
x_range = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"  ReLU demonstration:")
print(f"    Input:  {x_range}")
print(f"    Output: {relu(x_range)}")
print(f"    → Negative values become 0. Positive values pass through unchanged.")

# Show the difference: with vs without activation
h_linear = x_input @ W1 + b1
h_relu = relu(h_linear)
y_linear = h_linear @ W2 + b2
y_nonlinear = h_relu @ W2 + b2

print(f"\n  With vs without activation:")
print(f"    Linear only:           {np.round(y_linear, 4)}")
print(f"    With ReLU activation:  {np.round(y_nonlinear, 4)}")
print(f"    → Different outputs! Non-linearity changed everything.")


# ═══════════════════════════════════════════════════════════════════
# 0.4  What "training" actually means
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 0.4  TRAINING — How the weights get learned                     │
└─────────────────────────────────────────────────────────────────┘

All the W matrices and b vectors start as RANDOM NUMBERS. Training is
the process of gradually adjusting them to produce better outputs.

The loop:
  1. FORWARD PASS:   Input → network → prediction
  2. COMPARE:        How wrong was the prediction? (compute LOSS)
  3. BACKWARD PASS:  Which weights caused the most error? (gradients)
  4. UPDATE:         Nudge weights to reduce error next time.
                     weight = weight - learning_rate × gradient

Repeat millions of times on millions of examples. The weights
gradually shift from random noise into meaningful patterns.

┌─────────────────────────────────────────────────────────────────┐
│ 0.5  WHY THIS MATTERS FOR TRANSFORMERS                          │
└─────────────────────────────────────────────────────────────────┘

A transformer is just a VERY SPECIFIC arrangement of linear layers
(with specific weight-sharing patterns) and activation functions.
Every component reduces to:

  1. MATRIX MULTIPLICATION (x @ W) — the mixing operation
  2. ELEMENT-WISE NON-LINEARITY (ReLU/GELU/softmax) — breaking linearity
  3. ADDITION/SUBTRACTION (residuals, biases) — routing information

The rest of this module explains these three primitives in detail.
Once you understand them, you understand every operation in a
transformer.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY NUMPY?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY ARE WE USING NUMPY?")
print("=" * 70)

print("""
We use NumPy rather than PyTorch for a deliberate reason:

  NumPy forces you to SEE every tensor shape and every operation.
  There's no magic — you write the matrix multiply, you write the
  softmax, you see the intermediate results.

  PyTorch hides the operations behind nn.Module and autograd.
  That's great for production, but terrible for UNDERSTANDING.

  Once you understand the NumPy version, the PyTorch version becomes
  obvious. The reverse is not true.

  Key difference: NumPy operations produce NEW arrays (no autograd).
  PyTorch operations build a computation graph for backward pass.
  But the FORWARD math is identical. This course is about the forward
  math — the architecture itself.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: TENSORS — Beyond "It's Just A Multidimensional Array"
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: TENSORS — WHY THEY MATTER")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  WHAT IS A TENSOR? (And why the fancy name?)                │
└─────────────────────────────────────────────────────────────────┘

The word "tensor" comes from physics and differential geometry, where
it means "something that transforms in a predictable way under
coordinate changes." In machine learning, it just means "a
multidimensional array of numbers."

Dimensionality guide:
  - scalar:   a single number         → shape ()       → loss value
  - vector:   a list of numbers       → shape (d,)      → one token's embedding
  - matrix:   a 2D grid of numbers    → shape (r, c)    → weight matrix W
  - 3D tensor: stack of matrices      → shape (B,S,D)   → a batch of sequences
  - 4D tensor: stack of 3D tensors    → shape (B,H,S,D) → multi-head attention

The three dimensions you'll see everywhere in transformer code:
  - BATCH (B):    how many sequences we process at once
  - SEQUENCE (S): how many tokens in each sequence
  - FEATURE (D):  d_model — the "width" of each token

Every tensor in a transformer has some subset of these axes.
""")

scalar = np.array(3.14)
print(f"  Scalar: shape={scalar.shape}, ndim={scalar.ndim}, value={scalar}")
print(f"    → A single loss value.")

vector = np.array([1.0, 2.0, 3.0, 4.0])
print(f"  Vector: shape={vector.shape}, ndim={vector.ndim}, values={vector}")
print(f"    → One token's embedding vector, or a bias term.")

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"  Matrix: shape={matrix.shape}, ndim={matrix.ndim}")
print(f"    → A weight matrix W, or one sequence of token embeddings.")

tensor_3d = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
print(f"  3D: shape={tensor_3d.shape}, ndim={tensor_3d.ndim}")
print(f"    → A batch: (batch=2, seq=3, features=2)")

tensor_4d = np.random.randn(2, 4, 3, 5)
print(f"  4D: shape={tensor_4d.shape}, ndim={tensor_4d.ndim}")
print(f"    → Multi-head attention: (batch, heads, seq, d_k)")


# 1.2 WHY SHAPE MATTERS
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  WHY SHAPE MATTERS — The shape IS the semantics             │
└─────────────────────────────────────────────────────────────────┘

In transformer code, the shape tells you EXACTLY what a tensor
represents. Getting a shape wrong silently produces nonsense.

Convention (batch-first):
  (B, S, D)    = B batches, S tokens, D features per token
  (B, H, S, D) = B batches, H attention heads, S tokens, D per head
  (B, S, V)    = B batches, S positions, V vocabulary logits

ALWAYS track your shapes. Print them. The most common transformer bug
is a shape mismatch that still runs but computes wrong attention.
""")


# 1.3 AXIS CONVENTIONS
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.3  AXIS CONVENTIONS — Why batch is first                     │
└─────────────────────────────────────────────────────────────────┘

Most code uses batch-first: (batch, seq, features).

Why?
  1. Broadcasting: (seq, features) added to (batch, seq, features)
     automatically applies to every batch item.
  2. Memory: tokens at the SAME position across batches are adjacent.
     Faster for per-position operations (like FFN).
  3. Slicing: x[0] gives first batch item as contiguous slice.

But PyTorch's nn.Transformer uses sequence-first (S,B,D). Always check.
""")

batch = np.random.randn(2, 4, 3)
bias = np.random.randn(4, 3)
result = batch + bias
print(f"  Broadcasting: (2,4,3) + (4,3) → {result.shape}")
print(f"    → Bias automatically applied to every batch item.")


# 1.4 GPUs and tensors
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.4  WHY GPUS LOVE TENSORS — Single Instruction, Multiple Data  │
└─────────────────────────────────────────────────────────────────┘

GPUs have thousands of cores. They're fast when all cores do the SAME
operation on DIFFERENT data (SIMD). Tensors are the perfect SIMD
structure: every operation is independent across batch/sequence/token.

Transformers map to GPUs perfectly because independent per-token
operations (FFN) are embarrassingly parallel, and cross-token
operations (attention) are batched matrix multiplies.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: MATRIX MULTIPLICATION — THE FUNDAMENTAL MIXING OPERATION
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: MATRIX MULTIPLICATION — WHY IT'S EVERYWHERE")
print("=" * 70)


print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  MATRIX MULTIPLICATION = WEIGHTED COMBINATION               │
└─────────────────────────────────────────────────────────────────┘

Recall from Part 0: a linear layer is y = x @ W + b. What does @ do?

  A @ B means:
    "Each row of the result is a WEIGHTED COMBINATION of the rows of B,
     where the weights come from the corresponding row of A."

  A is (m, k) → m "mixing recipes", each with k coefficients
  B is (k, n) → k "ingredients", each n-dimensional
  Result is (m, n) → m "mixtures", each n-dimensional

This is EXACTLY what attention does:
  - Attention weights = recipes (how much of each token?)
  - Value vectors = ingredients (what each token contributes?)
  - Attended output = mixtures (context-aware representations)
""")

A_mat = np.array([[1, 2, 3], [4, 5, 6]])
B_mat = np.array([[7, 8], [9, 10], [11, 12]])
C_mat = A_mat @ B_mat

print(f"\nMatrix multiply: (2,3) @ (3,2) → (2,2)")
print(f"  A (recipes):\n{A_mat}")
print(f"  B (ingredients):\n{B_mat}")
print(f"  C = A @ B:\n{C_mat}")
print(f"\n  C[0] = 1*[7,8] + 2*[9,10] + 3*[11,12]")
print(f"       = {A_mat[0,0]*B_mat[0] + A_mat[0,1]*B_mat[1] + A_mat[0,2]*B_mat[2]}")


# 2.2 Why matmul, not element-wise?
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  WHY MATMUL? — Three ways to multiply, one winner           │
└─────────────────────────────────────────────────────────────────┘

1. ELEMENT-WISE: C[i,j] = A[i,j] * B[i,j] → NO mixing. Isolated.
2. OUTER PRODUCT: all pairs → explodes dimensionality. Impractical.
3. MATRIX MULTIPLY: C[i,j] = sum_k A[i,k] * B[k,j] → k contracts.
   Every output depends on a WHOLE row and column. The sweet spot.

Matmul is the only efficient way to mix information across dimensions.
""")

x = np.array([[1., 2.], [3., 4.]])
y = np.array([[5., 6.], [7., 8.]])
print(f"\nElement-wise (x*y):\n{x * y}")
print(f"  → No cross-talk. x[0,0]*y[0,0], isolated.")
print(f"\nMatrix multiply (x@y):\n{x @ y}")
print(f"  → (0,1) = x[0,0]*y[0,1] + x[0,1]*y[1,1]. Information MIXED!")


# 2.3 Why matmul is fast
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.3  WHY MATMUL IS FAST — 50 years of optimization              │
└─────────────────────────────────────────────────────────────────┘

Matmul has been optimized for 50+ years by CPU/GPU designers and
algorithm researchers. Modern GPUs have "Tensor Cores" that do
4×4×4 matmul (64 multiply-adds) in ONE clock cycle.

A transformer spends >99% of FLOPs in matmul. The architecture was
DESIGNED around hardware efficiency (unlike RNNs/LSTMs which are
sequential and hard to parallelize).
""")

batch_a = np.random.randn(4, 3, 2)
batch_b = np.random.randn(4, 2, 5)
print(f"Batched matmul: (4,3,2) @ (4,2,5) → {(batch_a @ batch_b).shape}")
print(f"  → 4 independent matmuls, computed in parallel.")


# 2.4 Geometric interpretation
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.4  GEOMETRIC VIEW — Matmul transforms space                  │
└─────────────────────────────────────────────────────────────────┘

x @ W = apply linear transformation W to each row of x.

In transformers:
  x @ W_Q → "query space"  (what am I looking for?)
  x @ W_K → "key space"    (how do I describe myself?)
  x @ W_V → "value space"  (what do I contribute?)

Each projection gives the same token a different interpretation.
""")

point = np.array([1.0, 0.5])
W_proj = np.array([[2, 0], [0, 1]])
print(f"Point {point} → W → {point @ W_proj}")
print(f"  → x doubled, y kept. W defined the transformation.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: SOFTMAX — Converting Scores to Probabilities
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: SOFTMAX — THE DIFFERENTIABLE ARGMAX")
print("=" * 70)

# Bridge from matmul to softmax
print("""
┌─────────────────────────────────────────────────────────────────┐
│ FROM MATMUL TO SOFTMAX — Why we need probabilities              │
└─────────────────────────────────────────────────────────────────┘

Part 0 showed: network output = raw scores (logits). Part 2 showed:
matmul computes those scores efficiently.

But raw scores like [2.3, -0.5, 5.1] aren't useful as-is. We need:
  - Each value between 0 and 1 (interpretable as probability)
  - All values sum to exactly 1.0 (a proper distribution)
  - Higher score → higher probability (preserve ordering)

This is EXACTLY what softmax does.
""")


# 3.1 Why softmax
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  WHY SOFTMAX? — Three problems with raw scores              │
└─────────────────────────────────────────────────────────────────┘

1. INTERPRETATION: logit=5.3 vs logit=-2.1. What do they mean?
2. SUMMATION: raw numbers don't sum to anything. Can't say "70% sure."
3. TRAINING: argmax (pick biggest) has zero gradient → can't learn.

Softmax solves all three: sums to 1, preserves ordering, gradient flows.
""")


# 3.2 Why exp()
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.2  WHY exp()? — Four properties that make it the ONLY choice  │
└─────────────────────────────────────────────────────────────────┘

softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

1. POSITIVITY: exp(x) > 0 for ALL x. Outputs always positive.
2. MONOTONICITY: If a > b, exp(a) > exp(b). Ordering preserved.
3. AMPLIFICATION: exp grows faster than linear. Small differences
   in logits → large differences in probability. This is "selection."
4. DERIVATIVE: d(exp)/dx = exp(x). Simplest derivative possible.
   Gradient: ∂p_i/∂z_j = p_i(δ_ij - p_j).
   When p_i≈0.5 (uncertain): big gradient → learn fast.
   When p_i≈1.0 (confident): small gradient → don't overshoot.
""")


# 3.3 The max trick
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.3  THE MAX TRICK — Why we subtract before exp()               │
└─────────────────────────────────────────────────────────────────┘

Problem: exp(1000) ≈ 10^434. float64 max: 1.8×10^308. It OVERFLOWS.
→ infinity/infinity = NaN → training dies.

Solution: softmax(x - max(x)). Mathematically identical, but now
the largest exp is exp(0) = 1. No overflow possible.

Proof: exp(a-c)/sum(exp(a-c)) = exp(a)/sum(exp(a)) for any c.
Set c = max(x) for numerical safety.

This is the single most important numerical trick in ML.
""")


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


large_logits = np.array([1000.0, 999.0, 0.1])
naive = np.exp(large_logits) / np.sum(np.exp(large_logits))
stable = softmax(large_logits)
print(f"\nMax trick demo:")
print(f"  Naive:  {'OVERFLOW' if np.any(~np.isfinite(naive)) else np.round(naive, 4)}")
print(f"  Stable: {np.round(stable, 4)}")


# 3.4 2D softmax in attention
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.4  SOFTMAX IN ATTENTION — Each ROW is a distribution         │
└─────────────────────────────────────────────────────────────────┘

In attention, softmax applies to a MATRIX of scores (seq_len × seq_len).
Each ROW independently sums to 1.0:

  Row i: "How does token i distribute attention across all tokens?"
  Row sum must be 1 — you can't pay "more than 100% attention."
""")

np.random.seed(42)
attn_scores = np.random.randn(4, 4) * 2
attn_weights = softmax(attn_scores, axis=-1)
print(f"\nScores (4×4):\n{np.round(attn_scores, 2)}")
print(f"\nWeights (each row sums to 1):\n{np.round(attn_weights, 3)}")
print(f"Row sums: {attn_weights.sum(axis=1)}")


# 3.5 Temperature
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.5  TEMPERATURE — Controlling confidence and creativity        │
└─────────────────────────────────────────────────────────────────┘

softmax(logits / T) — divide by temperature T before softmax.

  T → 0:  one-hot (greedy, always pick best) — no creativity
  T = 1:  standard softmax (training default)
  T → ∞:  uniform (completely random) — no coherence

  Lower T = more "confident" (peakier).
  Higher T = more "creative" (flatter, explores more).

ChatGPT uses T≈0.7 for a balance of coherence and variety.
""")

scores = np.array([2.0, 1.0, 0.1, -1.0])
for T, label in [(0.2, "cold/greedy"), (1.0, "default"), (5.0, "hot/random")]:
    print(f"  T={T} ({label}): {np.round(softmax(scores / T), 4)}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: THE ATTENTION FORMULA, FULLY DECODED")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V       │
│                                                                 │
│   Q, K, V      → tensors (batch, seq, d_model)                  │
│   Q @ K^T      → matrix multiplication: pairwise similarities   │
│   / sqrt(d_k)  → scaling: keep variance from exploding          │
│   softmax(...)  → convert to attention probability distribution  │
│   @ V          → matrix multiplication: weighted sum of values  │
│                                                                 │
│   FOUR operations. All covered here. There is nothing else.     │
└─────────────────────────────────────────────────────────────────┘

Quick self-check:
  1. Why can't two linear layers without activation give more power?
  2. What does A @ B do when A is (4,3) and B is (3,2)?
  3. Why subtract max before exp in softmax?
  4. What happens at T=0.1 vs T=10?

If you can answer all four, proceed to Module 1 (Tokenization).
""")

if __name__ == "__main__":
    print("\nModule 0 complete! Next: 01_tokenization.py")
    print("Run with: uv run python course/transformer/01_tokenization.py")
