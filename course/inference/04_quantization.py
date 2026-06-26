"""
INFERENCE MODULE 4: QUANTIZATION — Making Models Fit in Memory
================================================================

A 7B parameter model in FP16 is ~14 GB. An RTX 4090 has 24 GB.
After loading the model, you have ~10 GB left for the KV cache.
What if you want to run a 70B model?

Quantization reduces the precision of model weights from 16-bit
floating point to 8-bit or 4-bit integers. A 4-bit model is ~4×
smaller — but only if we can do it without destroying quality.

This module builds quantization from scratch: why naive rounding fails,
how group-wise quantization fixes it, and how to dequantize on-the-fly
so you keep the memory savings during inference.

WHAT YOU'LL LEARN:
   1. Why FP16 → INT4 saves 4× memory, but at a quality cost
   2. Why naive per-tensor quantization FAILS (the outlier problem)
   3. How group-wise quantization solves this (per-group scales)
   4. How to dequantize on-the-fly during inference (never expand to FP16)
   5. The real-world formats: GPTQ, AWQ, GGUF — how they improve on our code

AFTER THIS MODULE:
   You'll understand exactly what those .gguf files do, what
   "Q4_K_M" means, and why your 7B model suddenly fits on a laptop.
"""

import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY QUANTIZATION EXISTS AT ALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY QUANTIZATION?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE MEMORY PROBLEM — Models are too big for most hardware       │
└─────────────────────────────────────────────────────────────────┘

Every weight in a transformer is stored as a floating-point number.
The precision determines both memory usage and quality:

    FORMAT   |  Bits  |  Exponent  |  Mantissa  |  Memory per weight
    ─────────┼────────┼────────────┼────────────┼───────────────────
    FP32     |  32    |  8 bits    |  23 bits   |  4 bytes
    FP16     |  16    |  5 bits    |  10 bits   |  2 bytes
    BF16     |  16    |  8 bits    |  7 bits    |  2 bytes
    INT8     |   8    |  N/A       |  N/A       |  1 byte
    INT4     |   4    |  N/A       |  N/A       |  0.5 bytes

For a 7B parameter model:
    FP32:  7B × 4 bytes = 28 GB  (barely fits 80GB A100 after KV cache)
    FP16:  7B × 2 bytes = 14 GB  (fits A100, doesn't fit RTX 4090 + cache)
    INT8:  7B × 1 byte  =  7 GB  (fits RTX 4090 with room for cache)
    INT4:  7B × 0.5 B   =  3.5 GB (fits almost any GPU!)

┌─────────────────────────────────────────────────────────────────┐
│ WHY QUANTIZATION MATTERS FOR INFERENCE SPECIFICALLY             │
└─────────────────────────────────────────────────────────────────┘

Quantization helps inference TWICE:

  1. MEMORY: A quantized model fits on cheaper/smaller GPUs.
     This is the most obvious benefit: without INT4, you can't
     run a 70B model on consumer hardware at all.

  2. BANDWIDTH: Module 0 showed that inference is memory-bandwidth-
     bound. Loading 4-bit weights from HBM takes 4× less time than
     loading 16-bit weights. → ~4× faster generation!

The question: can we represent each weight with just 4 bits (16
possible values) while keeping the model accurate?
""")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_per_tensor(weights, bits=8):
    """Naive per-tensor quantization — ROUND the whole matrix uniformly.

    Maps the global [min, max] range to integer values [0, 2^bits-1].
    Every weight in the matrix uses the SAME scale factor.

    PROBLEM: one outlier stretches the range for EVERYTHING.
    If weights = [0.1, 0.2, 0.3, 100.0], the range is dominated by
    100.0, wasting precision on the small values that are actually
    clustered near zero.
    """
    w_min = weights.min()
    w_max = weights.max()
    scale = (w_max - w_min) / (2**bits - 1)
    zero_point = np.round(-w_min / scale)

    quantized = np.clip(np.round(weights / scale + zero_point), 0, 2**bits - 1)
    quantized = quantized.astype(np.uint8)
    return quantized, scale, zero_point


def dequantize_per_tensor(quantized, scale, zero_point):
    """Reverse per-tensor quantization: integer → approximate float."""
    return (quantized.astype(np.float64) - zero_point) * scale


def quantize_groupwise(weights, bits=4, group_size=128):
    """Group-wise quantization — the ACTUAL approach used in production.

    WHY GROUPS? Instead of one scale for the entire matrix, we split
    weights into GROUPS (e.g., 128 columns per group) and quantize each
    group independently with its OWN scale factor.

    This means an outlier only affects ITS group — not the entire row.
    For most groups (without outliers), the scale is tight and 4 bits
    provide excellent precision.

    WHY group_size=128?
      - Smaller groups = more scales to store (metadata overhead)
      - Larger groups = worse precision (bigger range per group)
      - 128 is the empirically determined sweet spot from GPTQ/AWQ

    Memory breakdown for 4-bit with group_size=128:
      - Weight bytes: 4/8 = 50% of FP16 storage
      - Scale storage: 16 bits per scale × 1 scale / 128 weights
                      = 0.125 bits per weight
      - Total: ~4.125 bits per weight (vs 16 bits for FP16)
      - Compression: ~3.9×
    """
    original_shape = weights.shape
    weights_flat = weights.reshape(-1, original_shape[-1])

    out_features, in_features = weights_flat.shape
    num_groups = max(1, in_features // group_size)

    # Pad to group_size boundary so every group is full
    padded_in = num_groups * group_size
    if in_features < padded_in:
        pad = padded_in - in_features
        weights_flat = np.pad(weights_flat, ((0, 0), (0, pad)), mode='constant')

    max_val = 2**bits - 1

    quantized = np.zeros((out_features, padded_in), dtype=np.uint8)
    scales = np.zeros((out_features, num_groups), dtype=weights.dtype)
    zeros = np.zeros((out_features, num_groups), dtype=weights.dtype)

    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = weights_flat[row, start:end]

            g_min = group.min()
            g_max = group.max()
            g_range = max(g_max - g_min, 1e-8)

            scale = g_range / max_val
            scale = max(scale, 1e-8)
            zero_point = np.round(-g_min / scale)
            zero_point = np.clip(zero_point, 0, max_val)

            scales[row, g] = scale
            zeros[row, g] = zero_point

            q = np.round(group / scale + zero_point)
            q = np.clip(q, 0, max_val)
            quantized[row, start:end] = q.astype(np.uint8)

    return quantized, scales, zeros, original_shape, group_size


def dequantize_groupwise(quantized, scales, zeros, original_shape, group_size):
    """Reverse group-wise quantization."""
    out_features = original_shape[0]
    in_features = original_shape[1]
    num_groups = scales.shape[1]

    weights = np.zeros((out_features, in_features), dtype=np.float64)
    q_float = quantized.astype(np.float64)

    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            w = (q_float[row, start:end] - zeros[row, g]) * scales[row, g]
            weights[row, start:end] = w

    return weights.reshape(original_shape)


def quantized_matmul(x, q_weight, scales, zeros, group_size):
    """Matrix multiply with ON-THE-FLY dequantization.

    WHY ON-THE-FLY? If we dequantized the ENTIRE weight matrix back to
    FP16 before multiplying, we'd lose the memory savings — we'd be back
    to storing FP16 weights. Instead, we dequantize one GROUP at a time,
    multiply that group, accumulate, and move to the next group.

    This keeps the weight matrix in INT4 in memory while still producing
    the correct FP16 result. Each group is dequantized just-in-time,
    used, and discarded.

    Args:
        x: (..., in_features) input
        q_weight: quantized weight matrix (out_features, padded_in_features)
        scales: (out_features, num_groups) per-group scales
        zeros: (out_features, num_groups) per-group zero points
        group_size: columns per quantization group

    Returns:
        y: (..., out_features) result
    """
    out_features, padded_in = q_weight.shape
    num_groups = scales.shape[1]

    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])
    batch_size = x_flat.shape[0]

    if x_flat.shape[1] < padded_in:
        x_flat = np.pad(x_flat, ((0, 0), (0, padded_in - x_flat.shape[1])),
                        mode='constant')

    y = np.zeros((batch_size, out_features), dtype=np.float64)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size

        # Dequantize JUST this group (temporary FP16, discarded after use)
        w_group = (q_weight[:, start:end].astype(np.float64) - zeros[:, g:g+1]) * scales[:, g:g+1]

        x_group = x_flat[:, start:end]
        y += x_group @ w_group.T

    return y.reshape(*original_shape[:-1], out_features)


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: NUMERICAL PRECISION — The Memory Tradeoff
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: NUMERICAL PRECISION — THE MEMORY TRADEOFF")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 1.1  How floating point numbers work
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  FLOATING POINT vs INTEGER — What precision means           │
└─────────────────────────────────────────────────────────────────┘

Floating point numbers (FP16, FP32) store:
  sign × 2^exponent × (1 + mantissa)

This gives them a HUGE dynamic range: FP16 can represent numbers
from ~0.00000006 to 65,504. But the PRECISION is limited:
  - FP16: ~3-4 decimal digits of precision
  - FP32: ~7 decimal digits of precision

Integers (INT8, INT4) are uniform: they divide a range into equal
steps. With N bits, you get 2^N equally-spaced values:

  INT8: 256 values, e.g. [0, 1, 2, ..., 255]
  INT4: 16 values,  e.g. [0, 1, 2, ..., 15]

The key insight: neural network weights DON'T need 4 decimal digits
of precision for every single number. They need to represent the
RELATIVE differences between weights. As long as the quantization
preserves these relative differences, the model's behavior stays
nearly identical.
""")

# Show the precision difference
print("Concrete precision comparison:")
print(f"  FP16:  1/2^10 ≈ {1/2**10:.6f}  (mantissa precision)")
print(f"  INT8:  1/256  ≈ {1/256:.6f}   (uniform precision)")
print(f"  INT4:  1/16   ≈ {1/16:.6f}    (uniform precision)")
print(f"  → INT4 is {1/16 / (1/2**10):.0f}× coarser than FP16 mantissa.")
print(f"  → But neural nets are ROBUST to small perturbations —")
print(f"    that's why quantization works at all.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: NAIVE QUANTIZATION — Why It Fails
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: WHY NAIVE QUANTIZATION FAILS — The Outlier Problem")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 2.1  The outlier problem — One bad value ruins everything
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  THE OUTLIER PROBLEM — One weight can destroy precision     │
└─────────────────────────────────────────────────────────────────┘

Imagine you have 128 weights that are mostly between -1 and +1,
but ONE weight is 25.0. Per-tensor quantization maps the entire
range [-1, 25] to 16 integer values (for INT4):

  Scale = (25 - (-1)) / 15 = 1.73
  → Each integer step represents 1.73 units
  → The 127 small weights (-1 to +1) all map to steps 0 or 1
  → They're ALL quantized to nearly the same value!
  → All fine-grained differences between them are LOST.

The outlier has DESTROYED the precision for 127 other weights
because they all share the same scale factor.
""")

np.random.seed(42)
weights = np.random.randn(4, 128) * 0.5  # mostly ~0
weights[2, 50] = 25.0  # ONE outlier!

print(f"\nSample weights (4 rows × 128 cols):")
print(f"  99% of values in: [{np.percentile(weights, 0.5):.3f}, "
      f"{np.percentile(weights, 99.5):.3f}]")
print(f"  One outlier at: {weights[2, 50]:.1f}")
print(f"  → The outlier stretches the range 25× wider!")

# Naive per-tensor quantization
q_w, scale, zp = quantize_per_tensor(weights, bits=4)
dq_w = dequantize_per_tensor(q_w, scale, zp)
error = np.abs(weights - dq_w)

print(f"\nNaive per-tensor INT4 quantization:")
print(f"  Scale factor: {scale:.4f}")
print(f"  → Because range=[{weights.min():.1f}, {weights.max():.1f}], "
      f"each integer step = {scale:.4f}")
print(f"  → The outlier forces us to waste 15 levels on a huge range.")
print(f"  Mean absolute error: {error.mean():.4f}")
print(f"  Max absolute error:  {error.max():.4f}")

# Show that small values collapse
small_mask = np.abs(weights) < 1.0
small_original = weights[small_mask]
small_dequant = dq_w[small_mask]
print(f"  Small-value original std: {small_original.std():.4f}")
print(f"  Small-value dequant std: {small_dequant.std():.4f}")
print(f"  → The small values' VARIANCE is lost! They're all nearly equal.")


# ═══════════════════════════════════════════════════════════════════
# 2.2  The fix — Group-wise quantization
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  GROUP-WISE QUANTIZATION — Each group gets its own scale    │
└─────────────────────────────────────────────────────────────────┘

Instead of one global scale, split the row into groups of 128
columns. Each group gets its OWN scale factor. The outlier in
group 2 only affects group 2 — groups 0, 1, and 3 keep their
tight scales and excellent precision.
""")

q_gw, scales_gw, zeros_gw, orig_shape, gs = quantize_groupwise(
    weights, bits=4, group_size=32)
dq_gw = dequantize_groupwise(q_gw, scales_gw, zeros_gw, orig_shape, gs)
error_gw = np.abs(weights - dq_gw)

print(f"\nGroup-wise INT4 quantization (group_size=32):")
print(f"  Number of groups per row: {scales_gw.shape[1]}")
print(f"  Per-group scale range: [{scales_gw.min():.4f}, {scales_gw.max():.4f}]")
print(f"  → The outlier's group has a large scale; other groups have tiny scales")
print(f"  Mean absolute error: {error_gw.mean():.6f}")
print(f"  Max absolute error (in outlier group): {error_gw.max():.6f}")

# Check the small values in the non-outlier groups
# Row 2, columns 0-31 (group 0 — no outlier)
small_in_group0 = weights[2, :32]
small_dequant_group0 = dq_gw[2, :32]
print(f"  Non-outlier group (row 2, cols 0-31):")
print(f"    Original std: {small_in_group0.std():.4f}")
print(f"    Dequant std:  {small_dequant_group0.std():.4f}")
print(f"    → Variance is PRESERVED because the group's scale is tight!")

print(f"\nComparison:")
print(f"  Naive (per-tensor) MAE:    {error.mean():.4f}")
print(f"  Group-wise MAE:            {error_gw.mean():.6f}")
print(f"  Improvement:               {error.mean() / max(error_gw.mean(), 1e-10):.0f}× better!")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: QUANTIZING A REAL MODEL — Does It Still Work?
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: QUANTIZED INFERENCE — Does MiniGPT Survive Quantization?")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 3.1  Quantize all weights, then run inference
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)
model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                num_layers=2, max_seq_len=64)

# Collect original weights
original_weights = {}
original_weights['token_embed'] = model.token_embed.copy()
original_weights['pos_embed'] = model.pos_embed.copy()
original_weights['lm_head'] = model.lm_head.copy()

for i, block in enumerate(model.blocks):
    attn = block.attention
    ffn = block.ffn
    original_weights[f'layer_{i}_W_Q'] = attn.W_Q.copy()
    original_weights[f'layer_{i}_W_K'] = attn.W_K.copy()
    original_weights[f'layer_{i}_W_V'] = attn.W_V.copy()
    original_weights[f'layer_{i}_W_O'] = attn.W_O.copy()
    original_weights[f'layer_{i}_W1'] = ffn.W1.copy()
    original_weights[f'layer_{i}_W2'] = ffn.W2.copy()

# Baseline prediction
test_input = np.array([[1, 2, 3, 4, 5]])
logits_original = model.forward(test_input)
top_tokens_original = np.argsort(logits_original[0, -1, :])[-5:][::-1]
print(f"\nOriginal model top-5 predictions: {top_tokens_original}")

# Quantize ALL matrices to INT4 with group_size=32
print(f"Quantizing all weight matrices to INT4 (group_size=32)...")
for name, w in original_weights.items():
    q_w, sc, zp, osh, gs = quantize_groupwise(w, bits=4, group_size=32)
    dq_w = dequantize_groupwise(q_w, sc, zp, osh, gs)

    # Replace model weights with dequantized versions
    if name == 'token_embed':
        model.token_embed = dq_w
    elif name == 'pos_embed':
        model.pos_embed = dq_w
    elif name == 'lm_head':
        model.lm_head = dq_w
    elif name.startswith('layer_'):
        parts = name.split('_')
        layer_idx = int(parts[1])
        weight_name = parts[2]
        block = model.blocks[layer_idx]
        if weight_name in ('W_Q', 'W_K', 'W_V', 'W_O'):
            setattr(block.attention, weight_name, dq_w)
        elif weight_name in ('W1', 'W2'):
            setattr(block.ffn, weight_name, dq_w)

# Run quantized inference
logits_quantized = model.forward(test_input)
top_tokens_quantized = np.argsort(logits_quantized[0, -1, :])[-5:][::-1]
print(f"Quantized model top-5 predictions: {top_tokens_quantized}")

overlap = len(set(top_tokens_original) & set(top_tokens_quantized))
print(f"\nPrediction overlap: {overlap}/5 tokens match")
print(f"  → INT4 quantization preserves the ROUGH prediction ordering")
print(f"  → With a real trained model, overlap is much higher")
print(f"  → INT8 would give near-perfect overlap (essentially lossless)")

# Calculate the memory savings
original_bytes = sum(w.nbytes for w in original_weights.values())
quantized_bytes = original_bytes / 4  # FP64 → INT4 is 4× smaller (roughly)
print(f"  Original weights: {original_bytes / 1024:.0f} KB (FP64)")
print(f"  Quantized weights: ~{quantized_bytes / 1024:.0f} KB (INT4)")
print(f"  Memory savings: {(1 - quantized_bytes/original_bytes) * 100:.0f}%")


# ═══════════════════════════════════════════════════════════════════
# 3.2  On-the-fly dequantization — Memory savings in practice
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.2  THE KEY TRICK — On-the-fly dequantization                 │
└─────────────────────────────────────────────────────────────────┘

In the demonstration above, we dequantized ALL weights back to FP64
before running inference. This used MORE memory than the original!

In production, you NEVER fully dequantize. Instead, for every matmul:
  1. Load a SMALL block of quantized weights from HBM
  2. Dequantize just that block to FP16 in SRAM
  3. Multiply with the input
  4. Accumulate the result
  5. Discard the FP16 block and move to the next

This is what quantized_matmul() does. The weights stay in INT4 in
GPU memory (HBM) the entire time. Only small chunks are temporarily
expanded to FP16 in the fast SRAM for computation.
""")

# Demonstrate on-the-fly matmul
x = np.random.randn(2, 128).astype(np.float64)
W = np.random.randn(256, 128).astype(np.float64) * 0.5  # note: transposed for quantize_groupwise

# Standard matmul
y_standard = x @ W.T  # (2, 256)

# Quantize W, then use on-the-fly dequantization
q_W, scales_W, zeros_W, orig_shape_W, gs_W = quantize_groupwise(W, bits=4, group_size=32)
y_quantized = quantized_matmul(x, q_W, scales_W, zeros_W, gs_W)

error = np.abs(y_standard - y_quantized)
print(f"\nOn-the-fly quantized matmul: (2, 128) @ (128, 256)^T = (2, 256)")
print(f"  Shape: {y_quantized.shape}")
print(f"  Mean absolute error: {error.mean():.6f}")
print(f"  Relative error: {error.mean() / max(np.abs(y_standard).mean(), 1e-10) * 100:.2f}%")
print(f"  Memory for W: {W.nbytes:,} bytes (FP64) → "
      f"{q_W.nbytes + scales_W.nbytes + zeros_W.nbytes:,} bytes (INT4 storage)")
print(f"  → Weights stay compressed in memory. Only small blocks are expanded.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: WHY 4-BIT WORKS — Information Theory
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: WHY 4-BIT WORKS AT ALL — The Distribution Answer")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  WEIGHTS ARE NOT UNIFORMLY DISTRIBUTED                     │
└─────────────────────────────────────────────────────────────────┘

At first glance: "How can you represent 7 billion numbers with only
16 possible values each? That's only 4 bits × 7B = 28 GB of
information for a model that has 14 GB of FP16 weights. Are we
throwing away 50% of the information?"

The answer: neural network weights are HIGHLY REDUNDANT.

  Distribution of weights in a typical transformer layer:

      count ↑
            │    ╱‾‾‾‾╲
            │   ╱  most  ╲
            │  ╱  weights  ╲          ╱╲  ← very few
            │ ╱  near zero  ╲        ╱  ╲    large weights
            │╱               ╲──────╱    ╲───
            └────────────────────────────────→ value
           -1          0          +1       +5

  Most weights are clustered near 0 with a Gaussian-like distribution.
  The information content (entropy) is much less than 16 bits per weight.

  Within each group of 128 weights, the range is SMALL — typically
  ±0.5 from the mean. With 4 bits (16 levels), you can distinguish
  weights that are ~0.06 apart within a ±0.5 range.

  For well-trained models, weights that differ by <0.06 in a small
  group usually encode redundant information — rounding them to the
  same value has negligible effect on the output.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  EMPIRICAL RESULTS — What the literature shows             │
└─────────────────────────────────────────────────────────────────┘

Perplexity increase after quantization (lower is better):

  Method        |  INT8   |  INT4   |  INT3   |  Notes
  ──────────────┼─────────┼─────────┼─────────┼───────────────────
  Naive rounding | +0.05   | +3.2    | +15.0   | Unusable at 4-bit
  GPTQ          | +0.03   | +0.5    | +2.1    | Good at 4-bit
  AWQ           | +0.02   | +0.15   | +1.0    | Excellent at 4-bit
  GGUF Q4_K_M   | +0.03   | +0.3    | N/A     | Great for CPU

  Perplexity 5.0 → 5.3: negligible quality loss
  Perplexity 5.0 → 10.0: model is ~2× more "surprised" by every token

The key innovation that makes INT4 practical:
  - GPTQ: error compensation by adjusting remaining weights
  - AWQ: identify important channels and preserve them
  - GGUF: mixed precision (some layers at higher precision)

Our group-wise implementation is the FOUNDATION. GPTQ/AWQ add
sophisticated error minimization on top of the same group-wise
structure.
""")


# Restore original model for cleanliness
model.token_embed = original_weights['token_embed']
model.pos_embed = original_weights['pos_embed']
model.lm_head = original_weights['lm_head']
for i, block in enumerate(model.blocks):
    block.attention.W_Q = original_weights[f'layer_{i}_W_Q']
    block.attention.W_K = original_weights[f'layer_{i}_W_K']
    block.attention.W_V = original_weights[f'layer_{i}_W_V']
    block.attention.W_O = original_weights[f'layer_{i}_W_O']
    block.ffn.W1 = original_weights[f'layer_{i}_W1']
    block.ffn.W2 = original_weights[f'layer_{i}_W2']


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: THE REAL-WORLD QUANTIZATION LANDSCAPE
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: THE REAL QUANTIZATION FORMATS")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  GPTQ — Optimal Brain Quantization                         │
└─────────────────────────────────────────────────────────────────┘

GPTQ (Frantar et al., 2023) quantizes weights column-by-column,
but after quantizing each column, it ADJUSTS the remaining columns
to COMPENSATE for the quantization error. This is like solving
a system of linear equations to redistribute the error:

  "Quantize weight w_ij → introduces error ε.
   Adjust weights w_i(j+1)...w_in to cancel out ε."

This guarantees minimal output error in the least-squares sense.
Result: INT4 with typically <1% perplexity increase.

┌─────────────────────────────────────────────────────────────────┐
│ 5.2  AWQ — Activation-Aware Weight Quantization                │
└─────────────────────────────────────────────────────────────────┘

AWQ (Lin et al., 2023) observes: NOT ALL WEIGHTS ARE EQUALLY
IMPORTANT. Weights corresponding to input channels with large
activations contribute more to the output.

AWQ's insight: SCALE UP the important channels BEFORE quantization,
then scale them back AFTER. This preserves the important weights'
relative precision while letting less important weights be quantized
more aggressively.

Result: INT4 with NEAR-ZERO quality loss for LLaMA-family models.
This is why AWQ is the preferred format for GPU serving.

┌─────────────────────────────────────────────────────────────────┐
│ 5.3  GGUF — llama.cpp's format for CPU inference               │
└─────────────────────────────────────────────────────────────────┘

GGUF stores quantized models as a self-contained file with metadata.
Supports many quantization types:
  Q4_0: basic 4-bit, group_size=32
  Q4_K_M: "knowledgeable medium" — 4-bit with careful distribution
          between small and large weights. The community favorite.
  Q5_K_M: 5-bit version, slightly better quality, slightly larger
  Q8_0: 8-bit, essentially lossless

GGUF's innovation: it quantizes the KV cache too (INT8), not just
weights. For long contexts, the KV cache dominates memory — 8-bit
caching doubles the effective batch size.

┌─────────────────────────────────────────────────────────────────┐
│ 5.4  HARDWARE QUANTIZATION — FP8 and FP4 on H100/B200          │
└─────────────────────────────────────────────────────────────────┘

NVIDIA's H100 has hardware FP8 support — the tensor cores natively
do FP8 matmul. This means 2× faster matmuls without software
emulation. The B200 adds FP4 hardware support.

FP8 differs from INT8: it's floating point (keeps dynamic range)
rather than integer (needs explicit scale/zero). This makes
quantization-aware training easier — no need to find per-group
scales; the exponent bits handle the range naturally.

Hardware trends: quantization is moving from a desperate memory-
saving hack to a FIRST-CLASS feature of GPU design.
""")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ QUANTIZATION — The memory-bandwidth multiplier                  │
│                                                                 │
│ 1. FP16 → INT4 = 4× smaller model → 4× less HBM traffic        │
│    → ~4× faster inference (memory-bandwidth-bound, remember?)   │
│                                                                 │
│ 2. NAIVE per-tensor quantization FAILS because one outlier     │
│    stretches the scale for ALL weights, destroying precision.   │
│                                                                 │
│ 3. GROUP-WISE quantization fixes this: each group of ~128       │
│    weights gets its own scale. Outliers only hurt their group.  │
│                                                                 │
│ 4. ON-THE-FLY dequantization: keep weights in INT4 in HBM,     │
│    expand small blocks to FP16 in SRAM for computation.         │
│    Preserves memory savings during inference.                  │
│                                                                 │
│ 5. GPTQ adds error compensation. AWQ adds activation awareness. │
│    GGUF optimizes for CPU. All build on group-wise quantization.│
│                                                                 │
│ 6. INT8 is essentially lossless. INT4 with modern techniques    │
│    loses <1% quality. INT3 is usable with AWQ.                  │
│                                                                 │
│ IMPACT ON INFERENCE: works hand-in-hand with KV cache.          │
│    Smaller weights = faster per-token generation.               │
│    Smaller weights = more room for KV cache = longer contexts.  │
└─────────────────────────────────────────────────────────────────┘

Next: Module 5 — Attention Optimizations (FlashAttention & PagedAttention)
""")

if __name__ == "__main__":
    print("\nModule 4 complete! Next: i05_attention_optimizations.py")
    print("Run with: uv run python course/inference/i05_attention_optimizations.py")
