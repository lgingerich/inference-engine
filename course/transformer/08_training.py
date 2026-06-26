"""
MODULE 8: TRAINING — Loss, Backpropagation, and Optimization
==============================================================

Module 7 built a working transformer. But with random weights, it
generates gibberish. Training is the process that turns random
matrices into a language model.

This module explains the training pipeline — loss functions,
gradient flow, optimization algorithms, and hyperparameters.
Since we're in NumPy (no autograd), we focus on CONCEPTUAL
understanding with explicit calculations where possible.

WHAT YOU'LL LEARN:
   1. Why cross-entropy is THE loss for language modeling
   2. How teacher forcing enables parallel training
   3. What backpropagation actually computes (conceptually)
   4. Why AdamW is the standard optimizer
   5. Why learning rate scheduling matters (warmup + cosine)
   6. The real cost of training at scale

AFTER THIS MODULE:
   You'll understand what happens between `loss.backward()` and
   `optimizer.step()` — the engine that creates intelligence.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# BACKGROUND: What does "training" actually mean?
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHAT IS TRAINING?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING = FINDING GOOD WEIGHTS                                 │
│                                                                 │
│ 1. Start with random weights (all parameters are random)        │
│ 2. Show the model text (input sequence, no labels needed)       │
│ 3. Model predicts what comes next                               │
│ 4. Compare prediction to ACTUAL next word (compute LOSS)        │
│ 5. Figure out how to change each weight to reduce loss          │
│    (compute GRADIENTS via backpropagation)                      │
│ 6. Nudge weights in the gradient's OPPOSITE direction           │
│    (weight = weight - learning_rate × gradient)                 │
│ 7. Repeat 100K to 10B times                                    │
│ 8. Emergent behavior: the model CAN PREDICT LANGUAGE            │
└─────────────────────────────────────────────────────────────────┘

That's it. No magic. Just gradient descent on a massive scale.
Every ChatGPT response is the result of billions of tiny weight
adjustments following this exact recipe.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE TRAINING OBJECTIVE — Next Token Prediction
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: THE TRAINING OBJECTIVE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  SELF-SUPERVISED LEARNING — The text IS the label           │
└─────────────────────────────────────────────────────────────────┘

Deep learning typically needs labeled data:
  Image → [human labels "cat"]
  Audio → [human transcript "hello world"]

Language modeling is SELF-SUPERVISED:
  Input:  ["The", "cat", "sat", "on", "the"]
  Target: ["cat", "sat", "on",  "the", "mat"]

For each prefix, the next token IS the label. No human annotation
needed. All of Wikipedia, books, code, and Reddit are automatically
"labeled" training data. This is why LLMs can scale to trillions
of tokens — the data labels itself.
""")

sequence = np.array([10, 20, 30, 40, 50, 60])
inputs = sequence[:-1]   # all but last
targets = sequence[1:]   # all but first

print(f"Sequence:  {sequence}")
print(f"Input:     {inputs}  (predict next token from each position)")
print(f"Target:    {targets}  (what the model SHOULD predict)")
print(f"\n  → Position 0 (token {inputs[0]}): should predict {targets[0]}")
print(f"  → Position 4 (token {inputs[4]}): should predict {targets[4]}")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: CROSS-ENTROPY LOSS — The Universal Language Model Metric
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: CROSS-ENTROPY LOSS")
print("=" * 70)

def cross_entropy_loss(logits, targets):
    """Compute average cross-entropy loss for next-token prediction.

    logits:  (batch, seq_len, vocab_size) — model's raw scores
    targets: (batch, seq_len) — ground truth token IDs

    WHY cross-entropy and not mean squared error (MSE)?
      - MSE assumes continuous values and Gaussian errors.
        Token prediction is DISCRETE — either you picked the right
        token or you didn't. MSE is inappropriate.
      - Cross-entropy is the NATURAL loss for classification. It
        measures the negative log-likelihood of the correct class.
      - It's the same as maximizing P(correct token | context),
        which is exactly what we want the model to do.
      - Cross-entropy has nice gradient properties: when the model
        is WRONG, gradients are large (fix quickly). When it's RIGHT,
        gradients are small (don't overcorrect).

    MATHEMATICAL DEFINITION:
      L = -1/N * Σᵢ log(P(model assigns to correct tokenᵢ))

      If the model is 90% confident about the right answer:
        loss = -log(0.9) ≈ 0.105  (good)
      If the model is 10% confident about the right answer:
        loss = -log(0.1) ≈ 2.30   (bad)
      If the model is 0.01% confident:
        loss = -log(0.0001) ≈ 9.21 (very bad)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Numerically stable softmax (convert logits to probabilities)
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Extract probability of the CORRECT token at each position
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    correct_probs = probs[batch_idx, seq_idx, targets]

    # Negative log likelihood
    nll = -np.log(np.clip(correct_probs, 1e-10, 1.0))

    return np.mean(nll)


# Demonstrate
np.random.seed(42)
logits = np.random.randn(2, 5, 10)  # 2 batches, 5 positions, 10-token vocab
targets = np.random.randint(0, 10, (2, 5))

loss = cross_entropy_loss(logits, targets)

print(f"\n  Logits shape:  {logits.shape}")
print(f"  Targets shape: {targets.shape}")
print(f"  Cross-entropy loss: {loss:.4f}")

print(f"\n  Loss interpretation guide:")
print(f"    Random guessing ({logits.shape[-1]} tokens): -ln(1/{logits.shape[-1]}) = "
      f"{-np.log(1/logits.shape[-1]):.4f}")
print(f"    Very confident & correct (prob=0.9):  loss ≈ {-np.log(0.9):.4f}")
print(f"    Perfect (prob=1.0):                    loss ≈ 0.0")
print(f"    Our random model:                      loss = {loss:.4f}")
print(f"    → Close to random guessing. Expected for untrained model.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: PERPLEXITY — The Human-Readable Loss
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: PERPLEXITY")
print("=" * 70)

perplexity = np.exp(loss)

print(f"\n  Perplexity = exp({loss:.4f}) = {perplexity:.2f}")

print(f"""
  INTERPRETATION: The model is as uncertain as if it had to choose
  from {perplexity:.1f} equally likely options at each step.

  For a vocabulary of {logits.shape[-1]} tokens:
    - Random:                       perplexity ≈ {logits.shape[-1]}
    - GPT-2 Small (trained):        perplexity ≈ 30-40
    - GPT-3 (trained):              perplexity ≈ 20
    - Near-perfect language model:  perplexity → 1.0

  Perplexity is LOWER = BETTER. It measures "how surprised is the
  model on average?" A perplexity of 1 means the model is never
  surprised — it always predicts the right token with certainty.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: GRADIENT DESCENT — How Weights Actually Change
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: GRADIENT DESCENT — THE LEARNING ALGORITHM")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  BACKPROPAGATION — Computing "which way to move each weight"│
└─────────────────────────────────────────────────────────────────┘

The gradient ∂L/∂w tells us: "If I increase weight w by epsilon,
how much does the loss change?"

To compute this for every weight in a 175B-parameter model, we use
backpropagation (the chain rule applied to the computation graph).

CONCEPTUALLY:
  1. Forward pass: compute loss L
  2. Start from the loss and work BACKWARD through the network
  3. At each operation, the chain rule tells us how to combine
     the incoming gradient with the local gradient
  4. The result is ∂L/∂w for every weight

For attention (simplified):
  ∂L/∂W_Q = (∂L/∂output) · (∂output/∂attn_weights) · (∂attn_weights/∂scores)
           · (∂scores/∂Q) · (∂Q/∂W_Q)

Each · is a chain rule step. In practice, autograd frameworks
(PyTorch, JAX) do this automatically. We never implement it
by hand. But understanding the FLOW is essential.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  THE UPDATE RULE — Gradient descent in one line            │
└─────────────────────────────────────────────────────────────────┘

  weight = weight - learning_rate × gradient

The learning rate controls step size:
  - Too large: overshoots minimum, loss oscillates or diverges
  - Too small: converges too slowly, may get stuck in local minima
  - Just right: the art of training

For transformers, typical learning rates:
  - Base LR: 3e-4 (GPT-2/3), 1.5e-4 (LLaMA)
  - For smaller models (<100M params): can use up to 6e-4
  - For very large models (>10B params): often 1e-4 or lower

The learning rate is NOT constant! It follows a SCHEDULE:
  1. WARMUP: linearly increase from 0 to peak LR (first 1-5% of steps)
  2. DECAY: gradually decrease to ~10% of peak (cosine or linear)
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: OPTIMIZERS — Why AdamW is the standard
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: AdamW — THE STANDARD OPTIMIZER")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  Why not just plain SGD?                                   │
└─────────────────────────────────────────────────────────────────┘

SGD:  weight = weight - lr × gradient

Problems for transformers:
  1. DIFFERENT SCALES: Some parameters (embedding weights) have
     different gradient magnitudes than others (FFN weights).
     A single learning rate doesn't work well for all.

  2. SPARSE GRADIENTS: Some parameters are updated rarely
     (embedding for rare tokens). They need a different effective
     learning rate than frequent ones.

  3. NOISY GRADIENTS: Each mini-batch gives a noisy estimate of
     the true gradient. SGD with momentum helps by averaging
     over recent batches.

┌─────────────────────────────────────────────────────────────────┐
│ 5.2  AdamW — Adaptive moments with decoupled weight decay       │
└─────────────────────────────────────────────────────────────────┘

AdamW maintains two running averages per parameter:
  m_t = β₁·m_{t-1} + (1-β₁)·g_t     (first moment — momentum)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²    (second moment — variance)

Update:  w = w - lr × (m_t / √v_t) - lr × λ × w

  m_t / √v_t: adaptively scaled gradient (normalizes by variance)
  lr × λ × w: decoupled weight decay (L2 regularization)

Why decoupled weight decay?
  - L2 regularization adds λ·||w||² to the loss
  - But in Adam, this gets modified by the adaptive scaling
  - Decoupled weight decay applies regularization DIRECTLY to
    the weights, independent of the adaptive learning rate
  - Empirically better than Adam + L2 for transformers

Typical hyperparameters:
  β₁ = 0.9, β₂ = 0.95 (slightly higher than image models)
  λ = 0.1 (weight decay — yes, this is high!)
  ε = 1e-8 (for numerical stability)
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 6: THE TRAINING LOOP — Pseudocode for Reality
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 6: THE TRAINING LOOP")
print("=" * 70)

print("""
Here's what a REAL transformer training loop looks like:

```python
model = MiniGPT(vocab_size=50257, d_model=768, num_layers=12, ...)
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=2000, total_steps=100000)

# Gradient accumulation: simulate larger batch size with limited memory
accumulation_steps = 4

for step in range(total_steps):
    total_loss = 0

    for micro_step in range(accumulation_steps):
        batch = next(data_loader)  # e.g., (64 sequences × 1024 tokens)

        # Teacher forcing: predict ALL next tokens in parallel
        input_ids = batch[:, :-1]   # (64, 1023)
        targets = batch[:, 1:]     # (64, 1023)

        # Forward
        logits = model(input_ids)  # (64, 1023, 50257)
        loss = cross_entropy(logits, targets)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()  # accumulate gradients

        total_loss += loss.item()

    # Gradient clipping: prevent explosion
    # Typical max norm: 1.0
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Update learning rate
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}: loss={total_loss:.4f}, lr={scheduler.get_lr():.2e}")
```

CRITICAL DETAILS:

  Gradient Accumulation:
    - You want batch_size=512 but GPU memory only fits 64
    - Solution: run 8 forward/backward passes, SUM gradients,
      THEN update weights. The result is identical to batch=512.

  Gradient Clipping:
    - Occasionally, a batch produces enormous gradients (bad data,
      unstable training state). Without clipping, one bad batch
      can destroy weeks of training progress.
    - Typical max_norm=1.0 clips the gradient L2 norm.

  Mixed Precision (FP16/BF16):
    - Store weights in FP32, compute in FP16/BF16
    - 2× speedup, 2× memory savings
    - Requires gradient scaling to handle small FP16 values
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 7: WHY TRAINING COSTS MILLIONS
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 7: THE REAL COST OF TRAINING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING COSTS — Why you need a GPU cluster                     │
└─────────────────────────────────────────────────────────────────┘

For a 7B parameter model (LLaMA scale):

  MEMORY PER GPU:
    - Model weights (FP16):         14 GB
    - Optimizer states (AdamW):     42 GB (2× moment buffers + FP32 master)
    - Activations (seq=2048):      ~100 GB (stored for backward pass)
    - Total: ~156 GB for batch_size=1

    → Even an A100 80GB can't fit one batch. You need 4-8 GPUs
      with model parallelism, or use activation checkpointing
      (recompute activations instead of storing them).

  COMPUTE:
    - Training FLOPs ≈ 6 × params × tokens  (Kaplan et al., 2020)
    - For 7B model, 1T tokens: 6 × 7e9 × 1e12 = 4.2 × 10^22 FLOPs
    - On 2048 A100 GPUs: ~20 days
    - Estimated cost (cloud): $300K-$500K

  GPT-3 (175B):
    - Training tokens: ~300B (Meena et al.)
    - FLOPs: 6 × 175e9 × 300e9 ≈ 3.15 × 10^23
    - Cost: ~$4.6M (Microsoft's estimate, probably optimistic)
    - Carbon: ~552 tons CO2 (training only, US grid average)

  The rule of thumb: training a SOTA LLM costs between $1M and $100M.
  Inference costs are recurring (per query or per token) and can
  exceed training costs over the lifetime of a deployed model.
""")

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Training objective: predict next token (self-supervised)
2. Cross-entropy loss: -log P(correct token) — the universal metric
3. Perplexity = exp(loss) — "how many choices is the model uncertain among?"
4. Backpropagation: compute ∂L/∂w for every weight via chain rule
5. AdamW: adaptive learning rates + decoupled weight decay
6. LR schedule: warmup → peak → cosine decay
7. Gradient clipping + accumulation + mixed precision = production training

Training a language model is just a gradient descent loop, repeated
billions of times, on a mountain of text, powered by a warehouse of GPUs.

Next: Module 9 — Advanced Topics (what modern LLMs do differently)
""")

if __name__ == "__main__":
    print("\nModule 8 complete! Next: 09_advanced_topics.py")
