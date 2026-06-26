"""
MODULE 7: MINI GPT — A Complete Decoder-Only Transformer
==========================================================

Modules 0-6 built all the pieces. Now we assemble them into a working
language model — structurally identical to GPT-2, just smaller.

There are two fundamental transformer architectures:
  - ENCODER-DECODER (original paper, T5, BART): separate encoder and
    decoder stacks. Used for translation, summarization.
  - DECODER-ONLY (GPT, LLaMA, Claude): one stack, causal attention.
    Used for autoregressive language modeling. THIS is what we build.

WHAT YOU'LL LEARN:
   1. Why decoder-only dominates modern LLMs
   2. The complete embedding → blocks → LM head pipeline
   3. How next-token prediction shapes the architecture
   4. How autoregressive generation works (token by token)
   5. Why this architecture scales from 1K to 1T+ parameters

AFTER THIS MODULE:
   You have a working transformer that can generate text. It won't be
   GOOD (needs training), but EVERY operation is correct.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Reuse all building blocks (same implementations as Modules 4-6)
# ──────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, d_model, expansion=4):
        self.d_ff = d_model * expansion
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.02
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        batch, seq, d_model = x.shape
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        Q = Q.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = softmax(scores, axis=-1)
        out = attn @ V
        return out.transpose(0, 2, 1, 3).reshape(batch, seq, d_model) @ self.W_O


class TransformerBlock:
    def __init__(self, d_model, num_heads, ffn_expansion=4):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ffn_expansion)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attention.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


# ═══════════════════════════════════════════════════════════════════
# WHY DECODER-ONLY?
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY DECODER-ONLY WON")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ Two transformer architectures, one survivor                     │
└─────────────────────────────────────────────────────────────────┘

ENCODER-DECODER (original Transformer, T5, BART):
  - Encoder: bidirectional attention (sees full input)
  - Decoder: causal attention + cross-attention to encoder output
  - Best for: translation ("Je t'aime" → "I love you")
  - Problem: half the model is dedicated to encoding, which is
    wasted when the task is "just predict next word"

DECODER-ONLY (GPT, LLaMA, Claude, Gemini):
  - Single stack of causal attention blocks
  - No encoder, no cross-attention
  - Best for: language modeling ("Once upon a" → "time")
  - Advantage: EVERY parameter contributes to the single task of
    next-token prediction. No wasted capacity.

WHY DECODER-ONLY DOMINATES:
  1. FEWER PARAMETERS for the same depth (no encoder stack)
  2. SELF-SUPERVISED TRAINING — just predict the next word.
     All of the internet is training data. No labels needed.
  3. IN-CONTEXT LEARNING — the model learns to "understand"
     the task from the prompt itself. No task-specific heads.
  4. FLEXIBILITY — the same architecture does chat, code, math,
     reasoning, translation, summarization.

The decoder-only transformer is the simplest architecture that
can learn language. And simplicity scales.
""")


# ──────────────────────────────────────────────────────────────────
# MiniGPT IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: MiniGPT — COMPLETE IMPLEMENTATION")
print("=" * 70)


class MiniGPT:
    """A minimal GPT-style decoder-only transformer.

    PIPELINE:
        Token IDs → Token Embed + Position Embed
                  → [TransformerBlock × N] with causal mask
                  → Final LayerNorm
                  → LM Head (project to vocab logits)
                  → Softmax (to get probabilities)

    WHY final LayerNorm BEFORE LM head?
      The output of the last block may have varying scale. LayerNorm
      normalizes it before the final projection, improving training
      stability and ensuring the LM head sees well-behaved inputs.

    WHY learnable position embeddings?
      Simpler and equally effective as sinusoidal for fixed-length
      training. RoPE (Module 9) is the modern upgrade that handles
      variable lengths better.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 max_seq_len, ffn_expansion=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings: (vocab, d_model)
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Position embeddings: (max_seq, d_model)
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, ffn_expansion)
            for _ in range(num_layers)
        ]

        # Final normalization before output projection
        self.final_ln = LayerNorm(d_model)

        # LM Head: project hidden state to vocabulary logits
        # Shape: (d_model, vocab_size)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

        # Compute total parameters
        embedding_params = vocab_size * d_model + max_seq_len * d_model
        per_block = (4 * d_model * d_model  # attention
                     + 2 * d_model * ffn_expansion * d_model)  # FFN
        block_params = per_block * num_layers
        head_params = d_model * vocab_size
        total = embedding_params + block_params + head_params

        print(f"\n  MiniGPT: vocab={vocab_size}, d_model={d_model}, "
              f"heads={num_heads}, layers={num_layers}")
        print(f"  Embeddings:  {embedding_params:>10,}")
        print(f"  {num_layers} blocks:    {block_params:>10,}")
        print(f"  LM Head:     {head_params:>10,}")
        print(f"  Total:       {total:>10,}")

    def forward(self, token_ids):
        """Forward pass: token IDs → next-token logits.

        Args:
            token_ids: (batch_size, seq_len) integers in [0, vocab_size)

        Returns:
            logits: (batch_size, seq_len, vocab_size) raw scores

        WHY the same causal mask for all layers?
          The causal constraint (don't look at the future) is
          FUNDAMENTAL to autoregressive language modeling. It must
          be applied at EVERY layer, not just the first. If layer 5
          could attend to the future, the model would cheat.
        """
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, \
            f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        # 1. Token embeddings + positional embeddings
        # WHY add, not concatenate?
        #   Adding preserves d_model dimension. Concatenating would
        #   double it, requiring twice the params in every subsequent
        #   layer. Addition means the model must learn to disentangle
        #   token identity from position — which it does naturally.
        x = self.token_embed[token_ids]  # (B, S, D)
        positions = np.arange(seq_len)
        x = x + self.pos_embed[positions]  # broadcast over batch

        # 2. Causal mask — same shape, broadcast across batch and heads
        mask = np.triu(np.ones((1, 1, seq_len, seq_len)) * float('-inf'), k=1)

        # 3. Apply each transformer block
        # WHY sequential, not parallel?
        #   Each block transforms the representations. Later blocks
        #   depend on the output of earlier blocks. This is the
        #   "deep" in deep learning — each layer refines the signal.
        for block in self.blocks:
            x = block.forward(x, mask)

        # 4. Final layer norm
        x = self.final_ln.forward(x)

        # 5. Project to vocabulary
        # (B, S, D) @ (D, V) → (B, S, V)
        # Each position now has a score for every vocabulary token.
        logits = x @ self.lm_head

        return logits

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """Generate text autoregressively.

        This is the algorithm that runs when you call ChatGPT's API.

        WHY one token at a time?
          Each new token depends on ALL previous tokens. You can't
          parallelize generation across time steps because token N
          depends on the attention over tokens 0...N-1.

          This is the INFERENCE BOTTLENECK — the subject of the
          parallel "inference course" in course/inference/.

        WHY temperature != 1.0 changes output?
          Higher T → flatter distribution → more random sampling
          Lower T → peakier distribution → more deterministic
          T→0 → always pick the most likely token (greedy)

        WHY random sampling instead of always picking the best?
          Deterministic generation produces repetitive, boring text.
          Sampling introduces VARIETY. The best next token might lead
          to a boring continuation; a slightly worse token might
          enable a creative one. This is the exploration-exploitation
          tradeoff in language generation.
        """
        generated = list(prompt_ids)

        for step in range(max_new_tokens):
            # Truncate to context window
            context = generated[-self.max_seq_len:]
            context_batch = np.array([context])

            # Forward pass
            logits = self.forward(context_batch)

            # Only care about the LAST position's predictions
            # WHY the last position?
            # Position 0 predicts token 1, position 1 predicts token 2, etc.
            # For generation, we only need the prediction after the LAST
            # prompt token. Earlier predictions are for training, not
            # generation.
            next_logits = logits[0, -1, :]

            # Apply temperature and sample
            next_logits = next_logits / temperature
            probs = softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(int(next_token))

        return generated


# ──────────────────────────────────────────────────────────────────
# DEMONSTRATION
# ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("DEMONSTRATION: CREATE AND TEST MiniGPT")
print("=" * 70)

np.random.seed(42)

model = MiniGPT(
    vocab_size=100,
    d_model=32,
    num_heads=4,
    num_layers=2,
    max_seq_len=64,
    ffn_expansion=4,
)

# Shape sanity check
batch_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
logits = model.forward(batch_ids)

print(f"\n  Forward pass:")
print(f"    Input:  {batch_ids.shape}")
print(f"    Logits: {logits.shape}")
print(f"    → (batch={batch_ids.shape[0]}, seq={batch_ids.shape[1]}, vocab={logits.shape[2]})")

# Generate (random output — model is untrained!)
print(f"\n  Generating from [1, 2, 3]:")
generated = model.generate([1, 2, 3], max_new_tokens=12, temperature=0.8)
print(f"    {generated}")
print(f"    (random output — model needs training!)")


# ──────────────────────────────────────────────────────────────────
# WHY THIS ARCHITECTURE SCALES
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: WHY THIS ARCHITECTURE SCALES")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ From 23K to 1.76T parameters — same architecture                │
└─────────────────────────────────────────────────────────────────┘

Our MiniGPT:  100 vocab × 32 dim × 2 layers × 4 heads = ~23K params
GPT-2 Small:  50K vocab × 768 dim × 12 layers × 12 heads = 124M
LLaMA-7B:     32K vocab × 4096 dim × 32 layers × 32 heads = 7B
GPT-3:        50K vocab × 12288 dim × 96 layers × 96 heads = 175B

The only differences are SCALE: more layers, wider dimensions,
more heads, bigger vocabulary, more data.

WHAT CHANGES WITH SCALE:
  ✓ Emergent abilities appear (in-context learning, reasoning)
  ✓ Training becomes the bottleneck (months on thousands of GPUs)
  ✓ Inference costs dominate (KV cache, memory bandwidth)
  ✗ The ARCHITECTURE is identical
  ✗ The MATH is identical
  ✗ The training OBJECTIVE is identical (next-token prediction)

This is why understanding a 23K-param model = understanding GPT-4.
Scale doesn't change the fundamentals — it just makes them work
better and cost more.

The Chinchilla scaling law (Hoffmann et al., 2022) tells us:
  For a model with N parameters, optimal tokens ≈ 20×N.
  Our 23K model: ~460K tokens for optimal training.
  GPT-3 (175B): ~3.5 TRILLION tokens. They used ~300B — 10× less
  than optimal. This means GPT-3 is "undertrained" — bigger models
  need MORE data, not just more parameters.

LLaMA (Touvron et al., 2023) proved this: a smaller model (7B)
trained on more data (1T tokens) outperforms a larger model (13B)
trained on less data. DATA is the real bottleneck now.
""")

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Decoder-only = single stack of causal attention blocks (no encoder)
2. Architecture: Embed → [Block × N] → LayerNorm → LM Head → Softmax
3. Next-token prediction is self-supervised — no labels needed
4. Generation: sample one token, append, repeat (autoregressive)
5. Same architecture scales from 23K params to 1T+ params

Your MiniGPT is structurally complete. It has every component of GPT-2.
Next: Module 8 — How to actually TRAIN this model.
""")

if __name__ == "__main__":
    print("\nModule 7 complete! Next: 08_training.py")
