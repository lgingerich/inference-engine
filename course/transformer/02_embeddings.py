"""
MODULE 2: EMBEDDINGS — How Numbers Become Meaning
===================================================

Module 1 gave us token IDs: [15496, 3797, 6159, ...]. But these
are just arbitrary labels — ID 15496 isn't "more" or "less" than
ID 3797. They have no geometry, no similarity, no meaning.

Embeddings turn each token into a vector in a high-dimensional
space where distance = meaning. Similar words cluster together.
Arithmetic works. "king - man + woman ≈ queen" is real.

WHAT YOU'LL LEARN:
   1. Why an embedding is literally just a lookup table (no magic)
   2. How a table of random numbers LEARNS to encode meaning
   3. Why cosine similarity works and what high dimensions give us
   4. Why embedding dimension IS the model's "width" everywhere
   5. Why weight tying saves parameters and improves training

AFTER THIS MODULE:
   You'll understand the geometric intuition behind every token's
   vector — and why the rest of the transformer just moves these
   vectors around in space.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: The geometry of language
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: THE GEOMETRY OF MEANING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ WHY VECTORS? — The distributed representation hypothesis        │
└─────────────────────────────────────────────────────────────────┘

Think about how you describe a word's meaning:
  "dog" → animal, furry, pet, loyal, four-legged, barks...
  "cat" → animal, furry, pet, independent, four-legged, meows...

These are FEATURES. Now imagine encoding each word as a list of
numbers where each position represents a "feature dimension":

  dog  = [0.8, 0.9, 0.7, 0.6, 0.8, 0.9, ...]
  cat  = [0.7, 0.8, 0.3, 0.9, 0.8, 0.1, ...]
          ↑    ↑    ↑    ↑    ↑    ↑
         animal furry pet indep 4leg sound

This is what embeddings ARE — but instead of hand-picking the
features (which is impossible for 50K words), we let the model
DISCOVER the features through training. The model learns to
arrange words in space such that semantically similar words
are close together.

This is called the "distributional hypothesis" (Firth, 1957):
  "You shall know a word by the company it keeps."
Words that appear in similar contexts get similar embeddings.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE EMBEDDING LAYER — It's Just A Lookup Table
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: EMBEDDINGS — A FANCY WORD FOR LOOKUP TABLE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  WHAT AN EMBEDDING LAYER ACTUALLY IS                        │
└─────────────────────────────────────────────────────────────────┘

An embedding layer is a matrix with shape (vocab_size, d_model).

  embedding_matrix[token_id]  →  returns the row for that token

That's it. No computation. Just a memory lookup.

  Before training: random numbers
  During training: numbers shift to encode meaning
  After training: rows contain rich semantic information

The matrix starts random because we don't know the "right"
arrangement yet. During training, each weight update moves
tokens that should be similar CLOSER together, and tokens
that should be different FARTHER apart.
""")

vocab_size = 10       # tiny demonstration
d_model = 4           # tiny (real models: 768, 1024, 4096, ...)

np.random.seed(42)
embedding_table = np.random.randn(vocab_size, d_model)

print(f"Embedding table shape: {embedding_table.shape}")
print(f"  = {vocab_size} tokens × {d_model} features each")
print(f"\nTable (each row is one token's vector):")
for i in range(vocab_size):
    print(f"  Token {i}: {np.round(embedding_table[i], 3)}")

# The "embedding" operation = grab rows by index
token_ids = np.array([3, 1, 4, 1])  # four tokens
embeddings = embedding_table[token_ids]

print(f"\nToken IDs: {token_ids}")
print(f"Embeddings shape: {embeddings.shape}  (4 tokens × 4 dims)")
print(embeddings)
print(f"\n→ Token 1 appears twice (positions 1 and 3):")
print(f"  embeddings[1] = {embeddings[1]}")
print(f"  embeddings[3] = {embeddings[3]}")
print(f"  → Same token → same vector → identical meaning")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: HOW EMBEDDINGS LEARN MEANING
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: HOW RANDOM NUMBERS BECOME MEANINGFUL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  THE LEARNING PROCESS — Gradient descent in embedding space │
└─────────────────────────────────────────────────────────────────┘

The embedding table is just a big set of parameters — like weights
in a linear layer. During backpropagation, the gradient flows all
the way back to the embedding lookup:

  Loss → LM Head → Transformer Blocks → Embedding table[token_ids]

The gradient tells us: "If I increase this embedding dimension by
epsilon, how much does the loss decrease?"

Tokens that appear in similar contexts get similar updates, so
their embeddings drift in similar directions. Over millions of
examples, this creates the geometric structure we observe.

KEY INSIGHT: The embedding is LEARNED from the TASK, not from a
separate embedding objective. In a language model, the task is
"predict the next token." Words that help predict the same next
words become similar. This is why embeddings from different
models (GPT vs BERT) encode DIFFERENT kinds of similarity —
they were trained on different tasks.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: SIMILARITY IN VECTOR SPACE
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 3: MEASURING SIMILARITY — COSINE DISTANCE")
print("=" * 70)


def cosine_similarity(a, b):
    """How similar are two vectors? 1 = same direction, -1 = opposite.

    WHY cosine and not Euclidean distance?
      - Euclidean distance: affected by vector magnitude.
        A very frequent word might have a large vector just from
        seeing many gradients. We care about DIRECTION, not magnitude.
      - Cosine similarity: treats vectors as directions in space.
        Angle between vectors = semantic relationship.
        Normalizes out magnitude automatically.
      - Dot product without normalization: biased toward long vectors.
        "the" (very frequent) would be "similar" to everything.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# Simulate trained embeddings with meaning structure
d = 6  # small for demo
# We design these manually to show the geometric property
king  = np.array([0.8, 0.6, 0.1, 0.0, 0.0, 0.0])  # royalty + masculine
queen = np.array([0.1, 0.6, 0.8, 0.0, 0.0, 0.0])  # royalty + feminine
man   = np.array([0.8, 0.0, 0.1, 0.0, 0.0, 0.0])  # masculine, no royalty
woman = np.array([0.1, 0.0, 0.8, 0.0, 0.0, 0.0])  # feminine, no royalty

print(f"king  = {np.round(king, 2)}")
print(f"queen = {np.round(queen, 2)}")
print(f"man   = {np.round(man, 2)}")
print(f"woman = {np.round(woman, 2)}")

# The famous analogy: king - man + woman ≈ queen
analogy = king - man + woman
print(f"\nking - man + woman = {np.round(analogy, 2)}")
print(f"queen              = {np.round(queen, 2)}")
print(f"Similarity: {cosine_similarity(analogy, queen):.3f}  (should be close to 1)")
print(f"\n→ king and queen differ mainly by gender dimension (0 vs 1)")
print(f"  king[0]=0.8 (masculine), queen[0]=0.1 (not masculine)")
print(f"  quee[2]=0.8 (feminine),  king[2]=0.1  (not feminine)")
print(f"  Both share dim 1 = 0.6 (royalty!)")

# Verify relationships
print(f"\nking-queen similarity:   {cosine_similarity(king, queen):.3f} (related)")
print(f"king-man similarity:     {cosine_similarity(king, man):.3f} (related)")
print(f"queen-woman similarity:  {cosine_similarity(queen, woman):.3f} (related)")
print(f"king-woman similarity:   {cosine_similarity(king, woman):.3f} (unrelated)")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: WHY D_MODEL — The Dimension That Defines Your Model
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: WHY d_model DEFINES THE ENTIRE MODEL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  d_model — The "width" of every layer                      │
└─────────────────────────────────────────────────────────────────┘

d_model is the single most important hyperparameter. It's the
dimension of EVERY vector in the transformer:

  - Token embeddings:        (vocab, d_model)
  - Position embeddings:     (max_seq, d_model)
  - Q/K/V projections:       (d_model, d_model)
  - Attention output:        (batch, seq, d_model)
  - FFN hidden layer:        (d_model, 4*d_model) → back to d_model
  - Layer norm:              normalizes d_model features
  - LM head:                 (d_model, vocab)

Everything is d_model, always. This is why:

  LARGER d_model:
    + More capacity — can learn more complex patterns
    + More expressiveness per token
    - Quadratic cost in attention: O(n^2 × d_model)
    - Quadratic params: W_Q is (d_model × d_model)
    - Memory: every activation tensor grows with d_model

  SMALLER d_model:
    + Faster compute, less memory
    + Easier to train (fewer params)
    - Less capacity — might miss complex patterns
    - But you can compensate with MORE LAYERS

The d_model sweet spot is the art of transformer design. GPT-2
scales from 768 (small) to 1600 (XL). LLaMA-7B uses 4096.
GPT-3 uses 12288. There's no formula — it's empirical.
""")

print("Model                d_model    Heads    Layers    Total Params")
print("──────────────────────────────────────────────────────────────")
print("GPT-2 Small            768        12        12         124M")
print("GPT-2 Medium          1024        16        24         345M")
print("GPT-2 Large           1280        20        36         762M")
print("LLaMA-7B              4096        32        32           7B")
print("LLaMA-70B             8192        64        80          70B")
print("GPT-3                12288        96        96         175B")

print(f"""
Notice: d_model MUST be divisible by num_heads because
each head gets d_model/num_heads dimensions:

  GPT-2 Small: 768 / 12 = 64 dims per head
  LLaMA-7B:    4096 / 32 = 128 dims per head
  GPT-3:       12288 / 96 = 128 dims per head

This is why you see dimensions like 768, 1024, 4096, 8192 — they're
chosen to be cleanly divisible by typical head counts.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: WEIGHT TYING — Sharing Embeddings Between Input and Output
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: WEIGHT TYING — The Clever Parameter-Saving Trick")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  WHY SHARE WEIGHTS BETWEEN INPUT AND OUTPUT?                │
└─────────────────────────────────────────────────────────────────┘

In transformers, the input embedding (token → vector) and the
output projection (hidden → vocab logits) are often THE SAME MATRIX
(or transpose of it):

  Input:  x = embedding_matrix[token_id]          → (vocab,d_model)
  Output: logits = hidden @ embedding_matrix^T    → (d_model,vocab)

WHY THIS WORKS:
  If two tokens have similar embeddings (the model thinks they're
  similar), then the hidden state that predicts token A should
  ALSO produce a high score for token B. The shared matrix
  enforces this consistency.

  Without tying: The model could learn that "cat" and "feline" are
  similar in input space, but that predicting "feline" from a
  hidden state is completely different from predicting "cat".
  This inconsistency hurts generalization.

WHY THIS SAVES PARAMETERS:
  - Untied: vocab × d_model (embedding) + d_model × vocab (head)
            = 2 × vocab × d_model params
  - Tied:   vocab × d_model (shared)
            = vocab × d_model params (HALF the parameters!)

  For GPT-2: 50K × 768 × 2 = 77M → 38.5M saved (31% reduction!)

WHY SOMETIMES NOT TIED:
  - When vocab is small, the saving is negligible
  - When the task is classification (BERT), the LM head is
    task-specific and shouldn't share with embeddings
  - Some argue untying gives slightly better perplexity at the
    cost of memory — used in very large models for quality
""")

shared_embed = np.random.randn(vocab_size, d_model)
input_emb = shared_embed[token_ids]
hidden = np.random.randn(d_model)  # output of transformer
output_logits = hidden @ shared_embed.T  # (d_model,) @ (d_model, vocab) → (vocab,)

print(f"\nWeight tying demonstration (shared matrix):")
print(f"  Embed (token→vector):   use matrix (vocab,d_model)")
print(f"  Unembed (hidden→logits): use matrix^T  (d_model,vocab)")
print(f"  Both use: {shared_embed.shape}")
print(f"  Logits shape: {output_logits.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Embeddings = a lookup table: embedding_matrix[token_id]
2. Vectors in high-dimensional space encode meaning geometrically
3. Similarity = direction (cosine), not magnitude (Euclidean)
4. d_model is the universal dimension — every layer uses it
5. Weight tying = input and output share the same matrix

Why this matters:
  The transformer's job is to TRANSFORM these embedding vectors
  through attention and FFN layers. The embeddings are the INPUT
  to this transformation. Their quality determines the ceiling
  of what the model can learn.

Pipeline so far:
  "hello" → Tokenizer → [15496] → Embedding → [0.2, -0.5, 0.8, ...]

Next: Attention — how these vectors "talk" to each other.
""")

if __name__ == "__main__":
    print("\nModule 2 complete! Next: 03_simple_attention.py")
    print("Run with: uv run python course/transformer/03_simple_attention.py")
