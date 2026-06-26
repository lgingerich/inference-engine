"""
Shared MiniGPT model — used by both the transformers course and the inference course.

This module contains a clean, import-only implementation of MiniGPT.
No print() calls at module level — it's designed to be imported by both courses.

The classes here are identical to what was built in course/07_mini_gpt.py,
extracted into a reusable module so the inference course can build on them
without duplicating code.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def softmax(x, axis=-1):
    """Numerically stable softmax: subtract max before exp to avoid overflow.

    Without the max subtraction, exp() on large values produces infinity,
    which breaks the division. Subtracting the max makes the largest value
    exactly exp(0)=1, keeping everything in a safe range — but the result
    is mathematically identical because exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x)).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit — smooth activation used in GPT/BERT.

    Unlike ReLU which is a hard cutoff (0 if x<0 else x), GELU is smooth:
    it multiplies x by P(X <= x) where X ~ N(0,1). This smoothness allows
    gradients to flow even for slightly negative values.

    This is the fast tanh approximation — within 0.01% of the exact version.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class LayerNorm:
    """Normalize each token's feature vector to mean=0, std=1.

    Normalization is applied across the LAST dimension (the features/d_model
    dimension). For each token independently:
        y = (x - mean(x)) / std(x) * gamma + beta

    Why this matters in deep networks:
    - Without normalization, activations can drift — some layers produce tiny
      values while others explode. The next layer sees an unpredictable input
      distribution ("internal covariate shift").
    - LayerNorm forces every token's activations to have a consistent scale
      regardless of how deep in the network we are.
    - gamma and beta are LEARNED parameters that let the network undo the
      normalization if it's not helpful (e.g., gamma=std, beta=mean undoes it).

    Gamma starts at 1 (do nothing), beta starts at 0 (do nothing).
    """
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)   # learnable scale (if 1, no scaling)
        self.beta = np.zeros(d_model)   # learnable shift (if 0, no shifting)
        self.eps = eps                  # prevents division by zero

    def forward(self, x):
        """Normalize along the feature dimension (last axis)."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK (FFN)
# ═══════════════════════════════════════════════════════════════════════════════


class FeedForward:
    """Position-wise feed-forward network.

    Applies a two-layer MLP to EACH token INDEPENDENTLY:
        FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    The expansion ratio (typically 4x) creates a bottleneck-like structure:
    d_model → 4*d_model (expand, apply non-linearity) → d_model (compress).
    This is where most of the model's parameters live — the FFN is ~4x larger
    than the attention mechanism in each block.

    Why this works: Attention mixes information ACROSS tokens (weighted average).
    The FFN transforms information WITHIN each token (non-linear transformation).
    Together they form a complete information processing unit.
    """
    def __init__(self, d_model, expansion=4):
        self.d_ff = d_model * expansion
        # First linear layer: expand from d_model to d_ff
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.02
        self.b1 = np.zeros(self.d_ff)
        # Second linear layer: compress from d_ff back to d_model
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """x shape: (batch, seq_len, d_model) → output: same shape."""
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════


class MultiHeadAttention:
    """Multi-head scaled dot-product self-attention.

    Each token projects into Query (Q), Key (K), and Value (V) vectors.
    Attention weights are computed as:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    "Multi-head" means we do this H times in parallel, each with its own
    learned Q/K/V projections, then concatenate the results.
    This lets different heads learn different types of relationships.

    For GPT-2 Small: d_model=768, num_heads=12, d_k=64 per head.
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        # One combined weight matrix per projection (not per-head)
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        # Output projection: combine all heads back into d_model
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        """Compute multi-head attention.

        Args:
            x: (batch, seq_len, d_model)  — input token representations
            mask: (1, 1, seq_len, seq_len) or None  — causal/padding mask

        Returns:
            output: (batch, seq_len, d_model)  — context-aware representations
        """
        batch, seq, d_model = x.shape

        # Step 1: Compute Q, K, V projections
        Q = x @ self.W_Q   # (batch, seq, d_model)
        K = x @ self.W_K
        V = x @ self.W_V

        # Step 2: Reshape to split heads
        # From (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = Q.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Step 3: Scaled dot-product attention
        # (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) → (batch, heads, seq, seq)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask  # -inf forces attention to 0 after softmax

        attn = softmax(scores, axis=-1)

        # Step 4: Weighted sum of values
        # (batch, heads, seq, seq) @ (batch, heads, seq, d_k) → (batch, heads, seq, d_k)
        out = attn @ V

        # Step 5: Concatenate heads and apply output projection
        # (batch, heads, seq, d_k) → (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
        return out @ self.W_O


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════


class TransformerBlock:
    """One complete transformer block = Pre-LN → Attention → + → Pre-LN → FFN → +

    This uses Pre-LayerNorm architecture (norm BEFORE each sublayer), which
    is the current standard (used by GPT-2/3, LLaMA, etc.):

        x = x + Attention(LayerNorm(x))    ← residual around attention
        x = x + FFN(LayerNorm(x))          ← residual around FFN

    The residual connections are CRITICAL: they let gradients flow directly
    through the identity path, preventing vanishing gradients in deep stacks.
    Without them, you couldn't train more than ~10 layers.
    """
    def __init__(self, d_model, num_heads, ffn_expansion=4):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ffn_expansion)
        self.ln1 = LayerNorm(d_model)  # before attention
        self.ln2 = LayerNorm(d_model)  # before FFN

    def forward(self, x, mask=None):
        # Sublayer 1: Multi-head attention with residual
        x = x + self.attention.forward(self.ln1.forward(x), mask)
        # Sublayer 2: Feed-forward with residual
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# MINI GPT — COMPLETE DECODER-ONLY TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════


class MiniGPT:
    """A minimal GPT-style decoder-only transformer.

    Architecture:
        Token IDs
          → Token Embeddings + Position Embeddings
          → [TransformerBlock × N layers]
          → Final LayerNorm
          → LM Head (linear → vocab_size)
          → Logits (next-token scores)

    This is structurally identical to GPT-2/3, LLaMA, etc. — just scaled down.
    The same blocks, same attention, same residual connections.

    Parameters:
        vocab_size:   number of unique tokens in vocabulary
        d_model:      hidden dimension (the "width" of the model)
        num_heads:    number of attention heads (d_model must be divisible by this)
        num_layers:   number of transformer blocks (depth)
        max_seq_len:  maximum context length (position embeddings)
        ffn_expansion: FFN hidden size multiplier (typically 4)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 max_seq_len, ffn_expansion=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Token embedding matrix: each row is the vector for one token
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Position embedding matrix: one vector per absolute position
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, ffn_expansion)
            for _ in range(num_layers)
        ]

        # Final layer norm (applied after all blocks, before output projection)
        self.final_ln = LayerNorm(d_model)

        # LM Head: projects from d_model to vocab_size
        # Each row = "score" for how likely each token is
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_ids):
        """Forward pass: token_ids → logits.

        Args:
            token_ids: (batch_size, seq_len) integer token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size) scores per position
                    logits[b, s, v] = score for token v at position s in batch b
        """
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # 1. Token embeddings: map each token ID to its vector
        x = self.token_embed[token_ids]  # (batch, seq, d_model)

        # 2. Add positional embeddings (broadcasts over batch)
        positions = np.arange(seq_len)
        x = x + self.pos_embed[positions]

        # 3. Create causal mask: position i can only see positions ≤ i
        # For batch=1, single head: shape is (1, 1, seq_len, seq_len)
        mask = np.triu(np.ones((1, 1, seq_len, seq_len)) * float('-inf'), k=1)

        # 4. Pass through each transformer block
        for block in self.blocks:
            x = block.forward(x, mask)

        # 5. Final layer norm
        x = self.final_ln.forward(x)

        # 6. Project to vocabulary: each position gets vocab_size scores
        logits = x @ self.lm_head  # (batch, seq, vocab_size)

        return logits

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """Generate text autoregressively (one token at a time).

        This is the SIMPLEST possible generator — it re-runs the full
        forward pass for every new token. This is O(n²) and SLOW.
        The entire inference course is about optimizing this loop!

        Args:
            prompt_ids: Starting token IDs (1D list or array)
            max_new_tokens: How many tokens to generate
            temperature: Controls randomness (0 = greedy, >1 = more random)

        Returns:
            list of token IDs including the prompt
        """
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len (sliding window if context grows too long)
            context = generated[-self.max_seq_len:]

            # Forward pass on the ENTIRE context — this is what we'll optimize!
            context_batch = np.array([context])  # (1, seq_len)
            logits = self.forward(context_batch)

            # Only the LAST position predicts the next token
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply temperature scaling: lower = sharper, higher = flatter
            next_logits = next_logits / max(temperature, 1e-8)

            # Convert logits to probabilities and sample
            probs = softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)

            generated.append(int(next_token))

        return generated
