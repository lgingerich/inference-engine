"""
MODULE 2: KV CACHE — The Most Important Inference Optimization
===============================================================

Module 1 showed that the naive autoregressive loop re-computes Key
and Value projections for ALL previous tokens on every generation
step. That's the O(n²) trap — doubling the sequence length quadruples
the per-step compute cost.

The fix is beautifully simple: SAVE the K and V matrices so we only
compute them ONCE per token. This is the KV Cache — the single most
important optimization in LLM inference. Without it, you cannot serve
an LLM at any reasonable speed. Every production inference engine
(vLLM, TGI, SGLang, llama.cpp) is built on this mechanism.

WHAT YOU'LL LEARN:
   1. What K and V actually ARE (concrete matrix shapes)
   2. Why K and V of past tokens never change — the mathematical proof
   3. How to implement a KV cache step-by-step
   4. Why concatenation is correct (cached K/V for past, new Q attends both)
   5. Prefill vs decode: the two modes of cached inference
   6. Memory cost of the cache — and why it often exceeds the model
   7. Actual speedup measurement vs the naive loop

AFTER THIS MODULE:
   You'll understand the core mechanism behind every production
   inference engine and why the KV cache is the foundation upon
   which batching, paged attention, flash attention, and speculative
   decoding are all built.
"""

import time
import numpy as np
from course._model import MiniGPT, softmax, LayerNorm


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY KV CACHING IS THE MOST IMPORTANT OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY KV CACHING?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE O(n²) TRAP — Why naive generation is unsustainable         │
│                                                                 │
│ Every transformer layer computes attention as:                  │
│   Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V              │
│                                                                 │
│ In the naive autoregressive loop (Module 1):                    │
│   Step 1: 1 token →  compute K₁, V₁                             │
│   Step 2: 2 tokens → compute K₁, K₂, V₁, V₂   (K₁ re-computed!)│
│   Step 3: 3 tokens → compute K₁,K₂,K₃, V₁,V₂,V₃ (K₁,K₂ again!) │
│   Step N: N tokens → compute ALL N K/V pairs                    │
│                                                                 │
│ Total K/V computations across N steps: 1+2+3+...+N ≈ N²/2     │
│ For a 2048-token generation: ~2 MILLION redundant computations  │
└─────────────────────────────────────────────────────────────────┘

The KV cache solves this with a single insight: K and V for past
tokens DON'T CHANGE. So compute them once, store them, and reuse.

  Without cache: O(n²) K/V computations across generation
  With cache:    O(n)  K/V computations across generation

For a 7B model generating 512 tokens, this is the difference between
serving 1 request per second and serving 50 requests per second.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: WHAT ARE K AND V, PHYSICALLY? — The Matrices We're Caching
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: K AND V — THE MATRICES WE'RE CACHING")
print("=" * 70)

np.random.seed(42)
model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                num_layers=2, max_seq_len=64)

# ═══════════════════════════════════════════════════════════════════
# 1.1  The physical shape of K and V
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  K AND V SHAPES — What we're actually storing              │
└─────────────────────────────────────────────────────────────────┘

Every transformer layer has four weight matrices for attention:
  W_Q: projects input → Query  (what this token is LOOKING FOR)
  W_K: projects input → Key    (what this token CONTAINS, for matching)
  W_V: projects input → Value  (what this token CONTAINS, for output)

K and V are the matrices we cache because they represent the
CONTENT of each token, which is immutable once computed.
Q changes every step (it's the new token asking questions).
""")

# Let's look at the actual K and V shapes for one layer
block0 = model.blocks[0]
attn = block0.attention
print(f"\nModel config: d_model={model.d_model}, heads={attn.num_heads}, "
      f"d_k={attn.d_k}")
print(f"\nFor layer 0, attention weight shapes:")
print(f"  W_Q: {attn.W_Q.shape}  ({model.d_model} × {model.d_model})")
print(f"  W_K: {attn.W_K.shape}")
print(f"  W_V: {attn.W_V.shape}")
print(f"  W_O: {attn.W_O.shape}")

# Simulate what K and V look like for a 5-token prompt
prompt = np.array([[1, 2, 3, 4, 5]])
x = model.token_embed[prompt]  # (1, 5, 32)

# Compute K as if we were inside attention
K_raw = x @ attn.W_K  # (1, 5, 32)

# After reshaping to multi-head:
K_mh = K_raw.reshape(1, 5, attn.num_heads, attn.d_k).transpose(0, 2, 1, 3)

print(f"\nAfter multi-head reshape:")
print(f"  K shape: {K_mh.shape}  (batch=1, heads=4, tokens=5, d_k=8)")
print(f"  V shape: same as K shape")

# Show what one head's K matrix looks like
print(f"\n  Head 0's K matrix (5 tokens × 8 dims):")
print(f"  {np.round(K_mh[0, 0], 2)}")
print(f"\n  Each ROW is one token's Key vector.")
print(f"  Token 0's Key: 8 numbers representing 'what position 0 contains'")
print(f"  Token 1's Key: 8 numbers representing 'what position 1 contains'")
print(f"  ...etc.")

# ═══════════════════════════════════════════════════════════════════
# 1.2  Why K and V of past tokens NEVER change
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  WHY K AND V ARE IMMUTABLE — The mathematical proof        │
└─────────────────────────────────────────────────────────────────┘

The key (K) and value (V) for token at position p are computed as:

  K_p = embedding[token_p] · W_K
  V_p = embedding[token_p] · W_V

Two inputs to this computation:
  (a) embedding[token_p] — the token at position p
  (b) W_K, W_V — the trained weight matrices

During autoregressive generation:
  • The weight matrices W_K and W_V are FROZEN (no training during
    inference). They never change.
  • The token at position p is ALREADY GENERATED — it's fixed in
    the sequence history. It will never change either.

  Therefore: K_p and V_p are IDENTICAL every time we compute them.
  Computing them again would produce the EXACT SAME numbers.

This is fundamentally different from Q (query):
  Q_p = embedding[token_p] · W_Q

  At step N, we compute Q ONLY for the new token at position N.
  Q_N attends to ALL K vectors (past and present) to decide what
  the new token should pay attention to. Q_N itself is new each
  step, but it only needs the new token's embedding.
""")

# THE CRUCIAL INSIGHT
print(f"""
THE KEY INSIGHT:
  K[position 0] depends ONLY on:
    → embedding[token_at_position_0]
    → W_K (fixed weight matrix)
  Neither changes during generation!
  → K[position 0] is IDENTICAL every time we compute it.
  → So we compute it ONCE and reuse it forever.
  
  This means if we store K and V in a cache:
    Token 1 generation: compute K₁, V₁ → store
    Token 2 generation: compute K₂, V₂ → cache now has [K₁,K₂]
    Token 3 generation: compute K₃, V₃ → cache now has [K₁,K₂,K₃]
    ...
    Token N generation: compute K_N, V_N → cache has [K₁,...,K_N]
  
  For token 500, we compute K/V ONCE for the new token.
  Without cache: we'd recompute K/V for tokens 1-499 AGAIN.
  500× vs 1× — that's a 500× reduction in K/V computation!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# KV CACHE INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# We're building a NEW inference loop that wraps around MiniGPT.
# Instead of calling model.forward() which recomputes everything,
# we manually step through the layers with a cache.
#
# The cache structure:
#   kv_cache[layer_index] = {
#       'K': numpy array shape (1, num_heads, seq_so_far, d_k)
#       'V': numpy array shape (1, num_heads, seq_so_far, d_k)
#   }
#
# At each decode step, we compute K and V ONLY for the new token,
# then CONCATENATE with cached K and V for the full attention computation.
# ═══════════════════════════════════════════════════════════════════════════════


class KVCacheInference:
    """Efficient inference using a Key-Value cache.

    This is the O(n) replacement for the O(n²) raw autoregressive loop.
    It manually controls the forward pass to store and reuse K/V matrices
    from previous generation steps.

    Architecture:
        - Prefill: process ALL prompt tokens, build initial cache
        - Decode: process ONE token at a time, extending the cache
        - Each step only computes K/V for the NEW token(s)
    """

    def __init__(self, model):
        self.model = model
        self.num_layers = model.num_layers

        # Convenient access to model internals
        self.blocks = model.blocks
        self.token_embed = model.token_embed
        self.pos_embed = model.pos_embed
        self.final_ln = model.final_ln
        self.lm_head = model.lm_head
        self.max_seq_len = model.max_seq_len

        # The cache: one entry per layer. Empty list means "not initialized yet".
        self.cache: list = []  # will be populated on first prefill
        self.cached_seq_len = 0

    def _initialize_cache(self, batch_size, num_heads, d_k):
        """Create empty cache slots for each layer."""
        self.cache = []
        for _ in range(self.num_layers):
            self.cache.append({
                'K': np.empty((batch_size, num_heads, 0, d_k)),
                'V': np.empty((batch_size, num_heads, 0, d_k)),
            })
        self.cached_seq_len = 0

    def _attention_with_cache(self, layer_idx, x, block):
        """Compute attention for one layer, using/updating the cache.

        This is the CORE of the KV cache mechanism. Instead of letting
        the model compute attention normally (which would recompute K/V
        for ALL tokens), we manually:

        1. Compute Q, K, V ONLY for the new input tokens
        2. Concatenate with cached K, V from previous steps
        3. Store the new K, V for next time
        4. Compute attention with the full [cached + new] keys/values

        WHY concatenation works: The cached K/V represent past tokens.
        The new Q attends to ALL K vectors (both cached and new). Since
        the cached K vectors are identical to what we would have computed
        from scratch, concatenation produces the EXACT SAME attention
        scores as a full recomputation — but with O(1) K/V compute per
        new token instead of O(n).

        Args:
            layer_idx: which transformer layer (0 to num_layers-1)
            x: input to this layer's attention, shape (batch, new_seq, d_model)
            block: the TransformerBlock for this layer

        Returns:
            attn_output: attention output, shape (batch, new_seq, d_model)
        """
        attn = block.attention
        batch, new_seq, d_model = x.shape

        # ── Step 1: Compute Q, K, V for the NEW tokens only ──
        # This is the key: we don't recompute K/V for past tokens!
        Q = x @ attn.W_Q  # (batch, new_seq, d_model)
        K = x @ attn.W_K
        V = x @ attn.W_V

        # ── Step 2: Reshape to multi-head ──
        # From (batch, new_seq, d_model) → (batch, num_heads, new_seq, d_k)
        Q = Q.reshape(batch, new_seq, attn.num_heads, attn.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, new_seq, attn.num_heads, attn.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, new_seq, attn.num_heads, attn.d_k).transpose(0, 2, 1, 3)

        # ── Step 3: Concatenate with cached K, V ──
        # If we have cached K and V from previous steps, prepend them.
        # Now K = [old_K, new_K] for attention, so Q can attend to ALL tokens.
        if self.cache and self.cached_seq_len > 0:
            cached_K = self.cache[layer_idx]['K']
            cached_V = self.cache[layer_idx]['V']
            # Concatenate along the sequence dimension (axis=2)
            K = np.concatenate([cached_K, K], axis=2)
            V = np.concatenate([cached_V, V], axis=2)

        # ── Step 4: Update the cache ──
        # Store the FULL K and V (cached + new) for next time
        self.cache[layer_idx]['K'] = K.copy()
        self.cache[layer_idx]['V'] = V.copy()
        if layer_idx == 0:
            self.cached_seq_len += new_seq

        # ── Step 5: Scaled dot-product attention ──
        # Q shape: (batch, num_heads, new_seq, d_k)
        # K shape: (batch, num_heads, cached_seq_len + new_seq, d_k)
        # scores shape: (batch, num_heads, new_seq, cached_seq_len + new_seq)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(attn.d_k)

        # ── Step 6: Causal masking ──
        # During decode (new_seq=1): the single Q token can attend to ALL K
        # tokens (all are in the past). No mask needed!
        #
        # During prefill (new_seq>1): Q tokens must only attend to K tokens
        # with position ≤ their own. We create a mask where Q[i] can attend
        # to K[j] only if j ≤ i + cached_len.
        #
        # However, since prefill is the FIRST call (no cached tokens),
        # the cached_len is 0, and we just need the standard causal mask.
        if new_seq > 1:
            mask = np.triu(np.ones((1, 1, new_seq, K.shape[2])) * float('-inf'), k=1)
            scores = scores + mask

        # ── Step 7: Softmax and weighted sum ──
        attn_weights = softmax(scores, axis=-1)
        out = attn_weights @ V  # (batch, num_heads, new_seq, d_k)

        # ── Step 8: Concatenate heads and output projection ──
        out = out.transpose(0, 2, 1, 3).reshape(batch, new_seq, d_model)
        return out @ attn.W_O

    def forward_with_cache(self, token_ids):
        """Forward pass with KV cache support.

        This replaces model.forward() for cached inference. It runs the
        full pipeline (embeddings → transformer blocks → LM head) but
        uses the KV cache in each attention layer to avoid recomputation.

        Args:
            token_ids: (batch_size, seq_len) — integer token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # ── Embedding step (same as MiniGPT) ──
        x = self.token_embed[token_ids]  # (batch, seq, d_model)

        # Position encoding: continue from where we left off
        if not self.cache:
            positions = np.arange(seq_len)
        else:
            positions = np.arange(self.cached_seq_len, self.cached_seq_len + seq_len)
        x = x + self.pos_embed[positions]

        # ── Initialize cache on first call ──
        if not self.cache:
            d_k = self.blocks[0].attention.d_k
            num_heads = self.blocks[0].attention.num_heads
            self._initialize_cache(batch_size, num_heads, d_k)

        # ── Pass through each layer with caching ──
        for layer_idx, block in enumerate(self.blocks):
            # Pre-LN: normalize before attention
            x_norm = block.ln1.forward(x)

            # Attention WITH CACHE (this is the magic!)
            attn_out = self._attention_with_cache(layer_idx, x_norm, block)

            # Residual connection
            x = x + attn_out

            # FFN: compute normally — no caching needed here
            x_norm = block.ln2.forward(x)
            ffn_out = block.ffn.forward(x_norm)
            x = x + ffn_out

        # ── Final layer norm and LM head ──
        x = self.final_ln.forward(x)
        logits = x @ self.lm_head
        return logits

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """Generate text using KV cache for efficient autoregression.

        Args:
            prompt_ids: Starting token IDs (1D list or array)
            max_new_tokens: How many tokens to generate
            temperature: Controls randomness (0 = greedy, >1 = more random)

        Returns:
            list of token IDs including the prompt
        """
        # Reset cache for new generation
        self.cache = []
        self.cached_seq_len = 0
        generated = list(prompt_ids)

        # ── PREFILL: Process the entire prompt at once ──
        # This is the compute-heavy first step where we build the cache.
        # We only do this ONCE, no matter how many tokens we generate later.
        prompt_batch = np.array([prompt_ids])
        logits = self.forward_with_cache(prompt_batch)

        # Get the first generated token from the last prompt position
        next_logits = logits[0, -1, :]
        next_logits = next_logits / max(temperature, 1e-8)
        probs = softmax(next_logits)
        next_token = int(np.random.choice(self.model.vocab_size, p=probs))
        generated.append(next_token)

        # ── DECODE: Generate remaining tokens one at a time ──
        # Each step: forward pass with JUST the new token.
        # The cache means we don't recompute K/V for any past token.
        for _ in range(max_new_tokens - 1):
            # Only process the LAST token (all others are in cache)
            token_batch = np.array([[next_token]])
            logits = self.forward_with_cache(token_batch)

            # Sample next token
            next_logits = logits[0, -1, :]
            next_logits = next_logits / max(temperature, 1e-8)
            probs = softmax(next_logits)
            next_token = int(np.random.choice(self.model.vocab_size, p=probs))
            generated.append(next_token)

        return generated


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: KV CACHE IN ACTION — Step-by-Step Generation
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: KV CACHE IN ACTION — Generating Text Efficiently")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 2.1  Prefill — Building the initial cache from the prompt
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  PREFILL — Process all prompt tokens, build the cache      │
└─────────────────────────────────────────────────────────────────┘

Prefill is the FIRST phase of KV-cached inference. The model
processes the entire prompt at once (just like the naive loop
does on step 1) and stores K and V for ALL prompt tokens.

WHY process all at once: Attention is inherently parallel during
prefill — each token can attend to all previous tokens in one
batch. This makes prefill compute-bound (GPU at 100% utilization).
It's the only phase where the full O(n²) attention is computed,
but it only happens ONCE per request.
""")

inference = KVCacheInference(model)

prompt_ids = [1, 2, 3, 4, 5]
print(f"\nPrompt: {prompt_ids} ({len(prompt_ids)} tokens)")

# Step 1: Prefill — process all prompt tokens
print(f"\n>>> PREFILL (process all {len(prompt_ids)} prompt tokens)")
print(f"    Building initial KV cache...")
print(f"    Cache before prefill: EMPTY (self.cache = [])")

logits_prefill = inference.forward_with_cache(np.array([prompt_ids]))

print(f"\n    Cache after prefill:")
print(f"      Cached sequence length: {inference.cached_seq_len} tokens")
print(f"      Layer 0 K shape: {inference.cache[0]['K'].shape}")
print(f"      Layer 0 V shape: {inference.cache[0]['V'].shape}")
print(f"      Layer 1 K shape: {inference.cache[1]['K'].shape}")
print(f"      Layer 1 V shape: {inference.cache[1]['V'].shape}")
print(f"    Logits shape: {logits_prefill.shape}")

# ═══════════════════════════════════════════════════════════════════
# 2.2  Decode — Watch the cache GROW step by step
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  DECODE — Generate one token at a time, cache grows        │
└─────────────────────────────────────────────────────────────────┘

In decode mode, we feed ONLY the last generated token into the
forward pass. The cache provides K/V for all past tokens. We
compute K/V only for the new token and APPEND to the cache.

WHY this is correct: The new Q attends to the CONCATENATED set
of [cached K for past tokens, new K for current token]. Since
cached K is identical to what we'd compute from scratch, the
attention scores are mathematically identical to a full recompute.

Below, we trace the cache growth token by token — watch how the
K tensor along the sequence dimension (axis 2) grows from 5 to 10.
""")

print(f"\n>>> DECODE (generate tokens one at a time using cache)")

for step in range(5):
    # Generate the next token
    next_logits = logits_prefill[0, -1, :]  # get prediction
    next_token = int(np.argmax(next_logits))  # greedy for demo
    prompt_ids.append(next_token)

    # Now decode with JUST the new token
    new_token_batch = np.array([[next_token]])
    logits_prefill = inference.forward_with_cache(new_token_batch)

    print(f"\n  Step {step}: generated token {next_token}")
    print(f"    Input to forward: 1 token (shape [{new_token_batch.shape}])")
    print(f"    Cache now holds:   {inference.cached_seq_len} tokens")
    print(f"    K shape growth:    (1, 4, {inference.cached_seq_len}, 8)")

print(f"\nFinal sequence: {prompt_ids}")
print(f"  Starting length: 5 → Final length: {len(prompt_ids)}")
print(f"  Total decode calls: 5 (each processed just 1 token)")
print(f"  Total K/V computations: 5 prompts + 5 decode = 10 (not 55)")

# Show what the concatenation physically looks like
print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ WHY CONCATENATION GIVES THE RIGHT ATTENTION SCORES             │
│                                                                 │
│ At decode step 3 (cache has 8 tokens, new token is 9th):       │
│                                                                 │
│   Compute K_new from token 9 embedding → (1, 4, 1, 8)          │
│   Concat: K_full = [K_cache(1,4,8,8), K_new(1,4,1,8)]          │
│          K_full shape → (1, 4, 9, 8)                           │
│                                                                 │
│   Compute Q from token 9 → (1, 4, 1, 8)                        │
│   Scores = Q @ K_fullᵀ → (1, 4, 1, 9)                         │
│                                                                 │
│   These 9 scores are EXACTLY what you'd get from a full         │
│   recomputation because K_cache == the K values you'd           │
│   recompute. The math is identical — just faster.               │
└─────────────────────────────────────────────────────────────────┘
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: MEMORY COST — Why the KV Cache Often Exceeds the Model
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: THE MEMORY COST — KV Cache Can Exceed Model Size")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 3.1  Why the cache memory cost grows with sequence length
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  KV CACHE MEMORY — Linear growth, but at scale it's brutal │
└─────────────────────────────────────────────────────────────────┘

The KV cache is NOT free. Every generated token adds a new K
and V vector to EVERY layer's cache. The memory cost scales as:

  cache_bytes = 2 × num_layers × seq_len × num_heads × d_k × bytes_per_element

WHY this grows with sequence length: Each new token pushes a new
row into K and V for every layer. Doubling the sequence length
DOUBLES the cache memory. This is O(n) memory growth — which
sounds fine until you consider the numbers at scale.

For our MiniGPT:
  - 2 layers
  - 4 heads × 8 dims = 32 elements per token per head → 32 × 4 = 128 per token
  - K + V = 2 matrices
  - Per token cost: 2 × 2 layers × 128 × 8 bytes (float64) = 4,096 bytes
  
For LLaMA-7B:
  - 32 layers
  - 32 heads × 128 dims (d_model=4096 / 32 = 128)
  - K + V per layer per token: 2 × 32 × 128 = 8,192 floats
  - Per token: 32 layers × 8,192 × 2 bytes (FP16) = 524,288 bytes ≈ 0.5 MB
  - For 4096 context: 4096 × 0.5 MB = 2 GB KV cache!
  
  Model weights (FP16): ~14 GB
  KV cache (4096 context): ~2 GB  (14% of model size!)

For GPT-3 (175B):
  - 96 layers, 96 heads, 128 dims
  - Per token per layer: 2 × 96 × 128 = 24,576 floats
  - Per token: 96 layers × 24,576 × 2 = ~4.7 MB
  - For 2048 context: ~9.6 GB KV cache!
  
  This is why GQA (Grouped Query Attention) exists — fewer K/V heads
  means a proportionally smaller cache. We'll cover this in Module 5.

WHY this matters in production: If you're serving 10 concurrent
requests with 2048-token contexts on GPT-3, you need ~96 GB of
GPU memory JUST for KV caches — on top of the 350 GB model weights
(FP16). This is the fundamental memory bottleneck that paged
attention (vLLM) and prefix caching solve.
""")

# Calculate for our model
d_k = model.blocks[0].attention.d_k
num_heads = model.blocks[0].attention.num_heads
num_layers = model.num_layers
cache_size_per_token = 2 * num_layers * num_heads * d_k * 8  # float64 bytes
print(f"\nOur MiniGPT KV cache cost:")
print(f"  Per token: {cache_size_per_token:,} bytes")
print(f"  For 64 tokens (max context): {cache_size_per_token * 64 / 1024:.1f} KB")
print(f"  Model params: {model.vocab_size * model.d_model * 8 / 1024:.0f} KB")
print(f"  → KV cache is tiny for us, but for real models it's THE memory bottleneck")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: SPEED COMPARISON — Cached vs Uncached
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: SPEED BENCHMARK — KV Cache vs Raw Generation")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 4.1  Measuring the speedup
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  BENCHMARK — How much faster is it really?                 │
└─────────────────────────────────────────────────────────────────┘

We time both the naive autoregressive loop (recomputes everything)
and the KV-cached loop (reuses past K/V). The difference grows
with sequence length because the uncached loop does O(n²) work
while the cached loop does O(n).

With our tiny 100-vocab, 32-dim model, the overhead of the cache
management actually dominates — you'll see a modest speedup.
For a 7B model, the speedup is typically 10-100× per token.
""")

prompt_bench = [1, 2, 3, 4, 5]
num_tokens = 30
num_trials = 3

# Raw generation (naive loop — re-computes everything)
print(f"\nGenerating {num_tokens} tokens, {num_trials} trials each...")

# Uncached
uncached_times = []
for _ in range(num_trials):
    start = time.perf_counter()
    result_raw = model.generate(prompt_bench, max_new_tokens=num_tokens, temperature=0.8)
    uncached_times.append(time.perf_counter() - start)

# Cached
cached_times = []
for _ in range(num_trials):
    inference = KVCacheInference(model)
    start = time.perf_counter()
    result_cached = inference.generate(prompt_bench, max_new_tokens=num_tokens, temperature=0.8)
    cached_times.append(time.perf_counter() - start)

avg_uncached = np.mean(uncached_times)
avg_cached = np.mean(cached_times)

print(f"\n  Uncached: {avg_uncached:.4f}s ({num_tokens / avg_uncached:.1f} tok/s)")
print(f"  Cached:   {avg_cached:.4f}s ({num_tokens / avg_cached:.1f} tok/s)")
print(f"  Speedup:  {avg_uncached / avg_cached:.1f}×")

# ═══════════════════════════════════════════════════════════════════
# 4.2  How speedup scales with sequence length
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.2  SPEEDUP SCALES — Longer sequences benefit more            │
└─────────────────────────────────────────────────────────────────┘

The speedup from KV caching grows with sequence length because
the naive loop does more and more redundant work. At short
lengths, the overhead of cache management may even make it slower.
At long lengths, the cache wins massively.
""")

print(f"\n  Per-step cost growth (longer sequence = bigger difference):")
print(f"  Sequence  |  Uncached (ms/token)  |  Cached (ms/token)")
print(f"  ──────────┼──────────────────────┼───────────────────")

# Run a quick per-length benchmark
for length in [10, 30, 50]:
    prompt = list(range(min(length, 5)))
    tokens_to_gen = 15

    # Uncached
    t0 = time.perf_counter()
    model.generate(prompt, max_new_tokens=tokens_to_gen, temperature=0.8)
    uncached = (time.perf_counter() - t0) / tokens_to_gen * 1000

    # Cached
    inf = KVCacheInference(model)
    t0 = time.perf_counter()
    inf.generate(prompt, max_new_tokens=tokens_to_gen, temperature=0.8)
    cached_t = (time.perf_counter() - t0) / tokens_to_gen * 1000

    print(f"  {length:>9}  |  {uncached:>19.2f}  |  {cached_t:>17.2f}")

print(f"\n  Note: With our tiny model, the speedup is modest (overhead dominates).")
print(f"  For a 7B model, the speedup can be 10-100× because compute dominates.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: PREFILL VS DECODE — The Two Operating Modes
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 5: THE TWO OPERATING MODES")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 5.1  Why prefill and decode MUST be separate
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  WHY PREFILL AND DECODE ARE SEPARATE STEPS                 │
└─────────────────────────────────────────────────────────────────┘

You might wonder: why not just run in decode mode from the start,
feeding one prompt token at a time?

WHY prefill batches the prompt:
  1. EFFICIENCY: Attention is O(n²) per token pair. Computing
     attention over 5 prompt tokens simultaneously costs ~25
     operations. Doing it one at a time costs ~1+4+9+16+25=55
     operations (the 1+2²+...+n² sum).
  2. PARALLELISM: GPUs excel at batched matrix multiplication.
     Processing all prompt tokens in one big matmul saturates
     the GPU. Single-token decode underutilizes it.
  3. LATENCY: Users want the first token FAST. Prefill runs the
     full prompt in one shot, delivering the first response token
     as quickly as possible.

WHY decode processes one token at a time:
  1. DEPENDENCY: Each generated token depends on the previous one.
     You can't parallelize across generation steps — token N+1
     needs token N's output.
  2. MEMORY: Processing all generated tokens together would require
     storing ALL intermediate activations, which grows with sequence
     length. The cache only stores K/V — far less memory.
  3. EARLY STOPPING: Generation may stop at an EOS token. Decoding
     one at a time lets us stop immediately rather than wasting
     compute on tokens we won't use.
""")

# ═══════════════════════════════════════════════════════════════════
# 5.2  The two-phase inference architecture
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.2  THE COMPLETE TWO-PHASE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────┘

A KV-cached inference engine operates in TWO distinct modes:

PREFILL MODE (the prompt):
    Input:  ["The", "cat", "sat", "on", "the"]  (5 tokens)
    Action: Run ALL tokens through ALL layers
            Cache K and V from every layer
    Output: First generated token
    Cost:   O(prompt_len²) — but only ONCE per request
    GPU:    Compute-bound (big matmuls, lots of work)

DECODE MODE (the response):
    Input:  [new_token]  (just 1 token!)
    Action: Compute K and V for new token only
            Concatenate with cache → attention over all prior tokens
            Append new K and V to cache
    Output: Next token
    Cost:   O(1) for K/V compute, O(cache_len) for attention
    GPU:    Memory-bandwidth-bound (tiny matmul, big weight loads)

Together they form the complete inference pipeline:

    ┌──────────┐
    │  Prefill │  "What's the capital of France?"
    │  (once)  │  → 5 prompt tokens processed in one shot
    │  O(n²)   │  → KV cache built for all 5 tokens
    └────┬─────┘
         │
    ┌────▼─────────────────────────┐
    │  Decode × N                  │
    │  (repeated N times)          │  "The capital of France is Paris."
    │  O(n) per step               │  → 7 tokens generated, each using cache
    │  Each step: 1 new token      │
    └──────────────────────────────┘

This is the foundation of EVERY production inference engine.
Everything else (batching, quantization, FlashAttention, speculative
decoding) builds ON TOP of the KV cache.

Ready to scale up? Next: Module 3 — Batching multiple requests.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ SUMMARY — The KV Cache in 8 points                             │
│                                                                 │
│ 1. K and V matrices for past tokens NEVER change — cache them! │
│    Proof: K_p = embed[token_p] · W_K. Both inputs are frozen   │
│    during inference, so K_p is identical every time.            │
│                                                                 │
│ 2. Prefill: process all prompt tokens once, build initial KV   │
│    cache. This is the only O(n²) step — and it runs just once. │
│                                                                 │
│ 3. Decode: generate one token at a time, extend cache with     │
│    only the new token's K/V. Then concatenate and attend.      │
│                                                                 │
│ 4. Concatenation is correct: cached K == freshly-computed K.   │
│    The attention scores are mathematically identical to a full  │
│    recomputation, just with O(1) instead of O(n) K/V compute.   │
│                                                                 │
│ 5. Without cache: ~n²/2 K/V computations across generation.    │
│    With cache: n K/V computations. Speedup: 10-100× at scale.  │
│                                                                 │
│ 6. Memory cost scales with sequence length:                    │
│    cache_bytes = 2 × layers × seq × heads × d_k × precision    │
│    For GPT-3 at 2048 context: ~10 GB KV cache alone!           │
│                                                                 │
│ 7. Prefill and decode are separate because they have different  │
│    compute patterns: prefill is compute-bound (parallel),       │
│    decode is memory-bandwidth-bound (sequential).              │
│                                                                 │
│ 8. Every production inference engine (vLLM, TGI, llama.cpp)    │
│    is built on the KV cache. Batching, paged attention, GQA —   │
│    all optimizations that build on top of this foundation.      │
│                                                                 │
│ Pipeline now:                                                   │
│   Prefill → [KV Cache built] → Decode step 1 → step 2 → ...   │
│   Each decode step: O(1) new K/V, O(cache_len) attention       │
│                                                                 │
│ Next: Module 3 — Batching & Scheduling (serving multiple        │
│ requests at once by sharing the KV cache across requests)       │
└─────────────────────────────────────────────────────────────────┘
""")

if __name__ == "__main__":
    print("\nModule 2 complete! Next: i03_batching.py")
    print("Run with: uv run python course/inference/i03_batching.py")
