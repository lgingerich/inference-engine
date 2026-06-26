# Tensor Design Inefficiencies & Axis Convention Rethinks

Ongoing notes on where the standard (B, S, D) / (B, H, S, D) conventions leak,
and what alternative designs could look like.

### Quick reference: where each concern appears in the course

| Section | Primary course file(s) |
|---|---|
| 1. Reshape/transpose tax | `05_multi_head_attention.py`, `09_advanced_topics.py`, `i05_attention_optimizations.py` |
| 2. Redundant head projections | `05_multi_head_attention.py`, `09_advanced_topics.py` |
| 3. Dense (S,S) attention | `03_simple_attention.py`, `04_self_attention.py`, `09_advanced_topics.py`, `i05_attention_optimizations.py` |
| 4. Padding waste | `00_prerequisites.py`, `i03_batching.py` |
| 5. KV-cache vs batch dim | `i02_kv_cache.py`, `i05_attention_optimizations.py` |
| 6. Broadcasting temporaries | `00_prerequisites.py` |
| 7. State-space alternative | `09_advanced_topics.py` |
| 8. Attention as DB lookup | `03_simple_attention.py`, `04_self_attention.py`, `09_advanced_topics.py` |

---

## 1. The reshape/transpose tax in multi-head attention

The standard path:

```
(B, S, D) → Linear → (B, S, 3*D) → chunk → (B, S, H, 3*d_h) → permute → (B, H, S, d_h)
```

- `view`/`permute` are "free" in PyTorch (stride metadata only), but produce **non-contiguous memory**.
- Non-contiguous layouts prevent GPU tensor cores from using fused load instructions.
- `flash_attention` is 2–5x faster largely because it fuses the entire attention block into a single CUDA kernel, bypassing the reshape pipeline entirely.

**Course references:**
- `course/transformer/05_multi_head_attention.py` — where this reshape pipeline is first built by hand
- `course/transformer/09_advanced_topics.py` — Flash Attention as an advanced topic
- `course/inference/i05_attention_optimizations.py` — FlashAttention tiling and online softmax in detail

### Rethink

Fusion-first design: specify the computation (attention), let the compiler decide layout/materialization. This is the direction `torch.compile` + Triton is heading.

---

## 2. Redundant projection work per head

Multi-head computes `Q = x @ W_Q` where `W_Q` is `(D, H * d_h)` — one big matmul, all heads entangled.

- Mathematically equivalent to H separate smaller projections, but `x` is the same for all heads.
- With H=32, D=4096, it's a `(4096, 4096)` weight. Most heads learn near-redundant patterns.
- Multi-query attention (1 KV head) and grouped-query attention attack this directly by acknowledging KV is over-parameterized.

**Course references:**
- `course/transformer/05_multi_head_attention.py` — builds multi-head attention, where head redundancy first becomes visible
- `course/transformer/09_advanced_topics.py` — covers GQA/MQA as modern alternatives

### Rethink

Split head count from feature dimension. Don't force H to be a tensor axis — treat heads as a routing/parallelism concern, not a data layout concern.

---

## 3. The (S, S) attention matrix is dense by convention, sparse by nature

`softmax(Q @ K^T)` produces an (S, S) matrix — O(S²) compute and memory.

- Most attention weights are near-zero after softmax.
- Tokens mostly attend locally; a few "global" tokens dominate.
- The axis convention forces computing the full dense matrix before using it.
- Sparse variants (sliding window, block-sparse, linformer, reformer, Minference) all fight the convention.

**Course references:**
- `course/transformer/03_simple_attention.py` — introduces attention as a dense pairwise comparison
- `course/transformer/04_self_attention.py` — the full Q@K^T formulation that creates the (S, S) matrix
- `course/transformer/09_advanced_topics.py` — sparse attention variants as advanced topics
- `course/inference/i05_attention_optimizations.py` — attention optimization techniques

### Rethink

Treat the sequence as a **graph**, not a flat axis. Compute attention edges on demand. Hardware-accelerated approximate nearest-neighbor search (ANN) instead of Q@K^T.

---

## 4. Padding waste under fixed-shape conventions

Variable-length sequences force padding to `max(S)` in (B, S, D).

- 1×2048-token seq + 3×32-token seqs → waste `(2048 - 32) * 3 * D` floats per intermediate tensor.
- With D=4096, fp16: ~48 MB wasted per layer.

**Course references:**
- `course/transformer/00_prerequisites.py` — introduces the (B, S, D) shape convention
- `course/inference/i03_batching.py` — batching with padding and masking in practice

### Rethink

Native ragged/variable-length tensor dimensions:
- No padding waste.
- KV caches are naturally jagged.
- Attention skips padding without mask workarounds.

`torch.nested` and JAX `vmap` over variable-length arrays explore this, but ecosystem adoption is slow.

---

## 5. KV-cache: the batch dimension fights you at inference

Autoregressive decoding: KV cache has shape `(B, H, S_past, d_h)`. Each batch item has a different `S_past`.

Options:
- **Pre-allocate max context per item**: wastes `(max_len - prompt_len)` memory.
- **Dynamic concatenation**: reallocation or ring buffer overhead.
- **PagedAttention (vLLM)**: abandons the contiguous (S) axis. KV blocks are a linked list of fixed-size pages. This is an explicit rejection of the axis convention.

**Course references:**
- `course/inference/i02_kv_cache.py` — builds the KV cache from scratch, where the (B, H, S, d_h) shape problem first appears
- `course/inference/i05_attention_optimizations.py` — PagedAttention as the solution to the contiguous-axis problem

### Rethink

Decouple sequence length from tensor layout. Pages/blocks as a first-class data structure, not a contiguity hack.

---

## 6. Broadcasting: semantics that hide temporaries

```python
(B, S, D) + (S, D) → broadcasts to (B, S, D)
```

- Broadcasting conceptually "stretches" the smaller tensor to full size.
- On CPU, real allocation. On GPU, cuBLAS fuses it, but chained broadcasts can cascade temporaries.
- The convention makes this look costless when it isn't.

**Course references:**
- `course/transformer/00_prerequisites.py` — section 1.3 on axis conventions and broadcasting, where `(2,4,3) + (4,3)` is demonstrated

### Rethink

Fusion-first compilation that materializes fewer intermediates. Explicit in-place operations where safe.

---

## 7. State-space models ditch (B, S, D) entirely

Mamba, RWKV, etc. don't have an (S) axis in the core computation.

- State: `(B, D, d_state)` — fixed-size recurrent state, not sequence-length-dependent.
- Per-token cost: O(1), not O(S).
- The (B, S, D) convention was a design choice driven by the transformer's attention mechanism, not an inevitable fact about sequence modeling.

**Course references:**
- `course/transformer/09_advanced_topics.py` — advanced architectures that move beyond standard transformers

### Rethink

Don't store the sequence as a tensor dimension. The sequence is implicit in the recurrence.

---

## 8. Attention as a database operation, not a matmul

Conceptually, attention is: "for each query token, find the most relevant key tokens and retrieve their values."

- That's a **nearest-neighbor search**, not a matrix multiply.
- Current hardware forces a matmul implementation because matmul is fast.
- If hardware had accelerated ANN at the tensor level, you'd index K with Q and retrieve top-k values directly — O(S log S) instead of O(S²).

**Course references:**
- `course/transformer/03_simple_attention.py` — introduces attention as "fuzzy dictionary lookup" — the database metaphor is already there from the start
- `course/transformer/04_self_attention.py` — the full learned projection version
- `course/transformer/09_advanced_topics.py` — MoE routing and other indexing-like patterns

### Rethink

Product-key memory, routed attention, learned indexing structures instead of dense pairwise comparisons.

---

## Core tension

The (B, H, S, D) convention is a local optimum:
- Makes math expressible as matmul + softmax + element-wise ops.
- GPU hardware does these obscenely fast.
- Every inefficiency listed above is a tradeoff that buys this simplicity.

The open question: **do we keep optimizing matmul (bigger tensor cores, better tiling), or build hardware that natively supports the sparse, irregular, graph-structured operations attention actually represents?** Currently, the industry is doubling down on matmul (H100, B200), which reinforces the axis conventions we have.

**Course references:**
- `course/transformer/00_prerequisites.py` — section 2.3 explicitly frames transformers as designed *around* hardware efficiency and matmul
- `course/inference/i00_why_inference_is_hard.py` — the roofline model, memory hierarchy, and why hardware shapes software design
- `course/inference/i09_production.py` — surveys where the field is heading (vLLM, TGI, SGLang)
