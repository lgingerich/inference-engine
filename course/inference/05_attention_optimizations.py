"""
INFERENCE MODULE 5: ATTENTION OPTIMIZATIONS — FlashAttention & PagedAttention
===============================================================================

The attention mechanism is O(n²) in memory: the score matrix Q@K^T
scales with sequence_length². For 4096 tokens, that's ~16.8M entries —
and with 12 heads and FP16, that's ~400 MB for ONE attention matrix
in ONE layer. At inference time, this dominates GPU memory.

Two breakthrough papers fix this: FlashAttention (tiling + online
softmax, O(n) memory) and PagedAttention (virtual memory for KV
cache, zero fragmentation). Together, they're the foundation of
every production inference engine.

WHAT YOU'LL LEARN:
   1. Why materializing the N×N attention matrix is the bottleneck
   2. FlashAttention: tiling + online softmax (O(n) memory!)
   3. PagedAttention: managing KV cache like virtual memory
   4. How these combine to make efficient serving possible

AFTER THIS MODULE:
   You'll understand the two most important papers in LLM serving —
   and why every production engine implements both.
"""

import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY STANDARD ATTENTION DOESN'T SCALE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY ATTENTION DOESN'T SCALE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE N×N PROBLEM — Attention is quadratic in memory             │
└─────────────────────────────────────────────────────────────────┘

Standard attention computes the FULL N×N score matrix:

    scores = Q @ K^T / sqrt(d_k)     → (seq_len, seq_len)
    weights = softmax(scores)         → (seq_len, seq_len)
    output = weights @ V              → (seq_len, d_k)

For GPU inference, this means materializing THREE N×N matrices
(scores, softmax output, attention weights before @V). That's
3 × seq_len² entries per head, per layer.

For batch=8, seq_len=4096, num_heads=12:
  Entries per layer: 8 × 12 × 4096² = 1.6 BILLION floats
  In FP16: 1.6B × 2 bytes = 3.2 GB per layer!
  For a 32-layer LLaMA-7B: 102 GB (more than GPU memory!)

Standard attention CANNOT scale to long contexts. Even prefill
(which processes all prompt tokens at once) would overflow GPU
memory on a 4096-token prompt.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE N×N PROBLEM — Quantifying the Bottleneck
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: THE N×N ATTENTION BOTTLENECK — By the Numbers")
print("=" * 70)

# Concrete numbers
seq_len = 4096
num_heads = 12
batch_size = 8

entries = batch_size * num_heads * seq_len * seq_len
print(f"""
For realistic inference:
  Batch: {batch_size}, Seq: {seq_len}, Heads: {num_heads}

  Q@K^T entries: {batch_size} × {num_heads} × {seq_len} × {seq_len}
                = {entries:,} floats
  FP16 memory:   {entries * 2 / 1024**3:.1f} GB
  FP32 memory:   {entries * 4 / 1024**3:.1f} GB

Memory budget (A100 80GB):
  - Model (7B FP16):  ~14 GB
  - KV cache (4K ctx): ~2 GB
  - Attention matrix:  {entries * 2 / 1024**3:.1f} GB per layer

  → One layer's attention uses {entries * 2 / 1024**3 / 80 * 100:.0f}% of GPU memory!
  → With 32 layers, standard attention is IMPOSSIBLE at this scale.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: FLASH ATTENTION — Tiling + Online Softmax
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: FLASH ATTENTION — NEVER Materialize the Full Matrix")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 2.1  The key insight — GPU memory hierarchy
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  THE KEY INSIGHT — Never write the N×N matrix to HBM       │
└─────────────────────────────────────────────────────────────────┘

Recall from Module 0: GPUs have HBM (big, slow) and SRAM (tiny, fast).
FlashAttention's brilliant idea:

  STANDARD attention:
    Q@K^T [N×N] → written to HBM  (SLOW — 3.2 GB per layer!)
    softmax   → read from HBM, write to HBM (SLOW again)
    @V        → read from HBM, compute

  FLASH attention:
    Load a BLOCK of Q into fast SRAM  (e.g., 128 rows)
    Load a BLOCK of K into fast SRAM  (e.g., 128 columns)
    Compute attention for JUST this tile in SRAM
    Accumulate output using ONLINE softmax
    NEVER write the tile to HBM
    Repeat for all block combinations
    Write ONLY the final output to HBM

The N×N attention matrix is computed, consumed, and DISCARDED
entirely within SRAM. It never touches HBM.
""")


# ═══════════════════════════════════════════════════════════════════
# 2.2  Online softmax — The math that makes tiling possible
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  ONLINE SOFTMAX — Why tiling works mathematically          │
└─────────────────────────────────────────────────────────────────┘

The challenge: standard softmax needs the ENTIRE row to compute
the denominator:
    softmax(x)_i = exp(x_i) / sum(exp(x_j))

If we split x into two blocks x=[x₁, x₂], we can't compute
softmax(x₁) and softmax(x₂) independently — the denominators
would be wrong.

ONLINE SOFTMAX solves this by keeping RUNNING state:

  State: running_max (scalar), running_sum (scalar), running_out (vector)

  For each new block:
    1. Compute block_max and block_sum = sum(exp(x - block_max))
    2. If block_max > running_max:
       RESCALE running_sum and running_out by exp(running_max - new_max)
       Update running_max = block_max
    3. running_sum += block_sum
    4. running_out += exp_block
    5. Final: result = running_out / running_sum

The rescaling step is the magic: exp(x - old_max) * exp(old_max - new_max)
= exp(x - new_max). This lets us FIX all previous results when we
discover a larger max in a later block.

This produces EXACTLY the same result as standard softmax — it's
not an approximation. The proof follows from the identity:
    exp(a-c) / sum(exp(a-c)) = exp(a) / sum(exp(a)) for any c.
""")


def online_softmax_demo(blocks):
    """Show how online softmax fuses results from multiple blocks.

    Each block contains DIFFERENT elements of the input. We process
    blocks one at a time, keeping a running max/sum/output that
    gets RESCALED when we discover a new max in a later block.
    """
    running_max = float('-inf')
    running_sum = 0.0
    running_out_parts = []

    for block_idx, block in enumerate(blocks):
        block_max = block.max()
        new_max = max(running_max, block_max)

        if block_max > running_max:
            scale = np.exp(running_max - new_max) if running_max != float('-inf') else 1.0
            running_sum *= scale
            for i in range(len(running_out_parts)):
                running_out_parts[i] = running_out_parts[i] * scale
            running_max = new_max

        exp_block = np.exp(block - running_max)
        block_sum = exp_block.sum()
        running_sum += block_sum
        running_out_parts.append(exp_block)

        print(f"    Block {block_idx}: max={block_max:.2f}, "
              f"running_max={running_max:.2f}, running_sum={running_sum:.2f}")

    running_out = np.concatenate(running_out_parts)
    return running_out / running_sum


# Demonstrate online softmax
np.random.seed(42)
x = np.random.randn(12) * 2
blocks = [x[0:4], x[4:8], x[8:12]]

print(f"\nOnline softmax: 12 numbers, 3 blocks of 4:")
print(f"  Full vector:    {np.round(x, 2)}")
print(f"  Standard softmax: {np.round(softmax(x), 4)}")

online_result = online_softmax_demo(blocks)
print(f"  Online softmax:   {np.round(online_result, 4)}")
print(f"  Max difference:   {np.abs(softmax(x) - online_result).max():.10f}")
print(f"  → IDENTICAL! Online softmax is exact, not an approximation.")


# ═══════════════════════════════════════════════════════════════════
# 2.3  Tiled attention — A simplified FlashAttention
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: TILED ATTENTION — FlashAttention in NumPy")
print("=" * 70)


def tiled_attention_simplified(Q, K, V, block_size=2):
    """Simplified tiled attention — FlashAttention concept in NumPy.

    Processes Q@K^T@V in BLOCKS, using online softmax to accumulate
    results without ever materializing the full attention matrix.

    This is the same algorithm as FlashAttention, but in NumPy
    (which runs on CPU, not GPU). Real FlashAttention is a CUDA
    kernel that operates directly on SRAM.
    """
    seq_len, d_k = Q.shape
    output = np.zeros_like(Q)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        Q_block = Q[q_start:q_end]

        running_max = np.full((q_end - q_start), float('-inf'))
        running_sum = np.zeros(q_end - q_start)
        running_out = np.zeros((q_end - q_start, d_k))

        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start + block_size, seq_len)
            K_block = K[k_start:k_end]
            V_block = V[k_start:k_end]

            scores = Q_block @ K_block.T / np.sqrt(d_k)
            block_max = scores.max(axis=1)
            new_max = np.maximum(running_max, block_max)

            scale = np.exp(running_max - new_max)
            running_sum *= scale
            running_out *= scale[:, np.newaxis]

            exp_scores = np.exp(scores - new_max[:, np.newaxis])
            block_sum = exp_scores.sum(axis=1)
            running_sum += block_sum
            running_out += exp_scores @ V_block
            running_max = new_max

        output[q_start:q_end] = running_out / running_sum[:, np.newaxis]

    return output


# Test vs standard attention
np.random.seed(123)
test_seq, test_dk = 8, 4
Q_t = np.random.randn(test_seq, test_dk)
K_t = np.random.randn(test_seq, test_dk)
V_t = np.random.randn(test_seq, test_dk)

scores = Q_t @ K_t.T / np.sqrt(test_dk)
attn_standard = softmax(scores) @ V_t
attn_tiled = tiled_attention_simplified(Q_t, K_t, V_t, block_size=3)

print(f"\nTesting: seq={test_seq}, d_k={test_dk}, block_size=3:")
print(f"  Standard: {attn_standard.shape}")
print(f"  Tiled:    {attn_tiled.shape}")
print(f"  Max diff: {np.abs(attn_standard - attn_tiled).max():.10f}")
print(f"  → EXACT output, but N×N matrix never fully materialized!")
print(f"  → Memory: O(block_size²) instead of O(seq_len²)")


# ═══════════════════════════════════════════════════════════════════
# PART 4: PAGED ATTENTION — Virtual Memory for KV Cache
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: PAGED ATTENTION — Virtual Memory for the KV Cache")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE FRAGMENTATION PROBLEM                                 │
└─────────────────────────────────────────────────────────────────┘

In standard KV cache, each request gets CONTIGUOUS memory. Problem:

  Request A: needs 500 token slots → allocates [0...499]
  Request B: needs  10 token slots → allocates [500...509]
  Request A finishes → frees [0...499] (500 slots free!)
  Request B continues → still using [500...509]
  Request C: needs 200 slots → CAN'T FIT! (only 500 free, but not contiguous)
  
  AVAILABLE: 500 slots. USABLE: 0 slots. INTERNAL FRAGMENTATION!

Plus: you don't know output length in advance. Pre-allocating is
guesswork. Overallocate → waste. Underallocate → crash.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  THE SOLUTION — Page the KV cache                          │
└─────────────────────────────────────────────────────────────────┘

PagedAttention applies OS virtual memory to the KV cache:

  1. Divide KV cache into FIXED-SIZE BLOCKS (e.g., 16 tokens each)
  2. Each request gets a LIST of block indices (not contiguous memory)
  3. A "block table" maps request → list of physical block numbers
  4. When a request needs more space, allocate another block from
     the free pool — NO CONTIGUITY REQUIRED!

  Request A: [block_5, block_12, block_3, block_7]
  Request B: [block_1, block_9]
  Free pool: [block_0, block_2, block_4, block_6, block_8, ...]

  When A finishes: blocks 5,12,3,7 go back to free pool.
  Now ANY new request can use those blocks — no fragmentation!

During attention, the block table tells us where to find each
block's K and V. We gather them on-the-fly:
  K_full = [K_block[5], K_block[12], K_block[3], K_block[7]]
""")


class PagedKVCache:
    """Simplified paged KV cache — vLLM's core data structure."""

    def __init__(self, num_blocks, block_size, num_layers, num_heads, d_k):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.total_slots = num_blocks * block_size

        self.physical_blocks = {}
        for layer in range(num_layers):
            self.physical_blocks[layer] = np.zeros(
                (num_blocks, block_size, num_heads, d_k))

        self.block_table = {}
        self.free_blocks = list(range(num_blocks))

    def allocate(self, request_id, num_blocks_needed):
        if len(self.free_blocks) < num_blocks_needed:
            return False, "Not enough free blocks!"
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.block_table[request_id] = allocated
        return True, f"Allocated {num_blocks_needed} blocks: {allocated}"

    def free(self, request_id):
        if request_id in self.block_table:
            blocks = self.block_table.pop(request_id)
            self.free_blocks.extend(blocks)
            return len(blocks)
        return 0


# Demonstrate paging
cache = PagedKVCache(num_blocks=20, block_size=16,
                     num_layers=2, num_heads=4, d_k=8)

print(f"\nPaged KV Cache demo ({cache.total_slots} total slots):")

ok, msg = cache.allocate("req_A", 5)
print(f"  {msg}")
ok, msg = cache.allocate("req_B", 3)
print(f"  {msg}")

n = cache.free("req_A")
print(f"  Freed A: {n} blocks returned to pool")

ok, msg = cache.allocate("req_C", 7)
print(f"  {msg}")
print(f"  Block table: {cache.block_table}")
print(f"  → Blocks are NON-contiguous — like OS virtual memory!")

print(f"""
BENEFITS OF PAGING:
  1. ~40% higher memory utilization (eliminates fragmentation)
  2. Memory SHARING: two requests with same system prompt
     share physical blocks (copy-on-write)
  3. Arbitrary output lengths — just allocate more blocks as needed
  4. This is what makes vLLM's throughput so high
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: BEYOND FLASH ATTENTION — The Current Landscape
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: BEYOND FLASH ATTENTION — Modern Variants")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE FLASH ATTENTION FAMILY TREE                                │
└─────────────────────────────────────────────────────────────────┘

FlashAttention-2 (Dao, 2023):
  - Better parallelism strategy: parallelize over sequence length
    dimension rather than batch/heads.
  - 2× speedup over FlashAttention-1 on forward pass.
  - NOW THE DEFAULT in PyTorch:
    torch.nn.functional.scaled_dot_product_attention()

FlashAttention-3 (Shah et al., 2024):
  - Designed for H100 GPUs — exploits async data movement.
  - FP8 support: even faster with lower-precision attention.
  - 1.5-2× faster than FlashAttention-2 on H100.

GQA + FlashAttention:
  - GQA (Module 7 of transformers course) reduces K/V heads.
  - FlashAttention's tiling works naturally with GQA — just fewer
    K/V blocks to load per Q block.
  - This is the CURRENT STATE OF THE ART for production inference.

Ring Attention (for multi-GPU, ultra-long contexts):
  - Splits sequence length across GPUs in a ring topology.
  - Each GPU passes its K/V blocks to the next GPU.
  - Enables 1M+ token contexts across many GPUs.
  - Used by Gemini 1.5 Pro for its 2M token context window.
""")


print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ ATTENTION OPTIMIZATIONS — From O(n²) to O(n) memory            │
│                                                                 │
│ 1. The N×N attention matrix is GB-sized for long sequences.    │
│    Standard attention CANNOT serve long contexts.              │
│                                                                 │
│ 2. FLASH ATTENTION: tile computation with online softmax.      │
│    → O(n) memory instead of O(n²).                             │
│    → 2-4× faster (less HBM traffic).                           │
│    → EXACT same output (not an approximation!).                │
│    → Enables training/inference on 64K+ contexts.              │
│                                                                 │
│ 3. PAGED ATTENTION: virtual memory for KV cache.               │
│    → Eliminates internal fragmentation (~40% more utilization). │
│    → Enables KV cache sharing between requests.                │
│    → The core innovation that makes vLLM's throughput so high. │
│                                                                 │
│ 4. Together, they form the foundation of every production      │
│    inference engine (vLLM, TGI, SGLang, llama.cpp).            │
└─────────────────────────────────────────────────────────────────┘

Next: Module 6 — Speculative Decoding (draft + verify for speed)
""")

if __name__ == "__main__":
    print("\nModule 5 complete! Next: i06_speculative_decoding.py")
    print("Run with: uv run python course/inference/i06_speculative_decoding.py")
