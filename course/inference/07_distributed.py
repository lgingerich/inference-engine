"""
INFERENCE MODULE 7: DISTRIBUTED INFERENCE — Splitting Across GPUs
===================================================================

A 70B parameter model in FP16 is 140 GB. No single GPU has that much
VRAM (A100 max is 80 GB). Even with INT4 quantization (35 GB), it
doesn't fit on a consumer RTX 4090 (24 GB) once you add the KV cache.

To serve large models, we SPLIT them across multiple GPUs. This module
covers tensor parallelism (splitting weight matrices) and pipeline
parallelism (splitting layers), plus the communication that makes it work.

WHAT YOU'LL LEARN:
   1. Why single-GPU inference hits a hard memory ceiling
   2. Tensor parallelism: column-wise and row-wise weight splitting
   3. Pipeline parallelism: splitting layers across GPUs
   4. GPU communication primitives and their costs
   5. How production systems combine TP + PP for 70B+ models

AFTER THIS MODULE:
   You'll understand what `tensor_parallel_size=4` means in vLLM
   and why 8× A100 can serve a 175B model.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: THE MEMORY CEILING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: THE SINGLE-GPU CEILING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  WHY ONE GPU ISN'T ENOUGH — The memory budget              │
└─────────────────────────────────────────────────────────────────┘

Inference memory = model weights + KV cache + overhead

  GPU           |  VRAM    |  Max FP16 Model  |  Max INT4 Model
  ──────────────┼──────────┼──────────────────┼──────────────────
  RTX 3060      |  12 GB   |  ~5B  (barely)   |  ~20B
  RTX 4090      |  24 GB   |  ~10B            |  ~40B
  A100 80GB     |  80 GB   |  ~35B            |  ~140B
  H100 80GB     |  80 GB   |  ~35B            |  ~140B
  2× A100       | 160 GB   |  ~75B            |  ~300B
  8× A100       | 640 GB   |  ~300B           |  ~1.2T

LLaMA-70B (70B params): FP16 = 140 GB → needs at least 2× A100
                        INT4 = 35 GB → fits 1× A100 but not 4090

Even with INT4, the KV cache can push you over the limit.
For 4096 context, batch=32: KV cache adds ~8 GB → 43 GB total.

DISTRIBUTED INFERENCE needs two techniques combined:
  1. TENSOR PARALLELISM: split each LAYER's weights across GPUs
  2. PIPELINE PARALLELISM: split LAYERS across GPUs
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: TENSOR PARALLELISM — Splitting Weight Matrices
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: TENSOR PARALLELISM — Splitting Inside Each Layer")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 2.1  Column-wise split (the workhorse of TP)
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  COLUMN-WISE SPLIT — Each GPU stores half the columns      │
└─────────────────────────────────────────────────────────────────┘

For a linear layer y = x @ W with W shape (d_in, d_out):

  Split W = [W_left | W_right] across 2 GPUs

  GPU 0: W_left  (d_in, d_out/2)
  GPU 1: W_right (d_in, d_out/2)

  FORWARD:
    x is broadcast to both GPUs (same input, different weights)
    GPU 0: y_left  = x @ W_left   (batch, d_out/2)
    GPU 1: y_right = x @ W_right  (batch, d_out/2)
    ALL-GATHER: y = [y_left | y_right]   ← communication!

  WHY THIS FOR ATTENTION:
    W_Q, W_K, W_V are split column-wise across GPUs.
    Each GPU computes attention for its SUBSET of heads.
    Natural fit: heads are independent, so no communication needed
    during the attention computation itself!
""")

np.random.seed(42)
d_in, d_out = 4, 8
x = np.random.randn(2, d_in)
W = np.random.randn(d_in, d_out) * 0.5

# Single GPU baseline
y_single = x @ W

# Column-wise TP across 2 "GPUs"
split = d_out // 2
W_gpu0 = W[:, :split]
W_gpu1 = W[:, split:]

y_gpu0 = x @ W_gpu0
y_gpu1 = x @ W_gpu1
y_parallel = np.concatenate([y_gpu0, y_gpu1], axis=1)

print(f"\nColumn-wise TP: (2,{d_in}) @ ({d_in},{d_out})")
print(f"  GPU 0: stores {W_gpu0.shape}, computes {y_gpu0.shape}")
print(f"  GPU 1: stores {W_gpu1.shape}, computes {y_gpu1.shape}")
print(f"  Combined: {y_parallel.shape}")
print(f"  Match: {np.allclose(y_single, y_parallel)}")
print(f"  → IDENTICAL result, each GPU uses HALF the memory!")


# ═══════════════════════════════════════════════════════════════════
# 2.2  Row-wise split (for output projections)
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  ROW-WISE SPLIT — Used for the output projection W_O       │
└─────────────────────────────────────────────────────────────────┘

Split W = [W_top; W_bottom] across 2 GPUs

  GPU 0: W_top    (d_in/2, d_out) — stores top half of rows
  GPU 1: W_bottom (d_in/2, d_out) — stores bottom half

  FORWARD:
    x is split: x_top (on GPU 0), x_bottom (on GPU 1)
    GPU 0: y_0 = x_top @ W_top    (batch, d_out)
    GPU 1: y_1 = x_bottom @ W_bottom (batch, d_out)
    ALL-REDUCE: y = y_0 + y_1    ← communication!

  WHY THIS FOR W_O:
    After multi-head attention, each GPU has partial outputs for its
    subset of heads. W_O combines them. Row-wise split means each GPU
    computes part of the combination, then we sum (all-reduce).

┌─────────────────────────────────────────────────────────────────┐
│ 2.3  FFN SPLITTING — The efficient pattern                     │
└─────────────────────────────────────────────────────────────────┘

For FFN: W1 (d_model → d_ff) and W2 (d_ff → d_model)

  W1: column-wise split. W2: row-wise split.

  This is efficient because:
    GPU 0: y = (x @ W1_left) @ W2_top  (activation never leaves GPU!)
    GPU 1: y = (x @ W1_right) @ W2_bottom
    ALL-REDUCE: sum partial outputs

  The intermediate activations (d_ff-sized) stay on the same GPU
  between W1 and W2 — no communication needed between the two
  FFN matmuls. Only the final output needs all-reduce.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: PIPELINE PARALLELISM — Splitting Layers
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 3: PIPELINE PARALLELISM — Splitting Across Layers")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  HOW PIPELINE PARALLELISM WORKS                            │
└─────────────────────────────────────────────────────────────────┘

Instead of splitting WITHIN layers, split LAYERS across GPUs:

  GPU 0: Embeddings + Blocks 0-3  (4 complete layers)
  GPU 1: Blocks 4-7               (4 complete layers)
  GPU 2: Blocks 8-11              (4 complete layers)
  GPU 3: Blocks 12-15 + LM Head   (4 layers + output)

Single request flow: GPU 0 → GPU 1 → GPU 2 → GPU 3
Only ONE GPU is active at a time → terrible for single-batch latency!

┌─────────────────────────────────────────────────────────────────┐
│ 3.2  MICRO-BATCHING — How PP becomes efficient                  │
└─────────────────────────────────────────────────────────────────┘

With multiple concurrent requests, we use MICRO-BATCHES:

  Time →
  GPU 0: [A0][A1][A2][A3][B0][B1][B2][B3]...
  GPU 1:    [A0][A1][A2][A3][B0][B1][B2]...
  GPU 2:       [A0][A1][A2][A3][B0][B1]...
  GPU 3:          [A0][A1][A2][A3][B0]...

While GPU 1 processes A's later layers, GPU 0 is ALREADY working on
B's earlier layers. The "bubble" (idle time at start/end) shrinks as
micro-batches increase.

TRADEOFF:
  Tensor Parallelism: more communication, works within ONE layer
  Pipeline Parallelism: less communication, has pipeline bubbles
  → Production uses BOTH: TP within a node (NVLink fast), PP across
    nodes (network slower, but PP needs less communication)
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: GPU COMMUNICATION PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 4: GPU COMMUNICATION — What Makes It Possible")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE THREE KEY COLLECTIVE OPERATIONS                       │
└─────────────────────────────────────────────────────────────────┘

ALL-REDUCE (for row-wise TP):
  Each GPU has a partial sum. Sum them and broadcast.

  GPU 0: [1, 2, 3]  ─┐
  GPU 1: [4, 5, 6]  ─┤─ SUM → [7, 10, 15] → broadcast to all
  GPU 2: [2, 3, 4]  ─┘

  Cost for tensor size S across P GPUs: 2S × (P-1)/P bytes transferred

ALL-GATHER (for column-wise TP):
  Each GPU has a chunk. Concatenate and broadcast.

  GPU 0: [A, B]  ─┐
  GPU 1: [C, D]  ─┤─ CONCAT → [A, B, C, D] → broadcast to all

  Cost for tensor size S across P GPUs: S × (P-1)/P bytes transferred

REDUCE-SCATTER:
  Inverse of all-gather. Sum chunks and distribute different parts
  to different GPUs. Used in some fused attention patterns.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  INTERCONNECT SPEED — Why topology matters                 │
└─────────────────────────────────────────────────────────────────┘

  NVLink (A100, within a node):    600 GB/s per link
  NVSwitch (H100, within a node):  900 GB/s per GPU
  PCIe 4.0 (across nodes):         ~32 GB/s (much slower!)
  InfiniBand NDR400 (across nodes): 400 GB/s

This is why TP uses NVLink inside a node, while PP can use
slower inter-node connections (it communicates less data).
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: PRODUCTION EXAMPLE — LLaMA-70B on 4 GPUs
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: REAL-WORLD EXAMPLE — LLaMA-70B on 4× A100")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  CONFIGURATION: 2-way TP × 2-way PP                        │
└─────────────────────────────────────────────────────────────────┘

LLaMA-70B: d_model=8192, 80 layers, 64 Q heads, 8 K/V heads (GQA)

  ┌─────────────────────────────────────────┐
  │ Node (NVLink/NVSwitch)                  │
  │  ┌──────────┐  NVLink  ┌──────────┐    │
  │  │  GPU 0   │◄────────►│  GPU 1   │    │  ← TP pair: each stores
  │  │Layers 0-39│  600GB/s │Layers 0-39│   │    half of layers 0-39
  │  │(TP rank 0)│          │(TP rank 1)│   │
  │  └──────────┘          └──────────┘    │
  │  ┌──────────┐  NVLink  ┌──────────┐    │
  │  │  GPU 2   │◄────────►│  GPU 3   │    │  ← TP pair: each stores
  │  │Layers40-79│  600GB/s │Layers40-79│   │    half of layers 40-79
  │  │(TP rank 0)│          │(TP rank 1)│   │
  │  └──────────┘          └──────────┘    │
  └─────────────────────────────────────────┘
    GPU 0,2 ←→ GPU 1,3: pipeline communication (pass activations)

Memory per GPU (FP16):
  Model weights: 70B × 2 / 4 = 35 GB
  KV cache (4K ctx, batch=32): ~8 GB
  Total: ~43 GB (fits in 80 GB!)

Throughput: ~10-20 req/s (depends on prompt/output lengths)
Latency:   ~100-500ms for typical chat

┌─────────────────────────────────────────────────────────────────┐
│ 5.2  SCALING FURTHER — What 8 GPUs buys you                    │
└─────────────────────────────────────────────────────────────────┘

  8× A100 with TP=4, PP=2:
    - 175B models (like GPT-3) with 16K context
    - ~100+ req/s throughput depending on batch settings
    - $25-40/GPU-hour × 8 = $200-320/hr to run at full capacity

  Multi-node (16+ GPUs):
    - 400B+ models with 64K+ context
    - Expert parallelism for MoE models (Mixtral 8×7B)
    - Sequence parallelism for ultra-long contexts (1M+ tokens)
""")


print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ DISTRIBUTED INFERENCE — Scaling beyond one GPU                 │
│                                                                 │
│ 1. Single GPU memory is limited: 80 GB max (A100/H100).        │
│    A 70B model in FP16 = 140 GB → MUST split.                  │
│                                                                 │
│ 2. TENSOR PARALLELISM: split each layer's weight matrices      │
│    column-wise or row-wise. Each GPU stores a fraction.        │
│    Communication: all-gather / all-reduce after each matmul.   │
│    Works best with fast interconnects (NVLink 600 GB/s).       │
│                                                                 │
│ 3. PIPELINE PARALLELISM: split layers across GPUs.             │
│    Less communication, but GPUs idle between pipeline stages.  │
│    Micro-batching reduces idle time.                           │
│                                                                 │
│ 4. PRODUCTION: vLLM uses TP (intra-node, NVLink) + PP          │
│    (inter-node, InfiniBand) combined. This serves 70B-175B     │
│    models efficiently.                                          │
└─────────────────────────────────────────────────────────────────┘

Next: Module 8 — The Serving Stack (HTTP API, streaming, metrics)
""")

if __name__ == "__main__":
    print("\nModule 7 complete! Next: i08_serving.py")
    print("Run with: uv run python course/inference/i08_serving.py")
