"""
INFERENCE MODULE 3: BATCHING & SCHEDULING
===========================================

So far we've been serving ONE request at a time. That's like a restaurant
with one table — you're leaving your GPU (the kitchen) mostly empty.

Batching processes MULTIPLE requests together in one forward pass. The GPU
gets more work per step → higher throughput. But it introduces complexity:
requests have different lengths, start at different times, and need to
feel responsive to each user.

WHAT YOU'LL LEARN:
   1. Why batching is essential for throughput
   2. How to pad variable-length sequences and mask attention
   3. Static batching (wait for all to finish) vs continuous batching
   4. Building a simple continuous batching scheduler
   5. The throughput-latency tradeoff

After this module, you'll understand what "throughput" means for LLM serving
and why every production inference engine — vLLM, TensorRT-LLM, SGLang —
uses continuous batching as its core scheduling primitive.
"""

import time
import heapq
import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS BATCHING SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════
#
# The key insight of continuous batching:
#   - Don't wait for ALL requests in a batch to finish
#   - When one completes, immediately slot in a new one
#   - The batch is DYNAMIC — requests join and leave fluidly
#
# This keeps the GPU busy without blocking behind slow completions.
# ═══════════════════════════════════════════════════════════════════════════════


class ContinuousBatchingScheduler:
    """A miniature continuous batching scheduler.

    This manages multiple inference requests, processing them in
    a shared batch and dynamically adding/removing requests as
    they complete or new ones arrive.

    Each request has:
      - prompt_ids: the input tokens
      - max_new_tokens: how many tokens to generate
      - state: 'waiting' | 'prefilling' | 'decoding' | 'done'
      - kv_cache: per-layer K/V cache (like Module 2)
      - generated: list of generated token IDs so far
    """

    def __init__(self, model, max_batch_size=4):
        self.model = model
        self.max_batch_size = max_batch_size

    def run(self, requests, verbose=True):
        """Process a list of requests with continuous batching.

        Args:
            requests: list of dicts with 'prompt_ids' and 'max_tokens'
            verbose: whether to print per-step info

        Returns:
            list of generated token sequences
        """
        # Initialize all requests
        active = []  # requests currently being processed
        waiting = list(requests)  # requests not yet started

        if verbose:
            print(f"\nStarting scheduler with {len(requests)} requests, "
                  f"max batch size={self.max_batch_size}")

        step = 0
        while active or waiting:
            if verbose and step % 5 == 0:
                print(f"\n  Step {step}: {len(active)} active, "
                      f"{len(waiting)} waiting")

            # ── 1. Check which requests have finished ──
            newly_done = []
            for req in active:
                if len(req['generated']) >= req['max_tokens']:
                    newly_done.append(req)

            for req in newly_done:
                active.remove(req)
                req['state'] = 'done'
                if verbose:
                    print(f"    ✓ Request complete: {len(req['generated'])} tokens")

            # ── 2. Add new requests if there's room ──
            while waiting and len(active) < self.max_batch_size:
                new_req = waiting.pop(0)
                new_req['state'] = 'prefilling'
                new_req['generated'] = []
                new_req['prompt_processed'] = False
                active.append(new_req)
                if verbose:
                    print(f"    + New request added (prompt_len={len(new_req['prompt_ids'])})")

            # ── 3. If nothing active, we're done ──
            if not active:
                break

            # ── 4. Process one step for all active requests ──
            self._step(active, verbose=(verbose and step % 10 == 0))

            step += 1

        # Collect results
        results = []
        for req in requests:
            results.append(req['prompt_ids'] + req['generated'])

        return results

    def _step(self, active_requests, verbose=False):
        """Run one forward pass on the active batch.

        This is a SIMPLIFIED version that processes each request
        separately (no true batch matmul). In a real engine, all
        requests would be packed into a single tensor for one GPU
        launch. But the scheduling logic is identical.
        """
        for req in active_requests:
            if req['state'] == 'prefilling' and not req['prompt_processed']:
                # ── PREFILL: process entire prompt once ──
                batch = np.array([req['prompt_ids']])
                logits = self.model.forward(batch)
                req['prompt_processed'] = True
                req['state'] = 'decoding'

                # Generate first token
                next_logits = logits[0, -1, :]
                probs = softmax(next_logits / 0.8)
                next_token = int(np.random.choice(self.model.vocab_size, p=probs))
                req['generated'].append(next_token)
                req['last_logits'] = logits  # store for next step (simulating cache)

                if verbose:
                    print(f"    Prefilled (len={len(req['prompt_ids'])}), "
                          f"first token={next_token}")

            elif req['state'] == 'decoding':
                # ── DECODE: generate one more token ──
                # In a real engine with KV cache, we'd only process the new token.
                # Here we re-process everything (like the raw loop) to keep it simple.
                # Module 2 showed how to add KV cache; Module 3 is about scheduling.
                current = req['prompt_ids'] + req['generated']
                # Truncate if needed
                if len(current) > self.model.max_seq_len:
                    current = current[-self.model.max_seq_len:]

                batch = np.array([current])
                logits = self.model.forward(batch)

                next_logits = logits[0, -1, :]
                probs = softmax(next_logits / 0.8)
                next_token = int(np.random.choice(self.model.vocab_size, p=probs))
                req['generated'].append(next_token)
                req['last_logits'] = logits

                if verbose:
                    print(f"    Decode: token={next_token}, "
                          f"seq_len={len(current)}")

            # Check if this request has reached its max
            if len(req['generated']) >= req['max_tokens']:
                req['state'] = 'done'


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY BATCHING IS CRITICAL FOR LLM SERVING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY BATCHING?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE GPU UTILIZATION PROBLEM                                     │
│                                                                 │
│ A single inference request on a modern GPU is like driving a    │
│ single passenger in a 50-seat bus: the hardware is capable of   │
│ far more work per clock cycle than one sequence can fill.       │
│                                                                 │
│ GPU matmul units (Tensor Cores) are designed for LARGE matrix   │
│ multiplies — hundreds or thousands of rows at once. A single    │
│ sequence is at most a few thousand tokens wide — nowhere near   │
│ saturating the hardware.                                        │
│                                                                 │
│ Batching is the answer: pack multiple independent sequences    │
│ into the batch dimension so the GPU processes them together     │
│ in a single kernel launch.                                      │
└─────────────────────────────────────────────────────────────────┘

A BRIEF HISTORY OF LLM BATCHING:

  2017–2020: NO BATCHING
    Early transformer serving (GPT-1, BERT) processed one request
    at a time. Acceptable for research demos, unusable for production.

  2020–2022: STATIC BATCHING
    HuggingFace and early ONNX runtimes grouped requests into fixed
    batches. All requests in a batch must finish before the next batch
    starts. The slowest request becomes the bottleneck for everyone.

  2023: CONTINUOUS BATCHING (vLLM, "PagedAttention" paper)
    Kwon et al. introduced the key insight: don't wait for the whole
    batch to finish. As individual requests complete, immediately
    replace them with new ones from the queue. The batch is fluid.
    This one change nearly doubled throughput in production.

  2024–present: ADVANCED SCHEDULING
    Chunked prefill (split long prefills across steps), prefix caching
    (reuse KV cache for shared prompts), disaggregated prefill/decode
    (separate servers for the two phases). All build on the continuous
    batching foundation.

TODAY: Every production inference engine (vLLM, TensorRT-LLM, SGLang,
LMDeploy) uses continuous batching. It is the table stakes for LLM
serving — you cannot compete without it.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: WHY BATCHING MATTERS
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: WHY BATCHING MATTERS")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THE THROUGHPUT PROBLEM — Why one-at-a-time is wasteful    │
└─────────────────────────────────────────────────────────────────┘

When you serve requests sequentially, the GPU is IDLE for most of
the time that matters. Consider three requests arriving simultaneously:

Without batching:
    Request A: [===prefill===][decode][decode][decode][done]
    Request B:                              [===prefill===][decode][decode][done]
    Request C:                                                          [===prefill===][decode][done]
    → GPU idle between requests!

An A100 GPU can perform 312 TFLOPS of FP16 compute. A single GPT-2
forward pass uses ~0.01 TFLOPS. Without batching, you're using
~0.003% of the GPU's capacity. The rest is wasted.

WHY THIS MATTERS ECONOMICALLY:
  - GPU cloud instances cost $2–5/hour
  - At 1 request/second on an A100, your cost per token is astronomical
  - Batching 32 requests = 32× better cost efficiency
  - Your margins, latency SLAs, and user experience all depend on it
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  STATIC BATCHING — The first attempt (and why it fails)    │
└─────────────────────────────────────────────────────────────────┘

Static batching groups N requests, processes them together, and
prevents ANY from completing until ALL have finished decoding.

With static batching (group size=3):
    Request A: [===prefill===][decode][decode][decode][done]
    Request B: [===prefill===][decode][decode][done]   ...waiting...
    Request C: [===prefill===][decode][done]           ...waiting.........waiting...

Requests B and C finish early but must wait for A. This is the
"head-of-line blocking" problem — the slowest request in the batch
dictates latency for ALL requests in that batch.

THE FUNDAMENTAL PROBLEM:
  Output lengths are UNKNOWN in advance. You cannot predict which
  request will be the slowest. Static batching guarantees that
  some GPU cycles are wasted waiting.

With CONTINUOUS batching:
    Request A: [===prefill===][decode][decode][decode][decode][done]
    Request B:                 [===prefill===][decode][decode][done]
    Request C:                                [===prefill===][decode][decode][done]
    Request D:                                              [===prefill===][decode][done]
    → New requests start as soon as slots open up!
    → GPU stays busy, no one waits for slow requests!
    → Each request experiences latency proportional to its OWN length, not the max

This is the core insight that made continuous batching the industry
standard. The batch is never "stuck" — it's always evolving.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: PADDING AND MASKING — The Mechanics of Batching
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: HOW BATCHING ACTUALLY WORKS — Padding & Masking")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY PADDING IS NECESSARY — GPUs demand rectangles        │
└─────────────────────────────────────────────────────────────────┘

In a GPU, all sequences in a batch must be the SAME LENGTH because
matrix multiplication operates on rectangular tensors. You cannot
multiply a (3, 5) matrix by a (3, 3) matrix side by side — the
dimensions must align.

Since requests have naturally different lengths:

    Request A: [ 1,  2,  3,  4,  5]  → 5 tokens
    Request B: [10, 20, 30]          → 3 tokens
    Request C: [99]                  → 1 token

We PAD shorter sequences to match the longest:

    Batch tensor (3 × 5):
       [[ 1,  2,  3,  4,  5],   ← Request A (5 tokens)
        [10, 20, 30,  0,  0],   ← Request B (padded to 5)
        [99,  0,  0,  0,  0]]   ← Request C (padded to 5)

WHY ZERO? Zero is a natural choice for padding because:
  1. The embedding for token 0 is whatever the model learned during
     training (typically a neutral "no information" vector).
  2. Combined with the padding mask, zero embeddings contribute
     exactly zero to the attention output — they're invisible.
  3. In practice, <PAD> is often token ID 0, though GPT uses
     <|endoftext|> (50256) and doesn't pad at all (it packs
     sequences end-to-end instead).

THE COST OF PADDING: Padding wastes compute. In the example above,
we compute attention for 15 positions but only 9 are real. With
large batch sizes and widely varying lengths, padding overhead can
be 30-50% of total compute. This is why techniques like "packing"
(concatenating sequences) and "flash attention with variable
lengths" (no padding needed) are active research areas.
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  WHY MASKING IS NECESSARY — Padding must be invisible      │
└─────────────────────────────────────────────────────────────────┘

Padding tokens (0) are FAKE — they don't represent real content.
Without masking, the model would attend to them as if they were
meaningful words, corrupting the output.

The PADDING MASK is -inf for padding positions, 0 for real tokens:

           positions  →   0    1    2    3    4
    Request A (len=5): [  0,   0,   0,   0,   0]   ← All real (no padding)
    Request B (len=3): [  0,   0,   0, -∞,  -∞]   ← Last two are padding
    Request C (len=1): [  0,  -∞,  -∞,  -∞,  -∞]   ← Only first is real

HOW IT WORKS in the attention mechanism:

    Before softmax:  attention_scores = Q @ K^T / sqrt(d_k)
    Add mask:        attention_scores = attention_scores + mask
    After softmax:   attention_weights = softmax(attention_scores)

    -∞ in the mask becomes 0 after softmax (e^-∞ = 0),
    so padding tokens contribute NOTHING to the attention output.

    This works because attention is a WEIGHTED SUM:
      output = Σ(attention_weight_i × value_i)

    If attention_weight_i = 0, that value is completely ignored.
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.3  THE FULL COMBINED MASK — Causal + Padding together        │
└─────────────────────────────────────────────────────────────────┘

In a real inference engine, you always need TWO masks simultaneously:

  1. CAUSAL MASK: Prevents attending to future tokens.
     A lower-triangular matrix where position i can see positions
     0 through i (inclusive), but not i+1 and beyond.
     Shape: (seq_len, seq_len), values: 0 (allowed) or -∞ (blocked).

  2. PADDING MASK: Prevents attending to padding tokens.
     Broadcast from (batch, 1, 1, seq_len) to (batch, heads, seq_len, seq_len).
     Values: 0 (real) or -∞ (padding) for keys being attended TO.

The combined mask = causal_mask + padding_mask. Both are added
to the attention scores before softmax. The effect:

  - Causal mask blocks future positions (bottom-left stays -∞)
  - Padding mask blocks padding key positions (right columns become -∞)

For a batch with A=[a,b,c], B=[d,e,pad], C=[f,pad,pad]:

  Combined mask for sequence A (all real, lower triangular):
       keys →   0    1    2
    query 0: [  0, -∞,  -∞]   ← position 0 sees only itself
    query 1: [  0,   0,  -∞]   ← position 1 sees 0 and 1
    query 2: [  0,   0,   0]   ← position 2 sees 0, 1, and 2

  Combined mask for sequence B (last position is padding):
       keys →   0    1    2
    query 0: [  0, -∞,  -∞]   ← position 0 sees only itself
    query 1: [  0,   0,  -∞]   ← position 1 sees 0 and 1, but 2 is padding = blocked
    query 2: [  0,   0,  -∞]   ← position 2 is padding itself (ignored)

  Combined mask for sequence C (only first position real):
       keys →   0    1    2
    query 0: [  0, -∞,  -∞]   ← position 0 sees only itself
    query 1: [  0, -∞,  -∞]   ← position 1 is padding (ignored)
    query 2: [  0, -∞,  -∞]   ← position 2 is padding (ignored)

In practice, the mask tensor has shape (batch, 1, seq_len, seq_len)
and broadcasts across attention heads automatically.
""")

# Demonstrate padding
print("\nConcrete padding example:")
A = [1, 2, 3, 4, 5]
B = [10, 20, 30]
C = [99]

max_len = max(len(A), len(B), len(C))
print(f"  Max length: {max_len}")

batch = np.zeros((3, max_len), dtype=int)
batch[0, :len(A)] = A
batch[1, :len(B)] = B
batch[2, :len(C)] = C
print(f"  Padded batch:\n{batch}")

# Create padding mask
def create_padding_mask(batch, pad_token=0):
    """Create mask where padding positions get -inf.

    WHY expand to (batch, 1, 1, seq_len): In attention, the mask is
    added to the attention scores which have shape (batch, heads,
    query_len, key_len). Expanding to (batch, 1, 1, seq_len) allows
    broadcasting across both heads and query positions automatically.
    """
    mask = np.where(batch == pad_token, float('-inf'), 0.0)
    # Expand to (batch, 1, 1, seq_len) for attention
    return mask[:, np.newaxis, np.newaxis, :]

pad_mask = create_padding_mask(batch)
print(f"\n  Padding mask shape: {pad_mask.shape}")
print(f"  Request B mask: [0, 0, 0, -inf, -inf]")

# ── Better demonstration: Show the full combined mask ──
print(f"\n  Full padding mask for batch (showing which positions are padding):")
for i in range(3):
    mask_row = pad_mask[i, 0, 0, :]
    tokens = ["PAD" if m < -1e10 else "REAL" for m in mask_row]
    print(f"    Sequence {i}: {tokens}  (len={sum(1 for t in tokens if t=='REAL')})")

# Demonstrate the dramatic impact of variable-length batching
print(f"\n  Batch efficiency analysis:")
print(f"    Total positions:  {batch.shape[0] * batch.shape[1]}")
print(f"    Real tokens:      {len(A) + len(B) + len(C)}")
print(f"    Padding tokens:   {batch.shape[0] * batch.shape[1] - len(A) - len(B) - len(C)}")
print(f"    Padding overhead: {(1 - (len(A) + len(B) + len(C)) / (batch.shape[0] * batch.shape[1])) * 100:.0f}%")
print(f"    → This overhead grows with batch size × variance in sequence lengths.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: CONTINUOUS BATCHING IN ACTION
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: CONTINUOUS BATCHING DEMO")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  THE CONTINUOUS BATCHING SCHEDULER — Design and execution  │
└─────────────────────────────────────────────────────────────────┘

The scheduler manages a dynamic pool of requests, each tracked by
a lifecycle state machine:

    waiting ──→ prefill ──→ decode ──→ done
                   │                      ↑
                   └──────────────────────┘
              (after first token, all subsequent
               tokens come from the decode loop)

KEY DESIGN DECISIONS:

  1. MAX BATCH SIZE: Caps how many requests can be active at once.
     Set by GPU memory (KV cache for each request). Larger = better
     throughput but more memory pressure and potential OOM.

  2. PREFILL PRIORITY: When a slot opens, should we prefill a new
     request or let existing ones decode? Prefill is expensive
     (processes the ENTIRE prompt at once), so large prompts can
     cause latency spikes. Chunked prefill splits this across steps.

  3. EVICTION: What if the queue has more requests than batch slots?
     FIFO is simplest, but priority scheduling lets high-priority
     requests jump the queue.

  4. KV CACHE MANAGEMENT: Each active request needs its KV cache
     in GPU memory. When a request finishes, the cache is freed.
     vLLM's PagedAttention makes this efficient by allocating
     cache in small "pages" rather than a contiguous block.

In this demo, we process 4 requests with varying prompt/output
lengths using a max batch size of 2. Watch how the scheduler
keeps the batch full without blocking short requests behind
long ones.
""")

np.random.seed(123)
model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                num_layers=2, max_seq_len=64)

# Create some requests
requests = [
    {'prompt_ids': [1, 2, 3, 4, 5], 'max_tokens': 8},   # short prompt, moderate output
    {'prompt_ids': [10, 20, 30], 'max_tokens': 5},        # very short prompt
    {'prompt_ids': [50, 51, 52, 53, 54, 55], 'max_tokens': 12},  # long prompt, long output
    {'prompt_ids': [90, 91], 'max_tokens': 3},             # tiny request
]

print("\nRequests being processed:")
for i, req in enumerate(requests):
    print(f"  Request {i}: prompt_len={len(req['prompt_ids'])}, "  # ty: ignore
          f"max_tokens={req['max_tokens']}")

scheduler = ContinuousBatchingScheduler(model, max_batch_size=2)
results = scheduler.run(requests, verbose=True)

print(f"\nFinal results:")
for i, result in enumerate(results):
    prompt_len = len(requests[i]['prompt_ids'])  # ty: ignore
    generated_len = len(result) - prompt_len
    print(f"  Request {i}: {prompt_len} prompt + "
          f"{generated_len} generated = {result}")

print("""
  Notice: the short requests (Request 3 with 2 prompt + 3 output
  tokens) finish quickly and free up batch slots for waiting
  requests. The long request (Request 2 with 12 output tokens)
  doesn't block anyone — new requests join as slots open.

  With STATIC batching, Request 2 would have forced everyone to
  wait. Continuous batching eliminates that bottleneck.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: THE THROUGHPUT-LATENCY TRADEOFF
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: THE THROUGHPUT-LATENCY TRADEOFF")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE FUNDAMENTAL TENSION — You cannot maximize both        │
└─────────────────────────────────────────────────────────────────┘

Batching is the primary knob for throughput vs. latency in LLM
serving. There is no free lunch — every choice trades one for
the other.

  SMALL BATCH (size=1–2):
    + Low latency per request (fast Time-To-First-Token, TTFT)
    + Good for interactive chat where users expect instant response
    + Each request gets dedicated GPU attention
    - Low throughput (GPU mostly idle between compute bursts)
    - Expensive per token (you're paying for idle GPU time)
    - Cannot handle many concurrent users economically

  LARGE BATCH (size=32–64):
    + High throughput (GPU compute units saturated)
    + Cheap per token (more tokens processed per GPU-hour)
    + Good for batch processing, evaluations, document summarization
    - High latency per request (queue time + batch processing time)
    - Bad for interactive use — users wait seconds per token
    - TTFT spikes when large batches form

  THE MATH:
    Throughput (tokens/sec) ≈ batch_size × tokens_per_step / step_time
    Latency (per token)    ≈ step_time / batch_size

  As batch_size increases:
    - Step_time grows slowly (GPU gets more efficient, then saturates)
    - Throughput rises rapidly, then plateaus
    - Latency rises steadily (queue time dominates)

  The "optimal" batch size is where marginal throughput gain = latency
  cost you're willing to pay. For chat: batch=8–16. For batch jobs:
  batch=64–128.
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.2  WHY CONTINUOUS BATCHING SOLVES THE TRADEOFF               │
└─────────────────────────────────────────────────────────────────┘

Continuous batching gives the BEST of both worlds:

  - Requests don't wait for the slowest batchmate.
    Each request's latency is proportional to its OWN length,
    not the max of a fixed group.

  - GPU stays busy without a large fixed batch.
    As soon as one request finishes, another starts.
    GPU utilization stays high even with varying request patterns.

  - Throughput approaches large-batch levels.
    Because the batch is always full (or nearly so), throughput
    is close to what you'd get with a fixed batch of the same size.

  - Latency approaches small-batch levels.
    Because requests don't block behind slow ones, individual
    latency is close to what you'd get with a small dedicated batch.

This is why EVERY production inference engine uses continuous batching.
It was introduced by vLLM (PagedAttention paper, Kwon et al., 2023)
and is now standard in vLLM, TensorRT-LLM, SGLang, LMDeploy, TGI,
and every other production-grade inference framework.

Advanced techniques beyond our scope:
  - PRIORITY SCHEDULING: VIP users skip the queue. Important for
    paid tiers and SLAs in production APIs.
  - PREEMPTION: Suspend a long-running request mid-generation to
    make room for short, high-priority ones. The suspended request
    resumes later (KV cache must be preserved).
  - CHUNKED PREFILL: Split long prefills across multiple steps to
    avoid blocking the decode loop. A 4096-token prompt is split
    into 512-token chunks, processed over 8 steps.
  - PREFIX CACHING: If two requests share a system prompt ("You are
    a helpful assistant..."), compute the KV cache for it once and
    reuse it. Saves ~30% of compute for chat applications.
  - DISAGGREGATED PREFILL/DECODE: Run separate GPU clusters for
    prefill (compute-heavy, latency-sensitive) and decode
    (memory-bound, throughput-sensitive). State of the art in 2025.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: CONCURRENT SCHEDULING — Visualizing the Timeline
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 5: VISUALIZING THE SCHEDULER TIMELINE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  THROUGHPUT COMPARISON — Batching vs. sequential           │
└─────────────────────────────────────────────────────────────────┘

To quantify the benefit, we compare sequential processing (one
request at a time) against continuous batching for the same set
of requests.

WHY THIS MATTERS: The difference between sequential and batched
throughput is the difference between serving 10 users and 100 users
on the same hardware. For production inference APIs (OpenAI, Anthropic,
Together), this multiplier is the difference between profit and loss.

With our tiny model, the overhead of batching logic may dominate
(since there's almost no real GPU work to parallelize). With real
models (7B–405B parameters), the batching speedup is dramatic:
""")


# Run a simpler demo with timing
np.random.seed(42)
simple_requests = [
    {'prompt_ids': [1, 2, 3], 'max_tokens': 4},
    {'prompt_ids': [10, 20, 30, 40], 'max_tokens': 3},
    {'prompt_ids': [50, 51], 'max_tokens': 5},
]

# Show the request diversity
print(f"\n  Requests with different prompt/output lengths:")
for i, req in enumerate(simple_requests):
    print(f"    Request {i}: prompt_len={len(req['prompt_ids'])}, "  # ty: ignore
          f"output_len={req['max_tokens']}")
print(f"    → These have naturally different completion times.")
print(f"    → Sequential: each must wait for all previous to finish.")
print(f"    → Batching: overlap their execution for higher throughput.")

t0 = time.perf_counter()
scheduler_simple = ContinuousBatchingScheduler(model, max_batch_size=3)
results_simple = scheduler_simple.run(simple_requests, verbose=False)
batch_time = time.perf_counter() - t0

# Now run them sequentially for comparison
t0 = time.perf_counter()
for req in simple_requests:
    model.generate(req['prompt_ids'], max_new_tokens=req['max_tokens'])
seq_time = time.perf_counter() - t0

print(f"\n  Batching time:    {batch_time:.4f}s")
print(f"  Sequential time:  {seq_time:.4f}s")

if batch_time < seq_time:
    print(f"  Speedup: {seq_time / batch_time:.1f}×")
    print(f"  → Batching completed all requests faster (better throughput)")
else:
    print(f"  → For tiny models, overhead can dominate. Real models benefit more.")
    print(f"    A 7B model on an A100 typically sees 2-4× throughput improvement")
    print(f"    with continuous batching vs. sequential serving.")

print(f"\n  No requests were blocked behind slow ones:")
for i, r in enumerate(simple_requests):
    print(f"    Request {i}: {len(r['prompt_ids'])} prompt → "  # ty: ignore
          f"{r['max_tokens']} tokens generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ KEY TAKEAWAYS FROM MODULE 3                                    │
│                                                                 │
│ 1. BATCHING: Multiple requests in one GPU launch → higher      │
│    throughput. Without it, GPU utilization is below 1%.         │
│                                                                 │
│ 2. PADDING: Makes sequences the same length for rectangular    │
│    tensors. The cost is wasted compute on padding positions.    │
│                                                                 │
│ 3. MASKING: Prevents attention from seeing padding (and future) │
│    tokens. Combined causal + padding masks work together.       │
│                                                                 │
│ 4. STATIC BATCHING: Groups requests, waits for all to finish.  │
│    Head-of-line blocking → high latency for fast requests.      │
│                                                                 │
│ 5. CONTINUOUS BATCHING: Dynamic batch — requests join/leave as  │
│    they start/finish. Best of throughput and latency.           │
│                                                                 │
│ 6. SCHEDULER LIFECYCLE: waiting → prefill → decode → done.     │
│    The scheduler's job is to keep the GPU fed.                  │
│                                                                 │
│ 7. TRADEOFF: Throughput vs. latency is the fundamental tension  │
│    of inference serving. Continuous batching optimizes both.    │
└─────────────────────────────────────────────────────────────────┘

ARCHITECTURE RECAP:
    Queue → Scheduler picks N requests → PREFILL (batch them)
    → DECODE loops → As requests finish, new ones join
    → Continuous GPU utilization without blocking

The scheduling pipeline you've built is the same pattern used by
vLLM, TensorRT-LLM, and every major inference engine. The differences
in production are scale (hundreds of concurrent requests, gigabytes
of KV cache), efficiency (PagedAttention for memory management), and
features (priority queues, preemption, prefix caching).

NEXT MODULE:
    Module 4 — Quantization: making models fit in smaller memory
    by reducing parameter precision from FP16 to INT8/INT4.
    Learn how 4-bit quantization can shrink a model 4× with
    minimal quality loss, and WHY it works at the mathematical
    level.
""")


if __name__ == "__main__":
    print("\nModule 3 complete! Next: i04_quantization.py")
    print("Run with: uv run python course/inference/i04_quantization.py")
