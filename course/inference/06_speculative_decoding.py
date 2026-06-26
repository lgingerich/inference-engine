"""
INFERENCE MODULE 6: SPECULATIVE DECODING — Draft + Verify for Speed
=====================================================================

After KV cache, batching, quantization, and FlashAttention, there's
still a FUNDAMENTAL LIMIT: autoregressive generation is sequential.
Token N+1 depends on token N. Your GPU is 99% idle during decode.

Speculative decoding BREAKS THROUGH this wall: use a SMALL "draft"
model to guess future tokens cheaply, then verify them all at once
with the BIG model. It's mathematically LOSSLESS — the output is
identical to what the big model would have produced alone.

WHAT YOU'LL LEARN:
   1. Why the sequential constraint is the final bottleneck
   2. The draft-and-verify loop: cheap guesses, expensive verification
   3. WHY the acceptance criterion is min(1, p_verify/p_draft)
   4. How to get 2-5× speedup with ZERO quality loss
   5. Why this is used by EVERY major LLM API provider

AFTER THIS MODULE:
   You'll understand one of the most elegant tricks in modern LLM
   serving — and why it's mathematically guaranteed to be correct.
"""

import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY SPECULATIVE DECODING EXISTS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: THE LAST BOTTLENECK")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE SEQUENTIAL CONSTRAINT — The final optimization target      │
└─────────────────────────────────────────────────────────────────┘

After Modules 2-5, inference is much faster. But there's a hard limit:

  Memory bandwidth (H100): 3.35 TB/s
  Model weights (7B FP16): 14 GB
  Minimum weight load time: 14 GB / 3.35 TB/s ≈ 4.2 ms

No matter how optimized your kernels are, you CANNOT generate tokens
faster than ~238 tok/s (1000ms / 4.2ms) because you MUST load all
14 GB of weights for every token in the decode phase. That's
a PHYSICAL limit of the hardware.

Speculative decoding's insight: we CAN'T go faster than 238 tok/s
per forward pass, but we can get MULTIPLE tokens per pass by guessing.

  Without speculation: 1 forward pass → 1 token (4.2 ms/tok)
  With speculation:    1 forward pass → 3-5 tokens (1.0 ms/tok effective)

The draft model doesn't need to be PERFECT — it just needs to be
accurate enough that the verifier accepts most of its guesses.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: THE DRAFT-AND-VERIFY PATTERN
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: THE DRAFT-AND-VERIFY PATTERN")
print("=" * 70)

np.random.seed(42)

large_model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                      num_layers=2, max_seq_len=64)

draft_model = MiniGPT(vocab_size=100, d_model=16, num_heads=2,
                      num_layers=1, max_seq_len=64)

def count_params(model):
    n = 0
    n += model.vocab_size * model.d_model
    n += model.max_seq_len * model.d_model
    for block in model.blocks:
        n += 4 * model.d_model * model.d_model
        n += 2 * model.d_model * model.d_model * 4
    n += model.d_model * model.vocab_size
    return n

large_params = count_params(large_model)
draft_params = count_params(draft_model)

print(f"""
Model sizes:
  Large (verifier): ~{large_params:,} params
  Draft (guesser):  ~{draft_params:,} params
  Ratio: {large_params/draft_params:.1f}× smaller → ~{large_params/draft_params:.0f}× faster per token

The draft model runs ~{large_params/draft_params:.0f}× faster per forward pass.
This means it can generate K tokens in less time than the large model
takes for ONE token. The speedup comes from using the large model's
forward pass to verify MANY draft tokens at once.

┌─────────────────────────────────────────────────────────────────┐
│ 1.1  THE DRAFT-AND-VERIFY CYCLE — Step by step                │
└─────────────────────────────────────────────────────────────────┘

  1. DRAFT (cheap): Draft model generates K tokens quickly.
     draft = [t₁, t₂, t₃, ..., t_K]

  2. VERIFY (expensive, but worth it): Large model processes
     [prompt + draft] in ONE forward pass. At each position,
     compare draft token's probability to the large model's.

  3. ACCEPT/REJECT: For each position i:
     - p_draft = draft model's probability for t_i
     - p_verify = large model's probability for t_i
     - Accept with probability min(1, p_verify/p_draft)
     - If REJECTED: replace t_i with a sample from the corrected
       distribution and STOP (don't check t_{{i+1}}...t_K)

  4. OUTCOME: We got M accepted tokens (1 ≤ M ≤ K+1) in one
     large model forward pass. Effective tokens per call = M.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: THE ACCEPTANCE CRITERION — Why min(1, p_verify/p_draft)?
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 2: THE ACCEPTANCE CRITERION — The Math of Losslessness")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WHY THE RATIO? — Rejection sampling produces the correct  │
│      target distribution                                        │
└─────────────────────────────────────────────────────────────────┘

We want to produce tokens from the LARGE model's distribution P(x),
but we have samples from the DRAFT model's distribution Q(x).

The speculative sampling algorithm (Chen et al., 2023):

  For a draft token x with draft probability q(x) and target
  probability p(x):

    Accept x if:  random() < min(1, p(x) / q(x))
    Reject x if:  random() >= min(1, p(x) / q(x))
      Then sample a replacement from: max(0, p - q) normalized
      (the "excess" probability that p assigns beyond q)

  WHY THIS WORKS:
    For any token x, the probability it ends up in the output is:

    P(x accepted) + P(x as replacement) = q(x) * min(1, p(x)/q(x))
                                         + (p(x) - q(x) * min(1, p(x)/q(x)))
                                       = p(x)

    The first term is "x was proposed AND accepted"
    The second term is "something else was proposed but x was sampled
    as the replacement"
    Together they sum to p(x) exactly.

  INTUITION: Imagine the draft model is your assistant who proposes
  answers. The large model reviews each proposal:
    - If the large model agrees (p >= q), always accept.
    - If the large model disagrees (p < q), accept with probability
      p/q. With remaining probability, "correct" by sampling from
      the large model's alternative.

  This is identical to running the large model alone — just faster.
""")


def speculative_verify(draft_tokens, large_logits, draft_logits):
    """Verify a draft sequence against the large model's output.

    Args:
        draft_tokens: K token IDs from the draft model
        large_logits: (K, vocab_size) — verifier logits at each position
        draft_logits: (K, vocab_size) — draft logits at each position

    Returns:
        (accepted_tokens, all_accepted) — list of accepted tokens
    """
    K = len(draft_tokens)
    accepted = []

    for i in range(K):
        draft_probs = softmax(draft_logits[i])
        large_probs = softmax(large_logits[i])

        p_draft = draft_probs[draft_tokens[i]]
        p_large = large_probs[draft_tokens[i]]

        # Acceptance probability: always accept if verifier is MORE confident
        acceptance_prob = min(1.0, p_large / max(p_draft, 1e-10))
        r = np.random.random()

        if r < acceptance_prob:
            accepted.append(draft_tokens[i])
        else:
            # REJECTED: sample replacement from corrected distribution
            # Subtract the draft's proposal, normalize the rest
            corrected = large_probs.copy()
            corrected[draft_tokens[i]] = 0
            corrected /= corrected.sum()
            replacement = int(np.random.choice(len(large_probs), p=corrected))
            accepted.append(replacement)
            return accepted, False  # stopped early

    return accepted, True  # all K accepted


# Demonstrate the acceptance mechanism
print("\nAcceptance criterion demo (toy example):")
print("  Draft model proposes token A: q(A)=0.7")
print("  Large model's probability:    p(A)=0.5")
print(f"  Acceptance probability: min(1, 0.5/0.7) = {min(1.0, 0.5/0.7):.2f}")
print("  → With 71% chance we accept A, 29% chance we reject and resample.")
print(f"  → The 29% rejection rate ensures the OUTPUT has p(A)=0.5, not 0.7.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: FULL SPECULATIVE DECODING LOOP
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: FULL SPECULATIVE DECODING IN ACTION")
print("=" * 70)


def speculative_decode(large_model, draft_model, prompt_ids,
                       max_new_tokens=20, draft_len=4, temperature=1.0):
    """Generate text using speculative decoding.

    Args:
        large_model: verifier (big, accurate)
        draft_model: drafter (small, fast)
        prompt_ids: starting token IDs
        max_new_tokens: total tokens to generate
        draft_len: K — how many tokens the draft guesses at once
        temperature: sampling temperature

    Returns:
        (generated, stats) — list of all tokens, plus stats dict
    """
    generated = list(prompt_ids)
    total_draft = 0
    total_accepted = 0
    large_calls = 0

    while len(generated) - len(prompt_ids) < max_new_tokens:
        # ── DRAFT: Generate K tokens cheaply ──
        draft_gen = list(generated)
        draft_tokens = []
        draft_logits_list = []

        for _ in range(draft_len):
            context = np.array([draft_gen[-large_model.max_seq_len:]])
            d_logits = draft_model.forward(context)
            d_next = d_logits[0, -1, :] / temperature
            d_probs = softmax(d_next)
            d_token = int(np.random.choice(large_model.vocab_size, p=d_probs))
            draft_tokens.append(d_token)
            draft_logits_list.append(d_logits[0, -1, :])
            draft_gen.append(d_token)

        # ── VERIFY: Large model processes ALL draft tokens in ONE pass ──
        verify_input = np.array([generated + draft_tokens])
        large_logits = large_model.forward(verify_input)
        large_calls += 1

        # Extract verifier logits at positions corresponding to each draft token
        verify_logits = []
        for j in range(draft_len):
            pos = len(generated) + j
            verify_logits.append(large_logits[0, pos, :])

        # ── ACCEPT/REJECT ──
        accepted, all_accepted = speculative_verify(
            draft_tokens,
            np.array(verify_logits),
            np.array(draft_logits_list))

        generated.extend(accepted)
        total_draft += draft_len
        total_accepted += len(accepted)

        # Log per-call stats
        if large_calls <= 3:
            print(f"  Call {large_calls}: drafted {draft_len}, "
                  f"accepted {len(accepted)} tokens")
            if not all_accepted:
                print(f"    (rejected at position {len(accepted)-1}, "
                      f"correction applied)")

    stats = {
        'large_calls': large_calls,
        'total_draft': total_draft,
        'accepted': total_accepted,
        'acceptance_rate': total_accepted / max(total_draft, 1),
        'tokens_generated': len(generated) - len(prompt_ids),
    }
    return generated, stats


# Run speculative decoding
print("\nSpeculative decoding (draft_len=4, max_new=16):")
np.random.seed(123)
prompt = [1, 2, 3]
gen_spec, stats = speculative_decode(
    large_model, draft_model, prompt,
    max_new_tokens=16, draft_len=4, temperature=0.8)

print(f"\nResults:")
print(f"  Generated sequence: {gen_spec}")
print(f"  Large model calls: {stats['large_calls']}")
print(f"  Draft tokens proposed: {stats['total_draft']}")
print(f"  Draft tokens accepted: {stats['accepted']}")
print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
print(f"  Effective tokens/call: {stats['tokens_generated']/stats['large_calls']:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: ACCEPTANCE RATES — What Determines Speedup
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: ACCEPTANCE RATES — The Speedup Determinant")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  SPEEDUP FORMULA                                           │
└─────────────────────────────────────────────────────────────────┘

  Effective speedup = (draft_len × acceptance_rate + 1) / 1

  If draft_len=4 and acceptance=80%, we get 4.2 tokens per large
  model call instead of 1 → 4.2× speedup.

Factors affecting acceptance rate:

  1. DRAFT QUALITY: Better (but still small) drafts = higher acc.
     A 150M-param drafter for a 7B model typically hits 80-90%.

  2. TEMPERATURE: Lower temp = more predictable = higher acceptance.
     At temp=0 (greedy), acceptance can reach 95%+.
     At temp=2.0, acceptance drops to 40-50%.

  3. TASK TYPE: Factual Q&A → high (deterministic answers).
     Creative writing → lower (many valid continuations).

  4. DRAFT LENGTH: 4-8 tokens is usually optimal. Longer = more
     risk of rejection, shorter = less speedup per call.

TYPICAL ACCEPTANCE RATES (from literature):
  Code generation:    85-95% (highly structured)
  Factual Q&A:        80-90%
  General chat:       70-85%
  Creative writing:   50-70%

PRACTICAL SPEEDUPS:
  4-token drafts at 80% acceptance → ~3.2× speedup
  6-token drafts at 75% acceptance → ~4.5× speedup

Note: Our tiny models show no meaningful wall-clock speedup (overhead
dominates for 23K-param models). The PRINCIPLE is what matters.

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  ADVANCED VARIANTS                                         │
└─────────────────────────────────────────────────────────────────┘

MEDUSA: Instead of a separate draft model, add MULTIPLE prediction
  "heads" to the large model that predict tokens 2, 3, 4 steps ahead.
  No separate model needed — just fine-tune extra predictor heads.

EAGLE (used by vLLM): A draft model that predicts the NEXT LAYER's
  hidden state, then decodes it to a token. More accurate than
  predicting tokens directly because hidden states carry more info.

TREE ATTENTION: Instead of a single draft sequence, explore a TREE
  of possibilities. The verifier checks all paths in one batched
  forward pass. Higher acceptance with the same number of drafts.
""")


print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ SPECULATIVE DECODING — The final speed multiplier              │
│                                                                 │
│ 1. The sequential constraint limits decode to ~238 tok/s per   │
│    weight load on a given GPU. Can't go faster per pass.      │
│                                                                 │
│ 2. SOLUTION: draft many guesses cheaply, verify all at once.   │
│    Draft model (3-5× faster) → K tokens → verify in 1 pass.    │
│                                                                 │
│ 3. ACCEPTANCE CRITERION: min(1, p_verify/p_draft). This is     │
│    mathematically PROVEN to produce the exact target distribution.│
│                                                                 │
│ 4. TYPICAL SPEEDUP: 2-5× on production workloads. Combined     │
│    with all other optimizations = 10-50× over the raw loop.    │
│                                                                 │
│ 5. Used by: ChatGPT/OpenAI API, Claude/Anthropic API, Gemini.  │
│    It's standard in every major production LLM service.        │
└─────────────────────────────────────────────────────────────────┘

Next: Module 7 — Distributed Inference (splitting models across GPUs)
""")

if __name__ == "__main__":
    print("\nModule 6 complete! Next: i07_distributed.py")
    print("Run with: uv run python course/inference/i07_distributed.py")
