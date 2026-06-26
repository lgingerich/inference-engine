"""
MODULE 10: ALIGNMENT — From Document Completer to Helpful Assistant
======================================================================

Module 8 taught you how pre-training turns random weights into a
language model. But that model is a DOCUMENT COMPLETER — it predicts
what comes next on the internet, which includes rants, misinformation,
and half-finished thoughts. It's not an assistant.

This module covers the ENTIRE post-training pipeline that transforms
a base model into something useful: SFT, Reward Modeling, RLHF/PPO,
DPO, GRPO, and RL for reasoning.

WHAT YOU'LL LEARN:
   1. Pre-training vs Post-training — the full pipeline
   2. Why base models aren't assistants (and SFT isn't enough)
   3. Bradley-Terry preference model — the math behind "better/worse"
   4. PPO for RLHF — the original alignment recipe
   5. DPO — alignment without a reward model
   6. GRPO — DeepSeek's simpler, more stable alternative
   7. RL for reasoning — how GRPO sparked the DeepSeek-R1 "aha moment"

AFTER THIS MODULE:
   You'll understand the complete LLM lifecycle: pre-training (Module 8)
   → alignment (Module 10) → inference (Inference Course i00-i09).
   You'll know why ChatGPT says "I can't help with that" instead of
   completing your sentence with whatever the internet would say next.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# PART 1: PRE-TRAINING vs POST-TRAINING — The Full Pipeline
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: PRE-TRAINING vs POST-TRAINING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE COMPLETE LLM LIFECYCLE                                       │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │ PRE-TRAINING│ →  │    SFT      │ →  │  RLHF / DPO / GRPO   │ │
│  │ (Module 8)  │    │Supervised   │    │  Alignment from      │ │
│  │             │    │Fine-Tuning  │    │  human preferences    │ │
│  └─────────────┘    └─────────────┘    └──────────────────────┘ │
│       ↑                  ↑                      ↑               │
│  Next-token          (prompt,            "I prefer response A   │
│  prediction on       response)           over response B"       │
│  internet text       pairs                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ RESULT OF EACH STAGE:                                       ││
│  │                                                             ││
│  │ Pre-training:     "The capital of France is Paris. The      ││
│  │                    Eiffel Tower was built in..."             ││
│  │                    → Completes text like the internet        ││
│  │                                                             ││
│  │ After SFT:        User: "What's the capital of France?"     ││
│  │                    Model: "The capital of France is Paris."  ││
│  │                    → Answers questions, follows format       ││
│  │                                                             ││
│  │ After RLHF:       User: "How do I hack my neighbor's WiFi?" ││
│  │                    Model: "I can't help with that, but..."   ││
│  │                    → Helpful, harmless, honest               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

KEY DISTINCTION:

  PRE-TRAINING (Module 8):
    - Data: Trillions of raw internet tokens
    - Objective: Predict next token (self-supervised)
    - Output: A model that COMPLETES text
    - Cost: Millions of dollars, thousands of GPUs
    - What you get: Knowledge, grammar, reasoning — but no
      idea that it's supposed to be an assistant

  POST-TRAINING / ALIGNMENT (this module):
    - Data: Thousands to millions of human-labeled examples
    - Objective: Be helpful, harmless, honest
    - Output: A model that ASSISTS users
    - Cost: Hundreds of thousands, much smaller scale
    - What you get: A model that refuses harmful requests,
      follows instructions, admits uncertainty, and
      understands it's in a conversation

Why doesn't pre-training produce an assistant?

  Internet text includes:
    - "The mitochondria is the powerhouse of the cell"  (good!)
    - "Actually the earth is flat because..."           (misinformation)
    - "Here's how to make explosives..."                (dangerous)
    - Incomplete forum posts, rants, spam, SEO garbage  (noise)

  Pre-training's objective is "predict what the internet WOULD say."
  But we don't want the model to SAY what the internet would say.
  We want it to say what a helpful, honest, harmless assistant WOULD say.
  That's alignment.
""")


# ──────────────────────────────────────────────────────────────────
# 1.1  Pre-training vs SFT vs RL: Concrete Example
# ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("CONCRETE EXAMPLE: Same prompt, three stages")
print("=" * 70)

prompt = "How do I build a bomb?\n\nAnswer:"

print(f"""
  PROMPT: {prompt!r}

  Stage 1 — BASE MODEL (pre-training only):
    → "How do I build a bomb? Here's what you'll need:"
    [continues with detailed instructions from internet]
    It doesn't KNOW it should refuse. It was trained to complete
    text, and the internet contains bomb-building instructions.

  Stage 2 — SFT MODEL (fine-tuned on helpful examples):
    → "I'm sorry, I can't help with that. Building bombs is illegal
       and dangerous. If you're interested in chemistry..."
    It learned the PATTERN of a helpful refusal, but it may still
    be vulnerable to jailbreaks. It's MIMICKING helpful behavior.

  Stage 3 — RLHF MODEL (trained with preference optimization):
    → "I can't help with building weapons. If you're interested in
       chemistry as a science, I'd be happy to discuss that instead."
    It has internalized helpfulness as a PREFERENCE, not just a
    pattern. It knows WHY it should refuse and offers alternatives.
    Much harder to jailbreak.

  Each stage builds on the last. SFT teaches FORMAT. RL teaches VALUES.
""")


# ═══════════════════════════════════════════════════════════════════
# PART 2: SUPERVISED FINE-TUNING (SFT)
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: SUPERVISED FINE-TUNING (SFT)")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  SFT IS STILL NEXT-TOKEN PREDICTION                         │
└─────────────────────────────────────────────────────────────────┘

SFT uses the EXACT same loss function as pre-training (cross-entropy).
The only difference is the DATA:

  Pre-training data (trillions of tokens):
    "The mitochondria is the powerhouse of the cell. It generates
    ATP through oxidative phosphorylation..."

  SFT data (thousands to millions of examples):
    {
      "messages": [
        {"role": "user", "content": "What are mitochondria?"},
        {"role": "assistant", "content": "Mitochondria are organelles
         found in most eukaryotic cells. They're often called the
         'powerhouse of the cell' because they generate most of the
         cell's ATP through oxidative phosphorylation..."}
      ]
    }

During SFT, the model only computes loss on the ASSISTANT tokens, not
the user tokens. This teaches it: "When you see a user message, respond
helpfully." It's still pure next-token prediction, just on a curated
dataset of (prompt, ideal_response) pairs.

┌─────────────────────────────────────────────────────────────────┐
│ 2.2  WHY SFT ALONE ISN'T ENOUGH                                 │
└─────────────────────────────────────────────────────────────────┘

Three problems with SFT-only alignment:

  1. DISTRIBUTION MISMATCH: SFT trains the model to produce ideal
     responses, but at inference time, the model generates tokens
     autoregressively. Small errors compound (exposure bias).

  2. CAN'T LEARN PREFERENCES: SFT can't distinguish between "good"
     and "great" responses. Both are treated as ground truth targets.
     If you show the model two responses to the same prompt, SFT
     has no way to learn "A is better than B."

  3. NEGATIVE LEARNING: SFT only shows the model what TO do, never
     what NOT to do. It can't learn to actively avoid bad behavior
     — it can only learn to produce good behavior more often.

This is where RL comes in. RL can learn from COMPARISONS (A > B),
not just targets (output = A).
""")


# ──────────────────────────────────────────────────────────────────
# 2.3  SFT Loss: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("SFT LOSS DEMONSTRATION:")
print("-" * 70)

# Simulate: model predicts assistant tokens in a conversation
# Only compute loss on assistant positions (mask out user tokens)

np.random.seed(42)
vocab_size = 100

# Simulated logits for a 3-turn conversation (seq_len=15)
logits = np.random.randn(15, vocab_size)

# Ground truth token IDs
targets = np.array([5, 12, 8, 33, 7, 91, 44, 2, 56, 19, 73, 8, 41, 12, 99])

# Mask: 1 = assistant token (compute loss), 0 = user token (ignore)
assistant_mask = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1])
#                          ^--user--^  ^---assistant---^  ^-user-^  ^-assist-^

# Softmax
logits_max = np.max(logits, axis=-1, keepdims=True)
exp_logits = np.exp(logits - logits_max)
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Per-token loss
correct_probs = probs[np.arange(15), targets]
nll = -np.log(np.clip(correct_probs, 1e-10, 1.0))

# Masked loss: only count assistant tokens
masked_nll = nll * assistant_mask
masked_loss = np.sum(masked_nll) / np.sum(assistant_mask)

print(f"  Position | Target | Prob   | NLL    | Mask | Counted?")
print(f"  " + "-" * 56)
for i in range(15):
    role = "User     " if assistant_mask[i] == 0 else "Assistant"
    counted = "Yes" if assistant_mask[i] == 1 else "No "
    print(f"  {i:8d} | {targets[i]:6d} | {correct_probs[i]:.4f} | {nll[i]:.4f} | {assistant_mask[i]:4d} | {counted} ({role})")

print(f"\n  Average loss (unmasked): {np.mean(nll):.4f}")
print(f"  Average loss (assistant only): {masked_loss:.4f}")
print(f"  → SFT only trains on assistant tokens. The user tokens are context only.")


# ═══════════════════════════════════════════════════════════════════
# PART 3: BRADLEY-TERRY — The Math of Preferences
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: BRADLEY-TERRY PREFERENCE MODEL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  THE FOUNDATION — How do we model "better than"?            │
└─────────────────────────────────────────────────────────────────┘

The Bradley-Terry model (1952!) answers: given two responses y_A and
y_B, what's the probability a human prefers y_A over y_B?

  P(y_A ≻ y_B) = exp(r_A) / (exp(r_A) + exp(r_B))
               = σ(r_A - r_B)

where r_A and r_B are latent "quality scores" and σ is the sigmoid.

This is elegant: if r_A >> r_B, then P ≈ 1 (A strongly preferred).
If r_A ≈ r_B, then P ≈ 0.5 (toss-up).
If r_A << r_B, then P ≈ 0 (B strongly preferred).

┌─────────────────────────────────────────────────────────────────┐
│ 3.2  FROM PREFERENCES TO LIKELIHOOD                             │
└─────────────────────────────────────────────────────────────────┘

Given a dataset of human preferences {(prompt, y_chosen, y_rejected)}:
  - y_chosen: the response the human PREFERRED
  - y_rejected: the response the human DIDN'T prefer

The likelihood of observing these preferences is:

  L = Π P(y_chosen ≻ y_rejected)
    = Π σ(r_chosen - r_rejected)

Maximizing this is equivalent to minimizing:
  loss = -log σ(r_chosen - r_rejected)

This is the REWARD MODEL loss: train a model to predict r (quality)
such that chosen responses consistently score higher than rejected ones.
""")


# ──────────────────────────────────────────────────────────────────
# 3.3  Bradley-Terry: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("BRADLEY-TERRY DEMONSTRATION:")
print("-" * 70)

def bradley_terry_prob(r_a, r_b):
    """Probability that response A is preferred over B, given latent scores."""
    return 1.0 / (1.0 + np.exp(-(r_a - r_b)))

def reward_model_loss(r_chosen, r_rejected):
    """Binary cross-entropy loss for reward model training.
    We want r_chosen > r_rejected for each preference pair.
    """
    # P(chosen ≻ rejected) = sigmoid(r_chosen - r_rejected)
    p = bradley_terry_prob(r_chosen, r_rejected)
    # Loss = -log(p) since we want to maximize P(chosen preferred)
    return -np.log(np.clip(p, 1e-10, 1.0))

# Example: three preference pairs
scores_chosen = np.array([2.0, 0.5, -1.0])
scores_rejected = np.array([-1.0, -0.5, 0.5])

print(f"  {'Chosen':>8}  {'Rejected':>8}  {'Diff':>8}  {'P(chosen wins)':>15}  {'Loss':>8}")
print(f"  " + "-" * 60)
for r_c, r_r in zip(scores_chosen, scores_rejected):
    p = bradley_terry_prob(r_c, r_r)
    loss = reward_model_loss(r_c, r_r)
    print(f"  {r_c:8.1f}  {r_r:8.1f}  {r_c-r_r:8.1f}  {p:15.4f}  {loss:8.4f}")

avg_loss = np.mean([reward_model_loss(c, r)
                    for c, r in zip(scores_chosen, scores_rejected)])
print(f"\n  Average reward model loss: {avg_loss:.4f}")

print(f"""
  INTERPRETATION:
    - When r_chosen >> r_rejected: P ≈ 1.0, loss ≈ 0 (confident & correct)
    - When r_chosen ≈ r_rejected: P ≈ 0.5, loss ≈ 0.69 (uncertain)
    - When r_chosen << r_rejected: P ≈ 0, loss large (wrong — fix this!)

  Training a reward model means finding a function r(text) that
  assigns higher scores to preferred responses. Once trained, r
  becomes the "reward signal" for RL.
""")


# ═══════════════════════════════════════════════════════════════════
# PART 4: PPO — Proximal Policy Optimization for RLHF
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: PPO — THE ORIGINAL RLHF ALGORITHM")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  THE RLHF LOOP — Four models, one objective                 │
└─────────────────────────────────────────────────────────────────┘

Standard RLHF (InstructGPT / ChatGPT) uses Proximal Policy Optimization:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  POLICY  │    │REFERENCE │    │  REWARD  │    │  VALUE   │
  │  MODEL   │    │  MODEL   │    │  MODEL   │    │  MODEL   │
  │ (being   │    │ (frozen  │    │ (frozen  │    │(estimates│
  │ trained) │    │  copy of │    │  reward  │    │ baseline)│
  │          │    │  policy) │    │  model)  │    │          │
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  Generates         Used for        Scores         Estimates
  responses         KL penalty      responses      expected
                                    (quality)      return

The objective combines THREE terms:

  J = E[R(response)]                     ← maximize reward (quality)
    - β · KL(π_θ || π_ref)              ← stay close to reference (don't drift)
    + γ · entropy(π_θ)                  ← encourage exploration

Where π_θ is the policy (the model being trained) and π_ref is the
frozen reference (typically the SFT model).

┌─────────────────────────────────────────────────────────────────┐
│ 4.2  WHY THE KL PENALTY?                                        │
└─────────────────────────────────────────────────────────────────┘

Without a KL penalty, the model would maximize reward at all costs.
The reward model is an IMPERFECT proxy for human preferences — it has
blind spots. The model can learn to exploit these blind spots to get
high reward scores while producing gibberish or worse.

This is REWARD HACKING (also called "reward overoptimization" or
"Goodhart's Law in action"):

  Human:   "Tell me about dogs."
  Model:   "Dogs dogs dogs wonderful dogs excellent dogs dogs..."
  Reward model:  HIGH SCORE! (it detects positive sentiment words
                  but can't evaluate coherence at scale)

The KL penalty prevents the model from diverging too far from the
well-behaved SFT model. It says: "Get high reward, but don't forget
how to speak English."

┌─────────────────────────────────────────────────────────────────┐
│ 4.3  PPO CLIPPING — Trust region in action                      │
└─────────────────────────────────────────────────────────────────┘

PPO's key innovation is CLIPPING the policy update. Instead of taking
the full gradient step (which might change the policy too much), PPO
limits how much the probability ratio can change:

  ratio = π_θ(token | context) / π_old(token | context)

  clipped_ratio = clip(ratio, 1-ε, 1+ε)   ← usually ε = 0.2

  loss = -min(ratio × advantage, clipped_ratio × advantage)

This means: if the new policy would make a token 500% more likely
than the old policy, PPO caps the effective change at ±20%. This
prevents catastrophic forgetting and makes training stable.

┌─────────────────────────────────────────────────────────────────┐
│ 4.4  THE VALUE MODEL — Why we need it                            │
└─────────────────────────────────────────────────────────────────┘

The ADVANTAGE is: "how much better was this action than expected?"

  advantage = reward - V(state)

Where V(state) is the value model's estimate of expected future reward.
Without a value model (critic), we'd use the raw reward, which is noisy
and has high variance. The critic provides a BASELINE that makes the
advantage signal much cleaner:

  - Positive advantage → action was better than expected → encourage
  - Negative advantage → action was worse than expected → discourage
  - Zero advantage → action was exactly as expected → no change needed

PPO requires FOUR models running simultaneously (policy, reference,
reward, value), making it expensive and complex. This motivated the
search for simpler alternatives...
""")


# ──────────────────────────────────────────────────────────────────
# 4.5  Policy Gradient: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("POLICY GRADIENT DEMONSTRATION:")
print("-" * 70)

np.random.seed(42)

# Simulate a tiny policy: 5-token vocabulary, 3 positions
# The policy outputs log-probabilities for each token
vocab_size = 5
seq_len = 3

# "Old" policy (before update) — already softmaxed
old_log_probs = np.log(np.array([
    [0.30, 0.25, 0.20, 0.15, 0.10],  # token 0 is most likely
    [0.10, 0.15, 0.25, 0.25, 0.25],  # uniform-ish
    [0.40, 0.05, 0.05, 0.05, 0.45],  # token 0 and 4 compete
]))
# "New" policy (after a gradient step)
new_log_probs = np.log(np.array([
    [0.35, 0.20, 0.20, 0.15, 0.10],  # token 0 more likely
    [0.10, 0.10, 0.35, 0.25, 0.20],  # token 2 more likely now
    [0.50, 0.05, 0.05, 0.05, 0.35],  # token 0 pulled ahead
]))

# Tokens actually generated (sampled from old policy)
tokens = np.array([1, 3, 0])  # token chosen at each position

# Advantage: was this token better (+1) or worse (-1) than expected?
advantages = np.array([1.0, -0.5, 2.0])

# Compute PPO-style loss
ratio = np.exp(new_log_probs[np.arange(seq_len), tokens]
               - old_log_probs[np.arange(seq_len), tokens])
epsilon = 0.2
clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

print(f"  Token | Old logP | New logP | Ratio  | Clipped | Advantage | Loss term")
print(f"  " + "-" * 72)
for i in range(seq_len):
    loss_term = -min(ratio[i], clipped_ratio[i]) * advantages[i]
    print(f"  {tokens[i]:5d} | {old_log_probs[i, tokens[i]]:8.4f} | "
          f"{new_log_probs[i, tokens[i]]:8.4f} | {ratio[i]:6.3f} | "
          f"{clipped_ratio[i]:7.3f} | {advantages[i]:9.1f} | {loss_term:9.4f}")

print(f"""
  INTERPRETATION:
    - Token 1 (pos 0): advantage=+1.0, ratio=0.8 → this token was good,
      and the new policy made it LESS likely. PPO says: INCREASE it.
    - Token 3 (pos 1): advantage=-0.5, ratio=1.0 → this token was bad,
      no change in probability. PPO says: mild DECREASE.
    - Token 0 (pos 2): advantage=+2.0, ratio=1.25 → this token was
      VERY good, but ratio is clipped to 1.2. PPO prevents over-updating.

  Without clipping, position 2 would get a much larger update, risking
  catastrophic forgetting on other prompts.
""")


# ═══════════════════════════════════════════════════════════════════
# PART 5: DPO — Direct Preference Optimization
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: DPO — ALIGNMENT WITHOUT A REWARD MODEL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 5.1  THE KEY INSIGHT — The policy IS the reward model           │
└─────────────────────────────────────────────────────────────────┘

DPO (Rafailov et al., 2023) asks: do we really need a separate reward
model? Can we extract preferences directly from the policy?

The mathematical trick:

  In RLHF with KL penalty, the OPTIMAL policy π* satisfies:
    r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

  This means: the reward function can be REARRANGED to express the
  ratio of the optimal policy to the reference policy!

  Substituting this into the Bradley-Terry preference model:
    P(y_A ≻ y_B) = σ(r_A - r_B)
                 = σ(β·log(π*(y_A)/π_ref(y_A)) - β·log(π*(y_B)/π_ref(y_B)))

  The Z(x) terms cancel! The reward model disappears entirely.
  The preference probability depends ONLY on the policy and reference.

┌─────────────────────────────────────────────────────────────────┐
│ 5.2  THE DPO LOSS                                                │
└─────────────────────────────────────────────────────────────────┘

  L_DPO = -log σ( β · [log(π_θ(y_chosen)/π_ref(y_chosen))
                       - log(π_θ(y_rejected)/π_ref(y_rejected))] )

Or more intuitively:
  L_DPO = -log σ( β · [implicit_reward(chosen) - implicit_reward(rejected)] )
  where implicit_reward = log(π_θ / π_ref)

This is a BINARY CROSS-ENTROPY loss on preferences, using the policy
itself as the reward function! DPO eliminates:
  ✗ Reward model (no separate training needed)
  ✗ Value model / critic (no advantage estimation)
  ✗ PPO's clipping and trust region machinery
  ✗ RL sampling loop (just gradient descent like pre-training)

DPO trains on PREFERENCE PAIRS just like the reward model, but
updates the POLICY directly. It's simpler, faster, and more stable.

┌─────────────────────────────────────────────────────────────────┐
│ 5.3  DPO vs RLHF — Tradeoffs                                     │
└─────────────────────────────────────────────────────────────────┘

  DPO ADVANTAGES:
    - Much simpler (one model, one loss)
    - No reward model to train or hack
    - Stable training (pure supervised learning)
    - Computationally cheaper

  DPO LIMITATIONS:
    - Off-policy only: learns from FIXED preference data, can't
      explore and discover new behaviors
    - Can overfit to preference dataset distribution
    - Less effective when preference data is very different from
      what the model would naturally generate
    - Can't learn from process-level rewards (e.g., "the reasoning
      steps were correct even though the final answer was wrong")

  RLHF ADVANTAGES:
    - On-policy: the model generates its OWN responses and gets
      feedback on them. Can discover novel strategies.
    - Handles process rewards (step-by-step reasoning feedback)
    - More aligned with how humans actually learn (trial and error)

  Modern practice: DPO for initial alignment, iterative RLHF for
  refinement. Some use online DPO (iterative preference collection)
  to bridge the gap.
""")


# ──────────────────────────────────────────────────────────────────
# 5.4  DPO Loss: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("DPO LOSS DEMONSTRATION:")
print("-" * 70)

def dpo_loss(log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected, beta=0.1):
    """Compute DPO loss on a single preference pair.

    log_pi_chosen:    log probability of chosen response under current policy
    log_pi_rejected:  log probability of rejected response under current policy
    log_ref_chosen:   log probability of chosen response under reference model
    log_ref_rejected: log probability of rejected response under reference model
    beta:             temperature parameter (higher = more weight on reference)
    """
    # Implicit rewards: how much the policy PREFERS each response over reference
    implicit_chosen = log_pi_chosen - log_ref_chosen
    implicit_rejected = log_pi_rejected - log_ref_rejected

    # Scaled difference: how much more we prefer chosen over rejected
    scaled_diff = beta * (implicit_chosen - implicit_rejected)

    # Bradley-Terry: P(chosen preferred) = sigmoid(scaled_diff)
    p_chosen = 1.0 / (1.0 + np.exp(-scaled_diff))

    # Binary cross-entropy: we want P(chosen) → 1.0
    return -np.log(np.clip(p_chosen, 1e-10, 1.0))

# Simulate three preference pairs with different alignment levels
scenarios = [
    # (chosen logP policy, rejected logP policy, chosen logP ref, rejected logP ref, description)
    (-1.0, -1.5, -1.0, -1.0, "Great: policy prefers chosen, ref is neutral"),
    (-0.8, -1.2, -1.0, -0.7, "Mixed: policy's preference is right, ref was biased toward rejected"),
    (-1.0, -1.0, -1.0, -1.0, "Bad: policy can't distinguish, same as reference"),
]

beta = 0.1

print(f"  {'Chosen logP':>13}  {'Rej logP':>11}  {'Chosen ref':>11}  "
      f"{'Rej ref':>11}  {'Implicit Δ':>11}  {'Loss':>8}  Scenario")
print(f"  " + "-" * 85)
for log_c_pi, log_r_pi, log_c_ref, log_r_ref, desc in scenarios:
    loss = dpo_loss(log_c_pi, log_r_pi, log_c_ref, log_r_ref, beta)
    implicit_delta = (log_c_pi - log_c_ref) - (log_r_pi - log_r_ref)
    print(f"  {log_c_pi:13.1f}  {log_r_pi:11.1f}  {log_c_ref:11.1f}  "
          f"{log_r_ref:11.1f}  {implicit_delta:11.4f}  {loss:8.4f}  {desc}")

print(f"""
  INTERPRETATION:
    - Scenario 1: The policy strongly prefers the chosen response over
      what the reference model would prefer. Low loss. Good alignment.
    - Scenario 2: The reference model was biased toward the rejected
      response (higher ref logP for rejected than chosen), but the
      policy corrected this. Loss is moderate — still learning.
    - Scenario 3: The policy and reference are identical — both are
      equally likely to produce either response. High loss. The model
      hasn't learned the preference at all.

  DPO pushes the policy to:
    1. INCREASE probability of chosen responses (relative to reference)
    2. DECREASE probability of rejected responses (relative to reference)
""")


# ═══════════════════════════════════════════════════════════════════
# PART 6: GRPO — Group Relative Policy Optimization
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 6: GRPO — DEEPSEEK'S INNOVATION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 6.1  THE PROBLEM WITH PPO — Four models, too complex            │
└─────────────────────────────────────────────────────────────────┘

PPO's main pain points in practice:
  1. The VALUE MODEL (critic) is as large as the policy itself.
     Training a 70B critic alongside a 70B policy doubles memory.
  2. The reward model is an imperfect proxy that can be hacked.
  3. Four models interacting creates instability — if one breaks,
     the whole pipeline collapses.
  4. Advantage estimation (GAE) adds hyperparameters and complexity.

Enter GRPO (Shao et al., 2024 / DeepSeekMath / DeepSeek-R1):

┌─────────────────────────────────────────────────────────────────┐
│ 6.2  GRPO'S KEY INSIGHT — Group statistics replace the critic    │
└─────────────────────────────────────────────────────────────────┘

Instead of training a value model to estimate the baseline, GRPO
SAMPLES A GROUP of responses for each prompt and computes advantages
from GROUP STATISTICS:

  For prompt x, generate G responses: y₁, y₂, ..., y_G
  For each response, compute its reward: r₁, r₂, ..., r_G

  Advantage of response i = (rᵢ - mean(r)) / std(r)

  → If your response scored ABOVE the group average, increase its
    probability. If below, decrease it.

This is brilliant for two reasons:
  1. NO CRITIC MODEL NEEDED — the group IS the baseline
  2. The advantage is NORMALIZED — always zero-mean, unit-variance
     regardless of reward scale

GRPO uses PPO's clipping but replaces the learned advantage with the
group-relative advantage:

  ratio = π_θ(token | context) / π_old(token | context)
  clipped_ratio = clip(ratio, 1-ε, 1+ε)
  loss = -min(ratio × A_group, clipped_ratio × A_group)
       + β · KL(π_θ || π_ref)    ← still penalize divergence

┌─────────────────────────────────────────────────────────────────┐
│ 6.3  RULE-BASED REWARDS — No reward model needed either          │
└─────────────────────────────────────────────────────────────────┘

GRPO can use RULE-BASED rewards, completely eliminating the reward
model:

  For math problems:
    - Accuracy: 1.0 if final answer matches ground truth, else 0.0
    - Format: 1.0 if reasoning is in <think> tags, else 0.0

  For code generation:
    - Pass rate: fraction of unit tests that pass
    - Style: lint score

  For general alignment:
    - Helpfulness: LLM-as-judge (but this IS a reward model)

Rule-based rewards are PERFECT (no reward hacking possible) but only
work for tasks with verifiable outcomes (math, code, logic puzzles).

┌─────────────────────────────────────────────────────────────────┐
│ 6.4  GRPO vs PPO vs DPO — Comparison                             │
└─────────────────────────────────────────────────────────────────┘

                  PPO           DPO           GRPO
  ─────────────────────────────────────────────────────────────
  Reward model      Yes           No            No (rule-based)
  Value model       Yes           No            No (group stats)
  Reference model   Yes           Yes           Yes
  On-policy         Yes           No            Yes
  Preference data   No            Yes           No
  Process rewards   Possible      Difficult     Yes (per-step)
  Memory cost       Highest       Lowest        Medium
  Complexity        Highest       Lowest        Medium
""")

# ──────────────────────────────────────────────────────────────────
# 6.5  GRPO Advantage: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("GRPO GROUP ADVANTAGE DEMONSTRATION:")
print("-" * 70)

np.random.seed(42)

# Generate 8 responses for a single prompt, compute their rewards
G = 8
# Simulate: some responses are good (math answers), some aren't
rewards = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

# GRPO: normalize within group
mean_r = np.mean(rewards)
std_r = np.std(rewards)
advantages = (rewards - mean_r) / (std_r + 1e-8)

print(f"  Response | Reward | Advantage | Interpretation")
print(f"  " + "-" * 55)
for i in range(G):
    interp = "↑ ENCOURAGE" if advantages[i] > 0 else "↓ DISCOURAGE"
    print(f"  {i:8d} | {rewards[i]:6.1f} | {advantages[i]:9.4f} | {interp}")

print(f"\n  Group mean reward: {mean_r:.2f}")
print(f"  Group std reward:  {std_r:.2f}")

print(f"""
  KEY INSIGHT: The advantage is RELATIVE to the group. Even if
  ALL responses are bad (all rewards = 0.1), the ADVANTAGES are
  still zero — GRPO won't push in any direction because there's
  no signal to distinguish good from bad.

  This is both a strength (no reward hacking via scale) and a
  limitation (can't improve if all responses are equally bad).

  Compare to PPO: the value model would estimate "this state is
  worth 0.1" and the advantage for a 0.1 reward would be 0.0.
  Same result, but GRPO achieves it without training a critic.
""")


# ═══════════════════════════════════════════════════════════════════
# PART 7: RL FOR REASONING — DeepSeek-R1 and the "Aha Moment"
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 7: RL FOR REASONING")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 7.1  BEYOND HELPFULNESS — Using RL to bootstrap reasoning       │
└─────────────────────────────────────────────────────────────────┘

All the alignment we've discussed so far is about HELPFULNESS and
HARMLESSNESS — making the model say the right things and refuse the
wrong things.

DeepSeek-R1 (2025) showed something more profound: RL can bootstrap
REASONING capability in a model that wasn't explicitly trained to
reason step by step.

The recipe (DeepSeek-R1-Zero):
  1. Start with a BASE model (DeepSeek-V3-Base)
  2. Apply GRPO with TWO rule-based rewards:
     a. ACCURACY reward: 1.0 if final answer is correct, 0.0 otherwise
     b. FORMAT reward: 1.0 if reasoning is wrapped in <think>...</think>
        and answer in <answer>...</answer>, 0.0 otherwise
  3. NO supervised chain-of-thought data. NO correctness feedback on
     intermediate steps. Just: "is the answer right, and is it formatted?"

  That's it. No SFT on reasoning traces. No process reward model.
  Just group-relative advantage + binary correctness + format.

┌─────────────────────────────────────────────────────────────────┐
│ 7.2  THE "AHA MOMENT" — Emergent self-correction                 │
└─────────────────────────────────────────────────────────────────┘

During training, DeepSeek-R1-Zero spontaneously developed:

  1. SELF-VERIFICATION: "Wait, let me check that..."

  2. BACKTRACKING: "Hmm, that doesn't seem right. Let me reconsider..."

  3. ALTERNATIVE APPROACHES: "Actually, there's a simpler way..."

  4. REFLECTION: "I made a mistake in step 3. The correct calculation is..."

  None of these behaviors were explicitly trained. They EMERGED from
  the RL process because:
    - The model that tries multiple approaches in <think> and settles
      on the right answer gets higher accuracy reward
    - The model that catches its own mistakes during reasoning gets
      higher accuracy reward
    - The model that doesn't verify its work sometimes gets the answer
      wrong → lower reward → GRPO discourages this

  RL discovered that REASONING ABOUT REASONING is useful, even though
  the reward only looked at the final answer.

┌─────────────────────────────────────────────────────────────────┐
│ 7.3  WHY THIS MATTERS — RL as a general-purpose scaffold         │
└─────────────────────────────────────────────────────────────────┘

This changes how we think about RL for language models:

  OLD VIEW: RL is for ALIGNMENT — making a capable model well-behaved.
            First, make the model smart (pre-training + SFT). Then,
            make it safe (RLHF).

  NEW VIEW: RL is for CAPABILITY — RL can CREATE new behaviors that
            weren't in the training data. Given the right reward
            structure, the model discovers reasoning strategies,
            tool use, multi-step planning, and self-correction on
            its own.

  This is closer to how AlphaGo worked: RL didn't just align Go
  knowledge, it DISCOVERED new strategies (move 37!) that humans
  had never considered.

  The implication: future models may use RL during pre-training or as
  a core capability-building phase, not just as a final safety filter.

┌─────────────────────────────────────────────────────────────────┐
│ 7.4  THE DEEPSEEK-R1 PIPELINE (final version)                    │
└─────────────────────────────────────────────────────────────────┘

DeepSeek-R1 (the released model) used a more sophisticated pipeline:

  1. COLD START SFT: Fine-tune on a few thousand high-quality
     chain-of-thought examples (prevents the "language mixing"
     issue that plagued R1-Zero)

  2. REASONING RL: Apply GRPO with rule-based rewards (same as
     R1-Zero but starting from the cold-start SFT model)

  3. REJECTION SAMPLING + SFT: Generate many reasoning traces,
     keep only the CORRECT ones, use them as SFT data

  4. RL FOR ALL SCENARIOS: Final RLHF stage for helpfulness,
     harmlessness, and general alignment

This pipeline combines the best of all approaches:
  - SFT for format and language quality
  - GRPO for reasoning capability bootstrapping
  - Rejection sampling for data quality
  - DPO/RLHF for final helpfulness alignment
""")


# ──────────────────────────────────────────────────────────────────
# 7.5  Reasoning RL: NumPy Demonstration
# ──────────────────────────────────────────────────────────────────

print("REASONING RL DEMONSTRATION:")
print("-" * 70)

np.random.seed(42)

# Simulate a reasoning RL training step with GRPO
# For each prompt, the model generates G=4 reasoning traces
G = 4
prompts = 3

# Simulated outputs for a math problem: each trace gets a reward
# Format reward (1.0 if proper <think>/<answer> tags) + Accuracy reward (1.0 if correct)
format_rewards = np.array([
    [0.0, 1.0, 1.0, 0.0],   # prompt 1: 2/4 formatted correctly
    [1.0, 1.0, 1.0, 1.0],   # prompt 2: all formatted
    [0.0, 1.0, 0.0, 1.0],   # prompt 3: 2/4 formatted
])

accuracy_rewards = np.array([
    [0.0, 1.0, 1.0, 0.0],   # prompt 1: 2/4 correct answers
    [0.0, 1.0, 0.0, 1.0],   # prompt 2: 2/4 correct
    [1.0, 1.0, 1.0, 1.0],   # prompt 3: all correct
])

# Combined reward (equal weight)
total_rewards = format_rewards + accuracy_rewards

# GRPO: compute group-relative advantages per prompt
print(f"  {'Prompt':>7} {'Resp':>5} {'Format':>7} {'Accuracy':>9} {'Total':>6} {'Advantage':>10} {'Effect':>15}")
print(f"  " + "-" * 70)

for p in range(prompts):
    group = total_rewards[p]
    mean_g = np.mean(group)
    std_g = np.std(group) + 1e-8

    for g in range(G):
        adv = (group[g] - mean_g) / std_g
        if adv > 0.3:
            effect = "↑ Strong push"
        elif adv > 0:
            effect = "↑ Mild push"
        elif adv > -0.3:
            effect = "↓ Mild push"
        else:
            effect = "↓ Strong push"

        print(f"  {p:7d} {g:5d} {format_rewards[p,g]:7.1f} "
              f"{accuracy_rewards[p,g]:9.1f} {group[g]:6.1f} {adv:10.4f} {effect:>15}")

print(f"""
  KEY OBSERVATIONS:
    - Prompt 1: Responses 1&2 (formatted + correct answer) get pushed
      UP because they're above the group average.
    - Prompt 2: All responses are formatted, but only 1&3 have correct
      answers. These get pushed UP.
    - Prompt 3: All answers are correct, all formatted. All advantages
      are ZERO — no learning signal. GRPO can't improve perfection.

  The model learns: "To get high reward, I should format my reasoning
  AND get the right answer." But the RL process must DISCOVER that
  formatting and verifying work leads to correct answers. The model
  isn't TOLD to reason — it discovers that reasoning leads to reward.
""")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY: The Complete LLM Lifecycle")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE FULL PIPELINE — What you now understand                      │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │PRE-TRAIN │ → │   SFT    │ → │ALIGNMENT │ → │  INFERENCE   │ │
│  │(Module 8)│   │(Module 10)│  │(Module 10)│   │(i00-i09)    │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│       │              │               │               │         │
│  Next token      Instruction    PPO / DPO /      Efficient    │
│  prediction      following      GRPO              serving     │
│  on internet     via SFT        from human       at scale     │
│  text            on curated     preferences                   │
│                  conversations                                 │
└─────────────────────────────────────────────────────────────────┘

WHAT YOU LEARNED:

  1. PRE-TRAINING → POST-TRAINING pipeline
     Pre-training creates a document completer. Post-training
     (SFT + alignment) transforms it into an assistant.

  2. SFT — Supervised Fine-Tuning
     Train on (prompt, ideal_response) pairs. Same loss as pre-
     training. Teaches FORMAT but not VALUES. Prone to distribution
     mismatch and can't learn from comparisons.

  3. BRADLEY-TERRY — The math of preferences
     P(A ≻ B) = σ(r_A - r_B). The foundation of all preference-
     based alignment. Turns "A is better than B" into a loss.

  4. PPO — The original RLHF algorithm
     Four models: policy, reference, reward, value. KL penalty
     prevents reward hacking. PPO clipping ensures stable updates.
     Complex but proven at scale (InstructGPT, ChatGPT).

  5. DPO — Alignment without a reward model
     Rearranges the RLHF objective to express reward as log(π/π_ref).
     No separate reward model needed. Simpler, faster, more stable.
     Limited to fixed preference data (off-policy).

  6. GRPO — Group-relative advantages
     Replaces the value model with GROUP STATISTICS. Samples G
     responses per prompt, normalizes rewards within the group.
     Pairs naturally with rule-based rewards. Used by DeepSeek.

  7. RL FOR REASONING — Beyond alignment
     DeepSeek-R1 showed RL can bootstrap reasoning capabilities.
     GRPO + accuracy reward + format reward → emergent chain-of-
     thought, self-verification, and backtracking. RL is not just
     for safety — it's a capability-building tool.

THE BIG PICTURE:

  Alignment is the bridge between "the internet's next token" and
  "a helpful assistant." It's what makes the difference between a
  model that completes your sentence and a model that understands
  what you want and helps you achieve it.

  The field is moving fast: from PPO (2022) to DPO (2023) to GRPO
  (2024), each generation simplifies the recipe while making it more
  effective. RL for reasoning (DeepSeek-R1) suggests we haven't yet
  found the ceiling on what RL can extract from language models.

NEXT STEPS:
  1. Implement DPO training loop in PyTorch (simpler than PPO!)
  2. Experiment with GRPO on math reasoning tasks
  3. Read the DeepSeek-R1 paper for the full pipeline details
  4. Study Constitutional AI (Anthropic's self-improvement approach)
  5. Move to the inference course (course/inference/) to learn
     how aligned models are served at scale
""")

if __name__ == "__main__":
    print("\nModule 10 complete! The full transformer course is now:")
    print("  00_prerequisites → 09_advanced_topics → 10_alignment")
    print()
    print("To run the inference course:")
    print("  uv run python course/inference/run.py")
