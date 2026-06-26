# Fundamental Assumptions Worth Questioning

Every field has its "obvious" truths — design choices that got locked in early,
everyone accepted, and few people revisit. This is a running collection of those
assumptions in transformer architectures and LLM systems. For each: what's the
assumption, why question it, what alternatives exist, and where the course
touches on it.

### Quick reference

| # | Assumption | Primary course files |
|---|---|---|
| 1 | Autoregressive: one token at a time | `i01_autoregressive_loop.py`, `i06_speculative_decoding.py` |
| 2 | Softmax forces attention to sum to 1 | `03_simple_attention.py`, `04_self_attention.py`, `00_prerequisites.py` |
| 3 | Subword tokenization (BPE) | `01_tokenization.py` |
| 4 | Fixed `d_model` across all layers | `06_transformer_block.py`, `07_mini_gpt.py` |
| 5 | Pre-LN was settled empirically | `06_transformer_block.py`, `07_mini_gpt.py` |
| 6 | Purely feedforward, no recurrence | `06_transformer_block.py`, `07_mini_gpt.py`, `09_advanced_topics.py` |
| 7 | Cross-entropy penalizes all wrong tokens equally | `08_training.py` |
| 8 | Teacher forcing: train on ground truth, run on own output | `08_training.py`, `i01_autoregressive_loop.py` |
| 9 | Weight tying: embedding and LM head share weights | `02_embeddings.py`, `07_mini_gpt.py` |
| 10 | Dense weight matrices | `09_advanced_topics.py`, `i04_quantization.py` |

---

## 1. Autoregressive decoding: one token at a time — why?

**The assumption:** Text generation must be sequential. Token N+1 depends on token N,
so you generate one at a time, feeding the full sequence through the model at every
step. This is universally accepted in LLM inference and baked into the architecture.

**Why question it:**

- Speculative decoding achieves 2-5x speedups by *drafting multiple tokens in
  parallel* and only verifying sequentially. This proves the sequential
  bottleneck is partially artificial — a small draft model can guess tokens 2,
  3, 4 without seeing token 1's output from the large model.
- Image diffusion models generate all pixels in parallel.
- Human speech/writing isn't strictly sequential — we plan phrases, clauses, and
  sentences ahead of articulation. Why can't language models do the same?
- The architecture forces sequential generation; the semantics don't always
  require it.

**Alternatives being explored:**

- Speculative decoding (band-aid: draft model + verification)
- Jacobi decoding (iterative parallel refinement)
- Discrete diffusion for text (generate all tokens simultaneously, denoise)
- Non-autoregressive transformers (NAT) — predict all positions at once
- Blockwise parallel decoding with iterative refinement

**Course references:**

- `course/inference/i01_autoregressive_loop.py` — the raw O(n²) loop that makes the problem concrete
- `course/inference/i06_speculative_decoding.py` — the main practical workaround today

### Rethink

What if the model produced a "plan" (high-level semantic structure) first, then
filled in tokens in parallel? What if we generated text at the phrase or sentence
level rather than token level? What if the generation order wasn't left-to-right
at all?

---

## 2. Softmax attention: must every token attend?

**The assumption:** Every token must compute a compatibility score with every
other token, then softmax-normalize all scores so they sum to 1. "Attention is
all you need" — literally, every token must allocate 100% attention across all
other tokens.

**Why question it:**

- Softmax forces attention weights to sum to 1, which means tokens MUST
  allocate their entire attention budget even when nothing is relevant. A token
  paying "0.3% attention" to 3000 irrelevant tokens wastes compute and dilutes
  the signal that should go to the few truly relevant tokens.
- The sum-to-1 constraint is a mathematical convenience (probabilities must sum
  to 1), not a semantic requirement. Attention doesn't need to be a probability
  distribution — "nothing is relevant" is a valid answer.
- In practice, attention IS sparse — the softmax just makes it "soft sparse"
  (many near-zero values) rather than "hard sparse" (zeros).
- Softmax + squared attention scores can't represent "attend to token A OR
  token B" — it can only represent weighted combinations. Real reasoning
  sometimes requires discrete selection.

**Alternatives:**

- Sparsemax: projects onto the probability simplex with hard zeros for low-scoring tokens
- Entmax: learned sparsity via Tsallis entropy (generalizes softmax and sparsemax)
- ReLU attention: replace softmax with ReLU, no normalization — let tokens decide how much total attention to give
- Linear attention: approximate attention without softmax at all (Performer, Linformer)
- Mamba-style selective state spaces: no attention, no softmax — different mechanism entirely

**Course references:**

- `course/transformer/00_prerequisites.py` — sections 3.1-3.5 build softmax from first principles
- `course/transformer/03_simple_attention.py` — introduces attention as softmax over pairwise scores
- `course/transformer/04_self_attention.py` — the full Q@K^T + softmax formulation
- `course/transformer/05_multi_head_attention.py` — multi-head extends the softmax assumption

### Rethink

What if attention was formulated as a retrieval operation with a learned
threshold — "attend only to tokens with relevance > τ, and don't normalize"?
What if each token could say "I'm done, nothing else matters"?

---

## 3. Subword tokenization: chopping language into arbitrary fragments

**The assumption:** Text is preprocessed into subword tokens via BPE,
SentencePiece, or similar. This is the standard preprocessing step for virtually
every modern LLM.

**Why question it:**

- Subword tokenization introduces artifacts that shouldn't exist: "the" is one
  token but "th" + "e" is two; "pipeline" vs "pipe" + "line" have different
  representations despite sharing meaning; numbers get split unpredictably
  (1234 → "12" + "34" or "1" + "2" + "3" + "4").
- These artifacts propagate through the entire model. The model must learn to
  undo tokenization artifacts before it can reason about semantics.
- Tokenization is language-specific: the same word in different scripts gets
  radically different token counts, creating an unfair compute tax on
  non-English languages.
- Misspellings and Unicode variations create tokenization inconsistencies.
- Multi-turn conversations, code, and structured data don't fit cleanly into
  natural-language tokenizers.

**Alternatives:**

- Byte-level models: operate directly on UTF-8 bytes (MegaByte, ByT5) — no tokenization at all
- Character-level models: simpler, but longer sequences
- Learned tokenization: not a fixed vocabulary — the tokenization is part of the model and trained end-to-end
- Morpheme-based splitting: linguistically meaningful units rather than frequency-based BPE merges
- Patch-based approaches for images and video: a different modality analogy

**Course references:**

- `course/transformer/01_tokenization.py` — builds char, word, and BPE tokenizers from scratch

### Rethink

What if the model operated directly on raw bytes with a hierarchical architecture
(chunk-level, sentence-level, paragraph-level) rather than forcing everything
through a fixed 32k-100k vocabulary? The sequence would be longer but the
representations wouldn't need to un-learn tokenization artifacts.

---

## 4. Fixed dimension (`d_model`) throughout all layers

**The assumption:** Every transformer layer has the same hidden dimension. Input
embeddings are `d_model`, every FFN hidden layer is `4 * d_model`, output is
`d_model`. Respected models (GPT-2 through LLaMA-3) maintain this.

**Why question it:**

- Early layers do syntactic/structural work (parsing, entity boundaries).
- Later layers do semantic/reasoning/compositional work.
- Why should they have the same representational capacity? Different
  computational tasks need different representational bandwidth.
- CNNs routinely vary channel counts across depth: shallower layers have fewer
  channels, deeper layers have more. This is considered obvious good design in
  vision.
- Vision transformers sometimes vary dimension (pyramid ViTs), but text
  transformers almost never do.
- Fixed dimension might waste capacity in early layers (too many parameters for
  the task) and starve later layers (not enough for complex reasoning).

**Alternatives:**

- Pyramid transformers: gradually increase dimension with depth
- Progressive dimension expansion: d_model grows across blocks
- Dynamic width per layer: learned by pruning or architecture search
- Variable-width FFN: keep d_model fixed but vary FFN expansion ratio

**Course references:**

- `course/transformer/06_transformer_block.py` — builds the transformer block with fixed dimension
- `course/transformer/07_mini_gpt.py` — assembles the full model with uniform d_model

### Rethink

What if we started wide (capture lots of low-level features) then narrowed
(compress to semantic representations) then widened again (expand for reasoning)?
Or what if dimension was learned per-layer during training, not fixed upfront?

---

## 5. Layer normalization: Pre-LN because training was easier

**The assumption:** LayerNorm is placed before attention and FFN sublayers
(Pre-LN) because it made training more stable than after (Post-LN). This was
settled empirically around 2019-2020 and is now baked into every major model
family.

**Why question it:**

- Post-LN was the original Transformer design (Vaswani et al., 2017). A key
  finding of the original paper was that it needed careful scheduling (warmup).
- Pre-LN won because it made training easier — gradients don't explode in early
  stages. But there's evidence Post-LN produces better final quality when you
  get it to converge.
- RMSNorm (remove mean-centering, just divide by RMS) has largely replaced
  LayerNorm in modern models (LLaMA, Mistral, etc.). Is RMSNorm actually better,
  or just simpler/faster?
- Normalization destroys magnitude information that could be useful — if token
  A's pre-norm activations are 100x token B's, should we really squash that?

**Alternatives:**

- Post-LN with better initialization/warmup (DeepNet modifications)
- DeepNorm: modified residual connections for deeper models
- No normalization at all with careful initialization (Fixup, ReZero)
- Learned normalization parameters that vary by token position
- Sub-LN (normalize within residual sub-branches)

**Course references:**

- `course/transformer/06_transformer_block.py` — builds the block with layer norm
- `course/transformer/07_mini_gpt.py` — Pre-LN in the full model assembly

### Rethink

What if normalization was learned rather than fixed? Each token, each position,
each head could learn how much normalization it needs. What if we normalized
along different axes (head-wise, FFN-channel-wise) depending on the operation?

---

## 6. Purely feedforward: no feedback, no iteration

**The assumption:** Transformers process information in a purely feedforward
manner. Each layer transforms its input once and passes it forward. No
recurrence, no feedback loops, no iterative refinement within a forward pass.

**Why question it:**

- Human reasoning is iterative: we revise, backtrack, reconsider, loop.
- A transformer gets one pass through N layers. If layer 24 realizes something
  that layer 3 should have handled differently, it's too late — information
  only flows forward.
- Universal transformers (Dehghani et al., 2019) showed that a single recurrent
  block applied for an adaptive number of steps can match or beat deep
  transformers on many tasks. This suggests depth is often a coarse substitute
  for recurrence.
- Chain-of-thought and "thinking tokens" are workarounds: they add more
  sequential computation, but within the same forward-pass model. What if the
  model could loop internally?

**Alternatives:**

- Universal transformers: apply the same block repeatedly, depth decided per-input
- Adaptive computation time (ACT): learn when to stop processing a token
- Layer looping / block recurrence: apply same block 2-3x before moving forward
- Thinking tokens / pause tokens: insert "blank" positions where the model can do extra computation
- xLSTM and similar: bring back controlled recurrence alongside attention

**Course references:**

- `course/transformer/06_transformer_block.py` — the block is purely feedforward
- `course/transformer/07_mini_gpt.py` — stacks blocks without recurrence
- `course/transformer/09_advanced_topics.py` — covers alternatives beyond standard transformers

### Rethink

What if each block could decide "I need another pass" and loop on itself?
What if there was a global "refinement" phase after the forward pass where
tokens could exchange information bidirectionally? What if depth wasn't a
hyperparameter but something the model decided dynamically?

---

## 7. Cross-entropy loss: penalizing all wrong tokens equally

**The assumption:** The training objective is cross-entropy loss between the
model's predicted token distribution and the one-hot ground-truth next token.
Every incorrect token is equally wrong.

**Why question it:**

- Cross-entropy treats "cat" and "zygomorphic" as equally bad wrong answers
  when the correct word is "dog".
- Semantic similarity between tokens is completely ignored by the loss function.
- This means the model gets no partial credit for answers that are "close" in
  meaning space.
- Contrastive objectives could exploit token embedding similarity to provide
  richer gradients.
- For knowledge-intensive tasks, near-misses vs. complete fabrications have
  very different real-world costs but identical loss contributions.

**Alternatives:**

- Contrastive learning objectives (e.g., sentence-level, token-level)
- Margin-based losses: wrong answers that are "closer" get smaller penalties
- Distillation losses: train against a larger model's full distribution (not just the argmax), which naturally encodes similarity
- RLHF/DPO: addresses this indirectly through preference learning — the reward model captures semantic preference beyond token identity
- Multi-token prediction: predict the next N tokens, not just one, to capture longer-range structure

**Course references:**

- `course/transformer/08_training.py` — loss functions, backpropagation, and optimization

### Rethink

What if the loss function operated in embedding space rather than vocabulary
space — measure the distance between the model's predicted embedding and the
true token's embedding, weighted by semantic similarity to other tokens?
What if loss was defined over phrases/sentences rather than tokens?

---

## 8. Teacher forcing: train on ground truth, run on your own output

**The assumption:** During training, at each step the model receives the
ground-truth previous token (teacher forcing). At inference, it receives its
own previous prediction. This gap is well-known but treated as unavoidable.

**Why question it:**

- The model never learns to recover from its own mistakes because it never
  sees them during training.
- If the model produces a slightly-off token at test time, the compounding
  error cascades — each subsequent token is conditioned on an input
  distribution the model was never trained to handle.
- This is the root cause of autoregressive models degrading over long
  sequences: the input distribution drifts further and further from the
  training distribution.
- The field mostly accepted this and moved on to scaling data/model size as
  a workaround — bigger models make fewer mistakes, so they stay closer to
  the training distribution longer. But this doesn't fix the fundamental
  mismatch.

**Alternatives:**

- Scheduled sampling: during training, sometimes feed the model's own
  prediction instead of ground truth (probability increases over training)
  — known to have theoretical issues but can work in practice
- Professor forcing: adversarial training where a discriminator tries to
  distinguish training-time from inference-time hidden states
- RL-based fine-tuning: expose the model to its own outputs and reward
  correctness (RLHF does something like this)
- Diffusion-style generation: instead of autoregressive left-to-right,
  generate in parallel with iterative refinement, removing the mismatch
  entirely

**Course references:**

- `course/transformer/08_training.py` — training with teacher forcing
- `course/inference/i01_autoregressive_loop.py` — inference where the gap manifests

### Rethink

What if training explicitly included a "recovery from mistakes" objective?
What if the model learned a correction policy — "when I see a likely-wrong
token in my input, here's how to re-interpret it"? What if training alternated
between teacher-forced and self-generated rollouts?

---

## 9. Weight tying: embedding and LM head share weights

**The assumption:** The input embedding matrix and the output projection matrix
(LM head) are the same matrix. This saves `V * d_model` parameters (e.g., 262M
params for V=32k, d=8192). This is standard in most models.

**Why question it:**

- Input embeddings and output predictions are semantically different operations.
  Input: "given token ID, produce a vector representation for processing."
  Output: "given a processed representation, project to token probabilities."
- Weight tying forces the same matrix to serve both roles in opposite directions.
  The embedding maps token → vector; the LM head maps vector → token logits.
- The parameter savings are real but trivially small relative to total model
  size: 262M saved out of 70B total is ~0.37%.
- There's evidence that untied weights can improve performance, especially for
  larger models where the parameter savings matter less.
- The token embedding matrix is already one of the most parameter-intensive
  components (V * d_model) — but that's a tokenization artifact (see #3),
  not an architectural necessity.

**Alternatives:**

- Fully untied weights: separate embedding and LM head matrices
- Partially tied: shared base matrix + learnable projection for the output direction
- Factorized embeddings: embed tokens as lower-dimensional codes, then project up

**Course references:**

- `course/transformer/02_embeddings.py` — builds embedding layers
- `course/transformer/07_mini_gpt.py` — ties weights in the full model

### Rethink

If parameter savings is not the bottleneck (it isn't at scale), why tie?
What if the output projection was a learned function of the embedding,
not the embedding itself? What if we factorized V to reduce embedding
parameters, then untied the head to compensate?

---

## 10. Dense weight matrices: every parameter is stored and multiplied

**The assumption:** Weight matrices are dense — every parameter is stored in
memory, loaded into registers, and multiplied regardless of its magnitude or
importance. This is the default in every framework.

**Why question it:**

- After training, a large fraction of weights are extremely close to zero
  (often 50-80%, depending on the model and training regime).
- These near-zero weights consume memory bandwidth, compute cycles, and
  storage for negligible contribution to the output.
- Structured sparsity (NVIDIA's 2:4 pattern) exploits this at the hardware
  level but is fighting the dense assumption rather than rethinking it.
- MoE demonstrates at the layer level that different inputs need different
  parameters. But within each expert, weights are still dense.
- What if the architecture was designed for sparsity from the start —
  allocating capacity where needed rather than everywhere uniformly?

**Alternatives:**

- Unstructured pruning: zero out individual weights (hard to accelerate)
- Structured sparsity: zero out whole channels/heads/rows (hardware-friendly)
  - 2:4 sparsity: 2 nonzero per group of 4, hardware-accelerated on A100/H100
- Sparse training: train sparse networks from scratch (lottery ticket hypothesis)
- Dynamic sparse training: grow and prune connections during training
- MoE: sparsity at the layer/block level — different tokens route to different experts

**Course references:**

- `course/transformer/09_advanced_topics.py` — MoE as a sparsity mechanism
- `course/inference/i04_quantization.py` — reducing precision, which is complementary to sparsity

### Rethink

What if sparsity was a first-class design principle rather than a
post-training optimization? What if we trained with a parameter budget
per-operation rather than a fixed matrix shape? What if the model could
dynamically "activate" parameters based on the input, like a
fine-grained learned hash table?
