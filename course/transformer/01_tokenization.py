"""
MODULE 1: TOKENIZATION — How Text Becomes Numbers
===================================================

A transformer cannot process raw text. It only understands sequences
of numbers. Tokenization is the bridge that converts human language
into the model's native "language" of integer token IDs.

This module builds three tokenizers from scratch — character, word,
and subword (BPE) — so you understand not just HOW they work, but
WHY subword tokenization is the industry standard and HOW it affects
everything from vocabulary size to generation quality.

WHAT YOU'LL LEARN:
   1. Why tokenization exists (the text-to-numbers problem)
   2. Character-level: simplest, but too long and semantically empty
   3. Word-level: more semantic, but explodes with unknown words
   4. Subword (BPE): the Goldilocks solution — no unknowns, efficient
   5. How vocab size is a critical architectural decision
   6. Real-world tokenizer quirks that affect model behavior

AFTER THIS MODULE:
   You'll understand why GPT-4's tokenizer is a 100K-entry lookup
   table, why "SolidGoldMagikarp" famously broke GPT, and why
   switching tokenizers invalidates a trained model.
"""

import re
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY TOKENIZATION EXISTS AT ALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BACKGROUND: WHY TOKENIZATION?")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE FUNDAMENTAL DISCONNECT                                      │
│                                                                 │
│ Humans reason in:     "The cat sat on the mat"                  │
│ Computers compute on: [15496, 3797, 6159, 319, 262, 11652]     │
│                                                                 │
│ The tokenizer IS the bridge. It's not a preprocessing step —    │
│ it's the first half of the model's "language understanding."    │
│ The embedding layer (Module 2) handles the second half.         │
└─────────────────────────────────────────────────────────────────┘

Three constraints that shape tokenizer design:
  1. FINITE VOCABULARY: The model must know all possible inputs in
     advance. You can't dynamically add new tokens during inference.

  2. FIXED SEQUENCE LENGTH: Transformers have a maximum context
     length. Longer token sequences = less content fits in context.

  3. SEMANTIC GROUPING: Tokens should represent meaningful units.
     "The" and "cat" are useful; "T" and "h" and "e" are not.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: CHARACTER-LEVEL TOKENIZATION — Why It Exists and Why It Fails
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: CHARACTER-LEVEL TOKENIZATION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  CHARACTER-LEVEL: Every character is a token               │
└─────────────────────────────────────────────────────────────────┘

Strategy: Each unique character gets an integer ID.

Advantages:
  ✓ Vocabulary is TINY — ASCII has 128 chars, Unicode ~150K but
    most models use byte-level (~256). Never exceeds 256 entries.
  ✓ NO unknown tokens ever — every text is just characters.
  ✓ Trivially reversible — encode/decode is deterministic.

Disadvantages:
  ✗ LONG sequences — "hello" = 5 tokens. A 512-token context
    window can only see ~85 English words.
  ✗ NO semantic grouping — 'c', 'a', 't' individually mean
    nothing. The model must learn that [c,a,t] is a cat from
    scratch, using up attention capacity.
  ✗ INEFFICIENT attention — computing attention over 512 chars
    when it could be 50 words wastes compute.
""")

text = "hello world"
chars = sorted(set(text))
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

encoded = [char_to_id[ch] for ch in text]
decoded = "".join(id_to_char[i] for i in encoded)

print(f"\n  Text:    '{text}'")
print(f"  Vocab:   {char_to_id}")
print(f"  Encoded: {encoded} ({len(encoded)} tokens)")
print(f"  Decoded: '{decoded}'")

# Show the semantic inefficiency with a real example
longer = "The transformer architecture revolutionized NLP"
char_tokens = len(longer)
word_tokens = len(longer.split())
print(f"\n  '{longer}'")
print(f"  → {char_tokens} character tokens vs {word_tokens} word tokens")
print(f"  → Character-level is {char_tokens/word_tokens:.1f}× longer!")


# ═══════════════════════════════════════════════════════════════════
# 1.2  Why character-level was never a serious option for LLMs
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.2  BYTE-LEVEL TOKENIZATION — Characters' cousin               │
└─────────────────────────────────────────────────────────────────┘

Some recent models (ByT5, CANINE, MegaByte) use byte-level
tokenization: each BYTE (0-255) is a token. This avoids the
Unicode character explosion while keeping the "no unknowns"
property. But it makes sequences 4-8× longer than subword.

Byte-level models require special "patch" or "chunk" attention
mechanisms to handle the length — regular O(n²) attention would
be 64× more expensive for the same text as a subword model.

The tradeoff: subword = more upfront complexity, less compute.
Byte-level = zero vocab design, more compute during inference.
""")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: WORD-LEVEL TOKENIZATION — Better, But Fatal Flaw
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: WORD-LEVEL TOKENIZATION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.1  WORD-LEVEL: Each word is a token                          │
└─────────────────────────────────────────────────────────────────┘

Strategy: Split on whitespace, assign each unique word an ID.

Advantages:
  ✓ SHORTER sequences — "the cat sat" = 3 tokens.
  ✓ SEMANTIC units — "cat" is meaningful; "c","a","t" are not.
  ✓ Efficient attention — more content per attention computation.

Disadvantages:
  ✗ UNBOUNDED VOCABULARY — English has ~170K words in common use,
    millions with technical terms, and infinite with proper nouns.
  ✗ OOV PROBLEM — Out-Of-Vocabulary: what happens when a new word
    appears? The model must either crash, skip it, or use <UNK>.
  ✗ MORPHOLOGY LOST — "run", "runs", "running", "ran" are four
    unrelated tokens, even though they share meaning.
  ✗ LANGUAGE DEPENDENT — Agglutinative languages like Turkish
    or Finnish can form millions of words from a single root.
""")

sample = "The cat sat on the mat. The dog ran fast!"
# Elegant word tokenization (lowercase, handle punctuation)
words = re.findall(r"\w+|[^\w\s]", sample.lower())
word_vocab = {w: i for i, w in enumerate(sorted(set(words)))}

print(f"\n  Text:    '{sample}'")
print(f"  Words:   {words}")
print(f"  Vocab:   {word_vocab}")
print(f"  Tokens:  {len(words)} (vs {len(sample)} characters)")


# ═══════════════════════════════════════════════════════════════════
# 2.2  The OOV problem — The fatal flaw of word-level
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 2.2  THE OOV PROBLEM — Why word-level is unusable              │
└─────────────────────────────────────────────────────────────────┘

Here's what happens when a new word appears:

  Trained on: "the quick brown fox jumps"
  New text:   "the quick brown foxification jumps"

"foxification" is not in the vocabulary. Your options:

  OPTION 1: <UNK> token — "the quick brown <UNK> jumps"
    → Information is destroyed. The model loses the word entirely.

  OPTION 2: Crash / skip — just don't process it
    → Unacceptable for a production system.

  OPTION 3: Dynamic vocabulary — add new words on the fly
    → But the embedding layer has a fixed size! Can't add rows.

The OOV problem is FUNDAMENTAL to word-level tokenization. No
amount of vocabulary expansion solves it — language is infinite.
""")

new_text = "the foxification was unexpected"
try:
    new_words = re.findall(r"\w+|[^\w\s]", new_text.lower())
    new_tokens = [word_vocab[w] for w in new_words]
    print(f"\n  New text: '{new_text}'")
    print(f"  Tokens:   {new_tokens}")
except KeyError as e:
    print(f"\n  ✗ OOV ERROR: '{e}' not in vocabulary!")
    print(f"    This is the fatal flaw of word-level tokenization.")

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: SUBWORD TOKENIZATION — The Goldilocks Solution
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: SUBWORD TOKENIZATION (BPE)")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  SUBWORD TOKENIZATION — The best of both worlds            │
└─────────────────────────────────────────────────────────────────┘

Core insight: Split rare words into frequent sub-pieces.

  "unfortunately" → ["un", "fortunate", "ly"]
  "foxification"  → ["fox", "ification"]

The pieces themselves are in the vocabulary. ANY word can be
represented, even never-before-seen ones, as long as its
sub-pieces are known.

SUBWORD ALGORITHMS:
  BPE (Byte-Pair Encoding): Start with characters, repeatedly
      merge the most frequent adjacent pair. Used by GPT-2/3/4.
  WordPiece: Like BPE but merges based on likelihood improvement.
      Used by BERT.
  Unigram: Start with a large vocabulary and prune. Used by T5,
      SentencePiece, and LLaMA.
  SentencePiece: A tokenizer FRAMEWORK that can use BPE or Unigram.
      Handles whitespace as a token (reversible), language-agnostic.

BPE is the simplest to understand, so we implement it here.
The industry is moving toward BPE (OpenAI) or Unigram (Google/Meta).
""")

# ──────────────────────────────────────────────────────────────────────────────
# BPE IMPLEMENTATION WITH DEEP EXPLANATIONS
# ──────────────────────────────────────────────────────────────────────────────

class SimpleBPETokenizer:
    """Minimal BPE tokenizer. Not production-ready, but fully explains the algorithm.

    BPE algorithm in one sentence:
        "Start with individual bytes as tokens, then repeatedly merge
         the most frequent adjacent pair into a new token."
    """

    def __init__(self):
        self.merges = {}  # (token_a, token_b) → new_token_id
        self.vocab = {}   # token_id → token_string

    def _get_stats(self, ids):
        """Count how often each adjacent pair appears.

        WHY: The most frequent pair is the one that would save the
        most tokens if merged. This is a greedy optimization:
        each merge maximally reduces the sequence length.
        """
        counts = Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        """Replace all occurrences of a pair with its merged ID.

        WHY careful (non-overlapping): If the sequence is [a,b,a,b]
        and we merge (a,b)→X, we need [X,X], not [X,b] or some
        overlapping mess. We advance by 2 after each merge to
        avoid consuming the second element of one pair as the first
        element of the next.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2  # skip the consumed pair
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts, num_merges=50):
        """Learn BPE merges from a corpus.

        WHY multiple texts: We count pairs ACROSS all texts, not per-text.
        This ensures the merges are globally optimal rather than fitting
        to any single example.

        WHY start from bytes: Using ord(c) gives values 0-255 (or wider
        for Unicode). This is a byte-level start — every character is
        representable, so no OOV possible.
        """
        # Start with byte-level tokens using character ordinals
        flat = "".join(texts)
        chars = sorted(set(ord(c) for c in flat))
        self.vocab = {c: chr(c) for c in chars}
        # Next ID starts after the highest character code
        next_id = max(chars) + 1 if chars else 256

        print(f"\n  Training BPE tokenizer on {len(texts)} texts...")
        print(f"  Initial vocab: {len(self.vocab)} tokens (character-level)")
        print(f"  Target merges: {num_merges}")

        # Convert all texts to byte-level ID sequences
        text_ids = [[ord(c) for c in text] for text in texts]

        for merge_step in range(num_merges):
            # Count pairs across all texts
            stats = Counter()
            for ids in text_ids:
                stats.update(self._get_stats(ids))

            if not stats:
                break  # nothing left to merge

            # Greedy: merge the most frequent pair
            top_pair = max(stats, key=stats.get)

            # Create the new token string
            new_token = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            self.vocab[next_id] = new_token
            self.merges[top_pair] = next_id

            # Apply the merge to all texts
            text_ids = [self._merge(ids, top_pair, next_id) for ids in text_ids]
            next_id += 1

            if merge_step < 5 or merge_step >= num_merges - 3:
                print(f"    Merge {merge_step+1}: '{top_pair}' ({stats[top_pair]}×) "
                      f"→ '{new_token}' (ID {next_id-1})")

        print(f"  Final vocab: {next_id} tokens")
        return self.vocab

    def encode(self, text):
        """Tokenize a string into BPE IDs.

        WHY greedy application: A proper BPE tokenizer applies merges
        in priority order, not in the order they were learned. Our
        simplification re-applies all merges until no more apply.
        This works for small vocabularies but is O(merges × seq_len)
        for each application — real tokenizers use more efficient
        algorithms (e.g., trie-based matching).
        """
        ids = [ord(c) for c in text]

        changed = True
        while changed:
            changed = False
            for pair, new_id in sorted(self.merges.items(),
                                        key=lambda x: x[1]):
                new_ids = self._merge(ids, pair, new_id)
                if new_ids != ids:
                    ids = new_ids
                    changed = True
                    break  # restart with newly available pairs

        return ids

    def decode(self, ids):
        """Convert token IDs back to text.

        WHY simple concatenation: Each token is a string. Decoding
        just looks up each ID and concatenates. No special handling
        needed for the basic case — SentencePiece adds special
        whitespace handling.
        """
        return "".join(self.vocab.get(i, f"<{i}>") for i in ids)


# Train and demonstrate
print("\nTraining BPE on a small corpus:")
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog",
    "the mat and the log",
]

tokenizer = SimpleBPETokenizer()
vocab = tokenizer.train(corpus, num_merges=20)

# Test the tokenizer
test_text = "the cat sat on the log"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"\nEncoding test:")
print(f"  Text:    '{test_text}'")
print(f"  Encoded:  {encoded}")
print(f"  Decoded:  '{decoded}'")
print(f"  Tokens:   {len(encoded)} (vs {len(test_text)} characters)")

# Show vocabulary
print(f"\nLearned vocabulary (newest merges):")
for i, token in sorted(vocab.items(), key=lambda x: x[0])[-8:]:
    print(f"  ID {i:4d}: '{token}'")


# ──────────────────────────────────────────────────────────────────────────────
# WHY VOCABULARY SIZE IS A CRITICAL DESIGN CHOICE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: VOCABULARY SIZE — THE HIDDEN ARCHITECTURAL DECISION")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.1  WHY VOCAB SIZE MATTERS                                    │
└─────────────────────────────────────────────────────────────────┘

Vocab size affects three things simultaneously:

  1. MODEL SIZE: The embedding layer has vocab_size × d_model
     parameters. For GPT-2 (50K vocab, 768 d_model):
       50,257 × 768 = 38.6M params (31% of the model!)
     Double the vocab → double the embedding params.
     The LM head also has d_model × vocab_size params.
     Together, ~62% of GPT-2's params are vocab-related!

  2. COMPRESSION: Larger vocab = fewer tokens per text = more
     content fits in the context window.
     - 10K vocab: ~1.5 chars/token → 2K window ≈ 3K chars
     - 100K vocab: ~3.5 chars/token → 2K window ≈ 7K chars

  3. GENERALIZATION: Smaller vocab = more shared subwords = better
     generalization to rare words. Larger vocab = more specific
     tokens = harder to generalize.
     - "playing" as one token: great for common words
     - "play"+"ing" as two tokens: generalizes to "playing",
       "plays", "played", "playful" from a single root

REAL MODEL VOCABULARIES:
  GPT-2:      50,257  (50K BPE merges + special tokens)
  GPT-3/4:    ~100K   (larger, more efficient)
  LLaMA:      32,000  (sentencepiece unigram, deliberately small)
  BERT:       30,522  (WordPiece, ~30K)
  T5:         32,128  (sentencepiece unigram)

OpenAI went BIG (100K = more compression, higher throughput).
Meta went small (32K = less params, better multilingual).
There is no right answer — it's a design tradeoff.
""")


# ═══════════════════════════════════════════════════════════════════
# Special tokens
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.2  SPECIAL TOKENS — The model's control codes                │
└─────────────────────────────────────────────────────────────────┘

Every tokenizer reserves special IDs for control purposes:

  <PAD> or <|endoftext|>  — Separates documents/training examples.
      Tells the model "sequence ends here, don't attend across this."
      GPT uses <|endoftext|>  (token ID 50256).

  <BOS> / <EOS>  — Beginning/End of Sequence. BOS is often implicit,
      EOS tells the model when to STOP generating.

  <UNK>  — Unknown token. Fallback for characters not in the
      training data. Modern BPE tokenizers eliminate <UNK> by
      starting at the byte level — every byte is known.

  Chat markers — <|user|>, <|assistant|>, <|system|> separate
      roles in chat models. These are just tokens that appear in
      the sequence like any other word — the model learns to
      associate them with role shifts.

The tokenizer is the most fragile part of an LLM pipeline. A single
missing special token can cause the model to produce garbage, and
it's the first place to check when a model behaves unexpectedly.
""")


# ═══════════════════════════════════════════════════════════════════
# Real-world tokenizer quirks
# ═══════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 4.3  REAL-WORLD TOKENIZER QUIRKS                               │
└─────────────────────────────────────────────────────────────────┘

GPT tokenizers have known quirks that affect model behavior:

  1. TRAILING SPACES: "hello" and " hello" are DIFFERENT tokens!
     Words at the start of text get different tokens than words
     after a space. This is why GPT sometimes fails on "fill in
     the blank starting with a specific letter" tasks.

  2. CASING: "Apple" (company) and "apple" (fruit) are different
     tokens. The model must learn this from context.

  3. SOLIDGOLDMAGIKARP: A bizarre token in GPT-2/3's vocabulary
     that appears in the training data from a Reddit joke. The
     model has a dedicated token for this nonsense word — and it
     behaves unpredictably when prompted with it.

  4. NUMBER HANDLING: "123" might tokenize as "12" + "3" or "1" +
     "23". This inconsistent splitting is why LLMs struggle with
     arithmetic — the numbers aren't atomic.

  5. CODE: Python code tokenizes differently than English prose.
     Common patterns ("def ", "self.", "__init__") become single
     tokens, making code more "efficient" per token than English.

Lesson: The tokenizer IS part of the model. You cannot swap
tokenizers and keep the same weights — the embedding layer was
trained for a specific vocabulary mapping.
""")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
1. Character-level → simple but wasteful (long sequences, no semantics)
2. Word-level → efficient but broken (unbounded vocab, OOV problem)
3. Subword (BPE) → the industry solution: no unknowns, bounded vocab

Key design decisions:
  - Vocab size: bigger = more compression, more params
  - Algorithm: BPE (OpenAI) vs Unigram (Google/Meta)
  - Special tokens: control codes, chat markers, EOS

The tokenizer determines:
  - Your model's parameter count (~30% from embeddings!)
  - Your effective context window (tokens × chars/token = chars)
  - Your model's ability to handle new/rare words

After this module: you know how raw text becomes the integer IDs
that feed into the next stage — Embeddings (Module 2).
""")

if __name__ == "__main__":
    print("\nModule 1 complete! Next: 02_embeddings.py")
    print("Run with: uv run python course/transformer/02_embeddings.py")
