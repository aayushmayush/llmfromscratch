# Chapter 2: Working with Text Data

> **Status:** Complete | **Book sections:** 2.1–2.8 | **File:** `tokeniser_basics.ipynb`

## What this chapter covers

Preparing raw text for LLM training. The complete pipeline from a `.txt` file on disk to numerical tensors ready for the transformer.

## Pipeline (end-to-end)

```
the-verdict.txt (20,479 chars)
        │
        ▼
  Regex Tokenizer ──────► 4,690 tokens (words + punctuation)
        │                        │
        │                   Build Vocabulary
        │                   (1,130 unique tokens)
        │                        │
        │                   SimpleTokenizerV1
        │                   (fails on unknown words)
        │                        │
        │                   + <|unk|>, <|endoftext|>
        │                   SimpleTokenizerV2 (1,132 tokens)
        │
        ▼
  BPE Tokenizer (tiktoken) ───► 5,145 token IDs (50,257 vocab)
        │
        ▼
  GPTDatasetV1 (sliding window)
  max_length=256, stride=128
        │
        ▼
  1,286+ input-target pairs
        │
        ▼
  DataLoader (batch_size=8)
        │
        ▼
  Batches of shape [8, 256]
        │
        ▼
  Token Embedding (nn.Embedding)
  [8, 256] → [8, 256, 256]
        │
        ▼
  + Positional Embedding
  [256, 256] broadcast to [8, 256, 256]
        │
        ▼
  Input Embeddings [8, 256, 256]
  Ready for Chapter 3 (Attention)
```

## What's implemented

### 2.2 — Regex Tokenizer
Splits raw text into tokens (words + punctuation) using `re.split()`:

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
```

Handles commas, periods, question marks, exclamation marks, quotes, parentheses, colons, semicolons, and double dashes. Whitespace entries are stripped and filtered.

### 2.3 — Vocabulary & Token IDs
- Creates a sorted set of all unique tokens from the tokenized text
- Builds `vocab = {token_string: integer_id}` mapping (1,130 unique tokens)
- `SimpleTokenizerV1`: `encode()` converts text → token IDs, `decode()` converts back

### 2.4 — Special Context Tokens
- Adds `<|unk|>` (unknown word fallback) and `<|endoftext|>` (document boundary marker)
- `SimpleTokenizerV2`: replaces unknown words with `<|unk|>` instead of crashing
- Demonstrates concatenating unrelated texts with `<|endoftext|>` separators

### 2.5 — Byte Pair Encoding (tiktoken)
- Integrates OpenAI's `tiktoken` library — the real GPT-2 tokenizer
- Vocabulary size: **50,257 tokens** (vs. 1,132 for The Verdict's custom vocab)
- No `<|unk|>` token needed — BPE breaks unknown words into subwords or individual bytes
- `<|endoftext|>` = token ID 50256 (largest token in vocabulary)

### 2.6 — Data Sampling with Sliding Window
- `GPTDatasetV1`: PyTorch `Dataset` subclass that chunks tokenized text into training pairs
- **Input chunk:** `token_ids[i : i + max_length]`
- **Target chunk:** `token_ids[i + 1 : i + max_length + 1]` (inputs shifted right by 1)
- `stride` controls overlap: `stride=1` (maximum overlap), `stride=max_length` (no overlap)
- `create_dataloader_v1`: wraps dataset in a PyTorch `DataLoader` with configurable:
  - `batch_size`, `max_length`, `stride`
  - `shuffle` (randomize training order)
  - `drop_last` (discard incomplete final batch)
  - `num_workers` (parallel data loading processes)

### 2.7 — Token Embeddings
- `torch.nn.Embedding(vocab_size, output_dim)` — lookup table mapping token IDs → dense vectors
- Demo: 6-token vocab, 3-dim embeddings → shows how embedding is a row lookup
- Production: 50,257 vocab, 256-dim embeddings (GPT-2 small uses 768-dim)
- Shape transformation: `[batch, seq_len]` → `[batch, seq_len, emb_dim]`

### 2.8 — Positional Embeddings
- Self-attention is position-agnostic → positional info must be injected
- `torch.nn.Embedding(context_length, output_dim)` — learned per-position vectors
- Positional embeddings added to token embeddings: `input = token_emb + pos_emb`
- Shape: `[seq_len, emb_dim]` broadcast to `[batch, seq_len, emb_dim]`

## Key concepts

| Concept | Why it matters |
|---------|---------------|
| Tokenization | Neural networks can only process numbers, not raw text |
| Vocabulary | Fixed mapping from tokens to integers; size defines the model's "alphabet" |
| Special tokens | `<|unk|>` for unknown words, `<|endoftext|>` for document boundaries |
| BPE | Handles any text by breaking unknown words into subwords — no out-of-vocabulary problem |
| Sliding window | Converts one long text into many short training examples |
| Stride | Controls overlap between chunks; `stride=max_length` = no overlap |
| Shuffling | Prevents model from memorizing story order instead of learning language patterns |
| Embedding layer | Lookup table; each token ID retrieves a learned vector from a weight matrix |
| Positional encoding | Tells the model WHERE each token is (since attention is position-blind) |

## Shape transformations

Every operation in the pipeline changes tensor shape. Tracking shapes is how you debug:

```
Token IDs:           [batch, seq_len]        e.g., [8, 4]
Token Embeddings:    [batch, seq_len, emb]   e.g., [8, 4, 256]
Position Embeddings: [seq_len, emb]          e.g., [4, 256]
Input Embeddings:    [batch, seq_len, emb]   e.g., [8, 4, 256]
```

## Running the code

```bash
cd ch02
# Open tokeniser_basics.ipynb in VS Code
# Select venv interpreter: Ctrl+Shift+P → "Python: Select Interpreter" → ./venv/bin/python
# Run cells sequentially with Shift+Enter
```

Requires: `torch`, `tiktoken`, `numpy`, `matplotlib`, `requests`

## Next: Chapter 3 — Attention Mechanisms

The input embeddings from this chapter now enter the self-attention mechanism — the most important architectural component of the transformer. Four variants will be implemented:

1. Simplified self-attention (no trainable weights)
2. Scaled dot-product attention (with Q/K/V weight matrices)
3. Causal (masked) attention (prevents looking at future tokens)
4. Multi-head attention (parallel attention heads, efficient implementation)
