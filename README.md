# LLM From Scratch

Building a GPT-style large language model from first principles, following [Sebastian Raschka's book](https://www.manning.com/books/build-a-large-language-model-from-scratch). No HuggingFace, no transformers library — just PyTorch and the fundamentals.

## Goal

Understand how LLMs actually work by implementing every component: tokenization → embeddings → attention → transformer blocks → pretraining → finetuning. Not just using APIs — building the thing.

## Setup

```bash
git clone <this-repo>
cd llmfromscratch

python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install torch tiktoken numpy matplotlib requests ipykernel
```

Open `.ipynb` files in VS Code. Select the venv interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → `./venv/bin/python`.

## Project Structure

```
llmfromscratch/
├── venv/                         # Virtual environment (gitignored)
├── the-verdict.txt               # Training data — public domain short story
├── README.md
├── BOOK_SYNTHESIS.md             # Complete chapter-by-chapter synthesis
├── llm_book_full.txt             # Full text extraction from the book PDF
│
├── ch02/                         # Chapter 2: Working with Text Data
│   └── tokeniser_basics.ipynb    # Tokenization, vocabulary, BPE, data sampling
│
├── ch03/                         # Chapter 3: Attention Mechanisms
│   └── attention_basics.ipynb    # Simplified self-attention (no weights yet)
├── ch04/                         # Chapter 4: GPT Architecture
├── ch05/                         # Chapter 5: Pretraining
├── ch06/                         # Chapter 6: Classification Finetuning
└── ch07/                         # Chapter 7: Instruction Finetuning
```

## Progress

### Chapter 2: Working with Text Data ✅

**What's in `ch02/tokeniser_basics.ipynb`:**
- Regex-based tokenizer — splits text into words and punctuation
- Vocabulary mapping unique tokens ↔ integer IDs
- `SimpleTokenizerV1` — basic encode/decode (fails on unknown words)
- `SimpleTokenizerV2` — handles unknown words with `<|unk|>`, text boundaries with `<|endoftext|>`
- `tiktoken` BPE tokenizer integration — the real GPT-2 tokenizer (vocab size: 50,257)
- `GPTDatasetV1` — PyTorch Dataset that creates input-target pairs via sliding window
- `create_dataloader_v1` — configurable DataLoader with batch_size, max_length, stride
- Token embedding layer (`nn.Embedding`) — converts token IDs → dense vectors (256-dim)
- Absolute positional embeddings — added to token embeddings for position awareness

**Key concepts:**
- Why raw text must become numbers (neural networks do math, not language)
- Byte Pair Encoding: breaks unknown words into subwords/bytes — no `<|unk|>` needed in production
- Sliding window: inputs = `tokens[i : i+max_length]`, targets = `tokens[i+1 : i+1+max_length]`
- Stride controls overlap between training chunks
- Embeddings as a lookup table: each token ID retrieves a row from a trainable weight matrix
- Positional embeddings: inject position information since self-attention is position-agnostic

**Pipeline so far:**
```
Raw text → Tokenizer → Token IDs → Embeddings + Positional Encoding → Input tensors
                                                                          ↓
                                                          (ready for attention — Chapter 3 next)
```

### Chapter 3: Attention Mechanisms 🚧

**What's in `ch03/attention_basics.ipynb` (Sections 3.1–3.4):**
- Self-attention mechanism without trainable weights — the conceptual foundation
- Dot product as similarity measure between word embeddings
- Softmax normalization: converting raw similarity scores → attention weights (sum to 1)
- Context vector computation: weighted sum of all inputs, enriched by attention weights
- Efficient matrix implementation: `inputs @ inputs.T` replaces nested for-loops
- All context vectors computed simultaneously: `softmax(X @ Xᵀ) @ X`
- Trainable weight matrices Wq, Wk, Wv — project inputs into query/key/value spaces
- Scaled attention: dividing by `√d_k` before softmax to prevent vanishing gradients
- `SelfAttention_v1` — compact class using `nn.Parameter` (manual weight init)
- `SelfAttention_v2` — same class using `nn.Linear` (preferred: built-in init + bias control)

**Key concepts:**
- Dot product measures vector alignment — higher score = more similar words
- Attention weights are a probability distribution over the input sequence
- Context vector = enriched embedding containing information from ALL words, not just one
- Matrix multiplication makes attention O(n²) but vectorized and GPU-friendly
- Q/K/V projection: model learns different similarity measures for different purposes
- `nn.Linear` vs `nn.Parameter`: Linear includes built-in weight initialization (Kaiming uniform)

**Pipeline update:**
```
Embeddings → Q/K/V projection → Scaled dot-product scores → Softmax → Attention weights → Context vectors
                                                                                        ↓
                                                                (causal attention next — Section 3.5)
```

### Coming next
- **Chapter 3 (remaining)**: Scaled dot-product attention, causal attention, multi-head attention (Sections 3.4–3.6)
- **Chapter 4**: Full GPT architecture — LayerNorm, GELU, FeedForward, TransformerBlocks, GPTModel
- **Chapter 5**: Pretraining — loss functions, training loop, text generation, loading OpenAI weights
- **Chapter 6**: Classification finetuning — spam detection
- **Chapter 7**: Instruction finetuning — making models follow instructions

## References

- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
- [Official book repository](https://github.com/rasbt/LLMs-from-scratch)
- [BOOK_SYNTHESIS.md](BOOK_SYNTHESIS.md) — my complete chapter-by-chapter synthesis
- [llm_book_full.txt](llm_book_full.txt) — full text extraction from the book PDF

## Notes

Building in public. Learning this with no prior ML background — if something's wrong or confusing, open an issue.
