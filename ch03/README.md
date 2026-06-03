# Chapter 3: Coding Attention Mechanisms

> **Status:** Complete | **Book sections:** 3.1–3.6 complete | **File:** `attention_basics.ipynb`

## What this chapter covers

Implementing the attention mechanism — the core innovation behind transformers and LLMs. Building four progressively more sophisticated variants from scratch.

## Progress

### 3.1 — The problem with modeling long sequences ✅
- Pre-transformer RNNs compress entire input into a single hidden state → lose context in long sequences
- Attention solves this: decoder can selectively access ALL input positions at each step

### 3.2 — Capturing data dependencies with attention ✅
- Bahdanau attention (2014): RNNs with selective input access
- Self-attention (2017): RNNs not needed — each position attends to all positions in the SAME sequence

### 3.3 — Simple self-attention (no trainable weights) ✅

### 3.4 — Self-attention with trainable weights ✅

**4 attention variants in this chapter:**
1. ✅ Simplified self-attention — dot products + softmax + weighted sum (Sections 3.3.1–3.3.2)
2. ✅ Scaled dot-product attention — with trainable Q/K/V weight matrices (Sections 3.4.1–3.4.2)
3. ✅ Causal (masked) attention — prevents looking at future tokens, dropout for regularization (Section 3.5)
4. ✅ Multi-head attention — wrapper (concat) and efficient weight-split with out_proj (Section 3.6)

#### 3.3.1 — Single context vector (z²)
Using the sentence "Your journey starts with one step" with 3D embeddings:

```python
inputs = [
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # journey  ← query
    [0.57, 0.85, 0.64],  # starts
    [0.22, 0.58, 0.33],  # with
    [0.77, 0.25, 0.10],  # one
    [0.05, 0.80, 0.55],  # step
]
```

**Step 1 — Attention scores (dot products):**
```python
query = inputs[1]  # "journey"
# Compute query · each_input
attn_scores_2 = [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]
```

**Step 2 — Normalize (softmax):**
```python
# Three methods shown:
# 1. Simple division by sum
# 2. Naive softmax implementation
# 3. torch.softmax (use this in practice)
attn_weights_2 = [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]
# Sum = 1.0 ✓
```

**Step 3 — Context vector (weighted sum):**
```python
z² = sum(α_i × x_i)  # Blends all inputs by attention weights
# = [0.4419, 0.6515, 0.5683]
```

#### 3.3.2 — All context vectors (matrix form)
Generalizes single-query computation to all inputs simultaneously:

```python
# Step 1: All attention scores via matrix multiplication (replaces nested for-loops)
attn_scores = inputs @ inputs.T    # [6, 6] matrix

# Step 2: Normalize by row (dim=-1)
attn_weights = torch.softmax(attn_scores, dim=-1)  # [6, 6]

# Step 3: All context vectors in one operation
all_context_vecs = attn_weights @ inputs  # [6, 3]
```

## Key concepts

| Concept | Meaning |
|---------|---------|
| Dot product as similarity | Higher dot product = vectors more aligned = words more related |
| Softmax normalization | Converts raw scores to probabilities (sum to 1, always positive) |
| Context vector | Enriched embedding = weighted blend of all input vectors |
| Matrix multiplication | Replaces for-loops for efficiency; `inputs @ inputs.T` = all pairwise dot products |
| dim=-1 | Normalize along last dimension (row-wise for 2D tensors) |

## The attention formula (so far)

```
Context vectors = softmax(X @ Xᵀ) @ X

Where:
  X = input embeddings [n_tokens, d_in]
  X @ Xᵀ = attention scores (dot product of every token pair)
  softmax() = normalize to sum=1 per row
  softmax() @ X = weighted sum of inputs → context vectors
```

## Shape transformations

```
Inputs:                [6, 3]    6 tokens, 3D embeddings
Attention scores:     [6, 6]    6×6 pairwise similarity matrix
Attention weights:    [6, 6]    normalized scores (rows sum to 1)
Context vectors:      [6, 3]    enriched embeddings (same shape as inputs)
```

### 3.5 — Causal (masked) attention ✅
Prevents the model from looking at future tokens — essential for autoregressive text generation (predicting next word).

**The mask:**
```python
context_length = attn_scores.shape[0]
mask = torch.tril(torch.ones(context_length, context_length))
# [[1, 0, 0, 0, 0, 0],   ← "Your" can only see itself
#  [1, 1, 0, 0, 0, 0],   ← "journey" can see "Your" + itself
#  [1, 1, 1, 0, 0, 0],   ← "starts" can see first 3
#  [1, 1, 1, 1, 0, 0],   ...
#  [1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1]]   ← "step" can see all
```

**Applying the mask:**
```python
# Set future positions to -inf before softmax (so they become 0 after exp)
attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
# Or: zero out after softmax + renormalize
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
masked = attn_weights * mask
masked_norm = masked / masked.sum(dim=-1, keepdim=True)
```

**Dropout — preventing overfitting:**
```python
self.dropout = nn.Dropout(p=0.5)           # in __init__
attn_weights = self.dropout(attn_weights)  # in forward, after softmax
```
- Randomly zeroes attention weights during training → forces robust, redundant patterns
- Each forward pass drops different neurons (like training multiple sub-networks)
- Active only during `.train()`; automatically disabled during `.eval()` (inference)
- Typical dropout rate: 0.5 for attention weights

**Why causal matters:**
Without the mask, when generating text, the model would peek at the answer. With the mask enforced, each token can only use previous context — exactly like real text generation where you don't know what word comes next.

### 3.6 — Multi-head attention ✅

Two implementations showing the evolution:

**MultiHeadAttentionWrapper (educational):**
```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```
- Creates N independent `CausalAttention` instances
- Each head has its own full Q/K/V weight matrices
- Concatenates outputs: 2 heads × d_out=2 → output shape [B, T, 4]

**MultiHeadAttention (efficient, production-ready):**
- Single set of Q/K/V projections with full `d_out` dimension
- Splits the output into `num_heads` chunks via `.view()` — each head operates on a smaller subspace (`head_dim = d_out // num_heads`)
- Transposes to `[B, num_heads, T, head_dim]` for batched attention computation
- Merges heads back via `.view(B, T, d_out)` after attention
- Adds `out_proj` (Linear layer) to combine head outputs

**Shape transformations (efficient version):**
```
Input:     [B, T, d_in]           e.g., [2, 6, 3]
Q/K/V:     [B, T, d_out]          e.g., [2, 6, 4]   ← single projection
Split:     [B, T, num_heads, hd]  e.g., [2, 6, 2, 2] ← d_out=4, 2 heads, head_dim=2
Transpose: [B, num_heads, T, hd]  e.g., [2, 2, 6, 2]
Attn:      [B, num_heads, T, hd]  e.g., [2, 2, 6, 2] ← same shape
Merge:     [B, T, d_out]          e.g., [2, 6, 4]     ← .view() back
out_proj:  [B, T, d_out]          e.g., [2, 6, 4]     ← final projection
```

**Why the efficient version matters:**
- Wrapper creates N separate Linear layers → N× more parameters
- Efficient version uses ONE set of projections split across heads → fewer params, faster
- `out_proj` learns how to best combine the different head perspectives
- This is the version used in GPT-2 and all production transformers

## Next: Chapter 4 — Implementing a GPT model from scratch

Building the full GPT architecture: LayerNorm, GELU activation, FeedForward networks, TransformerBlocks, and the complete GPTModel class that ties everything together.
