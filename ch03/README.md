# Chapter 3: Coding Attention Mechanisms

> **Status:** In Progress | **Book sections:** 3.1–3.3 complete | **File:** `attention_basics.ipynb`

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

**4 attention variants in this chapter:**
1. ✅ Simplified self-attention — dot products + softmax + weighted sum (Sections 3.3.1–3.3.2)
2. ⬜ Scaled dot-product attention — with trainable Q/K/V weight matrices (Section 3.4)
3. ⬜ Causal (masked) attention — prevents looking at future tokens (Section 3.5)
4. ⬜ Multi-head attention — parallel heads, efficient weight-split implementation (Section 3.6)

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

## Next: Section 3.4 — Self-attention with trainable weights

Adding Q (query), K (key), V (value) weight matrices that the model learns. The formula becomes:

```
Context = softmax(Q @ Kᵀ / √d_k) @ V

Where:  Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
```

The `/√d_k` scaling prevents vanishing gradients in high dimensions (hence "scaled" dot-product attention).
