# Chapter 4: Implementing a GPT Model from Scratch

> **Status:** In Progress | **Book sections:** 4.1 complete | **File:** `transfer_block.ipynb`

## What this chapter covers

Taking all the components from Chapters 2 and 3 (embeddings, attention, dropout) and assembling them into a complete GPT-2 model. Starting with a skeleton, then filling in each component.

## Progress

### 4.1 — GPT model skeleton ✅

**GPT_CONFIG_124M — the configuration for GPT-2 small:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # BPE tokenizer vocabulary
    "context_length": 1024,     # Max tokens the model can see
    "emb_dim": 768,             # Embedding dimension (width of every token)
    "n_heads": 12,              # Number of attention heads per block
    "n_layers": 12,             # Number of transformer blocks
    "drop_rate": 0.1,           # Dropout probability
    "qkv_bias": False           # Whether Q/K/V Linear layers include bias
}
```

**The model architecture:**
```
DummyGPTModel
├── tok_emb        nn.Embedding(50257, 768)     ← token → vector
├── pos_emb        nn.Embedding(1024, 768)      ← position → vector
├── drop_emb       nn.Dropout(0.1)              ← regularization
├── trf_blocks     12× DummyTransformerBlock    ← placeholder stack
├── final_norm     DummyLayerNorm(768)          ← placeholder
└── out_head       nn.Linear(768, 50257)        ← vector → vocab scores
```

**The forward pass (end to end):**
```python
def forward(self, in_idx):
    tok_embeds = self.tok_emb(in_idx)                          # [B, T] → [B, T, 768]
    pos_embeds = self.pos_emb(torch.arange(seq_len))           # [T, 768]
    x = tok_embeds + pos_embeds                                # [B, T, 768]
    x = self.drop_emb(x)                                       # [B, T, 768]
    x = self.trf_blocks(x)                                     # [B, T, 768] (dummy)
    x = self.final_norm(x)                                     # [B, T, 768] (dummy)
    logits = self.out_head(x)                                  # [B, T, 50257]
    return logits
```

**Why start with dummies?**
- The skeleton shows the full architecture without getting lost in details
- Dummy blocks just return `x` unchanged — they're placeholders
- Each dummy gets replaced with a real implementation later in the chapter
- Same pattern as Ch3: `SelfAttention_v1` → `SelfAttention_v2` → `CausalAttention`

## Key concepts

| Concept | Meaning |
|---------|---------|
| `cfg` dictionary | Single source of truth for all hyperparameters |
| Transformer block | Attention + FeedForward + Residuals + LayerNorm |
| `out_head` | Projects 768-dim → 50,257-dim (one score per vocab word) |
| Logits | Raw scores before softmax; higher = model thinks more likely |
| Shape invariance | Every block outputs `[B, T, 768]` — same shape as input |
| Dummy pattern | Build the skeleton, then fill in components one at a time |

## Shape transformations

```
Token IDs:     [B, T]                e.g., [2, 6]
Token Emb:     [B, T, 768]           lookup table
Position Emb:  [T, 768]              broadcast to batch
Sum:           [B, T, 768]           element-wise addition
After dropout: [B, T, 768]           same shape (just some values zeroed)
After blocks:  [B, T, 768]           same shape in, same shape out
Final norm:    [B, T, 768]           same shape
Output logits: [B, T, 50257]         projects to vocabulary
```

## Up next in Chapter 4

- LayerNorm implementation (Section 4.2)
- GELU activation function
- FeedForward network (expands 768 → 3072 → 768)
- Real TransformerBlock with attention + feedforward + residuals
- Complete GPTModel assembling everything
