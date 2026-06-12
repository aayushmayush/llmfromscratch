# Chapter 4: Implementing a GPT Model from Scratch

> **Status:** In Progress | **Book sections:** 4.1–4.4 complete | **File:** `transfer_block.ipynb`

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

### 4.2 — LayerNorm ✅

Normalizes each token's 768-dim vector to mean=0, variance=1, then applies learnable scale and shift.

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # learn to stretch
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # learn to shift

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**Why normalize?**
- Deep networks suffer from internal covariate shift — activations drift in scale
- LayerNorm keeps values in a stable range, preventing gradient explosion/vanishing
- Learnable `scale` and `shift` let the model undo normalization if needed

### 4.3 — GELU activation ✅

Implemented from scratch using the tanh approximation, then compared against `torch.nn.GELU`:

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

**GELU vs ReLU:**
- ReLU: `max(0, x)` — sharp cutoff at zero. "You're either in or you're out."
- GELU: `x · Φ(x)` — smooth S-curve. "You're somewhat in, somewhat out."
- GELU's smoothness = cleaner gradients in deep networks like GPT-2 (12 layers)

### 4.4 — FeedForward network ✅

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),                                           # activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),   # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)                                 # [B, T, 768]
```

**The 4× expansion pattern:**
- Expands to 4× embedding dim (3072), applies GELU, contracts back to 768
- The expansion lets the network learn complex nonlinear features in a higher-dimensional space
- Same shape in, same shape out → stackable in TransformerBlock

## Up next in Chapter 4

- Residual connections (shortcut paths around attention and feedforward)
- Real TransformerBlock combining attention + feedforward + residuals + LayerNorm
- Complete GPTModel with all real components
