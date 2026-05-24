"""
One-Layer Transformer for Modular Arithmetic (Grokking)

Architecture matching Nanda et al. "Progress measures for grokking via mechanistic interpretability":
- Token embedding W_E + Positional embedding W_pos
- Single attention layer with 4 heads (no LayerNorm)
- Residual connection
- MLP: Linear(d_model, d_mlp) -> ReLU -> Linear(d_mlp, d_model)
- Residual connection
- Unembedding W_U
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum


class Attention(nn.Module):
    """Multi-head attention without LayerNorm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Separate weight matrices per head for interpretability
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / np.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / np.sqrt(d_model))

        self.b_Q = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_K = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_V = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_O = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        # Compute Q, K, V for all heads
        # x: (batch, seq, d_model)
        # W_Q: (n_heads, d_model, d_head)
        q = einsum(x, self.W_Q, "batch seq d_model, heads d_model d_head -> batch heads seq d_head") + self.b_Q[None, :, None, :]
        k = einsum(x, self.W_K, "batch seq d_model, heads d_model d_head -> batch heads seq d_head") + self.b_K[None, :, None, :]
        v = einsum(x, self.W_V, "batch seq d_model, heads d_model d_head -> batch heads seq d_head") + self.b_V[None, :, None, :]

        # Attention scores
        attn_scores = einsum(q, k, "batch heads seq_q d_head, batch heads seq_k d_head -> batch heads seq_q seq_k")
        attn_scores = attn_scores / np.sqrt(self.d_head)

        # Causal mask - only look at current and previous positions
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

        attn_pattern = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        z = einsum(attn_pattern, v, "batch heads seq_q seq_k, batch heads seq_k d_head -> batch heads seq_q d_head")

        # Project back through W_O and sum heads
        out = einsum(z, self.W_O, "batch heads seq d_head, heads d_head d_model -> batch seq d_model") + self.b_O

        return out, attn_pattern


class MLP(nn.Module):
    """MLP with ReLU activation."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model), pre_act, post_act
        """
        pre_act = x @ self.W_in + self.b_in  # (batch, seq, d_mlp)
        post_act = F.relu(pre_act)  # (batch, seq, d_mlp)
        out = post_act @ self.W_out + self.b_out  # (batch, seq, d_model)
        return out, pre_act, post_act


class OneLayerTransformer(nn.Module):
    """
    One-layer transformer for modular arithmetic.

    Input: sequence of token indices [a, b, =]
    Output: logits over p classes at the = position
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512, seq_len: int = 3):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.seq_len = seq_len

        # Vocabulary: 0..p-1 for numbers, p for '=' sign
        self.n_vocab = p + 1

        # Embeddings
        self.W_E = nn.Parameter(torch.randn(self.n_vocab, d_model) / np.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(seq_len, d_model) / np.sqrt(d_model))

        # Transformer layers
        self.attention = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)

        # Unembedding (only need p outputs, not p+1)
        self.W_U = nn.Parameter(torch.randn(d_model, p) / np.sqrt(d_model))
        self.b_U = nn.Parameter(torch.zeros(p))

    def forward(self, tokens, return_cache=False):
        """
        tokens: (batch, seq_len) — integer token indices
        Returns: logits (batch, p) at the last position
        """
        cache = {}

        # Embed tokens
        token_embed = self.W_E[tokens]  # (batch, seq, d_model)
        pos_embed = self.W_pos[:tokens.shape[1]]  # (seq, d_model)
        residual = token_embed + pos_embed  # (batch, seq, d_model)
        cache['resid_pre'] = residual

        # Attention + residual
        attn_out, attn_pattern = self.attention(residual)
        residual = residual + attn_out
        cache['attn_out'] = attn_out
        cache['attn_pattern'] = attn_pattern
        cache['resid_mid'] = residual

        # MLP + residual
        mlp_out, pre_act, post_act = self.mlp(residual)
        residual = residual + mlp_out
        cache['mlp_out'] = mlp_out
        cache['mlp_pre'] = pre_act
        cache['mlp_post'] = post_act
        cache['resid_post'] = residual

        # Unembedding (only at last position)
        logits = residual[:, -1, :] @ self.W_U + self.b_U  # (batch, p)
        cache['logits'] = logits

        if return_cache:
            return logits, cache
        return logits

    def sum_squared_weights(self):
        """Total sum of squared weights (Frobenius norm squared) for all parameters."""
        return sum((p ** 2).sum() for p in self.parameters())


class MultiTaskTransformer(nn.Module):
    """
    One-layer transformer for multi-task modular arithmetic (co-grokking).

    Input: sequence [a, op, b, =] where op distinguishes the operation.
    Vocabulary: 0..p-1 for numbers, p for '+', p+1 for '×', p+2 for '='
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.seq_len = 4  # [a, op, b, =]

        # Vocabulary: 0..p-1 numbers, p for '+', p+1 for '×', p+2 for '='
        self.n_vocab = p + 3

        # Embeddings
        self.W_E = nn.Parameter(torch.randn(self.n_vocab, d_model) / np.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(self.seq_len, d_model) / np.sqrt(d_model))

        # Transformer layers
        self.attention = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)

        # Unembedding
        self.W_U = nn.Parameter(torch.randn(d_model, p) / np.sqrt(d_model))
        self.b_U = nn.Parameter(torch.zeros(p))

    def forward(self, tokens, return_cache=False):
        """
        tokens: (batch, 4) — [a, op, b, =]
        Returns: logits (batch, p)
        """
        cache = {}

        token_embed = self.W_E[tokens]
        pos_embed = self.W_pos[:tokens.shape[1]]
        residual = token_embed + pos_embed
        cache['resid_pre'] = residual

        attn_out, attn_pattern = self.attention(residual)
        residual = residual + attn_out
        cache['attn_out'] = attn_out
        cache['attn_pattern'] = attn_pattern
        cache['resid_mid'] = residual

        mlp_out, pre_act, post_act = self.mlp(residual)
        residual = residual + mlp_out
        cache['mlp_out'] = mlp_out
        cache['mlp_pre'] = pre_act
        cache['mlp_post'] = post_act
        cache['resid_post'] = residual

        logits = residual[:, -1, :] @ self.W_U + self.b_U
        cache['logits'] = logits

        if return_cache:
            return logits, cache
        return logits

    def sum_squared_weights(self):
        return sum((p ** 2).sum() for p in self.parameters())
