"""
Minimal decoder-only transformer for Study 2 (from-scratch grokking).

Architecture matches the setup in Nanda et al. (2023): 2 layers, 4 heads,
manual (non-fused) attention so results are bit-reproducible across
hardware. This file is unchanged from the version tested end-to-end on
CPU before the full 15-run grid was executed on T4 (see repo README).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GrokkTransformer(nn.Module):
    """
    Sequence layout is [a, op, b, eq] -> predict c at the final position.
    Vocab: 0..p-1 are numbers, p is the 'op' token, p+1 is '='.
    """

    def __init__(self, p, d_model=128, n_heads=4, n_layers=2, seq_len=4):
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p + 2, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, x):
        h = self.embed(x) + self.pos[: x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.unembed(self.ln_f(h))[:, -1, :]

    def number_embeddings(self):
        """
        (p, d_model) numpy array: the embedding rows for the number tokens,
        used by the Fourier / Gini analysis.

        .copy() is REQUIRED here, not optional. On CPU, .numpy() shares
        storage with the underlying tensor, so without .copy() every
        snapshot taken during training becomes a live view of whatever the
        weights are at the moment the array is later read, not at the
        moment it was "saved". This bug produced identical Gini values at
        every logged checkpoint the first time this file was written, and
        would have shipped silently: on GPU, .cpu() happens to copy, so
        the bug would not have shown up in Colab and would have surfaced
        only if this code were reused somewhere that logs embeddings on
        CPU. Do not remove the .copy() call.
        """
        return self.embed.weight[: self.p].detach().cpu().numpy().copy()
