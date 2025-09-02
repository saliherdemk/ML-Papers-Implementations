import torch
import torch.nn as nn
import torch.nn.functional as F
from feedforward import FeedForward


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_dim, head_size)
        self.W_k = nn.Linear(embed_dim, head_size)
        self.W_v = nn.Linear(embed_dim, head_size)

    def forward(self, x, mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / (self.head_size**0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        return attn @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(embed_dim, head_size) for _ in range(num_heads)]
        )

    def forward(self, x, mask=None):
        return torch.cat([h(x, mask) for h in self.heads], dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_dim, head_size)
        self.W_k = nn.Linear(embed_dim, head_size)
        self.W_v = nn.Linear(embed_dim, head_size)

    def forward(self, x, K, V):
        Q = self.W_q(x)
        K = self.W_k(K)
        V = self.W_v(V)
        scores = Q @ K.transpose(-2, -1) / (self.head_size**0.5)
        attn = F.softmax(scores, dim=-1)
        return attn @ V


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [CrossAttention(embed_dim, head_size) for _ in range(num_heads)]
        )

    def forward(self, x, K, V):
        return torch.cat([h(x, K, V) for h in self.heads], dim=-1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mattention = MultiHeadSelfAttention(
            embed_dim, num_heads, embed_dim // num_heads
        )
        self.ff = FeedForward(embed_dim, 64)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.mattention(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mattention = MultiHeadCrossAttention(
            embed_dim, num_heads, embed_dim // num_heads
        )
        self.ff = FeedForward(embed_dim, 64)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, K, V):
        x = x + self.mattention(self.ln1(x), K, V)
        x = x + self.ff(self.ln2(x))
        return x
