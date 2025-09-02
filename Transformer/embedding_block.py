import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_len, n=10000):
        super().__init__()
        self.n = n
        pe = torch.zeros(max_len, embed_dim)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(self.n) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        pe = pe.unsqueeze(0)
        pe = pe.expand(x.size(0), -1, -1)
        return x + pe


class EmbeddingBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, pad_id, dropout=0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.pad_id = pad_id

    def forward(self, token_ids):
        tok_emb = self.token_emb(token_ids) * math.sqrt(self.embed_dim)
        padding_mask = (token_ids == self.pad_id).unsqueeze(-1)
        tok_emb = tok_emb.masked_fill(padding_mask, 0.0)

        x = self.pos_emb(tok_emb)
        x = x.masked_fill(padding_mask, 0.0)

        return self.dropout(x)
