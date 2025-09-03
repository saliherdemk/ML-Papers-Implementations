import torch
import torch.nn as nn
from attention import CrossAttentionBlock, SelfAttentionBlock
from embedding_block import EmbeddingBlock


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, max_seq_len, pad_id, num_heads, num_layers
    ):
        super().__init__()
        self.emb_block = EmbeddingBlock(vocab_size, embed_dim, max_seq_len, pad_id)
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, src):
        x = self.emb_block(src)
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, max_seq_len, pad_id, num_heads, num_layers
    ):
        super().__init__()
        self.emb_block = EmbeddingBlock(vocab_size, embed_dim, max_seq_len, pad_id)
        self.self_layers = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def generate_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, tgt, enc_out):
        x = self.emb_block(tgt)
        mask = self.generate_mask(x.size(1), x.device)

        for self_layer, cross_layer in zip(self.self_layers, self.cross_layers):
            x = self_layer(x, mask)
            x = cross_layer(x, enc_out, enc_out)

        return self.out_proj(x)
