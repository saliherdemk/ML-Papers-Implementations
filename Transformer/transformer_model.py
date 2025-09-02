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
