import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .transformer_block import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, dropout, forward_dim, attention_heads=8):
        super().__init__()

        self.mmha = MultiHeadAttention(embedding_dim, attention_heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim, dropout, forward_dim, attention_heads)

    def forward(self, x, k, v, source_mask, target_mask):
        x = x + self.mmha(x, x, x, target_mask)
        x = self.layer_norm1(x)
        return self.transformer_block(x, k, v, source_mask)