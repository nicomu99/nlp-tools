import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FFNetwork

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, dropout, forward_dim, attention_heads=8):
        super().__init__()

        # Multihead Attention
        self.mha = MultiHeadAttention(embedding_dim, attention_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = FFNetwork(embedding_dim, forward_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, q, k, v, mask=None):
        # Input has [batch, seq_len, em_dim]
        # Apply mha
        x = q + self.mha(q, k, v, mask)
        x = self.dropout(x)
        x = self.layer_norm1(x)

        x += self.ffn(x)
        x = self.dropout(x)

        x = self.layer_norm2(x)
        return x
