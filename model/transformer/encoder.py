import math

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock


def get_pos_embeddings(max_len, embedding_dimensions):
    encoding = torch.zeros((max_len, embedding_dimensions))
    for i in range(max_len):
        for j in range(embedding_dimensions):
            angle = i / 10000 ** (2 * (j // 2) / embedding_dimensions)
            if j % 2 == 0:
                encoding[i, j] = torch.sin(angle)
            else:
                encoding[i, j] = torch.cos(angle)
    return encoding


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers,
                 num_heads, forward_dim, dropout, max_len):
        super().__init__()
        self.max_len: int = max_len
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_pos_embeddings(max_len, embedding_dim), freeze=True
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_block = nn.ModuleList(
            [TransformerBlock(embedding_dim, dropout, forward_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        x = self.embedding(x) * math.sqrt(self.embedding_dim)

        matrix = torch.arange(seq_len, device=device)
        x += self.pos_embedding(matrix).unsqueeze(0).expand(batch_size, seq_len, -1)
        x = self.dropout(x)
        for encoder in self.encoder_block:
            x = encoder(x, x, x, mask)
        return x