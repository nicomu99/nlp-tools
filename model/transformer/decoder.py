import torch

import torch.nn as nn
from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, forward_dim, dropout, max_len):
        super().__init__()

        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(max_len, embedding_dim)
        self.decoders = nn.ModuleList(
            [DecoderBlock(embedding_dim, dropout, forward_dim, num_heads) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, encoder_out, source_mask, target_mask):
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        x = self.tok_embeddings(x)
        pos = torch.arange(seq_len, device=device)
        pos = self.pos_embeddings(pos)

        x += pos
        for decoder in self.decoders:
            x = decoder(x, encoder_out, encoder_out, source_mask, target_mask) # TODO: add mask
        return x