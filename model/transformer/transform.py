import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            source_vocab_size,
            target_vocab_size,
            embedding_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len
    ):
        super().__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        self.decoder = Decoder(target_vocab_size, embedding_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source, target):
        encoder_out = self.encoder(source, None)
        decoder_out = self.decoder(target, encoder_out, None, None)
        return self.softmax(decoder_out)