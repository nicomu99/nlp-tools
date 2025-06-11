import torch
import torch.nn as nn

class LSTMForARLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 rnn_size: int,
                 hidden_size: int,
                 padding_index: int,
                 num_layers: int
                 ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings  = vocab_size,
            embedding_dim   = embedding_dim,
            padding_idx     = padding_index
        )

        self.lstm = nn.LSTM(
            input_size      = embedding_dim,
            hidden_size     = rnn_size,
            num_layers      = num_layers,
            bidirectional   = False,
            batch_first     = True
        )

        self.linear = nn.Linear(
            in_features = hidden_size * 2,
            out_features = vocab_size
        )

    def forward(self, batch: dict):
        input_ids = batch['input_ids']

        hidden = self.embedding(input_ids)
        hidden, _ = self.bilstm(hidden)

        return self.linear(hidden)