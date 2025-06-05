import torch
import torch.nn as nn

from model import Embedding
from model import Dropout
from model import Linear
from model import LSTM

class CBiLSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 rnn_size: int,
                 hidden_size: int,
                 dropout: float,
                 padding_index: int,
                 ):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings  = vocab_size,
            embedding_dim   = embedding_dim,
            padding_idx     = padding_index
        )

        self.dropout_layer = Dropout(p=dropout)

        self.bilstm = LSTM(
            input_size      = embedding_dim,
            hidden_size     = rnn_size,
            num_layers      = 1,
            bidirectional   = True
        )

        self.linear1 = Linear(
            in_features     = 2 * rnn_size,
            out_features    = hidden_size,
            bias            = True
        )

        self.linear2 = Linear(
            in_features = hidden_size,
            out_features = 1
        )

    def forward(self, seqs, lengths):
        hidden = self.embedding(seqs)
        hidden = self.dropout_layer(hidden)

        # For computational efficiency pack the padded sequences
        hidden, (_, _) = self.bilstm(hidden)

        # BiLSTM returns hidden representation for each index -> since we are only doing classification
        # we can pool into a single dimension
        hidden = torch.mean(hidden, dim=1)

        # hidden = self.dropout_layer(hidden)
        hidden = torch.relu(self.linear1(hidden))

        # Single output, since we do binary classification
        return torch.sigmoid(self.linear2(hidden)).squeeze(1)