import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMForClassification(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 rnn_size: int,
                 hidden_size: int,
                 dropout: float,
                 padding_index: int,
                 ):
        super().__init__()

        # Acts as a lookup table mapping token indices to word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index
        )

        # Randomly zeros out some of the input tensors using a bernoulli distribution for regularization
        # The parameter p is the probability of an input tensor being zeroed
        self.dropout_layer = nn.Dropout(p=dropout)

        # A bidirectional LSTM cell, the outputs of the embedding dimension (with dropout applied to them)
        # are transformed. The output size is rnn_size if the LSTM is unidirectional, or 2 * rnn_size if
        # it is bidirectional
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Applies a simple linear transformation to the outputs of the previous layer (in our case the biLSTM)
        # The outputs have dimension hidden_size and optionally have an added bias. An activation function can be
        # added in the forward function
        self.linear1 = nn.Linear(
            in_features=2 * rnn_size,
            out_features=hidden_size,
            bias=True
        )

        # Bias is added by default. Since we are doing a binary classification task, we only need 1 out feature
        self.linear2 = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, batch):
        if "input_ids" not in batch:
            raise ValueError(
                "Invalid batch input format. Could not find key 'input_ids'.")
        if "lengths" not in batch:
            raise ValueError(
                "Invalid batch input format. Could not find key 'lengths'.")

        seqs = batch["input_ids"]
        lengths = batch["lengths"]

        hidden = self.embedding(seqs)
        hidden = self.dropout_layer(hidden)

        # For computational efficiency pack the padded sequences
        # From doc: lengths must be on cpu if provided as a tensor -> lengths.cpu()
        # We keep the batch dimension as first dimension with batch_first
        hidden = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, (_, _) = self.bilstm(hidden)

        # Inverse operation to unpack the sequences and pad them again
        # Second return is the lengths again
        # Keep batches as first dimension
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        # BiLSTM returns hidden representation for each index -> since we are only doing classification
        # we can pool into a single dimension
        hidden = torch.mean(hidden, dim=1)

        hidden = self.linear1(hidden)
        # hidden = self.dropout_layer(hidden)
        hidden = F.relu(hidden)

        # Single output, since we do binary classification
        return torch.sigmoid(self.linear2(hidden)).squeeze(1)
