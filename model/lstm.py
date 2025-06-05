import torch
import torch.nn as nn
from model.lstm_cell import LSTMCell

class LSTM(nn.Module):
    """
    A full LSTM with an optional bidirectional setting. Computes the full hidden representation across a full sequence
    of tokens.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = True):
        """
        Initializes the LSTM.

        :param input_size: The expected size of the input tensors.
        :param hidden_size: The hidden dimension size.
        :param num_layers: Number of layers in the LSTM to allow stacking the LSTM.
        :param bidirectional: Controls whether the LSTM should be bidirectional. If True, the input sequences are
        processed both from left-to-right and right-to-left. If False, only from left-to-right.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.forward_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size * (2 if bidirectional else 1), hidden_size)
            for i in range(self.num_layers)
        ])

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.backward_cells = nn.ModuleList([
                LSTMCell(input_size if i == 0 else hidden_size * (2 if bidirectional else 1), hidden_size)
                for i in range(self.num_layers)
            ])

    def forward(self, x: torch.Tensor):
        """
        Computes the hidden representation across the whole input sequence.

        :param x: The features of the input sequence of size (batch_size, seq_len, input_dim) or (seq_len, input_dim).
        :return output: The hidden representation of the sequences after being processed of size
        (batch_size, seq_len, input_dim)
        """
        # Add another dimension, if the input is not batched.
        type(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape

        output = x
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_b = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        c_b = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for layer in range(self.num_layers):
            h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)

            forward = []
            for i in range(seq_len):
                x_f = output[:, i, :]
                h_f, c_f = self.forward_cells[layer](x_f, h_f, c_f)
                forward.append(h_f.unsqueeze(1))
            forward = torch.cat(forward, dim=1)

            if self.bidirectional:
                h_b = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
                c_b = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

                backward = []
                for i in reversed(range(seq_len)):
                    x_b = output[:, i, :]
                    h_b, c_b = self.backward_cells[layer](x_b, h_b, c_b)
                    backward.append(h_b.unsqueeze(1))

                backward = list(reversed(backward))
                backward = torch.cat(backward, dim=1)

                output = torch.cat([forward, backward], dim=-1)
            else:
                output = forward


        # To match pytorch
        if self.bidirectional:
            h_n = torch.stack([h_f, h_b], dim=0)
            c_n = torch.stack([c_f, c_b], dim=0)
        else:
            h_n = h_f.unsqueeze(0)
            c_n = c_f.unsqueeze(0)

        return output, (h_n, c_n)