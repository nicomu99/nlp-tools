import torch
import torch.nn as nn
from lstm_cell import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = True):
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
        batch_size, seq_len, _ = x.shape

        output = x
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

        return output