import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(
            torch.randn(self.hidden_size, self.input_size)
        )

        self.V = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size)
        )

        self.b = nn.Parameter(
            torch.randn(self.hidden_size)
        )

    def forward(self, x: torch.Tensor):
        """

        :param x: Input tensor of shape (batch_size, seq_len, input_size) or (seq_len, input_size)
        :return:
        """
        # Reshape to match implementation
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f'Input must be of shape (seq_len, input_dim) or (batch_size, seq_len, input_dim), got {x.shape}')

        batch_size, seq_len, _ = x.size()

        output = []
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for i in range(seq_len):
            prev_h = h
            x_i = x[:, i, :]

            h = torch.tanh(
                x_i @ self.W.T + self.V @ prev_h + self.b
            )

            output.append(h)

        output = torch.stack(output, dim=1)
        return output