import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size, input_size if i == 0 else hidden_size))
            for i in range(num_layers)
        ])

        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size, self.hidden_size)) for i in range(num_layers)
        ])

        self.b = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size)) for _ in range(num_layers)
        ])

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
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_h_t = []
            for layer in range(self.num_layers):
                h_prev = h_t[layer]
                h_curr = torch.tanh(
                    x_t @ self.W[layer].T + self.V[layer] @ h_prev[layer] + self.b[layer]
                )
                new_h_t.append(h_curr)
                x_t = h_curr

            h_t = torch.stack(new_h_t, dim=0)
            output.append(h_t[-1])

        output = torch.stack(output, dim=1)
        return output, h_t