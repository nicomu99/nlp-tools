import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_f = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_i = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_o = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_g = nn.Parameter(torch.randn(hidden_dim, input_dim))

        self.V_f = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.V_i = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.V_o = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.V_g = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.bias_f = nn.Parameter(torch.randn(hidden_dim))
        self.bias_i = nn.Parameter(torch.randn(hidden_dim))
        self.bias_o = nn.Parameter(torch.randn(hidden_dim))
        self.bias_g = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x_t, h_prev, c_prev):
        # x_t: (batch_size, input_dim)
        # h_prev, c_prev: (batch_size, hidden_dim)

        # x_t @ self.W_f.T:     (batch_size, input_dim) * (hidden_dim, input_dim)^T -> (batch_size, hidden_dim)
        # h_prev @ self.V_f.T:  (batch_size, hidden_dim) * (hidden_dim, hidden_dim)^T -> (batch_size, hidden_dim)
        # f_t: (batch_size, hidden_dim)
        f_t = torch.sigmoid(
            x_t @ self.W_f.T + h_prev @ self.V_f.T + self.bias_f
        )

        i_t = torch.sigmoid(
            x_t @ self.W_i.T + h_prev @ self.V_i.T + self.bias_i
        )

        g_t = torch.tanh(
            x_t @ self.W_g.T + h_prev @ self.V_g.T + self.bias_g
        )

        o_t = torch.sigmoid(
            x_t @ self.W_o.T + h_prev @ self.V_o.T + self.bias_o
        )

        c_t = f_t * c_prev + i_t * g_t

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
