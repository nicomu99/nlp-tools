import math

import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """
    A single LSTM cell calculating the LSTM transformation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initializes the cell.

        :param input_dim: The embedding dimension of the input tensors.
        :param hidden_dim: The hidden dimension of the LSTM cell.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Use pytorch's initialization, to match behavior
        std = 1.0 / math.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0

        self.W_f = nn.Parameter(torch.empty(hidden_dim, input_dim).uniform_(-std, std))
        self.W_i = nn.Parameter(torch.empty(hidden_dim, input_dim).uniform_(-std, std))
        self.W_o = nn.Parameter(torch.empty(hidden_dim, input_dim).uniform_(-std, std))
        self.W_g = nn.Parameter(torch.empty(hidden_dim, input_dim).uniform_(-std, std))

        self.V_f = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-std, std))
        self.V_i = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-std, std))
        self.V_o = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-std, std))
        self.V_g = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-std, std))

        self.bias_f = nn.Parameter(torch.empty(hidden_dim).uniform_(-std, std))
        self.bias_i = nn.Parameter(torch.empty(hidden_dim).uniform_(-std, std))
        self.bias_o = nn.Parameter(torch.empty(hidden_dim).uniform_(-std, std))
        self.bias_g = nn.Parameter(torch.empty(hidden_dim).uniform_(-std, std))

    def forward(self, x_t, h_prev, c_prev):
        """
        Computes the LSTM gates, creating the short term and long term memory in the process.

        :param x_t: Input tensors of size (batch_size, input_dim).
        :param h_prev: The short-term memory of size (batch_size, hidden_dim), which in a multi-layer LSTM is passed on
        from the previous layer. For the first layer, all entries are 0.
        :param c_prev: The long-term memory of size (batch_size, hidden_dim), which in a multi-layer LSTM is passed on
        from the previous layer. For the first layer, all entries are 0.

        :return h_t: The transformed short-term memory of size (batch_size, hidden_dim)
        :return c_t: The transformed long-term memory of size (batch_size, hidden_dim)
        """
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

        # f_t * c_prev: (batch_size, hidden_dim) * (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        # i_t * g_t:    (batch_size, hidden_dm) * (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        c_t = f_t * c_prev + i_t * g_t

        # (batch_size, hidden_dim)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
