import torch
import torch.nn as nn

class Linear(nn.Module):
    """
    A simple linear layer, which applies the transformation y = x*w.t + bias.
    """

    def __init__(self, in_features: int, out_features: int, bias = True):
        """
        Initializes the linear layer. If bias is set to true, a bias is added to the output.

        :param in_features: Dimension of the input tensor.
        :param out_features: Dimension of the output tensor
        :param bias: Controls whether a bias should be added or not.
        """
        super().__init__()

        self.weights = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        self.apply_bias = bias
        if self.apply_bias:
            self.bias = nn.Parameter(
                torch.randn(out_features)
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear transformation y = x*w.t + bias.

        :param input_ids: Input tensor of shape (batch_size, ..., in_features)
        :return: Output tensor of shape (batch_size, ..., out_features) with a bias added optionally
        """
        out = input_ids @ self.weights.T

        if self.apply_bias:
            out += out + self.bias
        return out

