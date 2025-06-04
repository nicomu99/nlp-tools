import torch
import torch.nn as nn

class Dropout(nn.Module):
    """
    A simple dropout layer. The parameter p is the probability any single token is masked out to 0.
    The masking procedure utilizes a Bernoulli distribution. If an entry in a randomly generated
    tensor is below the threshold, the token gets masked. Dropout is only applied during training. In this
    case, the outputs are divided with 1 - p to keep expected values equal with evaluation.
    """

    def __init__(self, p: float):
        """
        Initializes the dropout layer.

        :param p: Probability of masking out some token.
        """
        super().__init__()

        if p < 0 or p > 1:
            raise RuntimeError(f'Dropout must be between 0 and 1')

        self.p = p

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the input ids with applied dropout. Only gets applied during training.

        :param input_ids: Tensor of size (batch_size, seq_len, hidden_dim) on which dropout is applied.

        :return: Either the input_ids, if training mode is off, or the masked input_ids with normalization.
        """
        # self.training is automatically updated when using model.train() or .eval()
        if not self.training or self.p == 0:
            return input_ids

        # Dividing with (1 - p) keeps the expected values equal during training and evaluation
        mask = (torch.rand_like(input_ids) > self.p).float()
        return mask * input_ids / (1.0 - self.p)