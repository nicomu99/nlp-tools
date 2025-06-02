import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    A simple embedding layer, which is just a lookup table of embeddings. The embeddings
    are updated during training.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        """
        Initializes the embedding matrix.

        :param num_embeddings: Size of the dictionary.
        :param embedding_dim: The dimension of each token embedding.
        :param padding_idx: Index used for padding.
        """

        super().__init__()

        self.padding_idx = padding_idx
        self.embedding_matrix = nn.Parameter(
            torch.randn(
                num_embeddings,
                embedding_dim
            )
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the embeddings for the tensor input_ids.

        :param input_ids: A tensor of size (batch_size, seq_len) for which the embeddings should be returned.
        :return: A tensor of size (batch_size, seq_len, embedding_dim), holding the embeddings for each token.
        Any padded token gets zeroed.
        """

        # Mask out the row of the padding index by checking which input ids
        # are not the padding idx. Since our mask has shape (batch_size, seq_len),
        # we unsqueeze to add another dimension (batch_size, seq_len, 1). This gets
        # broadcasted to (batch_size, seq_len, embedding_dim) to allow multiplication
        # with the embeddings
        # Using float, we simply convert boolean 1s and 0s to float 1s and 0s
        mask = (input_ids != self.padding_idx).unsqueeze(-1).float()

        embeddings = self.embedding_matrix[input_ids]
        return embeddings * mask