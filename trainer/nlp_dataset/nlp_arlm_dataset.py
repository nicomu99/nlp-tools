"""ARLM dataset that returns input/target pairs for autoregressive language modeling."""
from .nlp_dataset import NLPDataset


class NLPARLMDataset(NLPDataset):
    """Dataset for autoregressive language modeling (ARLM).

    This dataset prepares examples for next-token prediction. It flattens all
    tokenized input sequences into a single stream, chunks the stream into
    fixed-length blocks, and returns `(x, y)` pairs where `y` is `x` shifted by
    one token.

    For a block `[w0, w1, ..., wt]`, this dataset returns:
      - inputs:  `[w0, w1, ..., w(t-1)]`
      - targets: `[w1, w2, ..., wt]`
    """

    def __init__(self, input_ids: list[list[int]], block_size: int):
        """Initializes the ARLM dataset.

        The provided `input_ids` are flattened into a single token stream and
        then chunked into blocks of length `block_size`.

        Args:
            input_ids (list[list[int]]): Tokenized input sequences.
            block_size (int): Number of tokens per block before shifting.
                Each returned example has length `block_size - 1` for both
                inputs and targets.

        Raises:
            ValueError: If `block_size` is less than 2.
        """
        if block_size < 2:
            raise ValueError("block_size must be >= 2 for next-token prediction.")
        super().__init__(input_ids)

        flattened_text: list[int] = []
        for seq in input_ids:
            flattened_text.extend(seq)

        self.input_ids = [
            flattened_text[i:i + block_size]
            for i in range(0, len(flattened_text), block_size)
        ]

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        """Returns a single autoregressive training example.

        Args:
            index (int): Index of the example to retrieve.

        Returns:
            tuple[list[int], list[int]]: A tuple `(inputs, targets)` where
                `targets` is `inputs` shifted left by one token.
        """
        input_ids = self.input_ids[index]
        return input_ids[:-1], input_ids[1:]  # (0, 1, ..., t-1), (1, 2, ..., t)

    @classmethod
    def from_hf(cls, dataset: HFDataset, block_size: int) -> NLPARLMDataset:
        """Creates an ARLM dataset from a HuggingFace dataset.

        This method converts a HuggingFace `datasets.Dataset` into an
        `NLPARLMDataset` by extracting the required columns and validating
        that the dataset conforms to the expected schema.

        The input dataset must contain the following column:
          - `"input_ids"`: Tokenized input sequences

        Args:
            dataset (datasets.Dataset): A HuggingFace dataset containing
                tokenized input sequences.
            block_size (int): Number of tokens per block before shifting.
                Each returned example has length `block_size - 1` for both
                inputs and targets.

        Returns:
            NLPARLMDataset: A dataset instance suitable for autoregressive
            language modeling.

        Raises:
            ValueError: If the dataset does not contain the required columns
                or if `block_size` is less than 2.
        """
        if block_size < 2:
            raise ValueError("block_size must be >= 2 for next-token prediction.")
        required_columns = {"input_ids"}
        cls._validate_columns(dataset, required_columns)
        return cls(dataset["input_ids"], block_size)
