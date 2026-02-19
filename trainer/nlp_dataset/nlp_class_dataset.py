"""Classification dataset that returns a single label."""
from .nlp_dataset import NLPDataset


class NLPClassDataset(NLPDataset):
    """Dataset for single-label classification tasks.

    This dataset represents classification data where each input sequence
    is associated with exactly one label. It extends `NLPDataset` by
    adding label storage and returning `(input_ids, label)` pairs.

    Instances of this class are typically created from a HuggingFace
    dataset using the `from_hf` class method.
    """

    def __init__(self, input_ids: list[list[int]], labels: list[int]):
        """Initializes the classification dataset.

        Args:
            input_ids (list[list[int]]): Tokenized input sequences.
            labels (list[int]): Integer class labels corresponding to each
                input sequence.

        Raises:
            AssertionError: If the number of input sequences does not match
                the number of labels.
        """
        super().__init__(input_ids)
        if len(input_ids) != len(labels):
            raise ValueError("input_ids and labels must have the same length.")
        self.labels = labels

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        """Returns a single classification example.

        Args:
            index (int): Index of the example to retrieve.

        Returns:
            tuple[list[int], int]: A tuple containing the tokenized input
            sequence and its corresponding class label.
        """
        return self.input_ids[index], self.labels[index]

    @classmethod
    def from_hf(cls, dataset: HFDataset) -> NLPClassDataset:
        """Creates a classification dataset from a HuggingFace dataset.

        This method converts a HuggingFace `datasets.Dataset` into an
        `NLPClassDataset` by extracting the required columns and validating
        that the dataset conforms to the expected schema.

        The input dataset must contain the following columns:
          - `"input_ids"`: Tokenized input sequences
          - `"label"`: Class labels corresponding to each input sequence

        Args:
            dataset (datasets.Dataset): A HuggingFace dataset containing
                tokenized inputs and labels.

        Returns:
            NLPClassDataset: A dataset instance suitable for classification
            tasks.

        Raises:
            ValueError: If the dataset does not contain the required columns
                or if the dataset format is otherwise invalid.
        """
        required_columns = {"input_ids", "label"}
        cls._validate_columns(dataset, required_columns)
        return cls(dataset["input_ids"], dataset["label"])
