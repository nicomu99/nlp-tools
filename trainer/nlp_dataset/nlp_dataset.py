"""Base dataset class."""
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class NLPDataset(Dataset, ABC):
    """Base class for NLP datasets used in training and evaluation.

    This class defines a minimal interface and shared functionality for
    NLP-specific datasets built on top of PyTorch's `Dataset`. It stores
    tokenized input sequences and provides common validation utilities
    for adapting HuggingFace datasets.

    Subclasses are expected to:
      - implement `__getitem__`
      - define task-specific fields (e.g. labels)
      - ensure dataset validity at construction time
    """

    def __init__(self, input_ids: list[list[int]]):
        """Initializes the dataset with tokenized input sequences.

        Args:
            input_ids (list[list[int]]): A list of tokenized input sequences,
                where each sequence is represented as a list of token IDs.
        """
        self.input_ids = input_ids

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            int: The number of input sequences.
        """
        return len(self.input_ids)

    @abstractmethod
    def __getitem__(self, index: int):
        """Returns a single dataset item.

        This method must be implemented by subclasses to return a
        task-specific representation of a dataset example.

        Args:
            index (int): Index of the example to retrieve.

        Raises:
            NotImplementedError: If the subclass does not implement this
                method.
        """

    @staticmethod
    def _validate_columns(
            dataset: HFDataset,
            required_columns: set[str],
    ) -> None:
        """Validates that a HuggingFace dataset contains required columns.

        Args:
            dataset (datasets.Dataset): The HuggingFace dataset to validate.
            required_columns (set[str]): Column names required by the dataset
                implementation.

        Raises:
            ValueError: If any required columns are missing.
        """
        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"Invalid dataset format. Missing columns: {sorted(missing)}"
            )
