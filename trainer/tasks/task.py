"""Abstract base class defining a training task."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from datasets import Dataset as HFDataset
from trainer.nlp_dataset.nlp_dataset import NLPDataset


class Task(ABC):
    """Abstract base class defining a training task.

    A `Task` encapsulates all task-specific logic required during neural
    network training, such as classification or arlm. This includes
    batch-level processing, epoch-level metric aggregation, and dataset
    adaptation.

    Subclasses must implement all abstract methods and may maintain
    internal state that is reset at the beginning of each epoch.
    """

    @abstractmethod
    def init_epoch(self):
        """Initializes or resets epoch-level state.

        This method is called once at the beginning of each training or
        evaluation epoch. Implementations should reset all metric
        accumulators and counters used during the epoch.
        """

    @abstractmethod
    def process_step(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> float:
        """Processes a single batch.

        This method is called once per batch and implements the
        task-specific forward pass, loss computation, and metric updates.

        Args:
            model (nn.Module): The neural network model used for this task.
            batch (dict[str, torch.Tensor]): A batch of data produced by the task-specific dataset.

        Returns:
            float: A task-dependent loss value required by the training loop.
        """

    @abstractmethod
    def get_epoch_stats(self) -> dict:
        """Returns aggregated statistics for the current epoch.

        This method is called after all batches in the epoch have been
        processed.

        Returns:
            dict: A dictionary mapping metric names to aggregated values
                for the epoch.
        """

    @staticmethod
    @abstractmethod
    def get_dataset(dataset: HFDataset) -> NLPDataset:
        """Converts a HuggingFace dataset into a task-specific dataset.

        This method adapts a generic HuggingFace `datasets.Dataset` into
        an `NLPDataset` suitable for this task. Implementations may
        perform tokenization, label encoding, filtering, or field
        restructuring.

        Args:
            dataset (datasets.Dataset): The raw HuggingFace dataset.

        Returns:
            NLPDataset: A task-specific dataset ready for training or
                evaluation.
        """
