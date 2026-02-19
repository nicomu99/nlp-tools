"""Utility functions."""
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def c_pad_sequence(
    sequence: list[int],
    pad_token: int,
    max_length: int
) -> list[int]:
    """Pads a sequence to a certain length.

    Args:
        sequence (list[int]): Tokenized input ids.
        pad_token (int): The padding token.
        max_length (int): The maximum sequence length. If the sequence is shorter
            than this number, it will be padded.

    Returns:
        list[int]: Sequence padded to the desired length.
    """
    for _ in range(max_length - len(sequence)):
        sequence.append(pad_token)
    return sequence


def c_pad_sequences(
    sequences: list[list[int]],
    pad_token: int,
    max_length: int
) -> list[list[int]]:
    """Pads a list of tokenized sequences.

    Args:
        sequences (list[list[int]]): A list of tokenized input ids.
        pad_token (int): The padding token.
        max_length (int): The maximum sequence length. If the sequence is shorter
            than this number, it will be padded.

    Returns:
        list[list[int]]: Padded list of sequences.
    """
    return [c_pad_sequence(sequence, pad_token, max_length) for sequence in sequences]


def collate_batch(
    batch: list[tuple[list[int], int] | tuple[list[int], list[int]]]
) -> dict[str, torch.Tensor]:
    """Collate function that returns a dictionary with lengths.

    This function returns a dictionary with input ids, targets and the lengths of the input sequences.

    Args:
        batch (list[tuple[list[int], int] | tuple[list[int], list[int]]]): A batch of inputs.

    Returns:
        dict[str, torch.Tensor]: A dictionary with padded input_ids and lengths.
    """
    ids = [torch.tensor(index, dtype=torch.long) for (index, _) in batch]
    labels = torch.tensor([label for (_, label) in batch], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in ids], dtype=torch.long)
    # TODO: The padding value should be changeable
    padded_indices = pad_sequence(ids, batch_first=True, padding_value=1)

    return {
        "input_ids": padded_indices,
        "targets": labels,
        "lengths": lengths
    }


def f1_score(
    predictions: list[int],
    labels: list[int]
) -> float:
    """Computes the F1-score.

    Args:
        predictions (list[int]): Ordered list with model label predictions.
        labels (list[int]): Ordered list with ground truth labels.

    Returns:
        float: F1-score.
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    true_positives = np.sum((predictions == 1) & (labels == 1))
    predicted_positives = np.sum(predictions == 1)
    actual_positives = np.sum(labels == 1)

    precision = true_positives / predicted_positives if predicted_positives != 0 else 0
    recall = true_positives / actual_positives if actual_positives != 0 else 0

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)
