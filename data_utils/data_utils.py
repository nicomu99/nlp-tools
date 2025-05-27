from typing import List, Tuple

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def c_pad_sequence(sequence: list[int], pad_token: int, max_length: int) -> list[int]:
    for _ in range(max_length - len(sequence)):
        sequence.append(pad_token)

    return sequence

def c_pad_sequences(sequences: list[list[int]], pad_token: int, max_length: int) -> list[list[int]]:
    return [c_pad_sequence(sequence, pad_token, max_length) for sequence in sequences]

def collate_batch(batch: List[Tuple[List[int], int]]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    ids = [torch.tensor(index, dtype=torch.long) for (index, _) in batch]
    labels = torch.tensor([label for (_, label) in batch], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in ids], dtype=torch.long)
    padded_indices = pad_sequence(ids, batch_first=True, padding_value=1)
    return padded_indices, labels, lengths

def f1_score(predictions, labels):
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