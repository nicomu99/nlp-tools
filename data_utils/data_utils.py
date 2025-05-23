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

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    # Pad input_ids
    input_ids, labels = zip(*batch)

    pad_token_id = 1
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

    # Lengths of individual sequences
    lengths = torch.tensor([len(seq) for seq in input_ids])

    return padded_input_ids, labels, lengths

def f1_score(predictions, labels):
    true_positives = ((predictions == 1) == labels)
    total_positive_preds = np.sum((predictions == 1))
    precision = true_positives / total_positive_preds

    total_positives = np.sum(labels == 1)
    recall = true_positives / total_positives

    return 2 * (precision * recall) / (precision + recall)