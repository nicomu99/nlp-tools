from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

def c_pad_sequence(sequence: list[int], pad_token: int, max_length: int) -> list[int]:
    for _ in range(max_length - len(sequence)):
        sequence.append(pad_token)

    return sequence

def c_pad_sequences(sequences: list[list[int]], pad_token: int, max_length: int) -> list[list[int]]:
    return [c_pad_sequence(sequence, pad_token, max_length) for sequence in sequences]

def collate_batch(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    # Pad input_ids
    input_ids = [index for (index, _) in batch]

    pad_token_id = 1
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

    # Turn labels into tensors
    labels = torch.tensor([label for (_, label) in batch], dtype=torch.long)

    # Lengths of individual sequences
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)

    return padded_input_ids, labels, lengths