from typing import Tuple, List

import torch
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.input_ids[index]), torch.tensor(self.labels[index])

    def get_input_ids(self) -> List[List[int]]:
        return self.input_ids