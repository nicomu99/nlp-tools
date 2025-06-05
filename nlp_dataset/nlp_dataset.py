from typing import Tuple, List

from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.input_ids[index], self.labels[index]

    def get_input_ids(self) -> List[List[int]]:
        return self.input_ids