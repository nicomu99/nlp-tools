"""Base dataset class."""
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    def __init__(self, input_ids: list[list[int]]):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        raise NotImplementedError("Please use a subclass.")
