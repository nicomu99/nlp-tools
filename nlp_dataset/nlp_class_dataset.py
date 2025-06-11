from typing import Tuple, List, Optional

from nlp_dataset import NLPDataset

class NLPClassDataset(NLPDataset):
    def __init__(self, input_ids: List[List[int]], labels: List[int]):
        super().__init__(input_ids)
        assert len(input_ids) == len(labels)
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.input_ids[index], self.labels[index]