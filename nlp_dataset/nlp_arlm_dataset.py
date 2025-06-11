from typing import Tuple, List

from nlp_dataset import NLPDataset

class NLPARLMDataset(NLPDataset):
    def __init__(self, input_ids: List[List[int]], block_size: int):
        super().__init__(input_ids)

        ids = []
        for sequence in self.input_ids:
            ids.extend(sequence)

        self.input_ids = [ids[i : i + block_size + 1] for i in range(0, len(ids) - block_size, block_size)]

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        input_ids = self.input_ids[index]
        return input_ids[:-1], input_ids[1:]