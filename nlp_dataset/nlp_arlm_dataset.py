from typing import Tuple, List

from nlp_dataset import NLPDataset

class NLPARLMDataset(NLPDataset):
    def __init__(self, input_ids: List[List[int]], block_size: int):
        super().__init__(input_ids)

        flattened_text = []
        for input_id in input_ids:
            flattened_text.extend(input_id)

        self.input_ids = [flattened_text[i:i + block_size] for i in range(0, len(flattened_text), block_size)]

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        input_ids = self.input_ids[index]
        return input_ids[:-1], input_ids[1:]