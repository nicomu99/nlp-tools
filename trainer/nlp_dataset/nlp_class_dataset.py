"""Classification dataset that returns a single label."""
from trainer.nlp_dataset import NLPDataset


class NLPClassDataset(NLPDataset):
    def __init__(self, input_ids: list[list[int]], labels: list[int]):
        super().__init__(input_ids)
        assert len(input_ids) == len(labels)
        self.labels = labels

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.input_ids[index], self.labels[index]
