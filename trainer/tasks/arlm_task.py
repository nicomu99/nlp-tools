"""Implementation class for arlm tasks."""
from datasets import Dataset as HFDataset

from trainer.nlp_dataset import NLPARLMDataset, NLPDataset

from .task import Task


class ARLMTask(Task):
    def __init__(self, criterion):
        self.criterion = criterion

        self.loss = 0
        self.processed = 0

    def init_epoch(self):
        self.processed = 0
        self.loss = 0

    def process_step(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> float:
        if "targets" not in batch:
            raise ValueError("Invalid batch format, missing key 'targets'")

        batch_size = batch["targets"].shape[0]
        targets = batch["targets"].float()

        outputs = model(batch)
        loss = self.criterion(outputs, targets)

        self.processed += batch_size
        self.loss += loss.item() * batch_size

        return loss

    def get_epoch_stats(self) -> dict:
        metric_dir = {"loss": self.loss / self.processed}
        return metric_dir

    @staticmethod
    def get_dataset(dataset: HFDataset, **kwargs) -> NLPDataset:
        block_size = kwargs.get("block_size")
        if block_size is None:
            raise ValueError("block_size is required for NLPARLMDataset")
        return NLPARLMDataset.from_hf(dataset, block_size)
