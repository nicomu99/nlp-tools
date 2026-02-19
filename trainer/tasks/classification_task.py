"""Implementation class for classification tasks."""
from datasets import Dataset as HFDataset

from trainer.nlp_dataset import NLPClassDataset, NLPDataset
from trainer.utils import f1_score

from .task import Task


class ClassificationTask(Task):
    def __init__(self, criterion):
        self.criterion = criterion

        self.processed = 0
        self.loss = 0
        self.correct = 0
        self.pred_labels = []
        self.true_labels = []

    def init_epoch(self):
        self.processed = 0
        self.loss = 0
        self.correct = 0
        self.pred_labels = []
        self.true_labels = []

    def process_step(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> float:
        if "targets" not in batch:
            raise ValueError("Invalid batch format, missing key 'targets'")

        batch_size = batch["targets"].shape[0]
        targets = batch["targets"].float()

        outputs = model(batch)
        loss = self.criterion(outputs, targets)

        targets = targets.int()
        predictions = (outputs >= 0.5).int()
        correct = (predictions == targets).sum().item()

        self.processed += batch_size
        self.loss += loss.item() * batch_size
        self.correct += correct
        self.pred_labels.extend(predictions.cpu().tolist())
        self.true_labels.extend(targets.cpu().tolist())

        return loss

    def get_epoch_stats(self) -> dict:
        metric_dir = {
            "loss": self.loss / self.processed,
            "accuracy": self.correct / self.processed,
            "f1": f1_score(self.pred_labels, self.true_labels)
        }
        return metric_dir

    @staticmethod
    def get_dataset(dataset: HFDataset, **kwargs) -> NLPDataset:
        return NLPClassDataset.from_hf(dataset)
