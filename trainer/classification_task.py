from datasets import Dataset as HFDataset
from torch.utils.data import BatchSampler, DataLoader

from nlp_dataset import NLPClassDataset
from logger import Logger
from tokenization import Tokenizer
from nlp_sampler import GroupedSampler
from data_utils import *


class ClassificationTask:
    def __init__(self, run_name, criterion):
        self.logger = Logger(run_name)
        self.criterion = criterion

        self.processed = 0
        self.loss = 0
        self.correct = 0
        self.pred_labels = []
        self.true_labels = []

    def init_epoch(self):
        self.logger.init_epoch()

        self.processed = 0
        self.loss = 0
        self.correct = 0
        self.pred_labels = []
        self.true_labels = []

    def process_step(self, model, batch):
        if "targets" not in batch:
            raise ValueError("Invalid batch format, missing key 'targets'")
        if "input_ids" not in batch:
            raise ValueError("Invalid batch format, missing key 'input_ids'")

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

        self.logger.update()

        return loss

    def finish_epoch(self, epoch: int, stage: str = "train"):
        """
        Compute metrics and write to tensorboard.
        """
        metric_dir = {
            "epoch": epoch,
            "loss": self.loss / self.processed,
            "accuracy": self.correct / self.processed,
            "f1": f1_score(self.pred_labels, self.true_labels)
        }
        self.logger.log_epoch(stage, metric_dir)

    def close(self):
        self.logger.close()

    @staticmethod
    def get_data_loader(
            tokenizer: Tokenizer, dataset: HFDataset, batch_size: int, data_loaders: int, split: str = "train"
    ):
        dataset = NLPClassDataset(
            tokenizer(dataset["text"]),
            dataset["label"]
        )

        if split == "train":
            batch_sampler = BatchSampler(
                GroupedSampler(dataset.get_input_ids(), batch_size),
                batch_size=batch_size,
                drop_last=True
            )

            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_batch,
                num_workers=data_loaders
            )

        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_batch,
                num_workers=data_loaders
            )

        return data_loader
