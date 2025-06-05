import torch

from logger.logger import Logger
from data_utils.data_utils import f1_score

class ClassificationLogger(Logger):
    def __init__(self, run_name):
        super().__init__(run_name)

        # Loss metrics
        self.processed      = 0
        self.loss           = 0
        self.correct        = 0
        self.pred_labels    = []
        self.true_labels    = []

    def init_epoch(self, epoch):
        super().init_epoch(epoch)

        self.processed = 0
        self.loss = 0
        self.correct = 0
        self.pred_labels = []
        self.true_labels = []

    def update(self, outputs, targets, loss):
        super().update_memory()

        batch_size = targets.size(0)
        self.processed += batch_size
        self.loss += loss.item() * batch_size

        pred_labels = (outputs >= 0.5).int()
        self.correct += torch.eq(pred_labels, targets).sum().item()

        self.pred_labels.extend(pred_labels.cpu().tolist())
        self.true_labels.extend(targets.cpu().tolist())

    def log_epoch(self, stage):
        metric_dir = {
            'epoch': self.epoch,
            'loss': self.loss / self.processed,
            'accuracy': self.correct / self.processed,
            'f1': f1_score(self.pred_labels, self.true_labels)
        }

        self.save_metrics_and_memory(metric_dir, stage)