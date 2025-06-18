from logger import Logger

class ARLMLogger(Logger):
    def __init__(self, run_name):
        super().__init__(run_name)

        self.loss = 0
        self.processed = 0

    def init_epoch(self):
        super().init_epoch()

        self.loss = 0
        self.processed = 0

    def update(self, outputs, targets, loss):
        super().update_memory()

        batch_size = targets.size(0)
        self.processed += batch_size
        self.loss += loss.item() * batch_size

    def log_epoch(self, stage):
        metric_dir = {
            'epoch': self.epoch,
            'loss': self.loss / self.processed
        }

        self.save_metrics_and_memory(metric_dir, stage)