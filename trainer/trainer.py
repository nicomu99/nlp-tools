import os
from typing import Optional

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, BatchSampler, DataLoader

from datasets import Dataset as HFDataset

from tokenization import Tokenizer
from nlp_dataset import NLPDataset, NLPClassDataset, NLPARLMDataset
from nlp_sampler import GroupedSampler
from logger import Logger
from data_utils import *
from .classification_task import ClassificationTask


class Trainer:
    def __init__(
        self,
        task: str,
        model: nn.Module,
        tokenizer: Tokenizer,
        criterion: nn.Module,
        train_dataset: HFDataset,
        eval_dataset: HFDataset = None,
        test_dataset: HFDataset = None,
        optimizer: torch.optim.Optimizer = None,
        num_epochs: int = 3,
        batch_size: int = 32,
        data_loaders: int = 2,
        learning_rate: float = 1e-4,
        run_name: str = "",
        early_stop: bool = False,
        early_stop_limit: int = 3,
        block_size: Optional[int] = 128
    ):
        self.run_name = run_name
        self.logger = Logger(run_name)

        # Task is used control aspects of the training
        if self.task == 'classification':
            self.logger = ClassificationLogger(self.run_name)
        elif self.task == 'arlm':
            self.logger = ARLMLogger(self.run_name)
        if task is None:
            raise RuntimeError("Task is missing.")
        self.task = task

        if train_dataset is None:
            raise RuntimeError("Train dataset is missing.")

        if tokenizer is None:
            raise RuntimeError("Tokenizer is missing.")

        if criterion is None:
            raise RuntimeError("Criterion is missing.")

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_loaders = data_loaders
        self.block_size = block_size  # For ARLM

        self.early_stop = early_stop
        self.early_stop_limit = early_stop_limit

        self.train_dataloader = task.get_data_loader(self.tokenizer, train_dataset, batch_size, data_loaders)

        if eval_dataset is not None:
            self.eval_dataloader = task.get_data_loader(self.tokenizer, eval_dataset, batch_size, data_loaders, "eval")

        if test_dataset is not None:
            self.test_dataloader = task.get_data_loader(self.tokenizer, eval_dataset, batch_size, data_loaders, "test")

    def _prepare_dataset(self, dataset: HFDataset) -> NLPDataset:
        if self.task == "classification":
            return NLPClassDataset(
                self.tokenizer(dataset["text"]),
                dataset["label"]
            )
        else:
            return NLPARLMDataset(
                self.tokenizer(dataset["text"]),
                block_size=self.block_size
            )

    def _prepare_loader(self, dataset: Dataset, batch_sampling: bool = False) -> DataLoader:
        if batch_sampling and self.task == "classification":
            # On ARLM tasks, grouping is not required
            batch_sampler = BatchSampler(
                GroupedSampler(self.train_dataset.get_input_ids(), self.batch_size),
                batch_size=self.batch_size,
                drop_last=True
            )

            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_batch,
                num_workers=self.data_loaders
            )
        elif batch_sampling:
            data_loader = DataLoader(
                dataset,
                batch_size=32,
                collate_fn=collate_batch,
                num_workers=self.data_loaders
            )

        else:
            data_loader = DataLoader(
                dataset,
                collate_fn=collate_batch,
                shuffle=True,
                num_workers=self.data_loaders
            )

        return data_loader

    def process_one_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        stage: str = ""
    ):
        """
        Processes one training or validation step for a given model.
        :param loader: The data loader providing batches of input sequences and labels.
        :param optimizer: The optimizer used for training. If None, the function runs in evaluation mode.
        :param stage: A string representing the current training stage. Can be 'train', 'eval' or 'test'.
        """
        train = optimizer is not None
        self.model.train() if train else self.model.eval()

        self.task.init_epoch()
        self.logger.init_epoch()
        for batch in tqdm(loader, desc="Training" if train else "Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            torch.cuda.reset_peak_memory_stats()
            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                loss = self.task.process_step(self.model, batch)

                if train:
                    loss.backward()
                    optimizer.step()
                self.logger.update()

        metric_dir = self.task.get_epoch_stats()
        self.logger.log_epoch(stage, epoch, metric_dir)

    def train(self):

        # early_stop_epoch = 0
        # best_f1 = 0
        for epoch in range(self.num_epochs):
            self.current_epoch += 1
            print(f"Epoch {self.current_epoch} - Training")

            self.process_one_epoch(self.train_dataloader, self.optimizer, stage="train")

            with torch.no_grad():
                self.process_one_epoch(self.eval_dataloader, stage="eval")

        self.save_model()
        self.logger.close()
        # val_f1 = val_metrics['f1']
        # if val_f1 > best_f1:
        #     # Save best model performance
        #     best_f1 = val_f1
        #     self.logger.save_model(self.model.state_dict().copy())
        #
        # # Early stopping if 3 consecutive epochs are below the highest F1 score
        # if val_f1 > best_f1:
        #     best_f1 = val_f1
        #     early_stop_epoch = 0
        # else:
        #     early_stop_epoch += 1
        #
        # if early_stop_epoch >= self.early_stop_limit and self.early_stop:
        #     print(f"Training stopped early after epoch {epoch}")
        #     break

    def save_model(self):
        save_dir = f"runs/{self.run_name}/model"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict().copy(), f"{save_dir}/best_model.pt")
