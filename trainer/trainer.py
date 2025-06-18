from typing import Optional

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, BatchSampler, DataLoader

from datasets import Dataset as HFDataset

from logger import ClassificationLogger, ARLMLogger
from tokenization import Tokenizer
from nlp_dataset import NLPDataset, NLPClassDataset, NLPARLMDataset
from nlp_sampler import GroupedSampler
from data_utils import *


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 tokenizer: Tokenizer,
                 criterion:  nn.Module,
                 train_dataset: HFDataset,
                 eval_dataset: HFDataset = None,
                 test_dataset: HFDataset = None,
                 optimizer: torch.optim.Optimizer = None,
                 num_epochs: int = 3,
                 batch_size: int = 32,
                 data_loaders: int = 2,
                 learning_rate: float = 1e-4,
                 run_name: str = '',
                 early_stop: bool = False,
                 early_stop_limit: int = 3,
                 task: str = '',
                 block_size: Optional[int] = 128
                 ):

        # Task is used control aspects of the training
        self.run_name = run_name
        self.task = task
        if self.task == 'classification':
            self.logger = ClassificationLogger(self.run_name)
        elif self.task == 'arlm':
            self.logger = ARLMLogger(self.run_name)
        else:
            raise ValueError(f"Task {self.task} unknown. Must be 'classification' or 'arlm'")

        if train_dataset is None:
            raise RuntimeError("Train dataset is missing.")

        if tokenizer is None:
            raise RuntimeError("Tokenizer is missing.")

        if criterion is None:
            raise RuntimeError("Criterion is missing.")

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_loaders = data_loaders
        self.block_size = block_size        # For ARLM

        self.early_stop = early_stop
        self.early_stop_limit = early_stop_limit

        self.train_dataset = self._prepare_dataset(train_dataset)
        self.train_dataloader = self._prepare_loader(self.train_dataset, batch_sampling=True)

        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = self._prepare_dataset(eval_dataset)
            self.eval_dataloader = self._prepare_loader(self.eval_dataset)

        self.test_dataset = test_dataset
        if self.test_dataset is not None:
            self.test_dataset = self._prepare_dataset(test_dataset)
            self.test_dataloader = self._prepare_loader(self.test_dataset)

    def _prepare_dataset(self, dataset: HFDataset) -> NLPDataset:
        if self.task == 'classification':
            return NLPClassDataset(
                self.tokenizer(dataset['text']),
                dataset['label']
            )
        else:
            return NLPARLMDataset(
                self.tokenizer(dataset['text']),
                block_size=self.block_size
            )

    def _prepare_loader(self, dataset: Dataset, batch_sampling: bool = False) -> DataLoader:
        if batch_sampling and self.task == 'classification':
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

    def process_one_epoch(self,
                          loader: torch.utils.data.DataLoader,
                          optimizer: Optional[torch.optim.Optimizer] = None,
                          stage: str = ''
                          ):
        """
        Processes one training or validation step for a given model.
        :param loader: The data loader providing batches of input sequences and labels.
        :param optimizer: The optimizer used for training. If None, the function runs in evaluation mode.
        :param stage: A string representing the current training stage. Can be 'train', 'eval' or 'test'.
        """
        train = optimizer is not None
        self.model.train() if train else self.model.eval()

        self.logger.init_epoch()
        for batch in tqdm(loader, desc="Training" if train else "Validation"):
            torch.cuda.reset_peak_memory_stats()

            batch = {k: v.to(self.device) for k, v in batch.items()}

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = self.model(batch)
                # loss = self.criterion(outputs, batch['targets'].float()) # Classification
                loss = self.criterion(outputs, batch['targets'])

                if train:
                    loss.backward()
                    optimizer.step()

            self.logger.update(outputs, batch['targets'], loss)

        self.logger.log_epoch(stage)

    def train(self):
        # Create empty file for metric saving
        self.logger.create_folders_and_files()

        # early_stop_epoch = 0
        # best_f1 = 0
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1} - Training')

            self.process_one_epoch(self.train_dataloader, self.optimizer, stage='train')

            with torch.no_grad():
                self.process_one_epoch(self.eval_dataloader, stage='eval')

            self.logger.update_epoch()

        self.logger.save_model(self.model.state_dict().copy())
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