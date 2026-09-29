import os
from typing import Optional

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from datasets import Dataset as HFDataset

from .tokenization import Tokenizer
from .tasks import ClassificationTask, ARLMTask
from .utils import collate_batch
from .logger import Logger


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
            train_dataloader: DataLoader = None,
            optimizer: Optimizer = None,
            num_epochs: int = 3,
            train_batch_size: int = 32,
            eval_batch_size: int = 1,
            data_loaders: int = 2,
            learning_rate: float = 1e-4,
            run_name: str = "",
            early_stop: bool = False,
            early_stop_limit: int = 3,
            block_size: Optional[int] = 128
    ):
        self.run_name = run_name
        self.logger = Logger(run_name)

        if train_dataset is None:
            raise RuntimeError("Train dataset is missing.")

        if tokenizer is None:
            raise RuntimeError("Tokenizer is missing.")

        if criterion is None:
            raise RuntimeError("Criterion is missing.")

        if optimizer is None:
            optimizer = Adam(model.parameters(), lr=learning_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_loaders = data_loaders
        self.block_size = block_size  # For ARLM

        # Task is used control aspects of the training
        if task == "classification":
            self.task = ClassificationTask(self.criterion)  # TODO: Fix typing in task
        elif task == "arlm":
            self.task = ARLMTask(self.criterion)  # TODO: Fix typing in task
        else:
            raise ValuError(f"Unknown task identifier {task}. Must be \"classification\" or \"arlm\".")

        self.early_stop = early_stop
        self.early_stop_limit = early_stop_limit

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.eval_dataloader = None
        # self.task.get_data_loader(self.tokenizer, train_dataset, batch_size, data_loaders)
        # if eval_dataset is not None:
        #     self.eval_dataloader = self.task.get_data_loader(self.tokenizer, eval_dataset, batch_size, data_loaders,
        #                                                      "eval")
        #
        # if test_dataset is not None:
        #     self.test_dataloader = self.task.get_data_loader(self.tokenizer, eval_dataset, batch_size, data_loaders,
        #                                                   "test")

    def _prepare_loader(self, dataset: Dataset, batch_sampling: bool = False) -> DataLoader:
        if batch_sampling:
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

    def _prepare_for_training(self):
        # Tokenize the inputs
        self.train_dataset = self.tokenize(self.train_dataset)
        self.eval_dataset = self.tokenize(self.eval_dataset)
        # self.test_dataset = self.tokenize(self.test_dataset)

        # Create dataloaders
        kwargs = {"block_size": self.block_size}
        train_dataset = self.task.get_dataset(self.train_dataset, **kwargs)
        eval_dataset = self.task.get_dataset(self.eval_dataset, **kwargs)

        if self.train_dataloader is None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                collate_fn=collate_batch,
                num_workers=self.data_loaders
            )

        if self.eval_dataloader is None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.eval_batch_size,
                collate_fn=collate_batch,
                num_workers=self.data_loaders
            )

    def process_one_epoch(
            self,
            loader: DataLoader,
            optimizer: Optimizer | None = None,
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
        for batch in loader:
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
        self.logger.log_epoch(stage, self.current_epoch, metric_dir)

    def tokenize_row(self, features: str):
        return self.tokenizer(features)

    def tokenize(self, dataset: HFDataset):
        if "text" not in dataset.column_names:
            raise ValueError("Invalid dataset format, could not find column \"text\"")
        if "input_ids" in dataset.column_names:
            return dataset

        dataset = dataset.map(lambda examples: self.tokenize_row(examples["text"]))
        return dataset

    def train(self):
        self._prepare_for_training()

        # early_stop_epoch = 0
        # best_f1 = 0
        for _ in tqdm(range(self.num_epochs)):
            self.current_epoch += 1
            # print(f"Epoch {self.current_epoch} - Training")

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
