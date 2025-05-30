import os
import csv
from typing import Optional

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, BatchSampler, DataLoader

from datasets import Dataset as HFDataset

from tokenization.tokenizer import Tokenizer
from nlp_dataset.nlp_dataset import NLPDataset
from nlp_sampler.grouped_sampler import GroupedSampler
from data_utils.data_utils import *


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
                 early_stop_limit: int = 3
                 ):
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
        self.run_name = run_name
        self.metric_dir = f'out/{self.run_name}/metrics'
        self.model_dir = f'out/{self.run_name}/model'
        self.metric_file = f'{self.metric_dir}/metrics.csv'
        self.memory_dir = f'out/{self.run_name}/memory'
        self.memory_file = f'{self.memory_dir}/memory.csv'
        self.early_stop = early_stop
        self.early_stop_limit = early_stop_limit
        self.curr_step = 0      # Tracks how many steps have been trained

        self.train_dataset = self._prepare_dataset(train_dataset)
        self.train_dataloader = self._prepare_loader(self.train_dataset, add_batching=True)

        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = self._prepare_dataset(eval_dataset)
            self.eval_dataloader = self._prepare_loader(self.eval_dataset)

        self.test_dataset = test_dataset
        if self.test_dataset is not None:
            self.test_dataset = self._prepare_dataset(test_dataset)
            self.test_dataloader = self._prepare_loader(self.test_dataset)

    def _prepare_dataset(self, dataset: Dataset) -> NLPDataset:
        return NLPDataset(
            self.tokenizer(dataset['text']),
            dataset['label']
        )

    def _prepare_loader(self, dataset: NLPDataset, add_batching: bool = False) -> DataLoader:
        if add_batching:
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

        else:
            data_loader = DataLoader(
                dataset,
                collate_fn=collate_batch,
                shuffle=False,
                num_workers=self.data_loaders
            )

        return data_loader

    def process_one_epoch(self,
                          loader: torch.utils.data.DataLoader,
                          optimizer: Optional[torch.optim.Optimizer] = None
                          ) -> Tuple[float, float, float, dict]:
        """
        Processes one training or validation step for a given model.
        :param loader: The data loader providing batches of input sequences and labels.
        :param optimizer: The optimizer used for training. If None, the function runs in evaluation mode.
        :return: tuple: (avg_epoch_loss, accuracy)
                - avg_epoch_loss (float): The average loss over all batches in the epoch.
                - accuracy (float): The classification accuracy over all samples.
        """
        train = optimizer is not None
        self.model.train() if train else self.model.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        processed_labels = []
        processed_predictions = []

        # context = torch.enable_grad() if train else torch.no_grad()
        # with context:

        allocated_mem       = []
        max_allocated_mem   = []
        reserved_mem        = []
        max_reserved_mem    = []
        steps                = []
        for input_ids, labels, lengths in tqdm(loader, desc="Training" if train else "Validation"):
            torch.cuda.reset_peak_memory_stats()

            input_ids   = input_ids.to(self.device)
            labels      = labels.to(self.device)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                preds = self.model(input_ids, lengths)
                loss = self.criterion(preds, labels.float())

                if train:
                    loss.backward()
                    optimizer.step()

            pred_labels = (preds >= 0.5).int()
            # print(preds)
            correct_predictions += (pred_labels == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            processed_labels.extend(labels.cpu().tolist())
            processed_predictions.extend(pred_labels.cpu().tolist())

            allocated_mem.append(torch.cuda.memory_allocated())
            max_allocated_mem.append(torch.cuda.max_memory_allocated())
            reserved_mem.append(torch.cuda.memory_reserved())
            max_reserved_mem.append(torch.cuda.max_memory_reserved())
            steps.append(self.curr_step)
            self.curr_step += 1

        avg_epoch_loss = total_loss / total_predictions
        accuracy = correct_predictions / total_predictions
        f1 = f1_score(processed_predictions, processed_labels)

        memory_stats = {
            'steps': steps,
            'alloc_mem': allocated_mem,
            'max_alloc_mem': max_allocated_mem,
            'reserved_mem': reserved_mem,
            'max_reserved_mem': max_reserved_mem
        }

        return avg_epoch_loss, accuracy, f1, memory_stats

    def train(self):
        # Create empty file for metric saving
        self._create_new_metric_saving_file()

        early_stop_epoch = 0
        best_f1 = 0
        for epoch in range(self.num_epochs):
            train_loss, train_acc, train_f1, train_mem = self.process_one_epoch(self.train_dataloader, self.optimizer)

            with torch.no_grad():
                val_loss, val_acc, val_f1, val_mem = self.process_one_epoch(self.eval_dataloader)

            self._save_epoch_metrics(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)
            self._save_memory_stats(train_mem, val_mem)

            if val_f1 > best_f1:
                # Save best model performance
                best_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, f"{self.model_dir}/best_model.pt")

            # Early stopping if 3 consecutive epochs are below the highest F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                early_stop_epoch = 0
            else:
                early_stop_epoch += 1

            if early_stop_epoch >= self.early_stop_limit and self.early_stop:
                print(f"Training stopped early after epoch {epoch}")
                break

    def _create_new_metric_saving_file(self):
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

        # Initialize empty metric saving file with
        headers = ['epoch', 'train_loss', 'train_accuracy', 'train_f1', 'val_loss', 'val_accuracy', 'val_f1']
        with open(self.metric_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        mem_headers = ['steps', 'train_alloc', 'train_max_alloc', 'train_res', 'train_max_res',
                       'val_alloc', 'val_max_alloc', 'val_res', 'val_max_res']
        with open(self.memory_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(mem_headers)

    def _save_epoch_metrics(self, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
        metric_row = [epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1]
        with open(self.metric_file, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(metric_row)

    def _save_memory_stats(self, train_mem_stats, val_mem_stats):
        for i in range(len(train_mem_stats['steps'])):
            row = [train_mem_stats['steps'][i],
                   train_mem_stats['alloc_mem'][i],
                   train_mem_stats['max_alloc_mem'][i],
                   train_mem_stats['reserved_mem'][i],
                   train_mem_stats['max_reserved_mem'][i],
                   val_mem_stats['alloc_mem'][i],
                   val_mem_stats['max_alloc_mem'][i],
                   val_mem_stats['reserved_mem'][i],
                   val_mem_stats['max_reserved_mem'][i]
                   ]

            with open(self.memory_file, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)