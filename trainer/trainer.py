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
                 train_dataset: HFDataset,
                 criterion:  nn.Module,
                 optimizer: torch.optim.Optimizer = None,
                 eval_dataset: HFDataset = None,
                 test_dataset: HFDataset = None,
                 num_epochs: int = 3,
                 max_seq_length: int = 256,
                 batch_size: int = 32,
                 data_loaders: int = 2,
                 learning_rate: float = 1e-6,
                 ):
        if train_dataset is None:
            raise RuntimeError("Train dataset is missing.")

        if tokenizer is None:
            raise RuntimeError("Tokenizer is missing.")

        if criterion is None:
            criterion = optim.Adam(model.parameters(), lr=learning_rate)

        if optimizer is None:
            raise RuntimeError("Optimizer is missing.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_loaders = data_loaders

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
                          ) -> Tuple[float, float, float]:
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

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for input_ids, labels, lengths in tqdm(loader, desc="Training" if train else "Validation"):
                input_ids   = input_ids.to(self.device)
                labels      = labels.float().to(self.device)
                lengths     = lengths.to(self.device)

                if train:
                    optimizer.zero_grad()

                preds = self.model(input_ids, lengths)
                loss = self.criterion(preds, labels)

                pred_labels = (preds >= 0.5).int()
                correct_predictions += (pred_labels == labels).sum().item()
                total_predictions += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                processed_labels.extend(labels.cpu().numpy())
                processed_predictions.extend(preds.cpu().numpy())

        avg_epoch_loss = total_loss / total_predictions
        accuracy = correct_predictions / total_predictions
        f1 = f1_score(processed_predictions, processed_labels)

        return avg_epoch_loss, accuracy, f1


    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.process_one_epoch(self.train_dataloader, self.optimizer)

            val_loss, val_acc = self.process_one_epoch(self.eval_dataloader)


            # Calculate regualrized f1 score given the predefined function
            reg_f1_score = regularized_f1(train_f1, val_f1)
            if reg_f1_score > best_f1_score:
                # Save best model performance
                best_f1_score = reg_f1_score
                best_config = index
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, f"{path}best_model.pt")

            # Early stopping if 3 consecutive epochs are below the highest F1 score
            if val_f1 > best_config_f1:
                best_config_f1 = val_f1
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 3:
                print(f"Configuration {index} stopped at epoch {epoch}.")
                break


