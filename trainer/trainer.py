from torch.utils.data import Dataset, BatchSampler, DataLoader
import torch.nn as nn

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
                 eval_dataset: HFDataset = None,
                 test_dataset: HFDataset = None,
                 num_epochs: int = 3,
                 max_seq_length: int = 256,
                 batch_size: int = 32,
                 data_loaders: int = 2
                 ):
        if train_dataset is None:
            raise RuntimeError("Train dataset is missing.")

        if tokenizer is None:
            raise RuntimeError("Tokenizer is missing.")

        self.model = model
        self.tokenizer = tokenizer
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

        self.test_dataset = train_dataset
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



    def train(self):
        pass


