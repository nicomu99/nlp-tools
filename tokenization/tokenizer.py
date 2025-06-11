from typing import List

import torch

class Tokenizer:
    def __init__(self, vocab_size: int=10_000, max_length: int=256,
                 unknown_token: str='<UNK>', pad_token: str='<PAD>',
                 special_tokens=None):
        if special_tokens is None:
            special_tokens = []
        self.unknown_token = unknown_token
        self.pad_token = pad_token

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.special_tokens = [self.unknown_token, self.pad_token] + special_tokens
        self.vocab = {}
        self.idx_to_word = {}

        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.idx_to_word[idx] = token

    def _build_vocab(self, dataset: List[str]):
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def __call__(self, text: List[str]) -> List[List[int]]:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def get_pad_token(self) -> int:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def __len__(self):
        return len(self.vocab)

    def decode(self, input_ids: torch.Tensor) -> str:
        input_ids = input_ids.tolist()
        output = ''
        for idx in input_ids:
            output += self.idx_to_word[idx]
        return output