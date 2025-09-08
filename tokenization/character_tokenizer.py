from typing import List

from tqdm import tqdm

from tokenization.tokenizer import Tokenizer
from collections import Counter

class CharacterTokenizer(Tokenizer):
    def __init__(self, dataset: list[str], vocab_size: int=10_000, max_length: int=-1,
                 unknown_token: str='<UNK>', pad_token: str='<PAD', special_tokens=None):

        super().__init__(vocab_size, max_length, unknown_token, pad_token, special_tokens)

        self._build_vocab(dataset)


    def _build_vocab(self, dataset: list[str]):
        value_counts = Counter()
        for sequence in tqdm(dataset):
            for character in sequence:
                value_counts.update(character)

        for character, _ in value_counts.most_common(self.vocab_size - len(self.vocab)):
            next_idx = len(self.vocab)
            self.vocab[character] = len(self.vocab)
            self.idx_to_word[next_idx] = character


    def __call__(self, text: List[str]) -> List[List[int]]:
        tokens = []
        for sequence in text:
            tokens.append(self.tokenize(sequence))

        return tokens


    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for character in text:
            tokens.append(self.vocab.get(character, self.vocab[self.unknown_token]))

        return tokens

    def get_pad_token(self) -> int:
        return self.vocab[self.pad_token]