import re

from tokenization.tokenizer import Tokenizer
from collections import Counter

class WordTokenizer(Tokenizer):
    def __init__(self, dataset: list[str], vocab_size: int=10_000, max_length: int=-1,
                 unknown_token: str='<UNK>', pad_token: str='<PAD', special_tokens=None):

        super().__init__(vocab_size, max_length, unknown_token, pad_token, special_tokens)

        self._build_vocab(dataset)

    def _build_vocab(self, dataset: list[str]):
        value_counts = Counter()
        for sequence in dataset:
            tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", sequence.lower())
            value_counts.update(tokens)

        for word, _ in value_counts.most_common(self.vocab_size - len(self.vocab)):
            next_idx = len(self.vocab)
            self.vocab[word] = next_idx
            self.idx_to_word[next_idx] = word

    def __call__(self, text: list[str]) -> list[list[int]]:
        tokens = []
        for sequence in text:
            tokens.append(self.tokenize(sequence))

        return tokens

    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for word in re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower()):
            tokens.append(self.vocab.get(word, self.vocab[self.unknown_token]))

        if self.max_length == -1:
            return tokens
        else:
            return tokens[:self.max_length]

    def get_pad_token(self) -> int:
        return self.vocab[self.pad_token]