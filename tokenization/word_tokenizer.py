from tokenization.tokenizer import Tokenizer
from collections import Counter

class WordTokenizer(Tokenizer):
    def __init__(self, dataset: list[str], vocab_size: int=10_000, max_length: int=256,
                 unknown_token: str='<UNK>', pad_token: str='<PAD', special_tokens=None):

        super().__init__(vocab_size, max_length, unknown_token, pad_token, special_tokens)

        self._build_vocab(dataset)


    def _build_vocab(self, dataset: list[str]):
        value_counts = Counter()
        for sequence in dataset:
            text = sequence.lower().split()

            value_counts.update(text)

        for word, _ in value_counts.most_common(self.vocab_size - len(self.vocab)):
            self.vocab[word] = len(self.vocab)


    def __call__(self, text: list[str]) -> list[list[int]]:
        tokens = []
        for sequence in text:
            tokens.append(self.tokenize(sequence))

        return tokens


    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab[self.unknown_token]))

        return tokens[:self.max_length]

    def get_pad_token(self):
        return self.vocab[self.pad_token]