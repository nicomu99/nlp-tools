from typing import List, Dict, Tuple

from tokenization.tokenizer import Tokenizer
from collections import Counter, defaultdict

class BPETokenizer(Tokenizer):
    def __init__(self, dataset: list[str], vocab_size: int=10_000, max_length: int=-1, num_merges: int=10,
                 unknown_token: str='<UNK>', pad_token: str='<PAD', special_tokens=None):

        super().__init__(vocab_size, max_length, unknown_token, pad_token, special_tokens)

        self.num_merges = num_merges
        self.bpe_merges = []
        self.eow_token = '</w'
        self._build_vocab(dataset)

    @staticmethod
    def _determine_next_merge(vocab: Dict[Tuple[str, ...], int]) -> Tuple[str, str]:
        """
        Determines the most frequent adjacent pair of symbols in the given vocabulary. Each entry in the vocabulary
        represents a word as a tuple of symbols (typically characters or merged tokens) and its associated frequency.
        This function counts how often each adjacent symbol pair occurs across all words and returns
        the most frequent pair.

        :param vocab: Dictionary where keys are tuples of strings (symbol sequences) and values are their frequencies.
        :return: The most frequent adjacent pair as a tuple of two strings.
        """
        pairs = defaultdict(int)
        for word, frequency in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += frequency
        return max(pairs, key=pairs.get)

    @staticmethod
    def _merge_new_pair(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]):
        """
        Merges all occurrences of a given pair of adjacent symbols in the vocabulary.

        For each word (represented as a tuple of symbols) in the input vocabulary, this function merges all
        adjacent instances of the specified pair into a single token and returns a new vocabulary reflecting
        these updates.

        :param pair: A tuple of strings determining which strings to merge in the input dictionary.
        :param vocab: Dictionary where keys are tuples of strings (symbol sequences) and values are their frequencies.
        :return: A new dictionary with each occurrence of pair merged.
        """
        new_vocab = dict()
        for word in vocab:
            new_word = list()
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = vocab[word]
        return new_vocab

    def _build_vocab(self, dataset: List[str]):
        """
        Creates a Byte Pair Encoding (BPE) vocabulary by learning merges based on the most frequent adjacent symbol pairs
        in the input dataset.

        The process begins by counting word frequencies and building an initial vocabulary of individual characters.
        Then, for a pre-defined number of merge operations, the function iteratively identifies the most frequent
        adjacent symbol pair across the vocabulary and merges it into a new token. Each merge operation is recorded
        and later used for tokenizing new input text.

        :param dataset: The input corpus as a list of strings.
        """
        # Get most common words in text corpus
        char_vocab = set()          # For efficient lookup
        value_counts = Counter()

        for sequence in dataset:
            # Initially, sequence is a string, e.g. 'Hello, my name is Jane Doe.'
            sequence = sequence.lower()
            words = sequence.split() # ['Hello,', 'my'...]
            for word in words:
                for char in word:
                    if char not in char_vocab:
                        char_vocab.add(char)
                        self.bpe_merges.append(char)
            value_counts.update(words)

        # Get word frequencies
        new_vocab = dict()
        for word, frequency in value_counts.most_common(self.vocab_size - len(self.vocab)):
            word = word + self.eow_token
            new_vocab[tuple(word)] = frequency     # e.g. new_vocab[('h', 'e', 'l', 'l', 'o', ',', '</w')] = 10

        for i in range(self.num_merges):
            best_pair = self._determine_next_merge(new_vocab)
            new_vocab = self._merge_new_pair(best_pair, new_vocab)
            self.bpe_merges.append(best_pair)

        for merge in self.bpe_merges:
            token = ''.join(merge)
            next_idx = len(self.vocab)
            self.vocab[token] = len(self.vocab)
            self.idx_to_word[next_idx] = token


    def __call__(self, text: List[str]) -> List[List[int]]:
        tokens = []
        for sequence in text:
            tokens.append(self.tokenize(sequence))

        return tokens


    def tokenize(self, text: str) -> list[int]:
        tokens = []
        text = text.lower()
        words = text.strip().split()
        for word in words:
            word_tokens = list(word) + [self.eow_token]
            for merge in self.bpe_merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge:
                        word_tokens[i:i + 2] = [''.join(merge)]
                        if i != 0:
                            i -= 1
                    else:
                        i += 1
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.vocab[self.unknown_token]))

        return tokens

    def get_pad_token(self) -> int:
        return self.vocab[self.pad_token]