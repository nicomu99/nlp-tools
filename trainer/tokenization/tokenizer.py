"""Tokenizer base module."""
import torch


class Tokenizer:
    """Base class for tokenizers.

    This class defines a minimal interface for converting raw text into token
    IDs and decoding token IDs back into text. It also initializes a vocabulary
    with special tokens and provides helper utilities used by concrete
    tokenizer implementations.

    Attributes:
        unknown_token (str): String used to represent out-of-vocabulary tokens.
        pad_token (str): String used for padding up to a fixed sequence length.
        vocab_size (int): Target maximum vocabulary size for the tokenizer.
        max_length (int): Maximum sequence length produced by the tokenizer.
        special_tokens (list[str]): List of special tokens. By default, this
            contains `[unknown_token, pad_token]` plus any user-provided tokens.
        vocab (dict[str, int]): Mapping from token string to integer ID.
        idx_to_word (dict[int, str]): Reverse mapping from integer ID to token
            string.
    """

    def __init__(
            self,
            vocab_size: int = 10_000,
            max_length: int = 256,
            unknown_token: str = "<UNK>",
            pad_token: str = "<PAD>",
            special_tokens: list | None = None
    ):
        """Initializes the tokenizer with special tokens and an empty vocabulary.

        Args:
            vocab_size (int): Maximum vocabulary size.
            max_length (int): Maximum sequence length produced by the tokenizer.
            unknown_token (str): Token used to represent unknown or
                out-of-vocabulary tokens.
            pad_token (str): Token used to pad sequences to a uniform length.
            special_tokens (list | None): Additional special tokens to reserve
                IDs for. If `None`, no extra tokens are added.
        """
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

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unknown_token]

    def _build_vocab(self, dataset: list[str]):
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def __call__(self, text: str) -> dict:
        """Tokenizes/encodes a single input string.

        Subclasses must implement this method. The return value should be a
        dictionary containing tokenization results (commonly at least
        `input_ids`). Many implementations also include fields like
        `attention_mask`, `token_type_ids`, or other model-specific inputs.

        Args:
            text (str): Input text to encode.

        Returns:
            dict: A dictionary of encoded outputs.

        Raises:
            NotImplementedError: Always raised for the base class.
        """
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def __len__(self):
        """Returns the current vocabulary size.

        Returns:
            int: Number of tokens currently present in `self.vocab`.
        """
        return len(self.vocab)

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decodes token IDs back into a string.

        This method converts `input_ids` into the corresponding string tokens.

        Args:
            input_ids (torch.Tensor): A 1D tensor of token IDs.

        Returns:
            str: Decoded string formed by concatenating decoded tokens.

        Notes:
            - If an ID is not present in `idx_to_word`, this method inserts the unknown token.
        """
        input_ids = input_ids.tolist()
        output = " ".join([
            self.idx_to_word.get(idx, self.unk_token_id)
            for idx in input_ids
        ])
        return output
