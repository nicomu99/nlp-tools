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

        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx

    def _build_vocab(self, dataset: list[str]):
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def __call__(self, text: list[str]) -> list[list[int]]:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")

    def get_pad_token(self) -> int:
        raise NotImplementedError("Tokenizer should not be instantiated. Please use a subclass.")