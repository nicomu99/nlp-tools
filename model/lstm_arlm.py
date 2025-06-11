import torch
import torch.nn as nn

class LSTMForARLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 padding_index: int,
                 num_layers: int
                 ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings  = vocab_size,
            embedding_dim   = embedding_dim,
            padding_idx     = padding_index
        )

        self.lstm = nn.LSTM(
            input_size      = embedding_dim,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            bidirectional   = False,
            batch_first     = True
        )

        self.linear = nn.Linear(
            in_features = hidden_size,
            out_features = vocab_size
        )

    def forward(self, batch: dict):
        input_ids = batch['input_ids']

        hidden = self.embedding(input_ids)
        hidden, _ = self.lstm(hidden)

        return self.linear(hidden)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                batch = {'input_ids': generated}
                logits = self.forward(batch)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated