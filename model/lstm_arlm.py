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

    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            beam_width: int = 1
    ) -> torch.Tensor:
        """
        A simple text generation function. The default behavior is greedy search.
        If beam_width > 1, beam search is used.
        :param input_ids:
        :param max_new_tokens:
        :param beam_width:
        :return:
        """
        self.eval()

        generated = input_ids.clone()
        candidates = [(generated, 0) for _ in range(beam_width)]
        for _ in range(max_new_tokens):
            new_candidates = []
            for (inputs, score) in candidates:
                batch = {'input_ids': inputs}

                with torch.no_grad():
                    logits = self.forward(batch)

                next_token_logits = logits[:, -1, :]

                values, indices = torch.topk(next_token_logits, k=beam_width, dim=-1)
                values = values.squeeze(0)
                indices = indices.squeeze(0)

                for i in range(len(values)):
                    new_candidates.append((torch.cat([inputs, indices[i].unsqueeze(0).unsqueeze(0)], dim=1), score + values[i]))
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_width]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]