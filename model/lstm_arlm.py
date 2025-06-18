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

    def _generate_log_probs(
            self,
            input_ids: torch.Tensor,
            top_k: int = 1
    ):
        batch = {'input_ids': input_ids}

        with torch.no_grad():
            logits = self.forward(batch)

        next_token_logits = logits[:, -1, :]
        logs = torch.log_softmax(next_token_logits, dim=-1)

        values, indices = torch.topk(logs, k=top_k, dim=-1)
        return values, indices

    def generate_beam_search(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            beam_width: int = 1
            ) -> torch.Tensor:
        generated = input_ids.clone()
        candidates = [(generated, 0) for _ in range(beam_width)]
        for _ in range(max_new_tokens):
            new_candidates = []
            for (inputs, score) in candidates:
                values, indices = self._generate_log_probs(inputs, beam_width)
                values = values.squeeze(0)
                indices = indices.squeeze(0)

                for i in range(len(values)):
                    new_candidates.append(
                        (torch.cat([inputs, indices[i].unsqueeze(0).unsqueeze(0)], dim=1), score + values[i]))
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_width]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def generate_topk(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            top_k: int = 2
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            topk_logs, topk_indices = self._generate_log_probs(generated, top_k)

            topk_probs = torch.exp(topk_logs)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(topk_probs, num_samples=1)
            next_token_id = topk_indices.gather(dim=1, index=next_token)

            generated = torch.cat([generated, next_token_id], dim=1)

        return generated

    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            beam_width: int = 1,
            top_k: int = 0
    ) -> torch.Tensor:
        """
        A simple text generation function. The default behavior is greedy search.
        If beam_width > 1, beam search is used.
        :param top_k:
        :param input_ids:
        :param max_new_tokens:
        :param beam_width:
        :return:
        """
        self.eval()

        if top_k == 0:
            return self.generate_beam_search(input_ids, max_new_tokens, beam_width)

        return self.generate_topk(input_ids, max_new_tokens, top_k)

