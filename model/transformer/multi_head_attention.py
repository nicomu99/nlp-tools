import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_heads):
        super().__init__()
        self.attention_heads: int = attention_heads
        self.embedding_dim: int = embedding_dim
        self.head_dim = self.embedding_dim // self.attention_heads

        self.w_q = self.Linear(embedding_dim, self.attention_heads * self.head_dim)
        self.w_k = self.Linear(embedding_dim, self.attention_heads * self.head_dim)
        self.w_v = self.Linear(embedding_dim, self.attention_heads * self.head_dim)

        self.out_layer = nn.Linear(self.attention_heads * self.head_dim, self.embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # Input has [batch, seq_len, em_dim]
        batch_size, seq_len = q.shape[0], q.shape[1]

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Each project must be reshaped to fit (batch_size, seq_len, num_heads, head_dim)
        q = q.reshape((batch_size, seq_len, self.attention_heads, self.head_dim)).transpose(1, 2)
        k = k.reshape((batch_size, seq_len, self.attention_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape((batch_size, seq_len, self.attention_heads, self.head_dim)).transpose(1, 2)
        # They now have shape (batch_size, attn_heads, seq_len, head_dim)

        key_out = q @ k.transpose(3, 2)
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        out = key_out / (self.head_dim ** (1/2))

        # softmax is (batch, num_heads, query_seq_len, key_seq_len), v is (batch, num_heads, seq_len, head_dim)
        # pytorch treats the last two as matrices:
        # output: (batch, heads, seq_len, head_dim)
        out = self.softmax(out) @ v
        # reshape to (batch, seq_len, heads, head_dim) first and then reunite heads
        # to (batch, seq_len, em_dim)
        out = out.transpose(2, 1).reshape((batch_size, seq_len, self.embedding_dim))

        return self.linear(out)
