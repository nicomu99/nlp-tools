import torch.nn as nn

class FFNetwork(nn.Module):
    def __init__(self, embedding_dim, forward_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, embedding_dim)
        )
    def forward(self, x):
        return self.ffn(x)