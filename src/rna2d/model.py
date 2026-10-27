"""Static comparison baseline. No Ghost or RWKV contribution is claimed."""
import torch
from torch import nn


class PairBaseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(5, hidden, padding_idx=4)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.pair_bias = nn.Parameter(torch.tensor(-2.0))
        self.distance = nn.Embedding(32, 1)
        nn.init.zeros_(self.distance.weight)

    def encode(self, tokens):
        valid = tokens.ne(4).unsqueeze(-1)
        h = self.embedding(tokens) * valid
        # Mask after each convolution so padding cannot leak through a second layer.
        x = h.transpose(1, 2)
        for layer in self.encoder:
            x = layer(x) * valid.transpose(1, 2)
        return x.transpose(1, 2)

    def score(self, h):
        q = self.query(h)
        logits = torch.matmul(q, h.transpose(-1, -2)) / h.shape[-1] ** 0.5
        logits = (logits + logits.transpose(-1, -2)) * 0.5
        pos = torch.arange(h.shape[1], device=h.device)
        bins = torch.floor(torch.log2((pos[:, None] - pos[None, :]).abs().float() + 1) * 4).long().clamp_max(31)
        return logits + self.distance(bins).squeeze(-1) + self.pair_bias

    def forward(self,tokens):
        return self.score(self.encode(tokens))
