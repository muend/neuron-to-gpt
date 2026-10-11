"""Build a decoder-only Transformer in transparent, independently written steps."""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Reuse the real Week 7 baselines rather than subtly changing their definitions.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from week07.models import BigramLanguageModel, SingleHeadLanguageModel, CharacterLanguageModel


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        time = x.shape[1]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = (q @ k.transpose(-2, -1)) * self.head_size ** -.5
        weights = F.softmax(scores.masked_fill(~self.mask[:time, :time], float("-inf")), dim=-1)
        return self.dropout(weights) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config.n_embd, config.n_embd // config.n_head,
                                                 config.block_size, config.dropout)
                                    for _ in range(config.n_head)])
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Each small head learns its own matching rule; total channel width stays C.
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(concatenated))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config.n_embd, config.dropout)
        self.residual = config.stage in ("residual", "layernorm")
        self.norm1 = nn.LayerNorm(config.n_embd) if config.stage == "layernorm" else nn.Identity()
        self.norm2 = nn.LayerNorm(config.n_embd) if config.stage == "layernorm" else nn.Identity()

    def forward(self, x):
        attended = self.attention(self.norm1(x))
        x = x + attended if self.residual else attended
        computed = self.feedforward(self.norm2(x))
        return x + computed if self.residual else computed


class TransformerLanguageModel(CharacterLanguageModel):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        if config.stage == "multi_head":
            self.blocks = nn.Sequential(MultiHeadAttention(config))
        else:
            self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd) if config.stage == "layernorm" else nn.Identity()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        if config.initialize_small:
            self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, indices, targets=None):
        time = indices.shape[1]
        if time > self.block_size:
            raise ValueError("Input exceeds block_size; crop the generation context")
        positions = torch.arange(time, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def make_model(config):
    if config.stage == "bigram":
        return BigramLanguageModel(config.vocab_size, config.block_size)
    if config.stage == "single_head":
        return SingleHeadLanguageModel(config.vocab_size, config.block_size, config.n_embd, config.n_embd)
    return TransformerLanguageModel(config)
