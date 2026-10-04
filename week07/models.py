"""A bigram baseline and ONE causal self-attention head, not a full GPT."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CharacterLanguageModel(nn.Module):
    @torch.no_grad()
    def generate(self, indices, max_new_tokens: int, generator=None):
        previous_mode = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Positions and the causal-mask buffer only cover block_size.
                context = indices[:, -self.block_size:]
                logits, _ = self(context)
                probabilities = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
                indices = torch.cat((indices, next_token), dim=1)
            return indices
        finally:
            self.train(previous_mode)


class BigramLanguageModel(CharacterLanguageModel):
    def __init__(self, vocab_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets=None):
        logits = self.token_embedding(indices)
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


class AttentionHead(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x, return_weights=False):
        time = x.shape[1]
        if time > self.tril.shape[0]:
            raise ValueError("Input exceeds the causal mask's block_size")
        keys, queries, values = self.key(x), self.query(x), self.value(x)
        scores = (queries @ keys.transpose(-2, -1)) * self.head_size ** -0.5
        scores = scores.masked_fill(~self.tril[:time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = weights @ values
        return (output, weights) if return_weights else output


class SingleHeadLanguageModel(CharacterLanguageModel):
    def __init__(self, vocab_size: int, block_size: int, embedding_size=64, head_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.head = AttentionHead(embedding_size, head_size, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def embeddings(self, indices):
        time = indices.shape[1]
        if time > self.block_size:
            raise ValueError("Crop the context to block_size before calling the model")
        positions = torch.arange(time, device=indices.device)
        return self.token_embedding(indices) + self.position_embedding(positions)

    def forward(self, indices, targets=None):
        logits = self.lm_head(self.head(self.embeddings(indices)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def attention_weights(self, indices):
        _, weights = self.head(self.embeddings(indices), return_weights=True)
        return weights
