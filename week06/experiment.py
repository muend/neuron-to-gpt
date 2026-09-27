"""Shared data, model builders, training and evaluation for Week 6."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from layers import BatchNorm1d, Embedding, Flatten, Linear, Sequential, Tanh

ROOT = Path(__file__).resolve().parent.parent
ALPHABETS = {
    "en": "abcdefghijklmnopqrstuvwxyz",
    "tr": "abcçdefgğhıijklmnoöprsştuüvyz",
}


def load_splits(language: str, context: int) -> tuple:
    path = ROOT / "week03" / "data" / f"names_{language}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run week03/veri_hazirla.py first")
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(42).shuffle(words)  # Same split as Week 4.
    n1, n2 = int(.8 * len(words)), int(.9 * len(words))
    stoi = {char: i for i, char in enumerate("." + ALPHABETS[language])}
    itos = {i: char for char, i in stoi.items()}

    def dataset(items):
        xs, ys = [], []
        for word in items:
            window = [0] * context
            for char in word + ".":
                target = stoi[char]
                xs.append(window)
                ys.append(target)
                window = window[1:] + [target]
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    return dataset(words[:n1]), dataset(words[n1:n2]), itos, (n1, n2-n1, len(words)-n2)


def make_model(vocab: int, *, context: int, embedding: int, hidden: int,
               hierarchical: bool = False, incorrect_bn: bool = False,
               seed: int = 2147483647) -> Sequential:
    generator = torch.Generator().manual_seed(seed)
    layers = [Embedding(vocab, embedding, generator=generator)]
    if hierarchical:
        if context != 8:
            raise ValueError("The three-stage hierarchy requires context=8")
        channels = embedding
        for _ in range(3):
            layers.extend([
                Flatten(2), Linear(2 * channels, hidden, bias=False, generator=generator),
                BatchNorm1d(hidden, incorrect_3d=incorrect_bn), Tanh(),
            ])
            channels = hidden
        layers.append(Linear(hidden, vocab, generator=generator))
    else:
        layers.extend([
            Flatten(context), Linear(context * embedding, hidden, bias=True, generator=generator),
            BatchNorm1d(hidden), Tanh(), Linear(hidden, vocab, generator=generator),
        ])
    with torch.no_grad():
        if hierarchical:
            layers[-1].weight.mul_(0.1)
        else:
            # Match the Week 4 MLP initialization: tanh gain and small logits.
            layers[2].weight.mul_(5 / 3)
            layers[-1].weight.mul_(0.01 * hidden**0.5)
    return Sequential(layers)


def count_parameters(model: Sequential) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: Sequential, data: tuple, *, steps: int, lr: float = 0.1,
          batch_size: int = 32, seed: int = 2147483647) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    x, y = data
    generator = torch.Generator().manual_seed(seed)
    parameters = model.parameters()
    model.train()
    log_losses = []
    for step in range(steps):
        indices = torch.randint(len(x), (batch_size,), generator=generator)
        loss = F.cross_entropy(model(x[indices]), y[indices])
        for parameter in parameters:
            parameter.grad = None
        loss.backward()
        rate = lr if step < int(steps * .75) else lr * .1
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(parameter.grad, alpha=-rate)
        log_losses.append(float(loss.log10().item()))
    return log_losses


@torch.no_grad()
def evaluate(model: Sequential, data: tuple, batch_size: int = 4096) -> float:
    x, y = data
    model.eval()
    total = 0.0
    for start in range(0, len(x), batch_size):
        total += float(F.cross_entropy(model(x[start:start+batch_size]),
                                       y[start:start+batch_size], reduction="sum"))
    return total / len(x)


@torch.no_grad()
def sample(model: Sequential, itos: dict[int, str], context: int, *,
           count: int = 12, seed: int = 42, max_length: int = 24) -> list[str]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    names = []
    for _ in range(count):
        window = [0] * context
        letters = []
        for _ in range(max_length):
            probs = model(torch.tensor([window])).softmax(1)
            index = int(torch.multinomial(probs, 1, generator=generator))
            if index == 0:
                break
            letters.append(itos[index])
            window = window[1:] + [index]
        names.append("".join(letters))
    return names


def shape_trace(model: Sequential, context: int, batch_size: int = 4) -> list[tuple[str, tuple[int, ...]]]:
    model.eval()
    x = torch.zeros(batch_size, context, dtype=torch.long)
    result = [("input", tuple(x.shape))]
    with torch.no_grad():
        for layer in model.layers:
            x = layer(x)
            result.append((type(layer).__name__, tuple(x.shape)))
    return result
