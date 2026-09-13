from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BOUNDARY = "."
EN_LETTERS = "abcdefghijklmnopqrstuvwxyz"
TR_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"
HERE = Path(__file__).resolve().parent


def make_vocab(letters: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [BOUNDARY] + list(letters)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def load_words(language: str) -> List[str]:
    path = HERE.parent / "week03" / "data" / f"names_{language}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce 'python week03/veri_hazirla.py' çalıştır."
        )
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_words(
    words: Sequence[str], seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n1 = int(0.8 * len(shuffled))
    n2 = int(0.9 * len(shuffled))
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]


def build_dataset(
    words: Sequence[str], stoi: Dict[str, int], block_size: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[int] = []
    for word in words:
        context = [0] * block_size
        for ch in word + BOUNDARY:
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


class CharMLP(nn.Module):
    """Bengio tarzı karakter MLP'si; BatchNorm bilinçli olarak elle yazıldı."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 2,
        n_hidden: int = 100,
        block_size: int = 3,
        batchnorm: bool = False,
        init: str = "good",
        seed: int = 2147483647,
    ) -> None:
        super().__init__()
        if init not in {"naive", "good"}:
            raise ValueError("init, 'naive' veya 'good' olmalı")

        self.block_size = block_size
        self.n_embd = n_embd
        self.batchnorm = batchnorm
        self.eps = 1e-5
        self.momentum = 0.001
        g = torch.Generator().manual_seed(seed)
        fan_in = block_size * n_embd

        w1_scale = 1.0 if init == "naive" else (5 / 3) / math.sqrt(fan_in)
        w2_scale = 1.0 if init == "naive" else 0.01

        self.C = nn.Parameter(torch.randn((vocab_size, n_embd), generator=g))
        self.W1 = nn.Parameter(
            torch.randn((fan_in, n_hidden), generator=g) * w1_scale
        )
        self.b1 = nn.Parameter(torch.zeros(n_hidden))
        self.W2 = nn.Parameter(
            torch.randn((n_hidden, vocab_size), generator=g) * w2_scale
        )
        self.b2 = nn.Parameter(torch.zeros(vocab_size))

        if batchnorm:
            self.bn_gain = nn.Parameter(torch.ones(n_hidden))
            self.bn_bias = nn.Parameter(torch.zeros(n_hidden))
            self.register_buffer("running_mean", torch.zeros(n_hidden))
            self.register_buffer("running_var", torch.ones(n_hidden))

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.C[x]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ self.W1 + self.b1

        if self.batchnorm:
            if self.training:
                mean = hpreact.mean(0)
                var = hpreact.var(0, unbiased=False)
                with torch.no_grad():
                    self.running_mean.lerp_(mean, self.momentum)
                    self.running_var.lerp_(var, self.momentum)
            else:
                mean = self.running_mean
                var = self.running_var
            hpreact = (
                self.bn_gain * (hpreact - mean) / torch.sqrt(var + self.eps)
                + self.bn_bias
            )

        h = torch.tanh(hpreact)
        logits = h @ self.W2 + self.b2
        if return_activations:
            return logits, hpreact, h
        return logits


def train_model(
    model: CharMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    learning_rate: float,
    batch_size: int = 32,
    seed: int = 2147483647,
    print_every: int = 0,
) -> List[float]:
    g = torch.Generator().manual_seed(seed)
    history: List[float] = []
    model.train()

    for step in range(steps):
        ix = torch.randint(0, x.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(x[ix]), y[ix])
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= learning_rate * p.grad

        history.append(float(loss.item()))
        if print_every and (step == 0 or (step + 1) % print_every == 0):
            print(f"  adım {step + 1:5d}/{steps}: loss={loss.item():.4f}")
    return history


@torch.no_grad()
def evaluate(
    model: CharMLP, x: torch.Tensor, y: torch.Tensor, batch_size: int = 4096
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        total += float(F.cross_entropy(model(xb), yb, reduction="sum").item())
    return total / len(x)


@torch.no_grad()
def sample_mlp(
    model: CharMLP,
    itos: Dict[int, str],
    count: int = 10,
    seed: int = 2147483647,
    max_len: int = 30,
) -> List[str]:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    names: List[str] = []
    for _ in range(count):
        context = [0] * model.block_size
        chars: List[str] = []
        for _ in range(max_len):
            logits = model(torch.tensor([context]))
            ix = int(torch.multinomial(logits.softmax(1), 1, generator=g).item())
            if ix == 0:
                break
            chars.append(itos[ix])
            context = context[1:] + [ix]
        names.append("".join(chars))
    return names


def bigram_probs(words: Sequence[str], stoi: Dict[str, int]) -> torch.Tensor:
    counts = torch.ones((len(stoi), len(stoi)))
    for word in words:
        chars = BOUNDARY + word + BOUNDARY
        for ch1, ch2 in zip(chars, chars[1:]):
            counts[stoi[ch1], stoi[ch2]] += 1
    return counts / counts.sum(1, keepdim=True)


def bigram_loss(
    words: Sequence[str], probs: torch.Tensor, stoi: Dict[str, int]
) -> float:
    log_likelihoods = []
    for word in words:
        chars = BOUNDARY + word + BOUNDARY
        for ch1, ch2 in zip(chars, chars[1:]):
            log_likelihoods.append(probs[stoi[ch1], stoi[ch2]].log())
    return float(-torch.stack(log_likelihoods).mean().item())


def sample_bigram(
    probs: torch.Tensor,
    itos: Dict[int, str],
    count: int = 10,
    seed: int = 2147483647,
    max_len: int = 30,
) -> List[str]:
    g = torch.Generator().manual_seed(seed)
    names: List[str] = []
    for _ in range(count):
        ix = 0
        chars: List[str] = []
        for _ in range(max_len):
            ix = int(torch.multinomial(probs[ix], 1, generator=g).item())
            if ix == 0:
                break
            chars.append(itos[ix])
        names.append("".join(chars))
    return names


def save_embedding_plot(model: CharMLP, itos: Dict[int, str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model.C.shape[1] != 2:
        raise ValueError("Embedding grafiği için n_embd=2 olmalı")
    path.parent.mkdir(parents=True, exist_ok=True)
    points = model.C.detach()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], s=20)
    for i, label in itos.items():
        ax.text(float(points[i, 0]), float(points[i, 1]), label)
    ax.set_title("Öğrenilen 2B harf embedding'leri")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_activation_histogram(
    naive_hpre: torch.Tensor,
    naive_h: torch.Tensor,
    good_hpre: torch.Tensor,
    good_h: torch.Tensor,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, values, title in [
        (axes[0, 0], naive_hpre, "Naive: tanh öncesi"),
        (axes[0, 1], naive_h, "Naive: tanh sonrası"),
        (axes[1, 0], good_hpre, "Kaiming: tanh öncesi"),
        (axes[1, 1], good_h, "Kaiming: tanh sonrası"),
    ]:
        ax.hist(values.detach().view(-1).tolist(), bins=50)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
