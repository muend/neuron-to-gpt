from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

BOUNDARY = "."
EN_LETTERS = "abcdefghijklmnopqrstuvwxyz"
TR_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


def make_vocab(letters: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [BOUNDARY] + list(letters)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def read_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def count_bigrams_dict(words: Sequence[str]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for word in words:
        chars = [BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2 in zip(chars, chars[1:]):
            counts[(ch1, ch2)] = counts.get((ch1, ch2), 0) + 1
    return counts


def count_bigrams_tensor(words: Sequence[str], stoi: Dict[str, int]) -> torch.Tensor:
    v = len(stoi)
    n = torch.zeros((v, v), dtype=torch.int64)
    for word in words:
        chars = [BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2 in zip(chars, chars[1:]):
            n[stoi[ch1], stoi[ch2]] += 1
    return n


def build_bigram_dataset(
    words: Sequence[str], stoi: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[int] = []
    ys: List[int] = []
    for word in words:
        chars = [BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def counts_to_probs(n: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    counts = n.float() + smoothing
    row_sums = counts.sum(dim=1, keepdim=True)
    if torch.any(row_sums == 0):
        raise ValueError("En az bir satırın toplamı 0. Smoothing > 0 kullan.")
    p = counts / row_sums
    if not torch.allclose(p.sum(dim=1), torch.ones(p.shape[0]), atol=1e-6):
        raise RuntimeError("Satır olasılıkları 1'e toplamıyor; broadcasting hatası olabilir.")
    return p


def nll_from_probs(
    words: Sequence[str], probs: torch.Tensor, stoi: Dict[str, int]
) -> float:
    xs, ys = build_bigram_dataset(words, stoi)
    selected = probs[xs, ys]
    if torch.any(selected <= 0):
        return float("inf")
    return float((-selected.log().mean()).item())


def sample_from_probs(
    probs: torch.Tensor,
    itos: Dict[int, str],
    n_samples: int = 10,
    seed: int = 2147483647,
    max_len: int = 30,
) -> List[str]:
    g = torch.Generator().manual_seed(seed)
    out: List[str] = []
    for _ in range(n_samples):
        ix = 0
        chars: List[str] = []
        for _ in range(max_len):
            ix = int(torch.multinomial(probs[ix], 1, replacement=True, generator=g).item())
            if ix == 0:
                break
            chars.append(itos[ix])
        out.append("".join(chars))
    return out


def visualize_bigram_table(
    n: torch.Tensor,
    itos: Dict[int, str],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    v = n.shape[0]
    fig_size = 16 if v <= 27 else 18
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(n, cmap="Blues")
    for i in range(v):
        for j in range(v):
            pair = f"{itos[i]}{itos[j]}"
            value = int(n[i, j].item())
            ax.text(j, i, f"{pair}\n{value}", ha="center", va="center", fontsize=5)
    ax.set_title(title)
    ax.set_xlabel("Sonraki karakter")
    ax.set_ylabel("Mevcut karakter")
    ax.set_xticks(range(v), [itos[i] for i in range(v)])
    ax.set_yticks(range(v), [itos[i] for i in range(v)])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_bigram_nn(
    words: Sequence[str],
    stoi: Dict[str, int],
    steps: int = 300,
    learning_rate: float = 50.0,
    seed: int = 2147483647,
    print_every: int = 25,
) -> Tuple[torch.Tensor, List[float]]:
    xs, ys = build_bigram_dataset(words, stoi)
    v = len(stoi)
    xenc = F.one_hot(xs, num_classes=v).float()

    g = torch.Generator().manual_seed(seed)
    w = torch.randn((v, v), generator=g, requires_grad=True)
    history: List[float] = []

    for step in range(steps):
        logits = xenc @ w
        probs = torch.softmax(logits, dim=1)
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean()

        w.grad = None
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad

        loss_value = float(loss.item())
        history.append(loss_value)
        if step == 0 or (step + 1) % print_every == 0 or step == steps - 1:
            print(f"adım {step + 1:4d}/{steps}: NLL={loss_value:.6f}")

    return w.detach(), history


def probs_from_weights(w: torch.Tensor) -> torch.Tensor:
    return torch.softmax(w, dim=1)


def sample_from_weights(
    w: torch.Tensor,
    itos: Dict[int, str],
    n_samples: int = 10,
    seed: int = 2147483647,
    max_len: int = 30,
) -> List[str]:
    return sample_from_probs(
        probs_from_weights(w),
        itos,
        n_samples=n_samples,
        seed=seed,
        max_len=max_len,
    )
