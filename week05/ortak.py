from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

BOUNDARY = "."
LETTERS = "abcdefghijklmnopqrstuvwxyz"
BLOCK_SIZE = 3
HERE = Path(__file__).resolve().parent

TensorMap = Dict[str, torch.Tensor]


def make_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [BOUNDARY] + list(LETTERS)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def load_words() -> List[str]:
    path = HERE.parent / "week03" / "data" / "names_en.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce "
            "'.\\.venv\\Scripts\\python.exe week03\\veri_hazirla.py' çalıştır."
        )
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_dataset(
    words: List[str], stoi: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[int] = []
    for word in words:
        context = [0] * BLOCK_SIZE
        for ch in word + BOUNDARY:
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs), torch.tensor(ys)


def prepare_batch(
    batch_size: int = 32,
    data_seed: int = 42,
    parameter_seed: int = 2147483647,
) -> Tuple[torch.Tensor, torch.Tensor, TensorMap]:
    """İngilizce eğitim verisinden tekrarlanabilir bir minibatch ve parametreler kur."""
    words = load_words()
    random.Random(data_seed).shuffle(words)
    n_train = int(0.8 * len(words))
    stoi, _ = make_vocab()
    x_train, y_train = build_dataset(words[:n_train], stoi)

    g = torch.Generator().manual_seed(parameter_seed)
    ix = torch.randint(0, x_train.shape[0], (batch_size,), generator=g)
    xb, yb = x_train[ix], y_train[ix]

    vocab_size = len(stoi)
    n_embd = 10
    n_hidden = 64
    fan_in = n_embd * BLOCK_SIZE

    # Sıfır olmayan başlangıçlar, hatalı backward kodunun tesadüfen doğru
    # görünmesini engeller. Bu seçim egzersiz notebook'unu takip eder.
    parameters: TensorMap = {
        "C": torch.randn((vocab_size, n_embd), generator=g),
        "W1": torch.randn((fan_in, n_hidden), generator=g)
        * (5 / 3)
        / math.sqrt(fan_in),
        "b1": torch.randn(n_hidden, generator=g) * 0.1,
        "W2": torch.randn((n_hidden, vocab_size), generator=g) * 0.1,
        "b2": torch.randn(vocab_size, generator=g) * 0.1,
        "bngain": torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0,
        "bnbias": torch.randn((1, n_hidden), generator=g) * 0.1,
    }
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    return xb, yb, parameters


FORWARD_NAMES = [
    "emb",
    "embcat",
    "hprebn",
    "bnmeani",
    "bndiff",
    "bndiff2",
    "bnvar",
    "bnvar_inv",
    "bnraw",
    "hpreact",
    "h",
    "logits",
    "logit_maxes",
    "norm_logits",
    "counts",
    "counts_sum",
    "counts_sum_inv",
    "probs",
    "logprobs",
]


def forward_in_small_steps(
    xb: torch.Tensor,
    yb: torch.Tensor,
    parameters: TensorMap,
    eps: float = 1e-5,
) -> TensorMap:
    """MLP + BatchNorm + cross entropy ileri geçişini küçük adımlara böl."""
    C = parameters["C"]
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    bngain, bnbias = parameters["bngain"], parameters["bnbias"]
    n = xb.shape[0]

    # Embedding ve ilk Linear katman.
    emb = C[xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1

    # BatchNorm; her işlem backward zincirinde ayrı görülebilsin diye bölündü.
    bnmeani = (1 / n) * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = (1 / (n - 1)) * bndiff2.sum(0, keepdim=True)
    bnvar_inv = (bnvar + eps) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # Aktivasyon ve ikinci Linear katman.
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    # F.cross_entropy ile aynı hesap, türev zinciri görülsün diye açık yazıldı.
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), yb].mean()

    return {
        "emb": emb,
        "embcat": embcat,
        "hprebn": hprebn,
        "bnmeani": bnmeani,
        "bndiff": bndiff,
        "bndiff2": bndiff2,
        "bnvar": bnvar,
        "bnvar_inv": bnvar_inv,
        "bnraw": bnraw,
        "hpreact": hpreact,
        "h": h,
        "logits": logits,
        "logit_maxes": logit_maxes,
        "norm_logits": norm_logits,
        "counts": counts,
        "counts_sum": counts_sum,
        "counts_sum_inv": counts_sum_inv,
        "probs": probs,
        "logprobs": logprobs,
        "loss": loss,
    }


def run_autograd(graph: TensorMap, parameters: TensorMap) -> None:
    """Ara tensor gradient'lerini sakla ve PyTorch referans backward'unu çalıştır."""
    for parameter in parameters.values():
        parameter.grad = None
    for name in FORWARD_NAMES:
        graph[name].retain_grad()
    graph["loss"].backward()
