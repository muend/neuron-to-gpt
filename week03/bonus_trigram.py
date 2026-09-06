from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Sequence, Tuple

import torch

from ortak import (
    BOUNDARY,
    EN_LETTERS,
    TR_LETTERS,
    count_bigrams_tensor,
    make_vocab,
    read_names,
    sample_from_probs,
)
from veri_hazirla import prepare_english, prepare_turkish


def split_words(
    words: Sequence[str], seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n)
    return shuffled[:n_train], shuffled[n_train:n_dev], shuffled[n_dev:]


def count_trigrams(words: Sequence[str], stoi: Dict[str, int]) -> torch.Tensor:
    v = len(stoi)
    n = torch.zeros((v, v, v), dtype=torch.int64)
    for word in words:
        chars = [BOUNDARY, BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2, ch3 in zip(chars, chars[1:], chars[2:]):
            n[stoi[ch1], stoi[ch2], stoi[ch3]] += 1
    return n


def bigram_probs(n: torch.Tensor, alpha: float) -> torch.Tensor:
    counts = n.float() + alpha
    return counts / counts.sum(dim=1, keepdim=True)


def trigram_probs(n: torch.Tensor, alpha: float) -> torch.Tensor:
    counts = n.float() + alpha
    return counts / counts.sum(dim=2, keepdim=True)


def bigram_nll(
    words: Sequence[str], probs: torch.Tensor, stoi: Dict[str, int]
) -> float:
    total = 0.0
    count = 0
    for word in words:
        chars = [BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2 in zip(chars, chars[1:]):
            p = float(probs[stoi[ch1], stoi[ch2]].item())
            total += -math.log(p)
            count += 1
    return total / count


def trigram_nll(
    words: Sequence[str], probs: torch.Tensor, stoi: Dict[str, int]
) -> float:
    total = 0.0
    count = 0
    for word in words:
        chars = [BOUNDARY, BOUNDARY] + list(word) + [BOUNDARY]
        for ch1, ch2, ch3 in zip(chars, chars[1:], chars[2:]):
            p = float(probs[stoi[ch1], stoi[ch2], stoi[ch3]].item())
            total += -math.log(p)
            count += 1
    return total / count


def choose_alpha_bigram(
    n: torch.Tensor,
    dev: Sequence[str],
    stoi: Dict[str, int],
    candidates: Sequence[float],
) -> Tuple[float, float]:
    scored = []
    for alpha in candidates:
        loss = bigram_nll(dev, bigram_probs(n, alpha), stoi)
        scored.append((loss, alpha))
        print(f"  bigram alpha={alpha:<5g} dev NLL={loss:.6f}")
    loss, alpha = min(scored)
    return alpha, loss


def choose_alpha_trigram(
    n: torch.Tensor,
    dev: Sequence[str],
    stoi: Dict[str, int],
    candidates: Sequence[float],
) -> Tuple[float, float]:
    scored = []
    for alpha in candidates:
        loss = trigram_nll(dev, trigram_probs(n, alpha), stoi)
        scored.append((loss, alpha))
        print(f"  trigram alpha={alpha:<5g} dev NLL={loss:.6f}")
    loss, alpha = min(scored)
    return alpha, loss


def sample_trigram(
    probs: torch.Tensor,
    itos: Dict[int, str],
    n_samples: int = 20,
    seed: int = 42,
    max_len: int = 30,
) -> List[str]:
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n_samples):
        i1, i2 = 0, 0
        chars = []
        for _ in range(max_len):
            i3 = int(
                torch.multinomial(
                    probs[i1, i2], 1, replacement=True, generator=g
                ).item()
            )
            if i3 == 0:
                break
            chars.append(itos[i3])
            i1, i2 = i2, i3
        samples.append("".join(chars))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bonus: trigram, 80/10/10 split ve dev loss ile smoothing seçimi."
    )
    parser.add_argument("--dataset", choices=("en", "tr"), default="en")
    args = parser.parse_args()

    if args.dataset == "en":
        words = read_names(prepare_english())
        letters = EN_LETTERS
    else:
        words = read_names(prepare_turkish())
        letters = TR_LETTERS

    stoi, itos = make_vocab(letters)
    train, dev, test = split_words(words, seed=42)

    print("=" * 72)
    print("BONUS — BIGRAM vs TRIGRAM")
    print("=" * 72)
    print(f"dataset={args.dataset}")
    print(f"train/dev/test = {len(train):,}/{len(dev):,}/{len(test):,}")
    print("Split oranları yaklaşık %80/%10/%10; test tuning sırasında kullanılmıyor.")

    n2 = count_bigrams_tensor(train, stoi)
    n3 = count_trigrams(train, stoi)

    candidates = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

    print("\nDev loss ile bigram smoothing seçimi:")
    alpha2, dev2 = choose_alpha_bigram(n2, dev, stoi, candidates)

    print("\nDev loss ile trigram smoothing seçimi:")
    alpha3, dev3 = choose_alpha_trigram(n3, dev, stoi, candidates)

    p2 = bigram_probs(n2, alpha2)
    p3 = trigram_probs(n3, alpha3)
    test2 = bigram_nll(test, p2, stoi)
    test3 = trigram_nll(test, p3, stoi)

    print("\n" + "=" * 72)
    print("SONUÇ")
    print("=" * 72)
    print(f"Bigram : best alpha={alpha2:g}, dev NLL={dev2:.6f}, test NLL={test2:.6f}")
    print(f"Trigram: best alpha={alpha3:g}, dev NLL={dev3:.6f}, test NLL={test3:.6f}")
    print(f"Test NLL farkı (bigram - trigram): {test2 - test3:+.6f}")

    print("\nBigram örnekleri:")
    for name in sample_from_probs(p2, itos, n_samples=20, seed=42):
        print(" ", name)

    print("\nTrigram örnekleri:")
    for name in sample_trigram(p3, itos, n_samples=20, seed=42):
        print(" ", name)

    print(
        "\nYorum: trigram iki karakterlik bağlam tuttuğu için yerel hece/harf "
        "örüntülerini bigramdan daha ayrıntılı yakalayabilir. Buna karşılık bağlam "
        "sayısı V²'ye çıktığı için veri seyrekliği artar; bu yüzden smoothing ve "
        "dev/test ayrımı özellikle önemlidir."
    )


if __name__ == "__main__":
    main()
