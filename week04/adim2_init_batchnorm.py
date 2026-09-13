from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ortak import (
    EN_LETTERS,
    CharMLP,
    build_dataset,
    evaluate,
    load_words,
    make_vocab,
    save_activation_histogram,
    split_words,
    train_model,
)

HERE = Path(__file__).resolve().parent


def saturation(h: torch.Tensor) -> float:
    return 100 * float((h.abs() > 0.99).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 4 görev 5-6: init ve BatchNorm.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    words = load_words("en")
    stoi, _ = make_vocab(EN_LETTERS)
    train_words, dev_words, _ = split_words(words)
    x_train, y_train = build_dataset(train_words, stoi)
    x_dev, y_dev = build_dataset(dev_words, stoi)
    xb, yb = x_train[:256], y_train[:256]

    naive = CharMLP(len(stoi), n_embd=10, n_hidden=200, init="naive")
    good = CharMLP(len(stoi), n_embd=10, n_hidden=200, init="good")
    naive_logits, naive_hpre, naive_h = naive(xb, return_activations=True)
    good_logits, good_hpre, good_h = good(xb, return_activations=True)

    print("GÖREV 5 — Başlangıç ve tanh saturation")
    print(f"Beklenen başlangıç loss'u log(27): {math.log(len(stoi)):.4f}")
    print(f"Naive başlangıç loss:              {F.cross_entropy(naive_logits, yb).item():.4f}")
    print(f"İyileştirilmiş başlangıç loss:     {F.cross_entropy(good_logits, yb).item():.4f}")
    print(f"Naive doymuş tanh oranı:           %{saturation(naive_h):.2f}")
    print(f"Kaiming doymuş tanh oranı:         %{saturation(good_h):.2f}")
    print("Doymuş tanh'ta yerel türev 1-h² sıfıra yaklaşır; gradient küçülür.")

    hist_path = HERE / "outputs" / "activation_histograms.png"
    save_activation_histogram(naive_hpre, naive_h, good_hpre, good_h, hist_path)
    print("Histogram:", hist_path)

    print("\nGÖREV 6 — BatchNorm")
    plain = CharMLP(len(stoi), n_embd=10, n_hidden=200, init="good", batchnorm=False)
    bn = CharMLP(len(stoi), n_embd=10, n_hidden=200, init="good", batchnorm=True)
    train_model(
        plain,
        x_train,
        y_train,
        args.steps,
        args.lr,
        print_every=max(1, args.steps // 5),
    )
    train_model(
        bn,
        x_train,
        y_train,
        args.steps,
        args.lr,
        print_every=max(1, args.steps // 5),
    )
    print(f"BatchNorm yok dev loss: {evaluate(plain, x_dev, y_dev):.4f}")
    print(f"BatchNorm var dev loss: {evaluate(bn, x_dev, y_dev):.4f}")
    print("Eğitimde minibatch mean/variance, tahminde running mean/variance kullanıldı.")


if __name__ == "__main__":
    main()
