from __future__ import annotations

import argparse

import torch

from ortak import (
    EN_LETTERS,
    CharMLP,
    build_dataset,
    load_words,
    make_vocab,
    split_words,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus E02: BatchNorm'u Linear'a katla.")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    words = load_words("en")
    stoi, _ = make_vocab(EN_LETTERS)
    train_words, dev_words, _ = split_words(words)
    x_train, y_train = build_dataset(train_words, stoi)
    x_dev, _ = build_dataset(dev_words, stoi)

    model = CharMLP(
        len(stoi), n_embd=10, n_hidden=200, batchnorm=True, init="good"
    )
    train_model(model, x_train, y_train, args.steps, learning_rate=0.1)
    model.eval()

    xb = x_dev[:128]
    with torch.no_grad():
        logits_bn = model(xb)
        scale = model.bn_gain / torch.sqrt(model.running_var + model.eps)
        w1_folded = model.W1 * scale
        b1_folded = (model.b1 - model.running_mean) * scale + model.bn_bias

        embcat = model.C[xb].view(len(xb), -1)
        h = torch.tanh(embcat @ w1_folded + b1_folded)
        logits_folded = h @ model.W2 + model.b2
        max_difference = float((logits_bn - logits_folded).abs().max().item())

    print("BONUS E02 — BatchNorm Linear katmanına katlandı")
    print(f"En büyük logits farkı: {max_difference:.10f}")
    assert torch.allclose(logits_bn, logits_folded, atol=1e-5)
    print("Forward pass aynı kaldı.")


if __name__ == "__main__":
    main()
