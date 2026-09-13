from __future__ import annotations

import argparse

from ortak import (
    TR_LETTERS,
    CharMLP,
    bigram_loss,
    bigram_probs,
    build_dataset,
    evaluate,
    load_words,
    make_vocab,
    sample_bigram,
    sample_mlp,
    split_words,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 4 görev 7: Türkçe isim MLP'si.")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    words = load_words("tr")
    stoi, itos = make_vocab(TR_LETTERS)
    train_words, dev_words, _ = split_words(words)
    x_train, y_train = build_dataset(train_words, stoi)
    x_dev, y_dev = build_dataset(dev_words, stoi)

    model = CharMLP(
        len(stoi), n_embd=10, n_hidden=200, init="good", batchnorm=True
    )
    print("GÖREV 7 — Türkçe MLP eğitiliyor")
    train_model(
        model,
        x_train,
        y_train,
        args.steps,
        args.lr,
        print_every=max(1, args.steps // 5),
    )

    probs = bigram_probs(train_words, stoi)
    print(f"MLP dev loss:    {evaluate(model, x_dev, y_dev):.4f}")
    print(f"Bigram dev loss: {bigram_loss(dev_words, probs, stoi):.4f}")
    print("\nTürkçe örnekler:")
    for mlp_name, bigram_name in zip(
        sample_mlp(model, itos, count=20, seed=42),
        sample_bigram(probs, itos, count=20, seed=42),
    ):
        print(f"  MLP: {mlp_name:<18} bigram: {bigram_name}")


if __name__ == "__main__":
    main()
