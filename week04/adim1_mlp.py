from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ortak import (
    EN_LETTERS,
    CharMLP,
    bigram_probs,
    build_dataset,
    evaluate,
    load_words,
    make_vocab,
    sample_bigram,
    sample_mlp,
    save_embedding_plot,
    split_words,
    train_model,
)

HERE = Path(__file__).resolve().parent


def scan_learning_rate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_dev: torch.Tensor,
    y_dev: torch.Tensor,
    vocab_size: int,
    steps: int,
) -> float:
    candidates = [0.01, 0.03, 0.1, 0.3]
    results = []
    print("\nLearning rate taraması:")
    for lr in candidates:
        model = CharMLP(vocab_size, n_embd=2, n_hidden=100, init="good")
        train_model(model, x_train, y_train, steps=steps, learning_rate=lr)
        dev_loss = evaluate(model, x_dev, y_dev)
        results.append((dev_loss, lr))
        print(f"  lr={lr:<4} dev loss={dev_loss:.4f}")
    best_loss, best_lr = min(results)
    print(f"Seçilen lr: {best_lr} (dev loss={best_loss:.4f})")
    return best_lr


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 4 görev 1-4: embedding + MLP.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--scan-steps", type=int, default=300)
    parser.add_argument("--overfit-steps", type=int, default=500)
    args = parser.parse_args()

    words = load_words("en")
    stoi, itos = make_vocab(EN_LETTERS)
    train_words, dev_words, test_words = split_words(words)
    x_train, y_train = build_dataset(train_words, stoi)
    x_dev, y_dev = build_dataset(dev_words, stoi)
    x_test, y_test = build_dataset(test_words, stoi)

    print("GÖREV 1 — Üç harflik bağlam")
    print("X shape:", tuple(x_train.shape), "Y shape:", tuple(y_train.shape))
    for x, y in zip(x_train[:5], y_train[:5]):
        print(" ", "".join(itos[int(i)] for i in x), "->", itos[int(y)])

    model = CharMLP(len(stoi), n_embd=2, n_hidden=100, init="good")
    xb, yb = x_train[:32], y_train[:32]
    logits = model(xb)
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    manual_loss = -probs[torch.arange(len(yb)), yb].log().mean()
    cross_entropy_loss = F.cross_entropy(logits, yb)

    print("\nGÖREV 2 — MLP ve loss")
    print("Embedding tablosu:", tuple(model.C.shape))
    print("Düzleştirilmiş giriş:", (len(xb), 3 * 2))
    print(f"Elle loss:            {manual_loss.item():.6f}")
    print(f"F.cross_entropy loss: {cross_entropy_loss.item():.6f}")
    print("F.cross_entropy daha kısa ve sayısal olarak daha kararlı olduğu için tercih edilir.")
    assert torch.allclose(manual_loss, cross_entropy_loss, atol=1e-6)

    overfit = CharMLP(len(stoi), n_embd=2, n_hidden=100, init="good")
    before = evaluate(overfit, xb, yb)
    train_model(overfit, xb, yb, args.overfit_steps, learning_rate=0.1)
    after = evaluate(overfit, xb, yb)
    print("\nGÖREV 3 — Tek minibatch'i overfit et")
    print(f"Loss: {before:.4f} -> {after:.4f}")

    best_lr = scan_learning_rate(
        x_train, y_train, x_dev, y_dev, len(stoi), args.scan_steps
    )
    print(
        f"Split: train={len(train_words)}, dev={len(dev_words)}, test={len(test_words)} isim"
    )

    print("\nGÖREV 4 — Model boyutu")
    small = CharMLP(len(stoi), n_embd=2, n_hidden=100, init="good")
    train_model(
        small,
        x_train,
        y_train,
        args.steps,
        best_lr,
        print_every=max(1, args.steps // 5),
    )
    small_dev = evaluate(small, x_dev, y_dev)

    large = CharMLP(len(stoi), n_embd=10, n_hidden=200, init="good")
    train_model(
        large,
        x_train,
        y_train,
        args.steps,
        best_lr,
        print_every=max(1, args.steps // 5),
    )
    large_dev = evaluate(large, x_dev, y_dev)
    print(f"Küçük model (embedding=2, hidden=100) dev loss:  {small_dev:.4f}")
    print(f"Büyük model (embedding=10, hidden=200) dev loss: {large_dev:.4f}")
    print(f"Büyük model test loss (yalnızca son ölçüm):      {evaluate(large, x_test, y_test):.4f}")
    print(f"Rastgele tahminin beklenen loss'u:               {math.log(len(stoi)):.4f}")

    image_path = HERE / "outputs" / "embeddings_en.png"
    save_embedding_plot(small, itos, image_path)
    print("Embedding grafiği:", image_path)
    distances = torch.cdist(small.C.detach(), small.C.detach())
    distances.fill_diagonal_(float("inf"))
    pairs = []
    for i in range(len(stoi)):
        for j in range(i + 1, len(stoi)):
            pairs.append((float(distances[i, j]), itos[i], itos[j]))
    print("En yakın embedding çiftleri:")
    for distance, ch1, ch2 in sorted(pairs)[:8]:
        print(f"  {ch1}-{ch2}: {distance:.3f}")

    p_bigram = bigram_probs(train_words, stoi)
    print("\nMLP ve bigram örnekleri:")
    for mlp_name, bigram_name in zip(
        sample_mlp(large, itos, seed=42), sample_bigram(p_bigram, itos, seed=42)
    ):
        print(f"  MLP: {mlp_name:<18} bigram: {bigram_name}")


if __name__ == "__main__":
    main()
