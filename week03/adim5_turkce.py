from __future__ import annotations

import argparse
from pathlib import Path

from ortak import (
    TR_LETTERS,
    count_bigrams_tensor,
    counts_to_probs,
    make_vocab,
    nll_from_probs,
    probs_from_weights,
    read_names,
    sample_from_probs,
    sample_from_weights,
    train_bigram_nn,
    visualize_bigram_table,
)
from veri_hazirla import prepare_turkish

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Görev 5: count ve neural bigram modellerini Türkçe isimlerle çalıştır."
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=50.0)
    args = parser.parse_args()

    words = read_names(prepare_turkish())
    stoi, itos = make_vocab(TR_LETTERS)

    print("=" * 72)
    print("GÖREV 5 — TÜRKÇE İSİMLER")
    print("=" * 72)
    print(f"Temiz isim sayısı: {len(words):,}")
    print(f"Türkçe harf sayısı: {len(TR_LETTERS)}")
    print(f"Boundary ile vocab size: {len(stoi)}")
    print("Alfabe:", " ".join(TR_LETTERS))
    print("Özel Türkçe karakterler:", "ç ğ ı ö ş ü")
    print("İlk 20 isim:", words[:20])

    n = count_bigrams_tensor(words, stoi)
    image_path = OUT_DIR / "bigram_counts_tr.png"
    visualize_bigram_table(
        n,
        itos,
        image_path,
        title="Türkçe isimler — 30×30 bigram count table",
    )
    print("\nTürkçe bigram tablosu:", image_path)

    p_mle = counts_to_probs(n, smoothing=0.0)
    p_smooth = counts_to_probs(n, smoothing=1.0)
    count_mle_nll = nll_from_probs(words, p_mle, stoi)
    count_smooth_nll = nll_from_probs(words, p_smooth, stoi)

    print("\nCount model")
    print(f"  MLE NLL (smoothing=0): {count_mle_nll:.6f}")
    print(f"  NLL (smoothing=1):     {count_smooth_nll:.6f}")
    print("  20 örnek:")
    for name in sample_from_probs(p_smooth, itos, n_samples=20, seed=20260906):
        print("   ", name)

    print("\nNeural bigram model eğitiliyor...")
    w, history = train_bigram_nn(
        words,
        stoi,
        steps=args.steps,
        learning_rate=args.lr,
        seed=20260906,
        print_every=max(1, args.steps // 10),
    )
    p_nn = probs_from_weights(w)
    nn_nll = nll_from_probs(words, p_nn, stoi)

    print("\nNeural model")
    print(f"  İlk NLL:          {history[0]:.6f}")
    print(f"  Son train NLL:    {history[-1]:.6f}")
    print(f"  Eval NLL:         {nn_nll:.6f}")
    print(f"  Count MLE farkı:  {abs(nn_nll - count_mle_nll):.6f}")
    print("  20 örnek:")
    for name in sample_from_weights(w, itos, n_samples=20, seed=20260906):
        print("   ", name)


if __name__ == "__main__":
    main()
