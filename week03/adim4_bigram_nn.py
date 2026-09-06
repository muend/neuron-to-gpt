from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ortak import (
    EN_LETTERS,
    build_bigram_dataset,
    count_bigrams_tensor,
    counts_to_probs,
    make_vocab,
    nll_from_probs,
    probs_from_weights,
    read_names,
    sample_from_weights,
    train_bigram_nn,
)
from veri_hazirla import prepare_english


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Görev 4: bigram modelini tek katmanlı sinir ağıyla kur."
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=50.0)
    args = parser.parse_args()

    words = read_names(prepare_english())
    stoi, itos = make_vocab(EN_LETTERS)
    xs, _ = build_bigram_dataset(words, stoi)
    v = len(stoi)

    print("=" * 72)
    print("GÖREV 4 — one-hot -> logits -> softmax -> NLL -> backward -> update")
    print("=" * 72)
    print(f"Örnek sayısı (bigram): {xs.nelement():,}")
    print(f"Vocab size: {v}")

    # One-hot encoding: her karakter indeksi 27 boyutlu 0/1 vektöre dönüşür.
    xenc = F.one_hot(xs[:5], num_classes=v).float()
    print("\nİlk 5 input indeksi:", xs[:5].tolist())
    print("One-hot shape:", tuple(xenc.shape))
    print(xenc)

    # Count modelin maksimum likelihood (smoothing'siz) train loss'u.
    n = count_bigrams_tensor(words, stoi)
    p_count_mle = counts_to_probs(n, smoothing=0.0)
    count_mle_nll = nll_from_probs(words, p_count_mle, stoi)

    # Çok güçlü eşdeğerlik kontrolü:
    # P_count(j|i) = (N_ij + 1) / sum_j(N_ij+1)
    # W_ij = log(N_ij+1) seçersek softmax(W[i]) TAM AYNI dağılımı verir.
    p_count_smooth = counts_to_probs(n, smoothing=1.0)
    w_from_counts = (n.float() + 1.0).log()
    p_from_logits = torch.softmax(w_from_counts, dim=1)
    max_diff = float((p_count_smooth - p_from_logits).abs().max().item())

    print("\nSayım modeli <-> softmax ağı eşdeğerlik kontrolü")
    print("max |P_count(smoothing=1) - softmax(log(N+1))| =", f"{max_diff:.10f}")

    print("\nGradient descent ile W öğreniliyor...")
    w, history = train_bigram_nn(
        words,
        stoi,
        steps=args.steps,
        learning_rate=args.lr,
        print_every=max(1, args.steps // 10),
    )

    p_nn = probs_from_weights(w)
    nn_nll = nll_from_probs(words, p_nn, stoi)

    print("\n" + "=" * 72)
    print("LOSS KARŞILAŞTIRMASI")
    print("=" * 72)
    print(f"Count MLE NLL:        {count_mle_nll:.6f}")
    print(f"Neural bigram NLL:    {nn_nll:.6f}")
    print(f"Mutlak fark:           {abs(nn_nll - count_mle_nll):.6f}")
    print(f"İlk NN loss:           {history[0]:.6f}")
    print(f"Son NN loss:           {history[-1]:.6f}")

    print("\nSinir ağı modelinden 20 örnek isim:")
    for name in sample_from_weights(w, itos, n_samples=20):
        print(" ", name)

    print(
        "\nbackward() bağlantısı: geçen hafta Value.backward() ile computation graph'ı "
        "ters topolojik sırada dolaşıp chain rule uyguluyorduk. Burada loss.backward() "
        "aynı fikri PyTorch tensor operasyonlarının grafiğinde otomatik uygular. "
        "W.grad, loss'un her W_ij'e göre türevidir."
    )


if __name__ == "__main__":
    main()
