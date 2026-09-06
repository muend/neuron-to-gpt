from __future__ import annotations

import math
from pathlib import Path

import torch

from ortak import (
    EN_LETTERS,
    count_bigrams_dict,
    count_bigrams_tensor,
    counts_to_probs,
    make_vocab,
    nll_from_probs,
    read_names,
    sample_from_probs,
    visualize_bigram_table,
)
from veri_hazirla import prepare_english

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"


def main() -> None:
    data_path = prepare_english()
    words = read_names(data_path)
    stoi, itos = make_vocab(EN_LETTERS)

    print("=" * 72)
    print("GÖREV 1 — Python dictionary ile bigram sayımı")
    print("=" * 72)
    print(f"İsim sayısı: {len(words):,}")
    print("İlk 10 isim:", words[:10])
    print("En kısa / en uzun:", min(map(len, words)), "/", max(map(len, words)))

    bigrams = count_bigrams_dict(words)
    top20 = sorted(bigrams.items(), key=lambda item: item[1], reverse=True)[:20]
    print("\nEn sık 20 bigram:")
    for (ch1, ch2), count in top20:
        print(f"  {ch1!r} -> {ch2!r}: {count:,}")

    print("\n" + "=" * 72)
    print("GÖREV 1 — Aynı sayımı 27x27 torch tensor'da yapma")
    print("=" * 72)
    n = count_bigrams_tensor(words, stoi)
    assert n.shape == (27, 27), n.shape
    assert int(n.sum().item()) == sum(len(word) + 1 for word in words)
    print("Tensor shape:", tuple(n.shape))
    print("Toplam bigram:", int(n.sum().item()))

    image_path = OUT_DIR / "bigram_counts_en.png"
    visualize_bigram_table(
        n,
        itos,
        image_path,
        title="English names — 27×27 bigram count table",
    )
    print("Görselleştirme kaydedildi:", image_path)

    print("\n" + "=" * 72)
    print("GÖREV 2 — Sayım tablosundan olasılık ve sampling")
    print("=" * 72)

    # Laplace/add-one smoothing: her hücreye 1 sahte sayım.
    counts = n.float() + 1.0

    # DOĞRU: (27, 1). Her satır kendi toplamına bölünür.
    denominator = counts.sum(dim=1, keepdim=True)
    p = counts / denominator
    print("N shape:", tuple(counts.shape))
    print("N.sum(1, keepdim=True) shape:", tuple(denominator.shape))
    print("Doğru model row-sum min/max:",
          float(p.sum(1).min()), float(p.sum(1).max()))

    # SESSİZ HATA: shape (27,) sondan hizalanır ve sütunlara broadcast edilir.
    wrong_denominator = counts.sum(dim=1)
    p_wrong = counts / wrong_denominator
    wrong_row_sums = p_wrong.sum(dim=1)
    print("keepdim=False denominator shape:", tuple(wrong_denominator.shape))
    print(
        "Yanlış model row-sum min/max:",
        float(wrong_row_sums.min()),
        float(wrong_row_sums.max()),
    )
    print(
        "Yanlış modelde satırların 1'den maksimum sapması:",
        float((wrong_row_sums - 1).abs().max()),
    )

    # Yardımcı fonksiyon da aynı doğru broadcasting'i kullanır ve assertion yapar.
    p = counts_to_probs(n, smoothing=1.0)

    print("\nSayıma dayalı modelden 20 örnek isim:")
    for name in sample_from_probs(p, itos, n_samples=20):
        print(" ", name)

    print("\n" + "=" * 72)
    print("GÖREV 3 — Negative log likelihood ve smoothing")
    print("=" * 72)

    p_mle = counts_to_probs(n, smoothing=0.0)
    nll_mle = nll_from_probs(words, p_mle, stoi)
    nll_smooth = nll_from_probs(words, p, stoi)
    uniform_nll = math.log(len(stoi))

    print(f"Uniform rastgele model NLL = log(27): {uniform_nll:.6f}")
    print(f"Count MLE NLL (smoothing=0):   {nll_mle:.6f}")
    print(f"Count NLL (smoothing=1):       {nll_smooth:.6f}")

    print("\nİlk 3 isimde ilk birkaç bigramın olasılık/log-olasılıkları:")
    shown = 0
    for word in words[:3]:
        chars = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chars, chars[1:]):
            prob = float(p[stoi[ch1], stoi[ch2]].item())
            logp = math.log(prob)
            print(f"  {ch1}{ch2}: p={prob:.6f}, log(p)={logp:.6f}, -log(p)={-logp:.6f}")
            shown += 1
            if shown >= 12:
                break
        if shown >= 12:
            break

    print(
        "\nÖzet: yüksek olasılık -> log(p) 0'a yakın -> -log(p) küçük. "
        "Düşük olasılık -> büyük ceza. Ortalama almak veri boyutundan bağımsız "
        "karşılaştırılabilir tek sayı verir."
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
