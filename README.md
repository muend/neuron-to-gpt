# neuron-to-gpt

Tek bir nörondan başlayıp dil modeline kadar giden, **kütüphanesiz** öğrenme
günlüğü. Her hafta bir klasör, her klasörde çalıştırılabilir ve yorumlanmış
Python dosyaları.

Kural: `numpy`, `torch` yok. Sadece Python standart kütüphanesi (`math`,
`random`). Tek istisna grafik çizmek için opsiyonel `matplotlib` — kurulu
değilse kod terminale ASCII grafik basar ve yine çalışır.

## Kurulum

```bash
git clone https://github.com/muend/neuron-to-gpt
cd neuron-to-gpt
python3 --version          # 3.8+ yeterli

# opsiyonel, sadece PNG grafikler icin
pip install matplotlib
```

## Haftalar

| Hafta | Konu | Durum |
|-------|------|-------|
| [01](week01/) | Nöron, katman, loss, gradient descent, dil modeli fikri | ✅ |
| 02 | — | ⏳ |

## Hafta 1'i çalıştır

```bash
cd week01
python3 adim1_tek_noron.py
python3 adim2_katman.py
python3 adim3_loss.py
python3 adim4_loss_egrisi.py
python3 adim5_gradient_descent.py
python3 bonus_bigram_lm.py
```

Detay için [week01/README.md](week01/README.md).

## Kaynaklar

- 3Blue1Brown — [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- 3Blue1Brown — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- Andrej Karpathy — [The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)
