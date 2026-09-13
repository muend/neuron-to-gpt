# Hafta 4 — MLP karakter dil modeli

Bu hafta önceki üç harften sıradaki harfi tahmin eden Bengio tarzı bir MLP kuruyoruz.

## Dosyalar

- `adim1_mlp.py`: veri seti, 27x2 embedding, elle loss / cross entropy, minibatch, learning rate taraması, train/dev/test, model boyutu, embedding grafiği ve örnekler.
- `adim2_init_batchnorm.py`: başlangıç loss'u, tanh saturation, Kaiming init, aktivasyon histogramı ve BatchNorm karşılaştırması.
- `adim3_turkce.py`: aynı modelin Türkçe isimlerle eğitimi ve bigram karşılaştırması.
- `bonus_batchnorm_fold.py`: BatchNorm'u önceki Linear katmana katlayıp forward pass'in değişmediğini gösteren E02.
- `ortak.py`: ortak model ve veri yardımcıları.

## Çalıştırma

Önce veriyi hazırla:

```bash
python week03/veri_hazirla.py
```

Sonra:

```bash
python week04/adim1_mlp.py
python week04/adim2_init_batchnorm.py
python week04/adim3_turkce.py
python week04/bonus_batchnorm_fold.py
```

Hızlı deneme için adım sayısı azaltılabilir:

```bash
python week04/adim1_mlp.py --steps 100 --scan-steps 20 --overfit-steps 20
python week04/adim2_init_batchnorm.py --steps 100
python week04/adim3_turkce.py --steps 100
python week04/bonus_batchnorm_fold.py --steps 100
```

Grafikler `week04/outputs/` içine yazılır.

## Kaynaklar

- [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [A Neural Probabilistic Language Model (Bengio, 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Karpathy'nin makemore reposu](https://github.com/karpathy/makemore)
