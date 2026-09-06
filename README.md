# neuron-to-gpt

Tek bir nörondan başlayıp karakter-seviyesinde dil modeline ve oradan GPT'nin temel fikirlerine doğru ilerleyen öğrenme günlüğü. Her hafta bir klasör; her klasörde çalıştırılabilir, yorumlanmış Python dosyaları ve kavram notları var.

İlk iki haftada türev, gradient descent ve backpropagation mekanizmasını mümkün olduğunca **kütüphanesiz** kuruyoruz. Hafta 3'te görev gereği PyTorch'a geçiyoruz: amaç autograd'ı sihir gibi kullanmak değil, geçen hafta kendi yazdığımız `backward()` mekanizmasının tensor'larla nasıl ölçeklendiğini görmek.

## Kurulum

```bash
git clone https://github.com/muend/neuron-to-gpt
cd neuron-to-gpt
python3 --version
```

Hafta 1–2 çekirdek kodu standart kütüphane ile çalışır. Hafta 3 için:

```bash
pip install -r week03/requirements.txt
```

## Haftalar

| Hafta | Konu | Durum |
|-------|------|-------|
| [01](week01/) | Nöron, katman, loss, gradient descent, dil modeli fikri | ✅ |
| [02](week02/) | Backpropagation: `Value`, computation graph, `backward()`, MLP eğitimi (micrograd) | ✅ |
| [03](week03/) | Bigram character LM: counts, probability, sampling, NLL, one-hot, softmax, autograd; Türkçe model + trigram bonus | ✅ |

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

## Hafta 2'yi çalıştır

```bash
cd week02
python3 adim1_value.py
python3 adim2_elle_gradient.py
python3 adim3_backward.py
python3 adim4_dogrulama.py
python3 adim5_mlp.py
```

Detay için [week02/README.md](week02/README.md).

## Hafta 3'ü çalıştır

```bash
cd week03
python veri_hazirla.py
python adim1_2_3_bigram_count.py
python adim4_bigram_nn.py
python adim5_turkce.py
python bonus_trigram.py --dataset en
```

Detay için [week03/README.md](week03/README.md).

## Kaynaklar

- 3Blue1Brown — [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- 3Blue1Brown — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- Andrej Karpathy — [The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd) (referans repo)
- Andrej Karpathy — [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- Andrej Karpathy — [makemore](https://github.com/karpathy/makemore)
- PyTorch — [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
