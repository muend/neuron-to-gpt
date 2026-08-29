# Hafta 2 — Backpropagation: micrograd'ı sıfırdan yaz

Geçen hafta türevi **sayısal** aldık: parametreyi bir tık oynat, loss'a bak.
13.002 parametreli bir ağda tek adım = 26.004 forward pass. Bu hafta doğru
yolu kuruyoruz: **backpropagation** — bir forward, bir backward, hepsi bu.

Kaynak: Andrej Karpathy, [The spelled-out intro to neural networks and
backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)

## Bu hafta cevaplanan sorular

- Bir sayı kendi geçmişini nasıl hatırlar? → `Value`: `.data` + `_prev` + `_op`.
- Zincir kuralı kodda neye benziyor? → `cocuk.grad += yerel_türev * ebeveyn.grad`
- Neden ters topolojik sıra? → Bir düğümün gradient'i tamamlanmadan
  çocuklarına dağıtılamaz.
- Neden `+=` ve `=` değil? → Bir değişken birden fazla yerde kullanılıyorsa
  gradient birden fazla koldan gelir, toplanmalıdır.
- tanh'ı parçalarsam sonuç değişir mi? → Hayır. Atomik işlem seçimi hız
  meselesi, matematik meselesi değil.
- Neden `zero_grad()`? → `+=` kullandığımız için gradient'ler adımlar arası
  birikir; birikirse etkin adım boyu büyür ve ağ patlar.

## Dosyalar

| Dosya | Görev | Video | İçerik |
|---|---|---|---|
| `adim1_value.py` | 1 | 19:09–32:10 | `Value`, `+`, `*`, computation graph, topolojik sıra |
| `adim2_elle_gradient.py` | 2 | 32:10–51:10, 52:52–1:09:02 | Gradient'ler ELLE: basit ifade + tek nöron (tanh) |
| `adim3_backward.py` | 3 | 1:09:02–1:27:05 | Otomatik `backward()`, `+=` ile gradient birikmesi |
| `value.py` | 4 | 1:27:05–1:43:55 | Tam motor: `exp`, `/`, `**`, `tanh`, `relu` |
| `adim4_dogrulama.py` | 4 | 1:27:05–1:43:55 | tanh'ı parçala + backward vs sayısal vs PyTorch |
| `nn.py` | 5 | 1:43:55–2:14:03 | `Neuron`, `Layer`, `MLP`, `parameters()`, `zero_grad()` |
| `adim5_mlp.py` | 5 | 1:43:55–2:14:03 | Eğitim döngüsü, loss eğrisi, zero_grad bug'ı, lr taraması |

Sırayla çalıştır — her dosya bir öncekinin üstüne kuruyor:

```bash
python3 adim1_value.py
python3 adim2_elle_gradient.py
python3 adim3_backward.py
python3 adim4_dogrulama.py     # PyTorch varsa 3 yönlü, yoksa 2 yönlü karşılaştırır
python3 adim5_mlp.py
```

Bağımlılık yok. İki opsiyonel var, ikisi de kurulu değilse kod yine çalışır:
`graphviz` (adim1'de PNG graf), `torch` (adim4'te üçüncü referans).

```bash
pip install graphviz torch   # opsiyonel
```

> Repo kuralı `numpy`/`torch` yok. `torch` burada modelin parçası değil,
> sadece `adim4`'te kendi sonucumuzu **doğrulamak** için bağımsız referans.

## Akılda kalması gerekenler

**Value = sayı + geçmiş**

```python
out = Value(self.data + other.data, (self, other), '+')
#                                    ^^^^^^^^^^^^  ^^^ beni kim, hangi islemle uretti
```

**Üç yerel türev** — ezberlenecek tek şey bu:

| işlem | forward | geri (`out.grad` yukarıdan gelir) |
|---|---|---|
| toplama | `c = a + b` | `a.grad += out.grad` (aynen dağıt) |
| çarpma | `c = a * b` | `a.grad += b.data * out.grad` (çapraz kopya) |
| tanh | `o = tanh(n)` | `n.grad += (1 - o²) * out.grad` |
| üs | `c = a**k` | `a.grad += k * a^(k-1) * out.grad` |
| exp | `c = exp(a)` | `a.grad += c.data * out.grad` (kendisi!) |

**backward() üç satır fikir:**

```python
sira = topolojik_sirala(self)   # her dugum cocuklarindan sonra
self.grad = 1.0                 # dL/dL = 1
for dugum in reversed(sira):    # kokten yapraklara
    dugum._backward()
```

**Eğitim döngüsü — sıra önemli:**

```python
loss = ...            # 1. forward  (graf kurulur)
model.zero_grad()     # 2. eski gradient'leri sil   <- unutulan satir
loss.backward()       # 3. gradient'ler dolar
for p in model.parameters():
    p.data -= lr * p.grad   # 4. ters yone kucuk adim
```

## Sonuçlar (bu repodaki çıktılar)

- **Görev 2 kontrolü:** elle hesaplanan gradient'ler sayısal türevle birebir.
- **Görev 4 kontrolü:** tek parça `tanh` ile parçalanmış `tanh` arasında fark
  `~1e-16` (kayan nokta gürültüsü). Üç yönlü karşılaştırmada
  backward = PyTorch (`<1e-12`), sayısal türev `~1e-10` sapıyor — bu bug değil,
  kayan nokta hatası.
- **Görev 5 eğitimi:** `MLP(3, [4,4,1])`, 41 parametre, 100 adım, `lr=0.05`:
  loss `3.1407 → 0.0059` (~533 kat). Tahminler `[0.963, -0.963, -0.966, 0.955]`,
  hedefler `[1, -1, -1, 1]`.
- **zero_grad bug'ı:** `lr=0.1`'de `zero_grad`'siz sürüm önce daha hızlı düşüyor
  (gradient biriktiği için adım boyu sürekli büyüyor), sonra 20. adımda patlıyor
  ve loss `8.0`'da donuyor — tanh'lar ±1'e doymuş, gradient sıfır, ağ ölmüş.
  `max|grad|` `3.08 → 3.53` diye sürekli büyürken doğru sürümde `3.08 → 0.16`
  diye küçülüyor.

## Kendi kendine test

1. `a + a`'nın türevi neden 2? Kodda bunu sağlayan tek karakter hangisi?
2. `zero_grad()`'i `backward()`'dan SONRA çağırsan ne olur?
3. `tanh` yerine `relu` kullansan `adim5` yine öğrenir mi? (Dene: `nn.py`'de
   `act.tanh()` → `act.relu()`)
4. Neden çıkış katmanında genelde aktivasyon istemeyiz? (`Neuron(..., dogrusal=True)`)
5. `lr=1.0`'da loss neden tam olarak `8.0`'da takılıyor? (İpucu: 4 örnek,
   tahminlerin hepsi ±1'e doymuş, `(±1 ∓ 1)² = 4`)

## Takılırsan

- `adim3`'te `+=`'leri `=` yap, `test_biriktirme()` ne diyor?
- `adim4`'te `h`'yi `1e-1` ve `1e-12` yap — sayısal türev nerede bozuluyor?
- `adim5`'te `random.seed(1337)` satırını sil, birkaç kez çalıştır. Bazı
  başlangıçlar neden geç öğreniyor?
- `MLP(3, [4,4,1])` yerine `MLP(3, [1])` yap — tek nöron bu veriyi çözebiliyor mu?
