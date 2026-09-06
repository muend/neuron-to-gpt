# Hafta 3 — Bigram karakter dil modeli

Bu hafta tek bir karakterden sonraki karakteri tahmin eden **bigram character language model** kuruyoruz. Aynı modeli önce doğrudan frekans sayımıyla, sonra `one-hot -> W -> logits -> softmax -> NLL` zinciriyle tek katmanlı bir sinir ağı olarak öğreniyoruz.

Ana fikir: count tablosu ile neural weight matrix farklı görünen iki yöntem olsa da ikisi de aynı `P(next_char | current_char)` dağılımını temsil edebilir. Hafta 2'de kendi yazdığımız `Value.backward()` mekanizması da burada PyTorch autograd'ın tensor ölçekli karşılığı olarak tekrar karşımıza çıkıyor.

Kaynak video: Andrej Karpathy — [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)

## Dosyalar

| Dosya | Görev | İçerik |
|---|---:|---|
| `veri_hazirla.py` | hazırlık | Karpathy `names.txt` ve açık kaynak Türkçe isim listesini indirir/temizler. |
| `adim1_2_3_bigram_count.py` | 1–3 | Dictionary bigram sayımı, 27x27 tensor, görselleştirme, probability, broadcasting, sampling, NLL ve smoothing. |
| `adim4_bigram_nn.py` | 4 | One-hot, 27x27 `W`, logits, softmax, NLL, `backward()`, gradient descent ve count ↔ neural karşılaştırması. |
| `adim5_turkce.py` | 5 | Türkçe alfabe ile count ve neural bigram modelleri, loss ve isim örnekleri. |
| `bonus_trigram.py` | 6 bonus | Trigram, %80/%10/%10 split, dev loss ile smoothing seçimi, bigram/trigram karşılaştırması. |
| `ortak.py` | ortak | Veri, vocab, count, sampling, NLL ve görselleştirme yardımcıları. |

Repo yapısı önceki haftalarla aynı adlandırma standardını kullanır:

```text
week01/
week02/
week03/
```

ve görev dosyaları `adim...` biçimindedir.

## Çalıştırma

```bash
cd week03
python veri_hazirla.py
python adim1_2_3_bigram_count.py
python adim4_bigram_nn.py
python adim5_turkce.py
python bonus_trigram.py --dataset en
```

Bonus Türkçe veri üzerinde de çalışır:

```bash
python bonus_trigram.py --dataset tr
```

Neural eğitim adımını değiştirmek için:

```bash
python adim4_bigram_nn.py --steps 500 --lr 50
python adim5_turkce.py --steps 500 --lr 50
```

## 1 — Bigram sayımı

Bigram modeli yalnızca mevcut karaktere bakar:

```text
P(next_char | current_char)
```

Örneğin `emma` için boundary token ile çiftler:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

İngilizce veri için 26 harf + `.` olduğundan count matrisi 27x27'dir. `N[i,j]`, `i` karakterinden sonra `j` karakterinin kaç kez görüldüğünü tutar.

`adim1_2_3_bigram_count.py` aynı sayımı önce Python dictionary ile, sonra `torch.int64` tensor ile yapar. Script ayrıca Karpathy tarzı count tablosunu `outputs/bigram_counts_en.png` olarak üretir.

## 2 — Probability, broadcasting ve sampling

Count tablosunu satır satır normalize ediyoruz:

```python
counts = N.float() + smoothing
P = counts / counts.sum(dim=1, keepdim=True)
```

Shape kritik:

```text
counts.shape                      = (27, 27)
counts.sum(1, keepdim=True).shape = (27, 1)
```

`(27,1)` denominator her satırın tamamına doğru broadcast edilir.

`keepdim=False` kullanırsak denominator `(27,)` olur. PyTorch broadcasting boyutları sondan hizaladığı için kod hata vermeden yanlış eksende bölme yapabilir. Bu nedenle script hem shape'i gösterir hem de:

```python
assert torch.allclose(P.sum(dim=1), torch.ones(P.shape[0]))
```

ile her satırın gerçekten 1'e toplandığını doğrular.

Sampling sırasında mevcut karakterin probability satırından `torch.multinomial` ile yeni karakter seçilir. `.` gelince isim tamamlanır.

## 3 — Negative log likelihood

Doğru bigramlara modelin verdiği olasılıklar `p1, p2, ..., pk` ise likelihood:

```text
p1 * p2 * ... * pk
```

Log alınca:

```text
log(p1) + log(p2) + ... + log(pk)
```

ve kullandığımız loss:

```text
NLL = -mean(log(p_correct))
```

**Neden log?** Çok sayıda 0–1 arası olasılığı çarpmak çok küçük sayılar üretir. Log, çarpımı toplamaya çevirir ve sayısal olarak daha kararlı bir amaç fonksiyonu verir.

**Neden negatif?** `0 < p <= 1` için `log(p) <= 0`. Log-likelihood'u maksimize etmek yerine işareti çevirip NLL'i minimize etmek gradient descent formuna uyar.

**Neden mean?** Toplam loss veri miktarı arttıkça doğal olarak büyür. Ortalama NLL, tahmin başına ortalama cezayı verir ve modelleri daha anlamlı karşılaştırmamızı sağlar.

### Smoothing

Görülmemiş bigramın count'u 0 ise probability de 0 olur ve `log(0)` sonsuza gider. Additive smoothing:

```python
counts = N.float() + 1.0
```

ile her geçişe küçük bir sahte sayım ekler ve görülmemiş olayların olasılığını sıfır olmaktan çıkarır.

## 4 — Aynı bigram modeli sinir ağıyla

Karakter indexini one-hot yapıyoruz:

```python
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
probs = logits.softmax(dim=1)
loss = -probs[torch.arange(len(xs)), ys].log().mean()
```

Burada `W` boyutu 27x27'dir. One-hot ile matris çarpımı, `W`'nin mevcut karaktere karşılık gelen satırını seçmek gibidir. O satır logits'tir; softmax ile probability dağılımına dönüşür.

Eğitim döngüsü:

```python
W.grad = None
loss.backward()
with torch.no_grad():
    W -= lr * W.grad
```

Hafta 2'de yazdığımız `backward()` da computation graph üzerinde chain rule ile gradient topluyordu. PyTorch `loss.backward()` aynı matematiği scalar `Value` nesneleri yerine tensor operasyonları üzerinde uygular ve sonuçta `W.grad = dLoss/dW` üretir.

### Count ↔ neural eşdeğerliği

Add-one count modeli:

```text
P(j|i) = (N[i,j] + 1) / sum_j(N[i,j] + 1)
```

Eğer:

```python
W = torch.log(N.float() + 1)
```

seçersek:

```text
softmax(W[i]) = (N[i] + 1) / sum(N[i] + 1)
```

olur. `adim4_bigram_nn.py` bunu sayısal olarak da doğrular. Gradient descent sürümü ise bu dağılımı count formülünden doğrudan koymak yerine NLL'i minimize ederek öğrenir.

## 5 — Türkçe isim modeli

Türkçe alfabe:

```text
a b c ç d e f g ğ h ı i j k l m n o ö p r s ş t u ü v y z
```

29 harf + `.` boundary = **30 token**.

Özellikle şu karakterler ASCII'ye çevrilmeden korunur:

```text
ç ğ ı ö ş ü
```

`veri_hazirla.py`, açık kaynak Türkçe isim CSV'sini indirir, Unicode NFC normalizasyonu yapar ve `I/İ/ı/i` dönüşümünü açıkça ele alır. Ardından `adim5_turkce.py` aynı count ve neural bigram modellerini Türkçe veri üzerinde tekrar çalıştırır; loss değerlerini ve örnek isimleri basar.

Türkçe count görseli:

```text
outputs/bigram_counts_tr.png
```

## 6 — Bonus: trigram + train/dev/test

Bigram:

```text
P(x_t | x_{t-1})
```

Trigram:

```text
P(x_t | x_{t-2}, x_{t-1})
```

Veri sabit seed ile `%80 train / %10 dev / %10 test` olarak bölünür. Smoothing adayları train setinden kurulan model üzerinde dev NLL ile seçilir; test seti yalnızca son karşılaştırmada kullanılır.

Bu sayede hyperparameter seçerken test leakage oluşmaz. Trigram daha fazla bağlam yakalar ama context sayısı büyüdüğü için veri seyrekliği de artar; smoothing bu nedenle daha kritik hale gelir.
