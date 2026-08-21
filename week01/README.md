# Hafta 1 — Nörondan gradient descent'e

## Bu hafta cevaplanan sorular

- Bir nöron ne yapar? → Ağırlıklı toplam + aktivasyon. İki satır.
- Parametre nedir? → `w`'lar ve `b`'ler. Öğrenirken değişen tek şey.
- Loss nedir? → "Ne kadar yanlışım?" sorusunun tek sayılık cevabı.
- Model nasıl öğrenir? → Her parametrenin loss'u nasıl etkilediğini ölç,
  hepsini küçük bir adım ters yöne taşı, tekrarla.
- Dil modeli nedir? → "Bir sonraki karakter ne?" tahmini. Aynı loss, aynı döngü.

## Dosyalar

| Dosya | Görev | İçerik |
|---|---|---|
| `adim1_tek_noron.py` | 1 | Tek nöron forward pass, sigmoid/relu/tanh elle yazılmış |
| `adim2_katman.py` | 2 | `Katman` ve `Ag` sınıfları, 784→16→16→10 parametre sayımı |
| `adim3_loss.py` | 3 | MSE, MAE, BCE + veri seti üstünde loss |
| `adim4_loss_egrisi.py` | 4 | Parametre tarama, loss eğrisi, 2B loss yüzeyi |
| `adim5_gradient_descent.py` | 5 | Sayısal türev, gradient descent, learning rate, XOR ağı |
| `bonus_bigram_lm.py` | bonus | Bigram dil modeli: sayarak vs. öğrenerek |
| `plot_utils.py` | — | ASCII + matplotlib grafik yardımcısı |

Sırayla çalıştır — her dosya bir öncekinin üstüne kuruyor:

```bash
python3 adim1_tek_noron.py
python3 adim2_katman.py
python3 adim3_loss.py
python3 adim4_loss_egrisi.py
python3 adim5_gradient_descent.py
python3 bonus_bigram_lm.py
```

Grafikler `outputs/` klasörüne PNG olarak düşer (matplotlib varsa).

## Akılda kalması gerekenler

**Forward pass**

```
z = w1*x1 + w2*x2 + ... + b
a = aktivasyon(z)
```

**Loss** — tek sayı, küçükse iyi:

```
MSE = ortalama( (tahmin - gercek)^2 )
```

**Sayısal türev** — "bu parametreyi bir tık oynatırsam loss ne olur?"

```
dL/dw ≈ ( L(w+h) - L(w-h) ) / (2h)      h ≈ 1e-5
```

**Gradient descent** — tek satır:

```python
w = w - ogrenme_orani * dL_dw
```

## Kendi kendine test

`adim5`'i çalıştırdıktan sonra şunları cevaplayabiliyor musun:

1. `w`'yi *artırınca* loss artıyorsa, gradient descent `w`'yi ne yapar?
2. Öğrenme oranı çok büyük olursa ne olur, çok küçük olursa ne olur?
3. 13.002 parametreli bir ağda sayısal türev bir adımda kaç forward pass ister?
   (Cevap: 26.004 — backprop'un neden gerektiği burada.)
4. XOR'u neden tek nöron çözemez?

## Takılırsan denenecek şeyler

- `adim4`'te `X` ve `Y`'yi değiştir (ör. `y = -3x + 2`), eğrinin dibi kayıyor mu?
- `adim5`'te `H`'yi `1e-1` ve `1e-12` yap, türev bozuluyor mu?
- `adim5`'teki XOR ağında gizli katmanı kaldır (2 → 1), loss neden takılıyor?
- `bonus`'ta `ISIMLER` listesine kendi veri setini koy.
