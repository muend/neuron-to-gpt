# Hafta 5 — Backpropagation'ı elle yazmak

Bu hafta, Hafta 4'teki üç karakter bağlamlı MLP + BatchNorm modelinin
gradient'lerini `loss.backward()` kullanmadan adım adım hesaplıyoruz. PyTorch
autograd yalnızca doğru cevabı üreten referans olarak kullanılıyor.

## Dosyalar

- `ortak.py`: veri hazırlığı, parametreler ve küçük işlemlere ayrılmış forward pass.
- `adim1_autograd_referans.py`: bütün ara tensor'larda `retain_grad()` çağırır,
  `loss.backward()` çalıştırır ve PyTorch gradient'lerinin şekil/normlarını gösterir.
- `adim2_manual_backprop.py`: zincir kuralını sondan başa elle uygular ve 26
  gradient'i `cmp` fonksiyonuyla autograd sonucuna karşı doğrular.
- `requirements.txt`: yalnızca PyTorch bağımlılığı.

İsteğe bağlı Egzersiz 2–4 (cross entropy ve BatchNorm backward'unu tek ifadeye
indirme, modeli tamamen manuel gradient'lerle eğitme) bu teslimin zorunlu
kapsamına dahil edilmedi.

## Kurulum ve çalıştırma

Veri henüz hazırlanmadıysa:

```powershell
.\.venv\Scripts\python.exe week03\veri_hazirla.py
```

Sonra iki görevi çalıştır:

```powershell
.\.venv\Scripts\python.exe week05\adim1_autograd_referans.py
.\.venv\Scripts\python.exe week05\adim2_manual_backprop.py
```

İkinci komutun sonunda `26/26 gradient doğrulandı` yazmalıdır. `exact`, bit
düzeyinde aynı sonucu; `approximate`, kayan noktalı işlem sırası yüzünden küçük
fark olsa da tolerans içinde aynı sonucu ifade eder.

## Forward zinciri

Modelin ileri geçişi türevi görülebilecek küçük işlemlere ayrıldı:

```text
Xb -> emb -> embcat -> hprebn
   -> bnmeani -> bndiff -> bndiff2 -> bnvar -> bnvar_inv
   -> bnraw -> hpreact -> tanh -> logits
   -> logit_maxes -> norm_logits -> counts -> counts_sum
   -> counts_sum_inv -> probs -> logprobs -> loss
```

Backward sırasında bu sıra tersine izlenir. Bir tensor ileri geçişte iki farklı
işlemde kullanıldıysa iki koldan gelen gradient'ler toplanır.

## Broadcasting geri dönerken neden `sum` gerekir?

Broadcast edilen küçük tensor ileri geçişte sanal olarak birçok kez kullanılır.
Bu nedenle geri geçişte her kullanımdan gelen katkı özgün tensor şekline
toplanmalıdır:

- `logits = h @ W2 + b2`: `b2` bütün batch satırlarında kullanılır;
  `db2 = dlogits.sum(0)`.
- `hpreact = bngain * bnraw + bnbias`: gain ve bias batch boyunca kullanılır;
  gradient'leri `dim=0` üzerinde toplanır.
- `bndiff = hprebn - bnmeani`: tek satırlık mean bütün batch'e yayılır;
  `dbnmeani = (-dbndiff).sum(0, keepdim=True)`.
- `counts_sum = counts.sum(1, keepdim=True)`: ileri yöndeki satır toplamının
  gradient'i geri yönde o satırdaki bütün sütunlara yayılır.
- `emb = C[Xb]`: aynı harf indeksi tekrar kullanıldığında bütün kullanımların
  gradient katkıları `dC` içindeki aynı satırda toplanır.

## Temel yerel türevler

- `log(x)` → `1/x`
- `exp(x)` → `exp(x)`
- `x**-1` → `-x**-2`
- `x**2` → `2x`
- `tanh(x)` → `1 - tanh(x)**2`
- `A @ B` → `dA = dOut @ B.T`, `dB = A.T @ dOut`

Her yerel türev, zincirin devamından gelen gradient ile eleman bazında çarpılır.

## Kaynaklar

- [Andrej Karpathy — Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Karpathy'nin Part 4 egzersiz notebook'u](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
