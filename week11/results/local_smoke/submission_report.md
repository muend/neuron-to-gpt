# Hafta 11 — eğitim ve hızlandırma raporu

Bu rapor yalnızca aynı klasördeki ölçüm dosyalarını özetler.

Weight tying: 50.257 × 768 = **38.597.376** parametre tasarrufu. Bağlı model 124.439.808; bağımsız model 163.037.184 parametre içerir. 50.304 vocab ile bağlı model 124.475.904 parametredir.

Başlangıç: matrislerde std=0.02; her bloktaki iki residual çıkış projeksiyonunda std=0.02/√(2L). Toplanan katkıların varyansı derinlikle büyümesin diye her katkının ölçeğini küçültüyoruz. Bu bir başlangıç yaklaşımıdır; dalların bağımsız olduğunu garanti etmez.

Gradient accumulation: eşit sayıda token içeren K mikro batch'in ortalama loss'unu K'ya bölüyoruz. Böylece toplam gradient, birleşik batch'in ortalama loss gradient'ine eşit olur. Clipping ve AdamW bir kez uygulanır; FP16'da önce unscale edilir.

Donanım: `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`; küçük model kontrolü: **True**.

B,T=[2, 16]; warmup=2; ölçülen adım=5. Süre: veri aktarımı, forward, backward, clipping ve optimizer dahil; derleme/warmup hariç.

| Aşama | ms/adım (medyan) | token/s | Önceki ölçüme göre hız | Durum |
|---|---:|---:|---:|---|
| baseline | 5.260 | 6083.9 | 1.000× | ölçüldü |
| tf32 | — | — | — | skipped: TF32 requires an Ampere-or-newer CUDA device (T4 is sm75) |
| mixed_precision | — | — | — | skipped: This controlled AMP benchmark is CUDA-only; MPS uses FP32 |
| compile | — | — | — | skipped: CUDA Inductor comparison runs on Linux/Colab; skipped on this platform |
| flash_attention | — | — | — | skipped: Forced CUDA FlashAttention needs a supported GPU and FP16/BF16 |
| padded_vocab | 9.595 | 3335.1 | 0.548× | ölçüldü |

Aşamalar birikimlidir. Atlanan/başarısız bir aşama sonraki ayara eklenmez. Padding, tokenizer'ı değiştirmez; softmax'a ek sınıflar katar, bu yüzden loss tam olarak aynı problem değildir.

En yüksek artımlı hız oranı: **padded_vocab (0.548×)**. 1'in altındaki oran yavaşlama demektir; tek koşu belirsizliğini gidermez.

En az iki GPU iyileştirme ölçümü şartı: **henüz sağlanmadı**.

Eğitim durumu: **complete**; küçük model kontrolü: **True**; donanım: `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`.

AdamW: β=(0.9,0.95), ε=1e-8, decay=0.1; decay uygulanan parametre: **124,928** (10 matris). Decay dışı: 1,792.

B,T=2,16; etkin batch: **64 token**; accumulation=2. Başarılı optimizer güncellemesi: 12. Tam veri turu: 0.

Her adımın loss, clipping öncesi norm, LR, süre ve token/s değeri `training.json` dosyasındadır. FP16 atlanan güncellemeleri ayrıca kaydedilir.

![Warmup ve cosine öğrenme oranı](learning_rate.png)

Bu haftaki demo metni eğitim hattını gösterir. Ayrılmış validation verisi ve genelleme değerlendirmesi gelecek haftanın kapsamıdır. Küçük CPU koşuları 124M GPU ödevinin tamamlandığı anlamına gelmez.

Ek görev: DDP/Türkçe tokenizer eğitim karşılaştırması **yapılmadı**. Farklı tokenizer loss'ları doğrudan kıyaslanmaz; aynı ayrılmış metindeki toplam NLL / (ln(2) × UTF-8 byte sayısı) ile bits/byte ölçülebilir. Eşit süre için işlenen byte ve corpus turu da kaydedilmelidir.
