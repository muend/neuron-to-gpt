"""
ADIM 3 — Loss (kayıp) fonksiyonu

Loss tek bir soruya cevap verir:
    "Modelin tahminleri, olması gereken cevaplardan NE KADAR uzak?"

Tek sayı üretir. Küçükse iyi, büyükse kötü.
Öğrenme = bu sayıyı düşürmek. Hepsi bu.

Buradaki fonksiyonlar:
    mse  — Mean Squared Error : hataların karesinin ortalaması (regresyon)
    mae  — Mean Absolute Error: hataların mutlak değerinin ortalaması
    bce  — Binary Cross Entropy: 0/1 sınıflandırma için

Neden kare? Çünkü:
  1) işaret gitsin (+2 hata ile -2 hata aynı kötülükte olsun),
  2) büyük hatalar orantısız cezalansın,
  3) fonksiyon pürüzsüz olsun -> türevi alınabilsin (Adım 5).

Çalıştır:  python adim3_loss.py
"""

import math


def mse(tahminler, gercekler):
    """Ortalama Kare Hata."""
    _uzunluk_kontrol(tahminler, gercekler)
    toplam = 0.0
    for y_hat, y in zip(tahminler, gercekler):
        fark = y_hat - y
        toplam += fark * fark
    return toplam / len(gercekler)


def mae(tahminler, gercekler):
    """Ortalama Mutlak Hata. Aykırı değerlere MSE'den daha az tepkili."""
    _uzunluk_kontrol(tahminler, gercekler)
    toplam = 0.0
    for y_hat, y in zip(tahminler, gercekler):
        toplam += abs(y_hat - y)
    return toplam / len(gercekler)


def bce(tahminler, gercekler, eps=1e-12):
    """
    Binary Cross Entropy. Tahminler (0,1) arasında olasılık olmalı,
    gerçekler 0 veya 1.

    Sezgi: doğru sınıfa verdiğin olasılık ne kadar düşükse ceza o kadar büyük.
    y=1 iken tahmin 0.99 -> ceza ~0.01;  tahmin 0.01 -> ceza ~4.6
    """
    _uzunluk_kontrol(tahminler, gercekler)
    toplam = 0.0
    for p, y in zip(tahminler, gercekler):
        p = min(max(p, eps), 1 - eps)  # log(0) patlamasin
        toplam += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return toplam / len(gercekler)


LOSSLAR = {"mse": mse, "mae": mae, "bce": bce}


def _uzunluk_kontrol(a, b):
    if len(a) != len(b):
        raise ValueError(f"Uzunluklar esit degil: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Bos liste ile loss hesaplanamaz.")


# ---------------------------------------------------------------------------
# Model + veri üstünden loss: Adım 4 ve 5 bunu kullanacak
# ---------------------------------------------------------------------------


def veri_seti_loss(model, X, Y, loss_fn=mse):
    """
    model : girdi listesi alıp TEK sayı döndüren çağrılabilir bir şey
    X     : girdi örnekleri listesi, ör. [[1.0], [2.0], [3.0]]
    Y     : beklenen çıktılar listesi, ör. [3.0, 5.0, 7.0]
    """
    tahminler = [model(x) for x in X]
    return loss_fn(tahminler, Y)


if __name__ == "__main__":
    print("=" * 62)
    print("ADIM 3 — LOSS FONKSIYONU")
    print("=" * 62)

    gercekler = [3.0, 5.0, 7.0, 9.0]

    print("\nGercek degerler:", gercekler)
    print("\n%-28s %10s %10s" % ("Tahminler", "MSE", "MAE"))
    print("-" * 50)
    for tahmin in (
        [3.0, 5.0, 7.0, 9.0],      # kusursuz
        [3.1, 4.9, 7.2, 8.8],      # cok yakin
        [2.0, 6.0, 6.0, 10.0],     # her birinde 1 birim hata
        [0.0, 0.0, 0.0, 0.0],      # tamamen bos
        [3.0, 5.0, 7.0, 100.0],    # tek buyuk hata (MSE patlar, MAE daha sakin)
    ):
        print("%-28s %10.4f %10.4f" % (str(tahmin), mse(tahmin, gercekler),
                                       mae(tahmin, gercekler)))

    print("\nDikkat: son satirda tek bir buyuk hata MSE'yi ucuruyor,")
    print("MAE ise daha sakin kaliyor. Loss secimi bir tasarim karari.")

    print("\n--- Binary Cross Entropy (siniflandirma) ---")
    etiketler = [1, 1, 0, 0]
    print("Etiketler:", etiketler)
    for olasiliklar in (
        [0.99, 0.98, 0.01, 0.02],  # emin ve dogru
        [0.60, 0.55, 0.45, 0.40],  # kararsiz ama dogru tarafta
        [0.40, 0.45, 0.55, 0.60],  # kararsiz ve yanlis tarafta
        [0.01, 0.02, 0.99, 0.98],  # emin ve YANLIS -> en agir ceza
    ):
        print(f"  {olasiliklar}  ->  BCE = {bce(olasiliklar, etiketler):.4f}")

    print("\n--- Model uzerinden loss ---")
    X = [[1.0], [2.0], [3.0], [4.0]]
    Y = [3.0, 5.0, 7.0, 9.0]        # gercek kural: y = 2x + 1

    def model_yap(w, b):
        return lambda x: w * x[0] + b

    for w, b in [(2.0, 1.0), (2.0, 0.0), (1.0, 1.0), (0.0, 0.0)]:
        L = veri_seti_loss(model_yap(w, b), X, Y, mse)
        print(f"  w={w:.1f}, b={b:.1f}  ->  loss = {L:8.4f}")

    print("\nw=2, b=1 loss'u 0 yapiyor cunku veriyi ureten kural tam olarak o.")
    print("Sorumuz: bu degerleri BILMEDEN nasil buluruz? -> Adim 4 ve 5.")
