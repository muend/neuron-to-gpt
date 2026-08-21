"""
BONUS — Language model fikrine giriş (kütüphane YOK)

Bir dil modeli tek bir iş yapar:

    "Şu ana kadar gelen karakterlere bakarak, BİR SONRAKİ karakter ne olacak?"

Hepsi bu. GPT de bunu yapıyor, sadece bağlamı çok daha uzun ve modeli çok
daha büyük. Burada en küçük hâlini kuruyoruz: BIGRAM modeli — sadece bir
önceki karaktere bakar.

İki farklı yolla aynı şeyi yapacağız:
  1) SAYARAK   : veri setindeki geçişleri say, olasılığa çevir. Öğrenme yok.
  2) ÖĞRENEREK : aynı olasılıkları gradient descent ile bul. Adım 5'in aynısı.

İkisinin aynı sonuca varması, "öğrenme"nin sihir olmadığını gösterir.

Çalıştır:  python bonus_bigram_lm.py
"""

import math
import random

# ---------------------------------------------------------------------------
# Kucuk bir veri seti: Turkce isimler
# ---------------------------------------------------------------------------
ISIMLER = [
    "ahmet", "mehmet", "ayse", "fatma", "mustafa", "emine", "ali", "hatice",
    "huseyin", "zeynep", "hasan", "elif", "ibrahim", "meryem", "murat",
    "sultan", "omer", "havva", "kemal", "sevim", "yusuf", "melek", "osman",
    "hulya", "ismail", "nurten", "riza", "gulsum", "cemal", "sibel",
    "burak", "derya", "emre", "esra", "furkan", "gamze", "halil", "irem",
    "kadir", "leyla", "metin", "nazli", "onur", "pinar", "serkan", "tugce",
]

NOKTA = "."  # hem baslangic hem bitis isareti


def alfabe_kur(kelimeler):
    harfler = sorted(set("".join(kelimeler)))
    karakterler = [NOKTA] + harfler
    ktoi = {k: i for i, k in enumerate(karakterler)}
    return karakterler, ktoi


def bigramlari_uret(kelimeler):
    """'ali' -> ('.','a'), ('a','l'), ('l','i'), ('i','.')"""
    ciftler = []
    for kelime in kelimeler:
        dizi = [NOKTA] + list(kelime) + [NOKTA]
        for a, b in zip(dizi, dizi[1:]):
            ciftler.append((a, b))
    return ciftler


# ---------------------------------------------------------------------------
# YOL 1 — Sayarak
# ---------------------------------------------------------------------------

def sayarak_model(kelimeler, karakterler, ktoi, duzeltme=0):
    """
    duzeltme (smoothing): her sayaca eklenen sabit. 0 = ham sayim.
    1 yaparsan hic gorulmemis gecislere de kucuk bir sans birakirsin —
    yeni veride daha saglam olur ama egitim verisindeki loss artar.
    """
    n = len(karakterler)
    sayim = [[duzeltme] * n for _ in range(n)]
    for a, b in bigramlari_uret(kelimeler):
        sayim[ktoi[a]][ktoi[b]] += 1

    # Satirlari olasiliga cevir (her satir toplami 1 olsun)
    P = []
    for satir in sayim:
        toplam = sum(satir)
        P.append([c / toplam for c in satir])
    return P


def ornek_uret(P, karakterler, ktoi, rng, maks_uzunluk=12):
    """Modelden yeni bir isim ornekle."""
    sonuc = []
    i = ktoi[NOKTA]
    for _ in range(maks_uzunluk):
        i = agirlikli_sec(P[i], rng)
        if karakterler[i] == NOKTA:
            break
        sonuc.append(karakterler[i])
    return "".join(sonuc)


def agirlikli_sec(olasiliklar, rng):
    """random.choices'in elle yazilmis hali."""
    r = rng.random()
    birikimli = 0.0
    for i, p in enumerate(olasiliklar):
        birikimli += p
        if r < birikimli:
            return i
    return len(olasiliklar) - 1


def negatif_log_olabilirlik(P, ciftler, ktoi):
    """
    Dil modellerinin loss'u. MSE'nin yerini burada bu alir.
    Model dogru karaktere yuksek olasilik verdiginde kucuk olur.
    """
    toplam = 0.0
    for a, b in ciftler:
        p = P[ktoi[a]][ktoi[b]]
        toplam += -math.log(max(p, 1e-12))
    return toplam / len(ciftler)


# ---------------------------------------------------------------------------
# YOL 2 — Ogrenerek (gradient descent, Adim 5'teki ile ayni fikir)
# ---------------------------------------------------------------------------

def softmax(skorlar):
    """Ham skorlari olasiliga cevirir. Toplamlari 1 olur."""
    m = max(skorlar)                     # tasma korumasi
    us = [math.exp(s - m) for s in skorlar]
    t = sum(us)
    return [u / t for u in us]


def ogrenerek_model(kelimeler, karakterler, ktoi, adim_sayisi=500,
                    ogrenme_orani=25.0, tohum=1):
    """
    Model: her (onceki_karakter, sonraki_karakter) cifti icin bir SKOR.
    Skorlar softmax'tan gecerek olasiliga donusur.
    Bu skorlar modelin parametreleri — baslangicta rastgele, sonra ogrenilir.

    Burada analitik gradyan kullaniyoruz (cross-entropy + softmax icin
    cok temiz: dL/dskor = olasilik - hedef). Sayisal turev de calisirdi ama
    n*n = ~750 parametre x 2 forward pass x 120 adim cok yavas olurdu —
    tam da backpropagation'in neden gerekli oldugunu gosteren durum.
    """
    n = len(karakterler)
    rng = random.Random(tohum)
    W = [[rng.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(n)]

    ciftler = bigramlari_uret(kelimeler)
    gecmis = []

    for adim in range(adim_sayisi):
        grad = [[0.0] * n for _ in range(n)]
        toplam_loss = 0.0

        for a, b in ciftler:
            i, j = ktoi[a], ktoi[b]
            p = softmax(W[i])
            toplam_loss += -math.log(max(p[j], 1e-12))
            for k in range(n):
                grad[i][k] += p[k] - (1.0 if k == j else 0.0)

        N = len(ciftler)
        toplam_loss /= N
        gecmis.append(toplam_loss)

        # Gradient descent — Adim 5'teki satirin aynisi
        for i in range(n):
            for k in range(n):
                W[i][k] -= ogrenme_orani * (grad[i][k] / N)

    P = [softmax(satir) for satir in W]
    return P, gecmis


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("BONUS — BIGRAM DIL MODELI")
    print("=" * 62)

    karakterler, ktoi = alfabe_kur(ISIMLER)
    ciftler = bigramlari_uret(ISIMLER)
    print(f"\n{len(ISIMLER)} isim, {len(karakterler)} karakter, "
          f"{len(ciftler)} bigram ornegi")
    print("Alfabe:", " ".join(karakterler))

    # --- Yol 1 ---
    print("\n" + "-" * 62)
    print("YOL 1: SAYARAK")
    print("-" * 62)
    P_say = sayarak_model(ISIMLER, karakterler, ktoi, duzeltme=0)
    say_loss = negatif_log_olabilirlik(P_say, ciftler, ktoi)
    P_say_yumusak = sayarak_model(ISIMLER, karakterler, ktoi, duzeltme=1)
    print(f"Loss (ortalama negatif log olabilirlik) = {say_loss:.4f}")
    print("Karsilastirma icin: hicbir sey ogrenmeyip her karaktere esit")
    print(f"olasilik veren model -> {math.log(len(karakterler)):.4f}")
    print("Smoothing (duzeltme=1) ile     -> "
          f"{negatif_log_olabilirlik(P_say_yumusak, ciftler, ktoi):.4f}"
          "  (egitim verisinde daha kotu, yeni veride daha saglam)")

    print("\n'a' harfinden sonra en olasi 6 karakter:")
    i = ktoi["a"]
    en_olasi = sorted(range(len(karakterler)), key=lambda j: -P_say[i][j])[:6]
    for j in en_olasi:
        ad = "SON" if karakterler[j] == NOKTA else f"'{karakterler[j]}'"
        print(f"  {ad:>5}  {P_say[i][j] * 100:5.1f}%")

    rng = random.Random(42)
    print("\nModelden uretilen 10 yeni 'isim':")
    for _ in range(10):
        print("  ", ornek_uret(P_say, karakterler, ktoi, rng))

    # --- Yol 2 ---
    print("\n" + "-" * 62)
    print("YOL 2: OGRENEREK (gradient descent)")
    print("-" * 62)
    P_ogr, gecmis = ogrenerek_model(ISIMLER, karakterler, ktoi)
    print(f"Baslangic loss = {gecmis[0]:.4f}  (rastgele parametreler)")
    for adim in [10, 50, 100, 250, len(gecmis) - 1]:
        print(f"  adim {adim:4d} -> loss {gecmis[adim]:.4f}")

    print(f"\nSayarak bulunan loss  : {say_loss:.4f}   (teorik en iyi)")
    print(f"Ogrenerek bulunan loss: {gecmis[-1]:.4f}")
    print("\nIkisi ayni yere yakinsiyor. Gradient descent, sayarak")
    print("bulabilecegimiz cevabi 'kendi kendine' buldu.")

    rng2 = random.Random(42)
    print("\nOgrenilmis modelden 10 isim:")
    for _ in range(10):
        print("  ", ornek_uret(P_ogr, karakterler, ktoi, rng2))

    try:
        from plot_utils import cizgi_grafik
        cizgi_grafik(list(range(len(gecmis))), gecmis,
                     baslik="Bigram dil modeli egitimi",
                     x_ad="adim", y_ad="loss (NLL)",
                     dosya="outputs/bigram_loss.png")
    except Exception as e:  # grafik opsiyonel
        print(f"[grafik atlandi: {e}]")

    print("\n" + "=" * 62)
    print("BUYUK RESIM:")
    print("  bigram  : 1 onceki karaktere bakar   -> sacma ama Turkce'msi")
    print("  GPT     : binlerce token'a bakar     -> ayni loss, ayni dongu")
    print("Fark olcek ve mimaride; fikir birebir ayni.")
    print("=" * 62)
