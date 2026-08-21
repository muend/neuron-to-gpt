"""
ADIM 4 — Parametreyi elle oynat, loss nasıl değişiyor gör

Burada hiç öğrenme yok. Sadece BAKIYORUZ.

Yaptığımız şey: w'yi -1'den 5'e kadar küçük adımlarla gezdirip her değerde
loss'u hesaplamak. Ortaya çıkan eğri, gradient descent'in üzerinde
yuvarlanacağı "vadi"dir. Bunu bir kez gözle görmek, Adım 5'i baştan anlaşılır
kılar.

Üç deney var:
  A) Tek parametre (w) tara       -> parabol, tek bir dip nokta
  B) İki parametre (w, b) tara    -> terminale ısı haritası, vadi 2 boyutta
  C) Kötü başlangıç noktası       -> "hangi yöne gitmeliyim?" sorusu doğar

Çalıştır:  python adim4_loss_egrisi.py
"""

from adim3_loss import mse, veri_seti_loss
from plot_utils import cizgi_grafik

# ---------------------------------------------------------------------------
# Oyuncak veri seti: gercek kural y = 2x + 1
# ---------------------------------------------------------------------------
X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
Y = [3.0, 5.0, 7.0, 9.0, 11.0]


def dogrusal_model(w, b):
    """Tek girdili, tek nöronlu, aktivasyonsuz model: y_hat = w*x + b"""
    return lambda x: w * x[0] + b


def loss_w(w, b=1.0):
    """b sabitken w'nin fonksiyonu olarak loss."""
    return veri_seti_loss(dogrusal_model(w, b), X, Y, mse)


def loss_wb(w, b):
    return veri_seti_loss(dogrusal_model(w, b), X, Y, mse)


def aralik(bas, son, adim):
    """range() ondalık sayı kabul etmiyor, kendi jeneratörümüzü yazıyoruz."""
    n = int(round((son - bas) / adim)) + 1
    return [bas + i * adim for i in range(n)]


# ---------------------------------------------------------------------------
# A) Tek parametre taramasi
# ---------------------------------------------------------------------------

def deney_a():
    print("\n" + "=" * 62)
    print("A) b=1 sabit, w'yi -1'den 5'e tariyoruz")
    print("=" * 62)

    ws = aralik(-1.0, 5.0, 0.05)
    losslar = [loss_w(w) for w in ws]

    print("\nBirkac ornek deger:")
    print(f"{'w':>8} {'loss':>12}")
    print("-" * 22)
    for w in aralik(-1.0, 5.0, 0.5):
        isaret = "   <-- en dusuk" if abs(w - 2.0) < 1e-9 else ""
        print(f"{w:8.2f} {loss_w(w):12.4f}{isaret}")

    en_iyi_i = min(range(len(losslar)), key=lambda i: losslar[i])
    en_iyi_w, en_iyi_loss = ws[en_iyi_i], losslar[en_iyi_i]

    print(f"\nTaramanin en dusuk noktasi: w = {en_iyi_w:.2f}, loss = {en_iyi_loss:.6f}")

    cizgi_grafik(
        ws, losslar,
        baslik="Loss egrisi: b=1 sabit, w degisiyor",
        x_ad="w", y_ad="loss (MSE)",
        dosya="outputs/loss_egrisi_w.png",
        isaretle=(en_iyi_w, en_iyi_loss),
    )

    print("\nOKUNACAK SEY: egri bir PARABOL. Tek bir dip nokta var ve orasi")
    print("w=2. Yani 'en iyi parametre' aramasi, bu vadinin dibini bulmak.")


# ---------------------------------------------------------------------------
# B) Iki parametre: terminalde isi haritasi
# ---------------------------------------------------------------------------

def deney_b():
    print("\n" + "=" * 62)
    print("B) Hem w hem b degisiyor — loss yuzeyi")
    print("=" * 62)

    ws = aralik(0.0, 4.0, 0.25)
    bs = aralik(-2.0, 4.0, 0.5)

    izgara = [[loss_wb(w, b) for w in ws] for b in bs]
    duz = [v for satir in izgara for v in satir]
    v_min, v_max = min(duz), max(duz)

    # Loglamsi olcek: kucuk farklar da gorunsun
    semboller = " .:-=+*#%@"

    def sembol(v):
        # 0..1 arasi normalize, sonra karekok ile kontrasti artir
        t = (v - v_min) / ((v_max - v_min) or 1.0)
        t = t ** 0.35
        return semboller[min(int(t * (len(semboller) - 1)), len(semboller) - 1)]

    print(f"\n(bos = dusuk loss, @ = yuksek loss; min={v_min:.3f}, max={v_max:.1f})\n")
    print("      b\\w    " + "".join(f"{w:>5.1f}" for w in ws))
    for b, satir in zip(bs, izgara):
        print(f"    {b:6.1f}   " + "".join(f"{sembol(v):>5}" for v in satir))

    # En iyi nokta
    en_iyi = min(
        ((w, b, loss_wb(w, b)) for b in bs for w in ws),
        key=lambda t: t[2],
    )
    print(f"\nIzgaradaki en iyi nokta: w={en_iyi[0]:.2f}, b={en_iyi[1]:.2f}, "
          f"loss={en_iyi[2]:.6f}")
    print("Gercek kural y = 2x + 1 oldugu icin beklenen: w=2, b=1.")
    print("\nOKUNACAK SEY: bos bolge bir 'vadi'. 13 bin parametreli bir agda")
    print("bu vadi 13 bin boyutlu — cizemeyiz ama mantik aynen bu.")


# ---------------------------------------------------------------------------
# C) Kotu bir noktadan bakinca: hangi yon asagi?
# ---------------------------------------------------------------------------

def deney_c():
    print("\n" + "=" * 62)
    print("C) w=4.5'ten bakiyoruz: saga mi sola mi gitmeliyim?")
    print("=" * 62)

    w = 4.5
    h = 0.1
    print(f"\n  loss(w - {h}) = loss({w - h:.1f}) = {loss_w(w - h):.4f}")
    print(f"  loss(w)      = loss({w:.1f}) = {loss_w(w):.4f}")
    print(f"  loss(w + {h}) = loss({w + h:.1f}) = {loss_w(w + h):.4f}")

    print("\nSola gidince loss dusuyor, saga gidince artiyor -> SOLA gitmeliyiz.")
    print("Bu kucuk deney aslinda SAYISAL TUREV'in ta kendisi:")
    print("    egim ~= (loss(w+h) - loss(w-h)) / (2h)")
    egim = (loss_w(w + h) - loss_w(w - h)) / (2 * h)
    print(f"    egim ~= {egim:.4f}   (pozitif => sola git)")
    print("\nAdim 5'te bunu bir donguye koyup modeli kendi kendine indireceğiz.")


if __name__ == "__main__":
    print("=" * 62)
    print("ADIM 4 — PARAMETREYI OYNAT, LOSS'U IZLE")
    print("=" * 62)
    print(f"\nVeri: X={[x[0] for x in X]}  Y={Y}   (gercek kural: y = 2x + 1)")

    deney_a()
    deney_b()
    deney_c()
