"""
ADIM 5 — Sayısal türev + gradient descent döngüsü

Adım 4'te "sola mı sağa mı?" sorusunu elle sorduk. Şimdi bunu otomatikleştirip
modeli kendi kendine öğrenir hale getiriyoruz.

TÜREV = "bu parametreyi bir tık artırırsam loss ne kadar değişir?"

Sayısal (numerical) türev, Karpathy'nin videosunun ilk 19 dakikasındaki fikir:
türevi formülden çıkarmak yerine, gerçekten küçük bir h ile deneyip ölçmek.

    ileri fark   : (f(x+h) - f(x))   / h        -> kaba
    merkezi fark : (f(x+h) - f(x-h)) / (2h)     -> belirgin sekilde daha isabetli

GRADIENT = her parametre için ayrı ayrı hesaplanmış türevlerin listesi.
    "loss'u en hızlı ARTIRAN yön" -> o yüzden TERSİNE gidiyoruz.

GÜNCELLEME KURALI (gradient descent):
    parametre = parametre - ogrenme_orani * turev

Not: sayısal türev öğretici ama pahalıdır — her parametre için modeli 2 kez
çalıştırır. 13.002 parametre = adım başına 26.004 forward pass. Gerçek ağlar
bu yüzden backpropagation kullanır (ilerideki hafta).

Çalıştır:  python adim5_gradient_descent.py
"""

from adim3_loss import mse, veri_seti_loss
from plot_utils import cizgi_grafik

H = 1e-5  # sayısal türev adımı: çok büyük -> hatalı, çok küçük -> yuvarlama hatası


# ---------------------------------------------------------------------------
# 1) Sayısal türev
# ---------------------------------------------------------------------------

def turev_ileri(f, x, h=H):
    """(f(x+h) - f(x)) / h"""
    return (f(x + h) - f(x)) / h


def turev_merkezi(f, x, h=H):
    """(f(x+h) - f(x-h)) / (2h) — pratikte tercih edilen."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradyan(loss_fn, parametreler, h=H):
    """
    Her parametre için kısmi türev. Tek seferde tek parametre oynatılır,
    diğerleri sabit tutulur.

    loss_fn      : parametre listesi alıp tek sayı döndüren fonksiyon
    parametreler : [p0, p1, ...]
    donus        : [dL/dp0, dL/dp1, ...]
    """
    grad = []
    for i in range(len(parametreler)):
        arti = list(parametreler)
        eksi = list(parametreler)
        arti[i] += h
        eksi[i] -= h
        grad.append((loss_fn(arti) - loss_fn(eksi)) / (2 * h))
    return grad


# ---------------------------------------------------------------------------
# 2) Sayısal türev gerçekten doğru mu? Bilinen bir fonksiyonla test
# ---------------------------------------------------------------------------

def turev_dogrulama():
    print("\n" + "=" * 62)
    print("1) SAYISAL TUREV DOGRU MU?")
    print("=" * 62)

    # Karpathy'nin ornegi
    def f(x):
        return 3 * x ** 2 - 4 * x + 5

    def f_turev_gercek(x):
        return 6 * x - 4  # elle alinmis analitik turev

    x = 3.0
    print(f"\nf(x) = 3x^2 - 4x + 5,  x = {x}")
    print(f"Analitik turev (6x-4) = {f_turev_gercek(x)}")
    print(f"\n{'h':>10} {'ileri fark':>16} {'merkezi fark':>16}")
    print("-" * 44)
    for h in [1.0, 0.1, 0.01, 1e-4, 1e-6, 1e-8]:
        print(f"{h:10.0e} {turev_ileri(f, x, h):16.8f} {turev_merkezi(f, x, h):16.8f}")

    print("\nOKUNACAK SEY: h kuculdukce ikisi de 14'e yaklasiyor ama merkezi")
    print("fark cok daha hizli yakinsiyor. h=1e-8'de ise kayan nokta yuvarlama")
    print("hatasi devreye giriyor — 'daha kucuk h her zaman daha iyi' degil.")


# ---------------------------------------------------------------------------
# 3) Asıl olay: gradient descent ile doğrusal modeli eğit
# ---------------------------------------------------------------------------

X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
Y = [3.0, 5.0, 7.0, 9.0, 11.0]        # gercek kural: y = 2x + 1


def parametrelerden_loss(p):
    """p = [w, b] -> veri setindeki MSE."""
    w, b = p
    return veri_seti_loss(lambda x: w * x[0] + b, X, Y, mse)


def egit(baslangic, ogrenme_orani=0.01, adim_sayisi=200, yaz_her=20):
    p = list(baslangic)
    gecmis = [parametrelerden_loss(p)]

    print(f"\n{'adim':>6} {'w':>9} {'b':>9} {'loss':>12} {'dL/dw':>10} {'dL/db':>10}")
    print("-" * 60)

    for adim in range(1, adim_sayisi + 1):
        g = gradyan(parametrelerden_loss, p)

        # ===== GRADIENT DESCENT'IN TEK SATIRI =====
        p = [pi - ogrenme_orani * gi for pi, gi in zip(p, g)]
        # ==========================================

        L = parametrelerden_loss(p)
        gecmis.append(L)

        if adim == 1 or adim % yaz_her == 0 or adim == adim_sayisi:
            print(f"{adim:6d} {p[0]:9.4f} {p[1]:9.4f} {L:12.6f} "
                  f"{g[0]:10.4f} {g[1]:10.4f}")

    return p, gecmis


def egitim_deneyi():
    print("\n" + "=" * 62)
    print("2) GRADIENT DESCENT DONGUSU")
    print("=" * 62)
    print(f"\nVeri: X={[x[0] for x in X]}  Y={Y}   (gercek kural: y = 2x + 1)")

    baslangic = [-1.0, 4.0]   # bilerek kotu bir baslangic
    print(f"Baslangic: w={baslangic[0]}, b={baslangic[1]}, "
          f"loss={parametrelerden_loss(baslangic):.4f}")

    p, gecmis = egit(baslangic, ogrenme_orani=0.05, adim_sayisi=600, yaz_her=50)

    print(f"\nSONUC: w={p[0]:.6f}, b={p[1]:.6f}, loss={gecmis[-1]:.8f}")
    print("Beklenen: w=2, b=1. Model kurali hic soylemedigimiz halde buldu.")

    cizgi_grafik(
        list(range(len(gecmis))), gecmis,
        baslik="Egitim sirasinda loss (gradient descent)",
        x_ad="adim", y_ad="loss (MSE)",
        dosya="outputs/egitim_loss.png",
    )

    print("\nTahminler:")
    print(f"{'x':>5} {'gercek y':>10} {'tahmin':>10}")
    for x, y in zip(X, Y):
        print(f"{x[0]:5.1f} {y:10.2f} {p[0] * x[0] + p[1]:10.4f}")

    return gecmis


# ---------------------------------------------------------------------------
# 4) Öğrenme oranı (learning rate) neden önemli?
# ---------------------------------------------------------------------------

def ogrenme_orani_deneyi():
    print("\n" + "=" * 62)
    print("3) OGRENME ORANI (LEARNING RATE) SECIMI")
    print("=" * 62)

    baslangic = [-1.0, 4.0]
    print(f"\n{'lr':>8} {'50 adim sonra loss':>22}  yorum")
    print("-" * 62)

    for lr in [0.0005, 0.005, 0.02, 0.06, 0.1]:
        p = list(baslangic)
        patladi = False
        for _ in range(50):
            g = gradyan(parametrelerden_loss, p)
            p = [pi - lr * gi for pi, gi in zip(p, g)]
            if abs(p[0]) > 1e6:
                patladi = True
                break

        if patladi:
            print(f"{lr:8.4f} {'PATLADI (inf)':>22}  adim cok buyuk, vadiden disari firladi")
        else:
            L = parametrelerden_loss(p)
            if L > 1.0:
                yorum = "cok yavas, hala inmedi"
            elif L > 0.01:
                yorum = "iyi gidiyor"
            else:
                yorum = "hizli ve stabil"
            print(f"{lr:8.4f} {L:22.6f}  {yorum}")

    print("\nOKUNACAK SEY: cok kucuk lr = sonsuza kadar bekle.")
    print("Cok buyuk lr = vadinin dibini asip yukari zipla ve dagil.")
    print("Ogrenme orani bir 'hyperparameter': model ogrenmez, sen secersin.")


# ---------------------------------------------------------------------------
# 5) Aynı döngü, bu sefer gerçek bir sinir ağıyla (doğrusal olmayan problem)
# ---------------------------------------------------------------------------

def sinir_agi_deneyi():
    print("\n" + "=" * 62)
    print("4) AYNI DONGU, GERCEK BIR AG: XOR")
    print("=" * 62)
    print("\nXOR duz bir cizgiyle ayrilamaz — tek noron bunu COZEMEZ.")
    print("Gizli katman + dogrusal olmayan aktivasyon sart. Ag: 2 -> 4 -> 1")

    from adim1_tek_noron import sigmoid, tanh

    XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    XOR_Y = [0.0, 1.0, 1.0, 0.0]

    # Parametreleri tek bir duz listede tutuyoruz: gradyan fonksiyonu boyle istiyor.
    # Yerlesim: gizli katman 4 noron x (2 agirlik + 1 bias) = 12
    #           cikti katmani 1 noron x (4 agirlik + 1 bias) = 5   -> toplam 17
    def ag_ileri(p, x):
        gizli = []
        for j in range(4):
            taban = j * 3
            z = p[taban] * x[0] + p[taban + 1] * x[1] + p[taban + 2]
            gizli.append(tanh(z))
        # cikti noronu: bias p[12], agirliklar p[13..16]
        z = p[12]
        for j in range(4):
            z += p[13 + j] * gizli[j]
        return sigmoid(z)

    def ag_loss(p):
        return mse([ag_ileri(p, x) for x in XOR_X], XOR_Y)

    import random
    rng = random.Random(3)
    p = [rng.uniform(-1, 1) for _ in range(17)]

    print(f"\nBaslangic loss: {ag_loss(p):.6f}  ({len(p)} parametre)")
    gecmis = [ag_loss(p)]
    lr = 0.5
    for adim in range(1, 3001):
        g = gradyan(ag_loss, p, h=1e-5)
        p = [pi - lr * gi for pi, gi in zip(p, g)]
        gecmis.append(ag_loss(p))
        if adim % 500 == 0:
            print(f"  adim {adim:5d}  loss = {gecmis[-1]:.6f}")

    print("\nTahminler:")
    print(f"{'x1':>4} {'x2':>4} {'beklenen':>10} {'tahmin':>10}")
    for x, y in zip(XOR_X, XOR_Y):
        print(f"{x[0]:4.0f} {x[1]:4.0f} {y:10.0f} {ag_ileri(p, x):10.4f}")

    cizgi_grafik(
        list(range(len(gecmis))), gecmis,
        baslik="XOR agi egitimi (2 -> 4 -> 1)",
        x_ad="adim", y_ad="loss (MSE)",
        dosya="outputs/xor_loss.png",
    )

    print("\nOKUNACAK SEY: kod Adim 1-5'te yazdigimizin AYNISI. Sadece")
    print("parametre sayisi artti. GPT'ye giden yol da bu — ayni dongu,")
    print("milyarlarca parametre ve backprop ile.")


if __name__ == "__main__":
    print("=" * 62)
    print("ADIM 5 — SAYISAL TUREV VE GRADIENT DESCENT")
    print("=" * 62)

    turev_dogrulama()
    egitim_deneyi()
    ogrenme_orani_deneyi()
    sinir_agi_deneyi()

    print("\n" + "=" * 62)
    print("HAFTA 1 TAMAM.")
    print("Ogrenme = loss'u olc, her parametrenin loss'u nasil etkiledigini")
    print("bul, hepsini kucuk bir adim ters yone tasi. Tekrarla.")
    print("=" * 62)
