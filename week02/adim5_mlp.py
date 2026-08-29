"""
ADIM 5 — Neuron / Layer / MLP ve gerçek eğitim (Video: 1:43:55 - 2:14:03)

Artık her şey elimizde. Eğitim döngüsü DÖRT satır:

    1. forward   : tahminleri hesapla, loss'u bul       -> graf kurulur
    2. zero_grad : gradient'leri sıfırla                -> ESKI adimin izini sil
    3. backward  : loss.backward()                      -> gradient'ler dolar
    4. update    : p.data -= lr * p.grad                -> parametreleri oynat

Sıra önemli: zero_grad MUTLAKA backward'dan ÖNCE. Değilse gradient'ler
birikir ve adımlar giderek büyür — video 2:03 civarındaki meşhur bug.
Bu dosyanın sonunda o bug'ı kasten açıp ne olduğunu gösteriyoruz.

Loss: MSE (mean squared error) — (tahmin - gercek)^2 toplamı.
    Neden kare? Hem işareti yok eder hem büyük hataları daha çok cezalandırır
    hem de türevi temizdir: d/dtahmin (tahmin-y)^2 = 2(tahmin-y).

Calistir:  python3 adim5_mlp.py
"""

import random

from nn import MLP

random.seed(1337)   # tekrarlanabilirlik: her calistirmada ayni sonuc

# Videodaki minik veri seti: 4 ornek, 3 girdi, hedef +1 / -1
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
Y = [1.0, -1.0, -1.0, 1.0]


def mse_loss(model, X, Y):
    """Tek bir Value dondurur — grafin koku. Buradan backward baslar."""
    tahminler = [model(x) for x in X]
    return sum((t - y) ** 2 for t, y in zip(tahminler, Y)), tahminler


def ascii_egri(kayiplar, yukseklik=10, genislik=60):
    """matplotlib olmadan loss egrisi."""
    if len(kayiplar) > genislik:
        adim = len(kayiplar) / genislik
        kayiplar = [kayiplar[int(i * adim)] for i in range(genislik)]
    ust, alt = max(kayiplar), min(kayiplar)
    aralik = (ust - alt) or 1.0
    izgara = [[" "] * len(kayiplar) for _ in range(yukseklik)]
    for x, v in enumerate(kayiplar):
        y = int((1 - (v - alt) / aralik) * (yukseklik - 1))
        izgara[y][x] = "*"
    print(f"  loss {ust:.4f} |" + "".join(izgara[0]))
    for satir in izgara[1:-1]:
        print("              |" + "".join(satir))
    print(f"       {alt:.4f} |" + "".join(izgara[-1]))
    print("              +" + "-" * len(kayiplar) + "> adim")


def egit(model, adim_sayisi=100, ogrenme_orani=0.05, zero_grad_yap=True,
         yazdir=True):
    kayiplar = []
    for adim in range(adim_sayisi):
        # 1) FORWARD — her cagrida yeni bir computation graph kurulur
        loss, tahminler = mse_loss(model, X, Y)

        # 2) ZERO_GRAD — eski adimin gradient'lerini sil
        #    (Value'da '+=' kullaniyoruz; silmezsek birikirler)
        if zero_grad_yap:
            model.zero_grad()

        # 3) BACKWARD — tek satirda tum aga gradient dagit
        loss.backward()

        # 4) UPDATE — gradient'in TERS yonune kucuk adim
        for p in model.parameters():
            p.data -= ogrenme_orani * p.grad

        kayiplar.append(loss.data)
        if yazdir and (adim < 3 or adim % 20 == 19):
            print(f"    adim {adim:>3}: loss = {loss.data:.6f}")
    return kayiplar, tahminler


def bolum_dogru_egitim():
    print("=" * 70)
    print("1) MLP(3, [4,4,1]) — dogru egitim")
    print("=" * 70)

    model = MLP(3, [4, 4, 1])
    print(f"  {model}")
    print(f"  toplam parametre: {len(model.parameters())}")
    print()
    print("  Not: sayisal turevle bir adim = 2 x 41 = 82 forward pass.")
    print("       backprop ile bir adim = 1 forward + 1 backward. Fark bu.")
    print()

    ilk_loss, ilk_tahmin = mse_loss(model, X, Y)
    print(f"  egitimden ONCE: loss = {ilk_loss.data:.6f}")
    print(f"    tahminler = {[round(t.data, 3) for t in ilk_tahmin]}")
    print(f"    hedefler  = {Y}")
    print()

    kayiplar, tahminler = egit(model, adim_sayisi=100, ogrenme_orani=0.05)

    print()
    print(f"  egitimden SONRA: loss = {kayiplar[-1]:.8f}")
    print(f"    tahminler = {[round(t.data, 4) for t in tahminler]}")
    print(f"    hedefler  = {Y}")
    print()
    ascii_egri(kayiplar)
    print()
    print(f"  loss {kayiplar[0]:.4f} -> {kayiplar[-1]:.8f}  "
          f"({kayiplar[0] / max(kayiplar[-1], 1e-12):.0f} kat kucuk)")
    print()
    return kayiplar


def bolum_zero_grad_bug():
    print("=" * 70)
    print("2) MESHUR BUG — zero_grad() unutulursa")
    print("=" * 70)
    print("  Ayni model, ayni tohum, ayni learning rate. Tek fark: gradient")
    print("  sifirlanmiyor. Value'da '+=' kullandigimiz icin her adimda eski")
    print("  gradient'in USTUNE ekleniyor -> etkin adim boyu buyuyup gidiyor.")
    print()

    LR = 0.1
    random.seed(1337)
    k_ok, _ = egit(MLP(3, [4, 4, 1]), 60, LR, zero_grad_yap=True, yazdir=False)
    random.seed(1337)
    k_bug, _ = egit(MLP(3, [4, 4, 1]), 60, LR, zero_grad_yap=False, yazdir=False)

    print(f"  lr = {LR}")
    print(f"  {'adim':>5} {'zero_grad VAR':>16} {'zero_grad YOK':>16}")
    print("  " + "-" * 40)
    for i in [0, 9, 19, 29, 39, 59]:
        print(f"  {i:>5} {k_ok[i]:>16.6f} {k_bug[i]:>16.6f}")
    print()
    print("  Bug'li surum once daha HIZLI dusuyor (adim boyu suruyor), sonra")
    print("  loss 8.0'da donuyor: tanh'lar +-1'e doymus, gradient sifir,")
    print("  ag olmus. Geri donusu yok.")
    print()

    # Kaniti gozle gorelim: bug'li surumde gradient'lerin buyuklugu
    print("  Gradient'lerin buyudugunu gorelim (max |grad|, ilk 6 adim):")
    for etiket, sifirla in [("zero_grad VAR", True), ("zero_grad YOK", False)]:
        random.seed(1337)
        model = MLP(3, [4, 4, 1])
        satir = []
        for _ in range(6):
            loss, _ = mse_loss(model, X, Y)
            if sifirla:
                model.zero_grad()
            loss.backward()
            satir.append(max(abs(p.grad) for p in model.parameters()))
            for p in model.parameters():
                p.data -= LR * p.grad
        print(f"    {etiket}: " + "  ".join(f"{v:6.3f}" for v in satir))
    print()
    print("  Kural: forward -> zero_grad -> backward -> update. Her adimda.")
    print()


def bolum_ogrenme_orani():
    print("=" * 70)
    print("3) Learning rate'in etkisi (100 adim sonundaki loss)")
    print("=" * 70)
    for lr in [0.001, 0.01, 0.05, 0.2, 1.0]:
        random.seed(1337)
        model = MLP(3, [4, 4, 1])
        kayiplar, _ = egit(model, 100, lr, yazdir=False)
        son = kayiplar[-1]
        yorum = ("cok yavas" if son > 0.2 else
                 "iyi" if son < 0.01 else
                 "idare eder")
        if son > kayiplar[0]:
            yorum = "PATLADI (adim cok buyuk, tepeden asiyor)"
        print(f"  lr={lr:<6} son loss = {son:>12.8f}   {yorum}")
    print()
    print("  Cok kucuk lr: dogru yon, ama omur yetmez.")
    print("  Cok buyuk lr: her adimda vadinin karsi yamacina zipliyorsun.")
    print()


if __name__ == "__main__":
    bolum_dogru_egitim()
    bolum_zero_grad_bug()
    bolum_ogrenme_orani()
    print("=" * 70)
    print("HAFTA 2 TAMAM. Yazdigimiz sey: micrograd. ~150 satir.")
    print("PyTorch'un yaptigi da bu — sadece skaler yerine tensor uzerinde,")
    print("C++/CUDA ile ve yuzlerce operasyonla. Fikir birebir ayni.")
    print("=" * 70)
