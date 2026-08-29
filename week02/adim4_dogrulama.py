"""
ADIM 4 — tanh'ı parçala + 3 yönlü doğrulama (Video: 1:27:05 - 1:43:55)

İki iş var:

A) tanh'ı atomik bir düğüm olmaktan çıkar, exp / toplama / bölme / üs ile
   yeniden kur. Aynı gradient'i alıyor muyuz? Almalıyız — çünkü zincir kuralı
   grafı nasıl parçaladığınızı umursamaz. İstediğin kadar ince böl, sonuç aynı.
   Bu, "hangi işlemi atomik yazacağım?" sorusunun bir HIZ/kolaylık sorusu
   olduğunu, matematik sorusu olmadığını gösterir.

B) Aynı ifadeyi üç bağımsız yolla hesapla ve karşılaştır:
      1. kendi backward()'ımız      (analitik, geriye tek geçiş)
      2. sayısal türev              (geçen haftanın yöntemi, merkezi fark)
      3. PyTorch autograd           (endüstri standardı referans)
   Üçü eşleşiyorsa motor doğru demektir.

Sayısal türev neden birebir aynı çıkmaz?
   Kayan nokta (floating point) hatası. h çok büyükse yaklaşım kabası,
   çok küçükse çıkarmada anlamlı basamak kaybı olur. 1e-5 civarı tatlı nokta.
   Bu yüzden "eşit mi" diye değil, "farkı 1e-6'dan küçük mü" diye bakıyoruz.

Calistir:  python3 adim4_dogrulama.py     (torch kurulu degilse o kismi atlar)
"""

from value import Value, tanh_parcali


# ---------------------------------------------------------------------------
# A) tanh: tek parça mı, parçalanmış mı — fark eder mi?
# ---------------------------------------------------------------------------


def bolum_a():
    print("=" * 70)
    print("A) tanh'i parcalamak gradient'i degistirir mi?")
    print("=" * 70)

    def noron(tanh_fonksiyonu):
        x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
        w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
        b = Value(6.8813735870195432, label="b")
        n = x1 * w1 + x2 * w2 + b
        o = tanh_fonksiyonu(n)
        o.backward()
        return o, {"w1": w1.grad, "w2": w2.grad, "x1": x1.grad, "b": b.grad}

    o1, g1 = noron(lambda n: n.tanh())          # tek dugum
    o2, g2 = noron(tanh_parcali)                 # exp + toplama + us(-1) + carpma

    print(f"  tek parca tanh   : o = {o1.data:.10f}")
    print(f"  parcalanmis tanh : o = {o2.data:.10f}")
    print()
    print("  gradient karsilastirmasi:")
    for k in g1:
        fark = abs(g1[k] - g2[k])
        durum = "OK " if fark < 1e-9 else "HATA"
        print(f"    [{durum}] {k:>2}: tek={g1[k]:+.10f}  parcali={g2[k]:+.10f}  fark={fark:.2e}")

    print()
    print("  Sonuc: ayni. Atomik islem secimi sadece HIZ ve kolaylik meselesi.")
    print("  (PyTorch da boyle yapar: tanh'i tek kernel yazar, cunku hizli.)")
    print()


# ---------------------------------------------------------------------------
# B) Üç yönlü doğrulama
# ---------------------------------------------------------------------------

# Test edilecek ifade — kasten karışık: toplama, çarpma, üs, bölme, exp, tanh
#   L = tanh(a*b + c) * (a / (b**2 + 1)) + (a - c).exp()
DEGERLER = {"a": 2.0, "b": -3.0, "c": 0.5}


def ifade_value(a, b, c):
    return (a * b + c).tanh() * (a / (b ** 2 + 1)) + (a - c).exp()


def ifade_duz(a, b, c):
    """Ayni ifade, saf Python float ile — sayisal turev icin."""
    import math
    return math.tanh(a * b + c) * (a / (b ** 2 + 1)) + math.exp(a - c)


def kendi_backward():
    a = Value(DEGERLER["a"], label="a")
    b = Value(DEGERLER["b"], label="b")
    c = Value(DEGERLER["c"], label="c")
    L = ifade_value(a, b, c)
    L.backward()
    return L.data, {"a": a.grad, "b": b.grad, "c": c.grad}


def sayisal(h=1e-5):
    """Merkezi fark: ( f(x+h) - f(x-h) ) / 2h — tek yonlu farktan daha hassas."""
    d = dict(DEGERLER)
    L = ifade_duz(**d)
    gradler = {}
    for k in d:
        arti = dict(d); arti[k] += h
        eksi = dict(d); eksi[k] -= h
        gradler[k] = (ifade_duz(**arti) - ifade_duz(**eksi)) / (2 * h)
    return L, gradler


def pytorch():
    try:
        import torch
    except ImportError:
        return None, None
    a = torch.tensor([DEGERLER["a"]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([DEGERLER["b"]], dtype=torch.double, requires_grad=True)
    c = torch.tensor([DEGERLER["c"]], dtype=torch.double, requires_grad=True)
    L = torch.tanh(a * b + c) * (a / (b ** 2 + 1)) + torch.exp(a - c)
    L.backward()
    return L.item(), {"a": a.grad.item(), "b": b.grad.item(), "c": c.grad.item()}


def bolum_b():
    print("=" * 70)
    print("B) Uc yonlu dogrulama")
    print("=" * 70)
    print("  L = tanh(a*b + c) * (a / (b^2 + 1)) + exp(a - c)")
    print(f"  a={DEGERLER['a']}, b={DEGERLER['b']}, c={DEGERLER['c']}")
    print()

    L1, g1 = kendi_backward()
    L2, g2 = sayisal()
    L3, g3 = pytorch()

    print(f"  forward:  bizim={L1:.10f}   duz python={L2:.10f}", end="")
    print(f"   torch={L3:.10f}" if L3 is not None else "   torch=(kurulu degil)")
    print()

    baslik = f"  {'':>3} {'backward()':>16} {'sayisal':>16} {'pytorch':>16}   {'durum':>6}"
    print(baslik)
    print("  " + "-" * (len(baslik) - 2))

    hepsi_ok = True
    for k in ["a", "b", "c"]:
        t = f"{g3[k]:+16.10f}" if g3 else f"{'-':>16}"
        # sayisal turevden 1e-6, torch'tan 1e-12 hassasiyet bekliyoruz
        ok = abs(g1[k] - g2[k]) < 1e-6 and (g3 is None or abs(g1[k] - g3[k]) < 1e-12)
        hepsi_ok = hepsi_ok and ok
        print(f"  {k:>3} {g1[k]:+16.10f} {g2[k]:+16.10f} {t}   {'OK' if ok else 'HATA':>6}")

    print()
    if g3 is None:
        print("  (PyTorch kurulu degil: 'pip install torch' ile ucuncu sutun dolar.")
        print("   Repo kurali geregi torch sadece bu DOGRULAMA dosyasinda opsiyonel.)")
    kaynak_sayisi = 3 if g3 is not None else 2
    print("  SONUC:", f"{kaynak_sayisi}/{kaynak_sayisi} kaynak eslesiyor — motor dogru."
          if hepsi_ok else "!!! fark var, bak.")
    print()
    print("  Not: sayisal turev tam esit degil, ~1e-10 sapiyor. Bu bug degil,")
    print("       kayan nokta hatasi. Zaten sayisal turevi test icin kullaniyoruz,")
    print("       egitimde degil — 13.000 parametre icin 26.000 forward pass ederdi.")
    print()


if __name__ == "__main__":
    bolum_a()
    bolum_b()
    print("Motor hazir. Simdi ustune ag kuruyoruz -> Adim 5.")
