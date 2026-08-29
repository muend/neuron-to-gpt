"""
ADIM 2 — Gradient'leri ELLE doldur (Video: 32:10 - 51:10 ve 52:52 - 1:09:02)

Otomatikleştirmeden önce zincir kuralını (chain rule) elle uyguluyoruz.
Amaç: backward() yazdığımızda "sihir" olmasın, ne yaptığını bilelim.

Sözlük:
    grad (dL/dx) : "x'i 1 birim artirirsam L ne kadar artar?" — L'nin x'e
                   gore turevi. Her Value'da .grad alani olarak durur.
    chain rule   : L, d uzerinden e'ye baglıysa
                       dL/de = dL/dd * dd/de
                   yani "yerel turev" ile "yukaridan gelen gradient"i CARP.
    yerel turev  : bir islemin kendi turevi. Toplama ve carpmada ezberi kolay:
                       c = a + b  ->  dc/da = 1,   dc/db = 1     (aynen gecir)
                       c = a * b  ->  dc/da = b,   dc/db = a     (capraz kopya)

Iki ornek var:
    A) L = (a*b + c) * f          -> saf zincir kurali
    B) tek noron: o = tanh(w1x1 + w2x2 + b)   -> tanh'in turevi de girer

Calistir:  python3 adim2_elle_gradient.py
"""

import math


# ---------------------------------------------------------------------------
# Value: Adım 1'in aynısı + .grad alanı + tanh
# ---------------------------------------------------------------------------


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0          # YENI: dL/dself. Baslangicta "etkisi yok" = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def tanh(self):
        """tanh(x) = (e^2x - 1)/(e^2x + 1). Sikistirir: (-inf,inf) -> (-1,1)."""
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        return Value(t, (self,), "tanh")


def sayisal_turev(fonksiyon, h=1e-6):
    """Kontrol icin: L(x+h) - L(x) / h. Gecen haftanin yontemi."""
    L1 = fonksiyon(0.0)
    L2 = fonksiyon(h)
    return (L2 - L1) / h


# ---------------------------------------------------------------------------
# ORNEK A — L = (a*b + c) * f
# ---------------------------------------------------------------------------


def ornek_a():
    print("=" * 66)
    print("ORNEK A:  e = a*b,  d = e + c,  L = d * f")
    print("=" * 66)

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    f = Value(-2.0, label="f")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    L = d * f; L.label = "L"

    print(f"forward: e={e.data}, d={d.data}, L={L.data}")
    print()
    print("Simdi GERIYE dogru, elle:")

    # 1) Kokun kendine gore turevi: dL/dL = 1. Her backprop buradan baslar.
    L.grad = 1.0
    print("  dL/dL = 1                      (bir seyin kendine gore turevi 1)")

    # 2) L = d * f  -> carpma: capraz kopya
    #    dL/dd = f = -2 ,  dL/df = d = 4
    d.grad = f.data * L.grad
    f.grad = d.data * L.grad
    print(f"  dL/dd = f * dL/dL = {d.grad}      (carpmada digerinin degeri)")
    print(f"  dL/df = d * dL/dL = {f.grad}")

    # 3) d = e + c  -> toplama: gradient'i AYNEN gecir (yerel turev 1)
    #    dL/de = dL/dd * dd/de = dL/dd * 1
    e.grad = 1.0 * d.grad
    c.grad = 1.0 * d.grad
    print(f"  dL/de = 1 * dL/dd = {e.grad}     (toplama = gradient dagitici)")
    print(f"  dL/dc = 1 * dL/dd = {c.grad}")

    # 4) e = a * b -> yine carpma, ama yukaridan gelen gradient ile CARP
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad
    print(f"  dL/da = b * dL/de = {a.grad}     <- ZINCIR KURALI burada")
    print(f"  dL/db = a * dL/de = {b.grad}")

    print()
    print("Kontrol (sayisal turev ile):")
    for isim, dugum, oynat in [
        ("a", a, lambda h: ((Value(2.0 + h) * Value(-3.0)) + Value(10.0)) * Value(-2.0)),
        ("b", b, lambda h: ((Value(2.0) * Value(-3.0 + h)) + Value(10.0)) * Value(-2.0)),
        ("c", c, lambda h: ((Value(2.0) * Value(-3.0)) + Value(10.0 + h)) * Value(-2.0)),
        ("f", f, lambda h: ((Value(2.0) * Value(-3.0)) + Value(10.0)) * Value(-2.0 + h)),
    ]:
        sayisal = sayisal_turev(lambda h: oynat(h).data)
        print(f"  d{isim}: elle={dugum.grad:+.4f}   sayisal={sayisal:+.4f}")

    print()
    print("Sezgi kontrolu: a.grad = 6 > 0 -> a'yi ARTIRIRSAM L artar.")
    print("               Loss'u kucultmek istiyorsak a'yi AZALTMALIYIZ.")
    print("               (gradient descent tam olarak bu: ters yone kucuk adim)")
    print()


# ---------------------------------------------------------------------------
# ORNEK B — Tek noron:  o = tanh(x1*w1 + x2*w2 + b)
# ---------------------------------------------------------------------------


def ornek_b():
    print("=" * 66)
    print("ORNEK B:  Tek noron -> o = tanh(x1*w1 + x2*w2 + b)")
    print("=" * 66)

    # Videodaki degerler (b, sonuclar yuvarlak ciksin diye secilmis)
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1; x1w1.label = "x1*w1"
    x2w2 = x2 * w2; x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1+x2w2"
    n = x1w1x2w2 + b; n.label = "n"     # n = ham skor (logit)
    o = n.tanh(); o.label = "o"          # o = noronun ciktisi

    print(f"forward: n = {n.data:.4f} -> o = tanh(n) = {o.data:.4f}")
    print()
    print("Geriye, elle:")

    o.grad = 1.0
    print("  do/do = 1")

    # tanh'in turevi: d/dx tanh(x) = 1 - tanh(x)^2
    # Neden guzel? Ciktinin kendisiyle yaziliyor -> forward'da hesapladigimiz
    # o.data'yi tekrar kullaniyoruz, yeni exp hesabi yok.
    n.grad = (1 - o.data ** 2) * o.grad
    print(f"  do/dn = (1 - o^2) = {n.grad:.4f}    <- tanh'in turevi")

    # Buradan sonrasi hep toplama (gradient'i aynen gecir)
    x1w1x2w2.grad = n.grad
    b.grad = n.grad
    x1w1.grad = x1w1x2w2.grad
    x2w2.grad = x1w1x2w2.grad
    print(f"  do/db = {b.grad:.4f}                 (toplama: aynen gecir)")

    # ve carpmalar (capraz kopya * yukaridan gelen)
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad
    print(f"  do/dw1 = x1 * {x1w1.grad:.4f} = {w1.grad:.4f}")
    print(f"  do/dw2 = x2 * {x2w2.grad:.4f} = {w2.grad:.4f}   <- x2=0 oldugu icin 0!")
    print(f"  do/dx1 = w1 * {x1w1.grad:.4f} = {x1.grad:.4f}")

    print()
    print("Kontrol (sayisal turev):")

    def noron(dw1=0.0, dw2=0.0, db=0.0):
        n_ = 2.0 * (-3.0 + dw1) + 0.0 * (1.0 + dw2) + (6.8813735870195432 + db)
        return math.tanh(n_)

    print(f"  w1: elle={w1.grad:+.4f}  sayisal={sayisal_turev(lambda h: noron(dw1=h)):+.4f}")
    print(f"  w2: elle={w2.grad:+.4f}  sayisal={sayisal_turev(lambda h: noron(dw2=h)):+.4f}")
    print(f"  b : elle={b.grad:+.4f}  sayisal={sayisal_turev(lambda h: noron(db=h)):+.4f}")

    print()
    print("Ders: w2'nin gradient'i 0, cunku x2=0. Girdi 0 ise o agirlik hicbir")
    print("      sey yapmiyor demektir -> ogrenme sinyali de almaz.")
    print()
    print("Ezberlenecek uc kural:")
    print("   +    : gradient'i cocuklara AYNEN dagit")
    print("   *    : gradient'i DIGER carpanla carpip dagit")
    print("   tanh : gradient'i (1 - cikti^2) ile carp")
    print()


if __name__ == "__main__":
    ornek_a()
    ornek_b()
    print("Bu isi 5 dugum icin elle yaptik. 13.000 parametre icin?")
    print("-> Adim 3: backward() bunu otomatiklestiriyor.")
