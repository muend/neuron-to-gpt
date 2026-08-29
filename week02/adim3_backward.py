"""
ADIM 3 — backward(): otomatik backpropagation (Video: 1:09:02 - 1:27:05)

Adım 2'de gradient'leri elle yazdık. Şimdi aynı kuralları makineye devrediyoruz.

Fikir tek cümle:
    Her işlem, kendi yerel türevini bilen küçük bir fonksiyon (_backward)
    bıraksın. Sonra grafı TERS topolojik sırayla gezip bu fonksiyonları
    sırayla çağıralım.

Neden ters topolojik sıra?
    Bir düğümün gradient'ini çocuklarına dağıtmadan önce, o düğümün KENDİ
    gradient'i tamamlanmış olmalı. Yani ondan sonra gelen (onu kullanan) tüm
    düğümler işini bitirmiş olmalı. Ters topolojik sıra tam olarak bunu garanti
    eder.

Kritik ayrıntı — neden `+=` ve `=` değil:
    Bir değişken birden fazla yerde kullanılıyorsa (ör. b = a + a), gradient
    iki koldan gelir ve TOPLANMALIDIR. Üzerine yazarsak biri kaybolur.
    Bu, çok değişkenli zincir kuralının kendisidir. Dosyanın sonunda bu bug'ı
    canlı gösteriyoruz.

Calistir:  python3 adim3_backward.py
"""

import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None   # YENI: bu dugumun yerel geri adimi
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- ileri + geri, ikisi bir arada ----------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # toplama: yerel turev 1 -> gradient'i aynen dagit
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # carpma: yerel turev = digerinin degeri
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # Python kolaylıkları: 2 * a, a + 2 gibi yazımlar calissin diye
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    # --- otomatik backprop ----------------------------------------------

    def backward(self):
        """Bu dugumu kok kabul edip tum graf boyunca gradient'leri doldur."""
        # 1) topolojik sira: her dugum, cocuklarindan SONRA listeye girer
        sira, gorulen = [], set()

        def olustur(v):
            if v in gorulen:
                return
            gorulen.add(v)
            for cocuk in v._prev:
                olustur(cocuk)
            sira.append(v)

        olustur(self)

        # 2) kokun gradient'i 1 (dL/dL = 1)
        self.grad = 1.0

        # 3) TERS sirada gez, her dugumde yerel kurali uygula
        for dugum in reversed(sira):
            dugum._backward()


# ---------------------------------------------------------------------------
# Test 1 — Adım 2'nin A örneği, ama artık otomatik
# ---------------------------------------------------------------------------


def test_ornek_a():
    print("=" * 66)
    print("TEST 1 — L = (a*b + c) * f   [Adim 2 ile ayni cikmali]")
    print("=" * 66)

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    f = Value(-2.0, label="f")
    L = (a * b + c) * f
    L.label = "L"

    L.backward()   # tek satir — Adim 2'deki 8 satirin yerine

    beklenen = {"a": 6.0, "b": -4.0, "c": -2.0, "f": 4.0}
    for isim, v in [("a", a), ("b", b), ("c", c), ("f", f)]:
        isaret = "OK " if abs(v.grad - beklenen[isim]) < 1e-9 else "HATA"
        print(f"  [{isaret}] {isim}.grad = {v.grad:+.4f}   (elle: {beklenen[isim]:+.4f})")
    print()


# ---------------------------------------------------------------------------
# Test 2 — Tek nöron, otomatik
# ---------------------------------------------------------------------------


def test_noron():
    print("=" * 66)
    print("TEST 2 — o = tanh(x1*w1 + x2*w2 + b)")
    print("=" * 66)

    x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
    w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()

    beklenen = {"w1": 1.0, "w2": 0.0, "x1": -1.5, "b": 0.5}
    for isim, v in [("w1", w1), ("w2", w2), ("x1", x1), ("b", b)]:
        isaret = "OK " if abs(v.grad - beklenen[isim]) < 1e-9 else "HATA"
        print(f"  [{isaret}] {isim}.grad = {v.grad:+.4f}   (elle: {beklenen[isim]:+.4f})")
    print()


# ---------------------------------------------------------------------------
# Test 3 — Aynı değişkeni iki kez kullanmak: neden `+=`
# ---------------------------------------------------------------------------


def test_biriktirme():
    print("=" * 66)
    print("TEST 3 — Bir degisken iki kez kullanilirsa (neden '+=' )")
    print("=" * 66)

    # b = a + a  ->  b = 2a  ->  db/da = 2
    a = Value(3.0, label="a")
    b = a + a
    b.backward()
    print(f"  b = a + a   ->  a.grad = {a.grad}   (dogru cevap 2)")
    print("    '=' kullansaydik: ikinci kol birinciyi ezer, sonuc 1 cikardi.")

    # Daha sinsi hali: d = a*b + b   -> b hem carpimda hem toplamada
    a = Value(-2.0, label="a")
    b = Value(3.0, label="b")
    d = a * b + b
    d.backward()
    print(f"  d = a*b + b ->  b.grad = {b.grad}   (dogru cevap a+1 = -1)")
    print("    b'ye iki ayri koldan gradient geliyor: a (carpimdan) + 1 (toplamadan)")
    print()


if __name__ == "__main__":
    test_ornek_a()
    test_noron()
    test_biriktirme()
    print("Artik gradient'leri elle yazmiyoruz. Kalan is: tanh'i parcalamak")
    print("ve sonucu PyTorch ile karsilastirmak -> Adim 4.")
