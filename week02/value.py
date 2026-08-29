"""
value.py — Tamamlanmış Value motoru (micrograd'ın kalbi)

Adım 1-3'te parça parça kurduğumuz sınıfın son hali. Adım 4 ve 5 bunu
`from value import Value` diye kullanıyor.

Desteklenen işlemler ve yerel türevleri:

    out = a + b        da: 1                db: 1
    out = a * b        da: b                db: a
    out = a ** k       da: k * a^(k-1)      (k sabit sayı)
    out = a.exp()      da: e^a  = out.data  (kendi çıktısı!)
    out = a.tanh()     da: 1 - out^2
    out = a.relu()     da: 1 if a>0 else 0

Gerisi bunların birleşimi:
    -a      = a * (-1)
    a - b   = a + (-b)
    a / b   = a * b**(-1)

NOT: Repo kuralı gereği burada numpy/torch yok, sadece `math`.
"""

import math


class Value:
    """Tek bir skaler + gradient + kendisini üreten grafın izi."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    # ------------------------------------------------------------------
    # Temel işlemler (her biri: forward + kendi _backward'ı)
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        """a ** k  — k sabit bir sayı olmalı (Value degil)."""
        assert isinstance(k, (int, float)), "us sabit sayi olmali"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # kuvvet kurali: d/da a^k = k * a^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """e^x — turevi yine kendisi: d/dx e^x = e^x = out.data"""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Tek parca tanh. Adim 4'te bunu exp/bolme/us ile de kuruyoruz."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Türetilmiş kolaylıklar
    # ------------------------------------------------------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):   # 2 + a
        return self + other

    def __rmul__(self, other):   # 2 * a
        return self * other

    def __rsub__(self, other):   # 2 - a
        return Value(other) + (-self)

    def __rtruediv__(self, other):  # 2 / a
        return Value(other) * self ** -1

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self):
        sira, gorulen = [], set()

        def olustur(v):
            if v in gorulen:
                return
            gorulen.add(v)
            for cocuk in v._prev:
                olustur(cocuk)
            sira.append(v)

        olustur(self)
        self.grad = 1.0
        for dugum in reversed(sira):
            dugum._backward()


# ---------------------------------------------------------------------------
# tanh'ın parçalanmış hali — Adım 4 bunu kullanıyor
# ---------------------------------------------------------------------------


def tanh_parcali(x):
    """
    tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

    Tek bir tanh dugumu yerine: carpma, exp, toplama, us(-1), carpma.
    Ayni sonucu ve ayni gradient'i vermeli — cunku zincir kurali
    hangi parcalanmayi sectiginizi umursamaz.
    """
    e = (2 * x).exp()
    return (e - 1) / (e + 1)
