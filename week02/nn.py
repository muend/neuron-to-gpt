"""
nn.py — Value'nun üstüne kurulan minik sinir ağı kütüphanesi

Hiyerarşi (hepsi Modul'den türüyor):

    Neuron  : n girdi -> 1 çıktı.   o = tanh(w·x + b)
    Layer   : aynı girdiyi alan n tane Neuron -> n çıktı
    MLP     : ard arda Layer'lar (Multi-Layer Perceptron)

Tek kural: her şey Value ile yapılır, dolayısıyla loss.backward() dediğimizde
gradient tüm ağ boyunca geriye akar. Ayrı bir "backprop kodu" yazmıyoruz —
Value zaten biliyor.

parameters() : ağdaki TÜM öğrenilebilir Value'ları tek listede verir.
               Eğitim döngüsü sadece bu listeyi gezer.
zero_grad()  : her adımdan önce gradient'leri sıfırlar. Unutulursa gradient'ler
               üst üste birikir (çünkü `+=` kullanıyoruz) — videodaki meşhur bug.
"""

import random

from value import Value


class Modul:
    """Ortak davranis: parametreleri listele, gradient'leri sifirla."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Modul):
    def __init__(self, girdi_sayisi, dogrusal=False):
        # Kucuk rastgele baslangic: hepsi 0 olsa tum noronlar ayni sey ogrenirdi
        # (simetri kirilmasi). Cok buyuk olsa tanh doyar, gradient olur.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(girdi_sayisi)]
        self.b = Value(0.0)
        self.dogrusal = dogrusal   # True ise aktivasyon yok (cikis katmani icin)

    def __call__(self, x):
        # w·x + b  — sum'in baslangici self.b, boylece bias de grafa girer
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act if self.dogrusal else act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        tur = "Dogrusal" if self.dogrusal else "TanH"
        return f"{tur}Neuron({len(self.w)})"


class Layer(Modul):
    def __init__(self, girdi_sayisi, noron_sayisi, **kw):
        self.noronlar = [Neuron(girdi_sayisi, **kw) for _ in range(noron_sayisi)]

    def __call__(self, x):
        cikis = [n(x) for n in self.noronlar]
        return cikis[0] if len(cikis) == 1 else cikis

    def parameters(self):
        return [p for n in self.noronlar for p in n.parameters()]

    def __repr__(self):
        return f"Layer[{', '.join(str(n) for n in self.noronlar)}]"


class MLP(Modul):
    def __init__(self, girdi_sayisi, katman_boylari):
        # ornek: MLP(3, [4, 4, 1]) -> 3 girdi, 4-4 gizli, 1 cikti
        boylar = [girdi_sayisi] + katman_boylari
        self.katmanlar = [
            Layer(boylar[i], boylar[i + 1])
            for i in range(len(katman_boylari))
        ]

    def __call__(self, x):
        for katman in self.katmanlar:
            x = katman(x)
        return x

    def parameters(self):
        return [p for k in self.katmanlar for p in k.parameters()]

    def __repr__(self):
        return "MLP(\n  " + "\n  ".join(str(k) for k in self.katmanlar) + "\n)"
