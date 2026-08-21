"""
ADIM 1 — Tek nöronun forward pass'i (kütüphane YOK, sadece math)

Bir nöron aslında iki satırlık bir fikir:

    z = w1*x1 + w2*x2 + ... + wn*xn + b      <- agirlikli toplam
    a = aktivasyon(z)                         <- dogrusal olmayan sikistirma

Sözlük:
    x  (input)      : nörona giren sayılar
    w  (weight)     : her girdinin ne kadar önemli olduğu — ÖĞRENİLİR
    b  (bias)       : nöronun "varsayılan eğilimi" — ÖĞRENİLİR
    z  (logit)      : ham skor
    a  (activation) : nöronun çıktısı
    parametre       : w'lar + b. Model "öğrenirken" değişen tek şey bunlar.

Çalıştır:  python adim1_tek_noron.py
"""

import math

# ---------------------------------------------------------------------------
# 1) Aktivasyon fonksiyonları — hepsi elle, math modülü dışında hiçbir şey yok
# ---------------------------------------------------------------------------


def sigmoid(z):
    """(-sonsuz, +sonsuz) -> (0, 1). Olasılık gibi okunur."""
    # Taşma (overflow) korumasi: exp(1000) Python'da hata verir.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def relu(z):
    """Negatifi kes, pozitifi aynen geçir. Modern ağların iş atı."""
    return z if z > 0 else 0.0


def tanh(z):
    """(-1, 1) arası. Sigmoid'in sıfır merkezli kardeşi."""
    if z > 20:
        return 1.0
    if z < -20:
        return -1.0
    e2 = math.exp(2 * z)
    return (e2 - 1) / (e2 + 1)


def linear(z):
    """Aktivasyon yok. Regresyon çıkışlarında kullanılır."""
    return z


AKTIVASYONLAR = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear,
}


# ---------------------------------------------------------------------------
# 2) Forward pass — önce en çıplak haliyle
# ---------------------------------------------------------------------------


def noron_forward(girdiler, agirliklar, bias, aktivasyon="sigmoid"):
    """
    Tek bir nöronun forward pass'i.

    girdiler   : [x1, x2, ...]
    agirliklar : [w1, w2, ...]  (girdilerle aynı uzunlukta)
    bias       : tek sayı
    """
    if len(girdiler) != len(agirliklar):
        raise ValueError(
            f"Girdi sayisi ({len(girdiler)}) ile agirlik sayisi "
            f"({len(agirliklar)}) esit olmali."
        )

    # Ağırlıklı toplam: zip + döngü. numpy'siz "dot product" budur.
    z = bias
    for x, w in zip(girdiler, agirliklar):
        z += x * w

    f = AKTIVASYONLAR[aktivasyon]
    return f(z)


# ---------------------------------------------------------------------------
# 3) Aynı şey, ama nesne olarak — Adım 2'de katman kurarken lazım olacak
# ---------------------------------------------------------------------------


class Noron:
    """Ağırlıkları ve bias'ı içinde tutan tek nöron."""

    def __init__(self, agirliklar, bias, aktivasyon="sigmoid"):
        self.agirliklar = list(agirliklar)
        self.bias = float(bias)
        self.aktivasyon = aktivasyon

    def forward(self, girdiler):
        return noron_forward(girdiler, self.agirliklar, self.bias, self.aktivasyon)

    # Nöronu fonksiyon gibi çağırabilmek için:  n(x)
    __call__ = forward

    def parametreler(self):
        """Öğrenilecek her şey tek listede. Adım 5'te bunu güncelleyeceğiz."""
        return self.agirliklar + [self.bias]

    def __repr__(self):
        return (f"Noron(w={[round(w, 4) for w in self.agirliklar]}, "
                f"b={self.bias:.4f}, akt={self.aktivasyon})")


# ---------------------------------------------------------------------------
# 4) Deneme
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("ADIM 1 — TEK NORON FORWARD PASS")
    print("=" * 62)

    girdiler = [0.5, -1.2, 2.0]
    agirliklar = [0.8, 0.3, -0.5]
    bias = 0.1

    # Ara adımları elle gösterelim ki "kara kutu" hissi kalmasın
    print("\nGirdiler   x =", girdiler)
    print("Agirliklar w =", agirliklar)
    print("Bias       b =", bias)

    z = bias
    print("\nAgirlikli toplam adim adim:")
    print(f"  z = b = {bias}")
    for i, (x, w) in enumerate(zip(girdiler, agirliklar), start=1):
        z += x * w
        print(f"  z += x{i}*w{i} = {x} * {w} = {x * w:+.4f}   ->  z = {z:.4f}")

    print(f"\nHam skor z = {z:.6f}")
    print("\nAyni z, farkli aktivasyonlardan gecince:")
    for ad, f in AKTIVASYONLAR.items():
        print(f"  {ad:<8} -> {f(z):+.6f}")

    # Nesne hali
    print("\n--- Noron sinifi ile ayni sonuc ---")
    n = Noron(agirliklar, bias, aktivasyon="sigmoid")
    print(n)
    print("n(x) =", round(n(girdiler), 6))

    # Küçük sezgi denemesi: bias'ı oynatınca çıktı ne oluyor?
    print("\n--- Bias'i oynatinca cikti nasil kayiyor? ---")
    for b in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        n.bias = b
        print(f"  b={b:+.1f}  ->  cikti = {n(girdiler):.6f}")

    print("\nSezgi: w'lar girdinin 'onemini', b ise 'esigi' belirliyor.")
    print("Ogrenmek = bu sayilari daha iyi degerlere tasimak. (Adim 5)")
