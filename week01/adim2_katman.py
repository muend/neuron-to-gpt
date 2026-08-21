"""
ADIM 2 — Katman (layer) ve çok katmanlı ağ (kütüphane YOK)

Fikir basit: bir KATMAN, aynı girdiyi paylaşan birden fazla nörondur.

    girdi x (3 sayi)
        |--> noron1 --> a1
        |--> noron2 --> a2      => katmanin ciktisi [a1, a2, a3]
        |--> noron3 --> a3

Bir AĞ ise katmanların üst üste dizilmesidir: bir katmanın çıktısı
bir sonrakinin girdisi olur. 3Blue1Brown videosundaki 784 -> 16 -> 16 -> 10
yapısı tam olarak budur.

Parametre sayısı:
    bir katman için  = (girdi_sayisi * noron_sayisi) + noron_sayisi
                       \_____ agirliklar _____/       \__ bias'lar __/

Çalıştır:  python adim2_katman.py
"""

import random

from adim1_tek_noron import Noron, AKTIVASYONLAR  # noqa: F401


class Katman:
    """Aynı girdiyi alan n tane nöron."""

    def __init__(self, girdi_sayisi, noron_sayisi, aktivasyon="sigmoid", tohum=None):
        self.girdi_sayisi = girdi_sayisi
        self.noron_sayisi = noron_sayisi
        self.aktivasyon = aktivasyon

        rng = random.Random(tohum)
        # Küçük rastgele başlangıç: hepsi 0 olsaydı tüm nöronlar aynı şeyi
        # öğrenirdi ("symmetry breaking" problemi).
        self.noronlar = [
            Noron(
                agirliklar=[rng.uniform(-1, 1) for _ in range(girdi_sayisi)],
                bias=rng.uniform(-1, 1),
                aktivasyon=aktivasyon,
            )
            for _ in range(noron_sayisi)
        ]

    def forward(self, girdiler):
        """Girdi listesi -> her nöronun çıktısından oluşan liste."""
        if len(girdiler) != self.girdi_sayisi:
            raise ValueError(
                f"Bu katman {self.girdi_sayisi} girdi bekliyor, "
                f"{len(girdiler)} geldi."
            )
        return [n.forward(girdiler) for n in self.noronlar]

    __call__ = forward

    def parametreler(self):
        p = []
        for n in self.noronlar:
            p.extend(n.parametreler())
        return p

    def parametre_sayisi(self):
        return self.girdi_sayisi * self.noron_sayisi + self.noron_sayisi

    def __repr__(self):
        return (f"Katman({self.girdi_sayisi} -> {self.noron_sayisi}, "
                f"{self.aktivasyon}, {self.parametre_sayisi()} parametre)")


class Ag:
    """Katmanları sırayla uygulayan basit ileri beslemeli (feed-forward) ağ."""

    def __init__(self, boyutlar, aktivasyon="sigmoid", son_aktivasyon=None, tohum=None):
        """
        boyutlar : ör. [3, 4, 4, 1]  ->  3 girdi, iki gizli katman, 1 cikti
        son_aktivasyon : None ise 'aktivasyon' ile aynı kullanılır.
        """
        self.boyutlar = list(boyutlar)
        self.katmanlar = []
        katman_sayisi = len(boyutlar) - 1

        for i in range(katman_sayisi):
            son_mu = (i == katman_sayisi - 1)
            akt = son_aktivasyon if (son_mu and son_aktivasyon) else aktivasyon
            self.katmanlar.append(
                Katman(boyutlar[i], boyutlar[i + 1], aktivasyon=akt,
                       tohum=None if tohum is None else tohum + i)
            )

    def forward(self, girdiler, ara_adimlari_yaz=False):
        a = list(girdiler)
        if ara_adimlari_yaz:
            print(f"  girdi      : {[round(v, 4) for v in a]}")
        for i, katman in enumerate(self.katmanlar, start=1):
            a = katman.forward(a)
            if ara_adimlari_yaz:
                print(f"  katman {i}   : {[round(v, 4) for v in a]}")
        return a

    __call__ = forward

    def parametreler(self):
        p = []
        for k in self.katmanlar:
            p.extend(k.parametreler())
        return p

    def parametre_sayisi(self):
        return sum(k.parametre_sayisi() for k in self.katmanlar)

    def __repr__(self):
        satirlar = [f"Ag({' -> '.join(map(str, self.boyutlar))})"]
        for k in self.katmanlar:
            satirlar.append("  " + repr(k))
        satirlar.append(f"  TOPLAM parametre: {self.parametre_sayisi()}")
        return "\n".join(satirlar)


if __name__ == "__main__":
    print("=" * 62)
    print("ADIM 2 — KATMAN VE COK KATMANLI AG")
    print("=" * 62)

    x = [0.5, -1.2, 2.0]

    print("\n--- Tek katman: 3 girdi -> 4 noron ---")
    k = Katman(girdi_sayisi=3, noron_sayisi=4, aktivasyon="sigmoid", tohum=42)
    print(k)
    for i, n in enumerate(k.noronlar, start=1):
        print(f"  noron{i}: {n}")
    print("\ncikti =", [round(v, 6) for v in k(x)])

    print("\n--- Ag: 3 -> 4 -> 4 -> 1 ---")
    ag = Ag([3, 4, 4, 1], aktivasyon="tanh", son_aktivasyon="sigmoid", tohum=7)
    print(ag)
    print("\nForward pass, katman katman:")
    cikti = ag.forward(x, ara_adimlari_yaz=True)
    print("\nNihai cikti =", round(cikti[0], 6))

    print("\n--- 3Blue1Brown'un rakam agi: 784 -> 16 -> 16 -> 10 ---")
    mnist_ag = Ag([784, 16, 16, 10], aktivasyon="sigmoid", tohum=1)
    print(mnist_ag)
    print("\nVideodaki '13.002 parametre' sayisi tam olarak bu.")
    print("Ogrenmek = bu 13.002 sayiyi ayarlamak demek.")
