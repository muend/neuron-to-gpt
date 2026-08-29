"""
ADIM 1 — Value sınıfı ve computation graph (Video: 19:09 - 32:10)

Geçen hafta türevi "parametreyi bir tık oynat, loss'a bak" diye SAYISAL
hesapladık. 13.002 parametreli bir ağda bu bir adım için 26.004 forward pass
demekti. Bu hafta doğru yolu kuruyoruz: her sayı, kendisinin nereden geldiğini
hatırlasın. O zaman türevi ileri değil, GERİ giderek tek seferde hesaplarız.

Sözlük:
    Value           : tek bir skaler sayıyı saran kutu. İçinde .data var,
                      ayrıca "beni kim üretti" bilgisi var.
    _prev (children): bu Value'yu üreten Value'lar. (a+b -> c ise c._prev={a,b})
    _op             : bu Value'yu üreten işlem. ('+', '*', 'tanh', ...)
    computation graph: bu bağlantıların oluşturduğu yönlü graf.
                      Yapraklar = girdiler/parametreler, kök = çıktı (loss).

Neden graf?
    Türev almak = zincir kuralını graf üstünde kökten yapraklara doğru
    uygulamak. Grafi saklamazsak zinciri geriye takip edemeyiz.

Çalıştır:  python3 adim1_value.py
"""


# ---------------------------------------------------------------------------
# 1) Value — bir skaler + onu üreten geçmiş
# ---------------------------------------------------------------------------


class Value:
    """Tek bir sayıyı saran kutu. Kendi geçmişini (grafını) hatırlar."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data          # asıl sayı
        self._prev = set(_children)  # beni üreten Value'lar
        self._op = _op            # beni üreten işlem ('' ise yaprağım)
        self.label = label        # sadece çizim/okuma kolaylığı için isim

    def __repr__(self):
        return f"Value(data={self.data})"

    # --- işlemler --------------------------------------------------------
    # Python'da a + b yazınca a.__add__(b) çağrılır. Yani operatörleri
    # kendi sınıfımıza öğretebiliyoruz. Sonucu yeni bir Value olarak dönerken
    # "beni (self, other) üretti ve işlem '+' idi" bilgisini içine koyuyoruz.

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out


# ---------------------------------------------------------------------------
# 2) Grafı gezmek — topolojik sıra
# ---------------------------------------------------------------------------
# Topolojik sıra: her düğüm, kendisini üreten düğümlerden SONRA gelir.
# Aşağıdaki DFS (derinlik öncelikli arama) tam olarak bunu üretir; Adım 3'te
# backward() bu listeyi TERSTEN gezecek.


def topolojik_sirala(kok):
    sira, gorulen = [], set()

    def gez(v):
        if v in gorulen:
            return
        gorulen.add(v)
        for cocuk in v._prev:
            gez(cocuk)
        sira.append(v)   # önce tüm çocuklar, sonra ben

    gez(kok)
    return sira


def grafi_yazdir(kok, girinti=0, gorulen=None):
    """Graphviz yoksa da grafı görmek için basit ASCII ağaç."""
    ad = kok.label or "?"
    op = f"  <- '{kok._op}'" if kok._op else "  (yaprak)"
    print("   " * girinti + f"{ad} = {kok.data}{op}")
    for cocuk in sorted(kok._prev, key=lambda v: v.label):
        grafi_yazdir(cocuk, girinti + 1)


def graphviz_ciz(kok, dosya="outputs/graf1"):
    """Opsiyonel: graphviz kuruluysa PNG çizer, değilse sessizce geçer."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("(graphviz kurulu degil -> PNG atlandi. 'pip install graphviz')")
        return None

    import os
    os.makedirs(os.path.dirname(dosya) or ".", exist_ok=True)

    dugumler, kenarlar = set(), set()

    def topla(v):
        if v in dugumler:
            return
        dugumler.add(v)
        for c in v._prev:
            kenarlar.add((c, v))
            topla(c)

    topla(kok)

    g = Digraph(format="png", graph_attr={"rankdir": "LR"})
    for n in dugumler:
        uid = str(id(n))
        g.node(uid, label=f"{{ {n.label} | data {n.data:.4f} }}", shape="record")
        if n._op:
            g.node(uid + n._op, label=n._op)
            g.edge(uid + n._op, uid)
    for c, p in kenarlar:
        g.edge(str(id(c)), str(id(p)) + p._op)

    try:
        yol = g.render(dosya, cleanup=True)
    except Exception as hata:
        # graphviz'in Python paketi kurulu ama 'dot' programi yoksa buraya duser
        print(f"(graf cizilemedi: {hata})")
        return None
    print(f"graf ciziliyor -> {yol}")
    return yol


# ---------------------------------------------------------------------------
# 3) Deneme — videodaki ifade:  L = (a*b + c) * f
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b        # e = -6
    e.label = "e"
    d = e + c        # d = 4
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f        # L = -8
    L.label = "L"

    print("=" * 62)
    print("1) Sonuc ve gecmis")
    print("=" * 62)
    print(f"L = {L}")
    print(f"L'yi ureten Value'lar (_prev): {L._prev}")
    print(f"L'yi ureten islem  (_op)     : '{L._op}'")

    print()
    print("=" * 62)
    print("2) Computation graph (kokten yapraklara)")
    print("=" * 62)
    grafi_yazdir(L)

    print()
    print("=" * 62)
    print("3) Topolojik sira (yapraklardan koke)")
    print("=" * 62)
    print(" -> ".join(v.label for v in topolojik_sirala(L)))
    print("Adim 3'te backward() bu listeyi TERSTEN gezecek: L, f, d, c, e, b, a")

    print()
    graphviz_ciz(L, "outputs/graf1_L")

    print()
    print("Akilda kalsin: Value sadece bir sayi degil, bir sayi + onun tarihi.")
