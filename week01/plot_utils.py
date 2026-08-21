"""
plot_utils.py — Küçük bir çizim yardımcısı.

Neden var?
    Görev 4'te "loss eğrisini çiz" deniyor. Nöron/loss/gradient kodunu
    kütüphanesiz yazıyoruz ama GRAFİK çizmek öğrenme konusunun bir parçası
    değil. Burası tek istisna: matplotlib varsa PNG üretir, yoksa
    terminale ASCII grafik basar. Yani repo hiçbir kurulum olmadan çalışır.

Kullanım:
    from plot_utils import cizgi_grafik
    cizgi_grafik(xs, ys, baslik="Loss egrisi", x_ad="w", y_ad="loss",
                 dosya="outputs/loss_egrisi.png")
"""

import os

# ---------------------------------------------------------------------------
# ASCII grafik (hiçbir kütüphane gerektirmez)
# ---------------------------------------------------------------------------


def ascii_grafik(xs, ys, genislik=64, yukseklik=18, baslik="", x_ad="x", y_ad="y"):
    """xs/ys listelerini terminalde kaba bir çizgi grafiğe çevirir."""
    if not xs:
        return "(veri yok)"

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Tüm değerler aynıysa 0'a bölmeyi engelle
    x_araligi = (x_max - x_min) or 1.0
    y_araligi = (y_max - y_min) or 1.0

    # Boş tuval
    tuval = [[" "] * genislik for _ in range(yukseklik)]

    for x, y in zip(xs, ys):
        sutun = int((x - x_min) / x_araligi * (genislik - 1))
        # y ekseni yukarıdan aşağıya olduğu için ters çeviriyoruz
        satir = int((1 - (y - y_min) / y_araligi) * (yukseklik - 1))
        tuval[satir][sutun] = "*"

    satirlar = []
    if baslik:
        satirlar.append(baslik)
        satirlar.append("")

    for i, satir in enumerate(tuval):
        if i == 0:
            etiket = f"{y_max:9.4f}"
        elif i == yukseklik - 1:
            etiket = f"{y_min:9.4f}"
        else:
            etiket = " " * 9
        satirlar.append(f"{etiket} |" + "".join(satir))

    satirlar.append(" " * 9 + " +" + "-" * genislik)
    alt = f"{x_min:.3f}".ljust(genislik // 2) + f"{x_max:.3f}".rjust(genislik - genislik // 2)
    satirlar.append(" " * 9 + "  " + alt)
    satirlar.append(" " * 9 + "  " + f"({x_ad} ekseni, dikey: {y_ad})")
    return "\n".join(satirlar)


# ---------------------------------------------------------------------------
# Ana çizim fonksiyonu
# ---------------------------------------------------------------------------


def cizgi_grafik(xs, ys, baslik="", x_ad="x", y_ad="y", dosya=None,
                 isaretle=None, isaret_etiketi="min"):
    """
    Grafiği hem terminale (ASCII) basar hem de matplotlib varsa PNG kaydeder.

    isaretle: (x, y) tuple'ı — grafikte kırmızı nokta olarak işaretlenir
              (ör. loss'un en düşük olduğu nokta).
    """
    print(ascii_grafik(xs, ys, baslik=baslik, x_ad=x_ad, y_ad=y_ad))

    if dosya is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")  # ekran olmayan ortamlarda da çalışsın
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[not] matplotlib kurulu degil, PNG uretilmedi.")
        print("      Kurmak icin: pip install matplotlib")
        return

    klasor = os.path.dirname(dosya)
    if klasor:
        os.makedirs(klasor, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, linewidth=2)
    if isaretle is not None:
        plt.scatter([isaretle[0]], [isaretle[1]], color="red", zorder=5)
        plt.annotate(f"{isaret_etiketi}: ({isaretle[0]:.3f}, {isaretle[1]:.5f})",
                     xy=isaretle, xytext=(10, 10), textcoords="offset points")
    plt.title(baslik)
    plt.xlabel(x_ad)
    plt.ylabel(y_ad)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dosya, dpi=130)
    plt.close()
    print(f"\n[kaydedildi] {dosya}")
