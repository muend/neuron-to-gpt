"""Render a summary of the verified Week 11/12 L4 experiments."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    benchmark = read("week11/results/nvidia_l4/benchmark.json")
    train = read("week12/results/nvidia_l4/training.json")
    own = read("week12/results/nvidia_l4/hellaswag_own.json")
    reference = read("week12/results/nvidia_l4/hellaswag_openai.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 9), facecolor="#f7f9fc")
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.96, bottom=0.10, top=0.78,
                          hspace=0.52, wspace=0.27)
    fig.text(0.08, 0.94, "YZ50 · Hafta 11 ve 12", fontsize=23, weight="bold", color="#132238")
    fig.text(0.08, 0.90, "Kendi GPT-2 uygulamamız · NVIDIA L4 · ölçülmüş sonuçlar", fontsize=12, color="#526278")
    for x, value, label in ((0.08, "124,5 milyon", "model parametresi"),
                             (0.40, "80,22 milyon", "işlenen eğitim token’ı · tekrarlar dahil"),
                             (0.75, "60,20 dakika", "eğitim duvar süresi")):
        fig.text(x, 0.842, value, fontsize=18, weight="bold", color="#1763b1")
        fig.text(x, 0.817, label, fontsize=9, color="#526278")
    speed = fig.add_subplot(gs[0, 0])
    rows = [row for row in benchmark["stages"] if row["status"] == "measured"]
    names = {"baseline": "FP32 başlangıç", "tf32": "+ TF32", "mixed_precision": "+ BF16",
             "compile": "+ torch.compile", "flash_attention": "+ Flash Attention", "padded_vocab": "+ vocab padding"}
    values = [row["median_ms"] for row in rows]
    speed.barh(range(len(rows)), values, color=["#a8b4c3"] * 3 + ["#347bc0"] * 3, height=0.63)
    speed.set_yticks(range(len(rows)), [names[row["stage"]] for row in rows])
    speed.invert_yaxis()
    speed.set_xlim(0, max(values) * 1.23)
    for index, value in enumerate(values):
        speed.text(value + 0.8, index, f"{value:.2f}", va="center", fontsize=9)
    speed.set(title="Hafta 11 · Birikimli hızlandırma", xlabel="Medyan ms/adım · daha düşük daha hızlı")
    speed.grid(axis="x", alpha=0.14)
    speed.set_axisbelow(True)
    hellaswag = fig.add_subplot(gs[0, 1])
    rates = [25.0, own["accuracy_norm"] * 100, reference["accuracy_norm"] * 100]
    hellaswag.barh(["Rastgele beklentisi", "Kendi modelimiz", "OpenAI GPT-2"], rates,
                   color=["#a8b4c3", "#347bc0", "#e78b35"], height=0.58)
    hellaswag.invert_yaxis()
    hellaswag.set_xlim(0, 42)
    for index, rate in enumerate(rates):
        hellaswag.text(rate + 0.7, index, f"%{rate:.1f}", va="center", weight="bold")
    hellaswag.set(title="Hafta 12 · Aynı 1.000 HellaSwag örneği", xlabel="Doğruluk · ortalama devam-token NLL ile seçim")
    hellaswag.grid(axis="x", alpha=0.14)
    hellaswag.set_axisbelow(True)
    loss = fig.add_subplot(gs[1, :])
    history = train["history"]
    steps = np.array([row["step"] for row in history])
    losses = np.array([row["loss"] for row in history])
    window = 100
    loss.plot(steps[window-1:], np.convolve(losses, np.ones(window) / window, mode="valid"),
              color="#347bc0", lw=1.4, label="Train · 100 adım ortalaması")
    validation = train["validation"]
    loss.plot([row["step"] for row in validation], [row["loss"] for row in validation],
              color="#e78b35", lw=1.7, label="Validation · sabit 32.768 token")
    loss.set(title="Hafta 12 · Gerçek veride öğrenme", xlabel="Başarılı optimizer güncellemesi", ylabel="Loss · nats/token")
    loss.grid(alpha=0.16)
    loss.legend(loc="upper right", frameon=False)
    loss.text(0.98, 0.50, f"Validation: {validation[0]['loss']:.3f} → {validation[-1]['loss']:.3f}\n"
              f"{train['successful_updates']:,} başarılı güncelleme", transform=loss.transAxes,
              ha="right", va="center", fontsize=11, color="#132238")
    fig.text(0.08, 0.035, "20M train token’ı + ayrı 131.072 validation token’ı · Hız tablosu: B=2, T=128 · HellaSwag: FP32, TF32 kapalı",
              fontsize=9, color="#526278")
    destination = ROOT / "week12/results/nvidia_l4/summary.png"
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
