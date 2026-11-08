"""Plot measured losses and summarize training plus matching HellaSwag runs."""

import argparse
from pathlib import Path

try:
    from .common import HERE, read_json
except ImportError:
    from common import HERE, read_json


def plot_training(training, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    history = [row for row in training["history"] if row["loss"] is not None and row["finite"]]
    validation = training["validation"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    if history:
        steps = [row["step"] for row in history]
        values = np.array([row["loss"] for row in history])
        window = min(50, len(values))
        average = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0].plot(steps, values, alpha=0.18, color="tab:blue", label="Train per update")
        axes[0].plot(steps[window-1:], average, color="tab:blue", label=f"Train mean ({window} updates)")
        axes[1].plot(steps, [row["learning_rate"] for row in history])
    if validation:
        axes[0].plot([row["step"] for row in validation], [row["loss"] for row in validation],
                     "o-", color="tab:orange", label="Fixed held-out validation")
    axes[0].set(xlabel="Successful optimizer update", ylabel="NLL (nats/token)", title="Train and validation loss")
    axes[0].legend()
    axes[1].set(xlabel="Successful optimizer update", ylabel="Learning rate", title="Warmup + wall-time cosine")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.savefig(destination / "training_curves.png", dpi=150)
    plt.close(fig)


def make_report(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# Hafta 12 — gerçek veri, eğitim ve HellaSwag", ""]
    training_path = directory / "training.json"
    if training_path.exists():
        training = read_json(training_path)
        plot_training(training, directory)
        history, validation = training["history"], training["validation"]
        lines += [f"Eğitim: **{training['status']}**; küçük model kontrolü: **{training['smoke_only']}**; "
                  f"donanım: `{training['environment']['hardware']}`.", "",
                  f"Model: {training['parameters']:,} parametre, sıfırdan rastgele başlangıç. "
                  f"B={training['settings']['batch_size']}, T={training['settings']['sequence_length']}, "
                  f"accumulation={training['settings']['accumulation_steps']}; "
                  f"etkin batch {training['effective_batch_tokens']:,} token.", "",
                  f"Başarılı güncelleme: **{training['successful_updates']:,}**; "
                  f"atlanan güncelleme: {training['skipped_updates']}. "
                  f"Görülen eğitim token'ı: **{training['tokens_seen']:,}** (tekrarlar dahil). "
                  f"Duvar süresi: **{training['elapsed_wall_seconds']/60:.2f} dakika**. "
                  f"Update süresine göre token/s: "
                  f"**{training['tokens_seen']/max(training['optimizer_seconds'],1e-9):,.1f}**. "
                  f"Tüm duvar süresine göre token/s: "
                  f"**{training['tokens_seen']/max(training['elapsed_wall_seconds'],1e-9):,.1f}**.", ""]
        if history and validation:
            lines += [f"Son train loss: **{history[-1]['loss']}** nats/token "
                      "(son güncellemenin mikro batch ortalaması). "
                      f"İlk/son validation loss: **{validation[0]['loss']:.5f} → "
                      f"{validation[-1]['loss']:.5f}**. "
                      f"Her kontrolde aynı ilk {validation[-1]['tokens']:,} validation token'ı ölçüldü.", ""]
        manifest = directory / "data_manifest.json"
        if manifest.exists():
            data = read_json(manifest)
            lines += [f"Hazırlanan veri: train **{data['splits']['train']['tokens']:,} token / "
                      f"{len(data['splits']['train']['shards'])} shard**, val "
                      f"**{data['splits']['val']['tokens']:,} token / "
                      f"{len(data['splits']['val']['shards'])} shard**. "
                      "GPT-2 tokenizer, uint16 NPY; belge sınırlarında EOT. "
                      "Train/val belgeleri ayrıdır, aynı metnin birebir tekrarları çıkarılır. "
                      "Yakın kopya temizliği veya HellaSwag kontaminasyon taraması yapılmadı.", ""]
        lines += ["![Train, val ve learning rate](training_curves.png)", "",
                  "## Aynı prompt ve örnekleme ayarlarıyla gelişim", ""]
        samples = training["samples"]
        selected = sorted(set([0, len(samples)//2, len(samples)-1])) if samples else []
        for index in selected:
            sample = samples[index]
            text = sample["text"].replace("```", "` ` `")
            lines += [f"Adım {sample['step']} (seed=42, temperature=0.8, top-k=50):", "",
                      "```text", text, "```", ""]
        lines += ["Örnekleri incelerken başlangıçtaki parçalı token'lardan kelime/cümle "
                  "düzenine geçiş olup olmadığına bak. Düşen loss tek başına tutarlı veya "
                  "doğru içerik üretildiğini kanıtlamaz; burada yalnız kaydedilmiş örnekler vardır.", ""]
    else:
        lines += ["Gerçek veri eğitimi: **bekliyor**.", ""]
    lines += ["## HellaSwag karşılaştırması", "",
              "Rastgele, eşit olasılıklı dört seçenek tahmininin beklenen doğruluğu **%25**. "
              "Ana ölçüt yalnız devam token'ları üzerinden ortalama NLL (acc_norm). "
              "Toplam NLL doğruluğu da kaydedilir. İki modelde ortak 50.257 vocab, FP32 ve TF32 kapalıdır.", "",
              "| Model | Tamamlanan örnek | Doğru (normalize) | acc_norm | acc (toplam NLL) |",
              "|---|---:|---:|---:|---:|"]
    results = {}
    for name in ("own", "openai"):
        path = directory / f"hellaswag_{name}.json"
        if not path.exists():
            lines.append(f"| {name} | — | — | bekliyor | — |")
            continue
        result = read_json(path)
        results[name] = result
        n = result["processed"]
        lines.append(f"| {name} | {n} ({result['status']}) | {result['correct_norm']} | "
                     f"{100*result['correct_norm']/max(1,n):.2f}% | {100*result['correct_sum']/max(1,n):.2f}% |")
    if set(results) == {"own", "openai"}:
        own, reference = results["own"], results["openai"]
        matched = all(own[key] == reference[key] for key in (
            "selection_sha256", "example_count", "dataset_revision", "scoring", "precision", "tf32"))
        complete = matched and all(row["status"] == "complete" for row in results.values())
        if complete:
            gap = 100 * (reference["accuracy_norm"] - own["accuracy_norm"])
            lines += ["", f"Aynı {own['example_count']} örnekte OpenAI − kendi modelimiz farkı: "
                      f"**{gap:+.2f} yüzde puan**. "
                      "Bu kısa eğitimin veri miktarı, güncelleme sayısı ve metin dağılımı "
                      "OpenAI'ın ön eğitiminden farklıdır. Farkı yalnız mimariye veya tek bir "
                      "hiperparametreye bağlayamayız; bu deney nedenleri tek tek ayırmıyor.", "",
                      "Örnek sayısı sınırlı olduğundan örneklem belirsizliği vardır. Her modelin "
                      "%95 Wilson aralığı ve örnek bazındaki tahminleri JSON'dadır; "
                      "bu sonuçlar farklı HellaSwag ön işleme yöntemleriyle doğrudan kıyaslanmamalıdır.", ""]
        else:
            lines += ["", "Karşılaştırma henüz tamamlanmadı veya iki ölçümün seçim/puanlama ayarları farklı.", ""]
    lines += ["", "Ek Türkçe eğitim: **bu koşuda yapılmadı**. Türkçe model için ayrılmış Türkçe "
              "metinde loss/bits-per-byte, Türkçe görevler ve insan değerlendirmesi kullanılır. "
              "İngilizce HellaSwag tek başına Türkçe kalitesini ölçmez.", "",
              "Kaynaklar: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), "
              "[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), "
              "[Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s).", ""]
    (directory / "submission_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/l4")
    make_report(parser.parse_args().output_dir)
