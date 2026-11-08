"""Verify the published G4 evidence and regenerate its analysis and figures."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT = HERE / "results/nvidia_rtx_pro6000"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def paired_comparison(own, reference):
    counts = dict(both=0, own_only=0, reference_only=0, neither=0)
    for left, right in zip(own["predictions"], reference["predictions"], strict=True):
        require(all(left[key] == right[key] for key in ("index", "ind", "label")),
                "Paired evaluation examples differ")
        a = left["predicted_norm"] == left["label"]
        b = right["predicted_norm"] == right["label"]
        counts["both" if a and b else "own_only" if a else "reference_only" if b else "neither"] += 1
    total = sum(counts.values())
    # Resample the paired correctness outcomes, not the two models separately.
    rng = np.random.default_rng(20261108)
    draws = rng.multinomial(total, np.array(list(counts.values())) / total, size=20_000)
    difference = (draws[:, 1] - draws[:, 2]) / total
    discordant = counts["own_only"] + counts["reference_only"]
    smaller = min(counts["own_only"], counts["reference_only"])
    exact_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant)
    return dict(counts=counts, difference_own_minus_reference=own["accuracy_norm"]-reference["accuracy_norm"],
                paired_bootstrap95=np.quantile(difference, [.025, .975]).tolist(),
                bootstrap_resamples=20_000, bootstrap_seed=20261108,
                mcnemar_exact_two_sided_p=exact_p)


def verify(directory):
    exported = read(directory / "export_manifest.json")
    for name, entry in exported["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
                f"Unsafe evidence path: {name}")
        path = directory / relative
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                f"Imported evidence changed: {name}")
    training = read(directory / "training.json")
    summary = read(directory / "summary.json")
    data = read(directory / "data_manifest.json")
    checkpoints = read(directory / "checkpoint_manifest.json")
    runtime = read(directory / "runtime_verification.json")
    selection = read(directory / "selection_manifest.json")
    require(training["status"] == "complete" and not training["smoke_only"], "Incomplete or tiny run")
    require(training["deadline"] is None and training["budget_seconds"] == 5400, "Unexpected final budget")
    require(digest(directory / "data_manifest.json") == training["provenance"]["data_manifest_sha256"],
            "Training data manifest changed")
    for name, expected in training["provenance"]["source_sha256"].items():
        require(digest(directory / "sources" / name) == expected, f"Runtime source differs: {name}")
    for key in ("runtime_source_hashes_verified", "all_shards_checksum_verified",
                "drive_tokens_verified", "drive_checkpoints_verified", "evaluation_predictions_verified"):
        require(runtime[key] is True, f"Runtime verification did not pass: {key}")
    require(runtime["inference_rerun"] is False, "Unexpected inference verification claim")
    require(digest(directory / "checkpoint_manifest.json") == runtime["checkpoint_manifest_sha256"],
            "Checkpoint manifest changed")
    old = read(ROOT / "week12/results/nvidia_l4/data_manifest.json")
    train_hashes, val_hashes = map(set, (data["document_sha256"]["train"], data["document_sha256"]["val"]))
    require(not train_hashes.intersection(val_hashes), "Train/val document overlap")
    require(not train_hashes.intersection(set(old["document_sha256"]["train"]) | set(old["document_sha256"]["val"])),
            "Fresh training set includes baseline documents")
    require(data["splits"]["val"] == old["splits"]["val"], "Validation shard differs from L4")
    require(all(sum(item["tokens"] for item in split["shards"]) == split["tokens"]
                for split in data["splits"].values()), "Shard token counts do not sum")
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    for expected_step, row in enumerate(rows, 1):
        require(row["step"] == expected_step and row["tokens_seen"] == expected_step * training["effective_batch_tokens"],
                f"Update accounting differs at step {expected_step}")
        require(row["finite"] and not row["skipped_update"], f"Invalid update {expected_step}")
        require(all(math.isfinite(row[key]) for key in ("loss", "gradient_norm_before_clip", "learning_rate",
                                                       "milliseconds", "tokens_per_second")), "Nonfinite update metric")
    require(len(rows) == training["successful_updates"] == runtime["raw_update_rows"], "Update count differs")
    require(rows[-1]["tokens_seen"] == training["tokens_seen"] == summary["additional_tokens"], "Final tokens differ")
    require(math.isclose(sum(row["milliseconds"] for row in rows) / 1000, training["optimizer_seconds"], abs_tol=1e-5),
            "Optimizer timing does not match raw updates")
    best = min(training["validation"], key=lambda row: row["loss"])
    require(best["step"] == summary["best_step"] == training["best_step"], "Best checkpoint step differs")
    require(best["loss"] == summary["best_validation_loss"] == training["best_validation_loss"], "Best loss differs")
    require(all(row["tokens"] == 32768 and row["batches"] == 32 for row in training["validation"]),
            "Validation protocol changed")
    require(checkpoints["best.pt"]["successful_updates"] == best["step"], "Selected checkpoint metadata differs")
    require(summary["total_tokens_including_L4"] == training["base_tokens"] + training["tokens_seen"],
            "Combined token count differs")
    evaluations, comparisons = {}, {}
    for cohort, start in (("course1000", 0), ("heldout1000", 1000)):
        for model in ("own", "openai"):
            record = read(directory / summary["evaluation_directory"] / f"hellaswag_{model}_{cohort}.json")
            require(record["status"] == "complete" and record["processed"] == len(record["predictions"]) == 1000,
                    "Incomplete evaluation")
            require(record["selection_sha256"] == selection["cohorts"][cohort]["selection_sha256"], "Selection differs")
            norm_correct, sum_correct = 0, 0
            for index, item in enumerate(record["predictions"], start):
                require(item["index"] == index and item["label"] in range(4), "Example order/label differs")
                require(len(item["mean_nll"]) == len(item["sum_nll"]) == 4, "Invalid choice scores")
                require(all(math.isfinite(v) for v in item["mean_nll"] + item["sum_nll"]), "Nonfinite choice score")
                require(item["predicted_norm"] == int(np.argmin(item["mean_nll"])), "Normalized prediction differs")
                require(item["predicted_sum"] == int(np.argmin(item["sum_nll"])), "Summed prediction differs")
                norm_correct += item["predicted_norm"] == item["label"]
                sum_correct += item["predicted_sum"] == item["label"]
            require(norm_correct == record["correct_norm"] and sum_correct == record["correct_sum"], "Correct count differs")
            require(math.isclose(record["accuracy_norm"], norm_correct / 1000, abs_tol=1e-12), "Accuracy differs")
            summary_label = ("G4" if model == "own" else "GPT2") + ("/course" if cohort == "course1000" else "/heldout")
            require(math.isclose(summary["hellaswag_percent"][summary_label], record["accuracy_norm"]*100, abs_tol=1e-10),
                    "Summary accuracy differs from per-example scores")
            if model == "own":
                require(record["checkpoint_sha256"] == checkpoints["best.pt"]["sha256"], "Evaluated weights differ")
            evaluations[(model, cohort)] = record
        require(evaluations[("own", cohort)]["source_sha256"] == evaluations[("openai", cohort)]["source_sha256"],
                "Models used different evaluation sources")
        comparisons[cohort] = paired_comparison(evaluations[("own", cohort)], evaluations[("openai", cohort)])
    analysis = dict(status="verified", imported_files=len(exported["files"]), update_rows=len(rows),
                    source_hashes_verified=True, predictions_recomputed_from_saved_scores=True,
                    model_inference_rerun_locally=False, checkpoints_in_git=False,
                    data_shards_in_git=False, corpus_passes=training["tokens_seen"] / data["splits"]["train"]["tokens"],
                    validation_loss_change=summary["best_validation_loss"]-summary["base_validation_loss"],
                    scores={cohort: {model: dict(correct=evaluations[(model, cohort)]["correct_norm"], total=1000,
                                accuracy=evaluations[(model, cohort)]["accuracy_norm"],
                                wilson95=evaluations[(model, cohort)]["accuracy_norm_wilson95"])
                                for model in ("own", "openai")} for cohort in ("course1000", "heldout1000")},
                    comparisons=comparisons,
                    note="Runtime verification checks Drive copies and token shards. Local verification checks exported bytes, sources, scores and accounting; it does not rerun GPU inference.")
    return training, summary, rows, evaluations, analysis


def plot(directory, training, summary, rows, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="#f7f9fc")
    fig.subplots_adjust(left=.07, right=.97, bottom=.22, top=.69, wspace=.29)
    fig.text(.07, .92, "From a neuron to GPT-2", fontsize=25, weight="bold", color="#132238")
    fig.text(.07, .865, "Measured G4 continuation of our 124M model · 8 November 2026", fontsize=12, color="#526278")
    for x, value, label in ((.07, "124.5M", "tied parameters"), (.36, "1.247B", "additional tokens · repeats included"),
                            (.72, "90.01 min", "G4 run wall time")):
        fig.text(x, .77, value, fontsize=20, weight="bold", color="#1763b1")
        fig.text(x, .732, label, fontsize=9, color="#526278")
    tokens = np.array([row["tokens_seen"] for row in rows]) / 1e9
    losses = np.array([row["loss"] for row in rows])
    width = 200
    axes[0].plot(tokens[width-1:], np.convolve(losses, np.ones(width)/width, mode="valid"),
                 color="#76a3cb", lw=1.2, label="Train · T1024 · 200-update mean")
    vals = training["validation"]
    axes[0].plot([v["step"]*training["effective_batch_tokens"]/1e9 for v in vals],
                 [v["loss"] for v in vals], "o-", markersize=3, color="#1763b1", lw=2,
                 label="Validation · fixed T256 / 32,768 targets")
    axes[0].axhline(summary["base_validation_loss"], color="#9c6b37", linestyle="--", lw=1, label="L4 final validation")
    axes[0].set(xlabel="G4 processed tokens (billions)", ylabel="Loss (nats / token)",
                title="Validation loss: 4.985 → 3.666", ylim=(3.1, 5.25))
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].grid(alpha=.15)
    positions = np.arange(2)
    for model, offset, color, label in (("own", -.18, "#1763b1", "Our G4 model"),
                                       ("openai", .18, "#e78b35", "OpenAI GPT-2")):
        records = [evaluations[(model, c)] for c in ("course1000", "heldout1000")]
        rates = np.array([r["accuracy_norm"] for r in records])*100
        intervals = np.array([r["accuracy_norm_wilson95"] for r in records])*100
        bars = axes[1].bar(positions+offset, rates, width=.32, color=color, label=label,
                           yerr=np.array([rates-intervals[:, 0], intervals[:, 1]-rates]), capsize=4)
        for bar, rate in zip(bars, rates):
            axes[1].text(bar.get_x()+bar.get_width()/2, rate+3.6, f"{rate:.1f}%", ha="center", fontsize=10, weight="bold")
    axes[1].scatter([-.18], [27.6], marker="D", s=40, color="#526278", zorder=5, label="Our L4 model · original set")
    axes[1].axhline(25, color="#aaa", linestyle="--", lw=1)
    axes[1].set(xticks=positions, xticklabels=["Original 1,000", "Additional held-out 1,000"],
                ylim=(0, 43), ylabel="HellaSwag accuracy (%)", title="Same examples and scoring for both models")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.15)
    axes[1].set_axisbelow(True)
    fig.text(.07, .105, "200M fresh training tokens processed 6.24 times. The G4 model remains below GPT-2 on both HellaSwag cohorts.", fontsize=10, color="#132238")
    fig.text(.07, .06, "HellaSwag: ending-only mean NLL · common 50,257-token vocabulary · FP32 / TF32 off · bars show marginal 95% Wilson intervals.", fontsize=8.5, color="#526278")
    fig.savefig(directory / "overview.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(directory, training, summary, evaluations, analysis):
    lines = ["# G4 devam eğitimi — ölçülmüş sonuçlar", "", "8 Kasım 2026 · NVIDIA RTX PRO 6000 Blackwell Server Edition", "",
             "Kendi L4 checkpoint'imizden devam eden 90 dakikalık deney tamamlandı. Model, OpenAI GPT-2 ağırlıklarından başlatılmadı. "
             "Validation loss ve ilk HellaSwag grubundaki başarı iyileşti; iki değerlendirme grubunda da GPT-2'nin puanının altında kaldı.", "",
             "![Ölçülmüş G4 sonuçları](overview.png)", "", "## Eğitim", "",
             "| Ölçüm | Değer |", "|---|---:|",
             "| Parametre | 124.475.904; weight tying, vocab 50.304 |",
             "| Yapı | 12 blok, 12 head, genişlik 768; pozisyon tablosu 1.024 |",
             f"| Başarılı G4 güncellemesi | {training['successful_updates']:,} |",
             f"| G4 işlenen token | {training['tokens_seen']:,} |",
             f"| L4 dahil toplam token | {summary['total_tokens_including_L4']:,} |",
             "| Yeni eğitim havuzu | 200.000.000 token |",
             f"| Yeni havuzun işlenme miktarı | {analysis['corpus_passes']:.4f} tur; token tekrarları dahil |",
             f"| G4 duvar süresi | {summary['training_minutes']:.4f} dakika |",
             f"| Güncelleme sürelerine göre hız | {summary['optimizer_tokens_per_second']:,.1f} token/s |",
             f"| Toplam duvar süresine göre hız | {summary['wall_tokens_per_second']:,.1f} token/s |",
             f"| L4 son / G4 son validation loss | {summary['base_validation_loss']:.6f} → {summary['best_validation_loss']:.6f} |", "",
             "B=8, T=1024, accumulation=4; etkin batch 32.768 token. BF16, SDPA, torch.compile ve fused AdamW kullanıldı. "
             "AdamW β=(0.9,0.95), ε=1e-8, matrislerde decay=0.1 ve clipping=1.0. LR 30 başarılı güncellemede ısınır; "
             "duvar zamanına bağlı cosine zarfı 3e-4'ten 3e-5'e iner.", "",
             "Süre; yükleme, ilk derleme, validation ve dönemsel kayıtları içerir. Kesinti arasında geçen süre, veri hazırlığı ve son HellaSwag testi bu bütçeye dahil değildir. "
             "Yaklaşık 9,57 dakikada bir kez devam kaydı açıldı; eski saat sınırı kaldırılıp toplam bütçe 90 dakikaya uzatıldı. "
             "3.989 tamamlanmış güncelleme korundu, 16.384 tamamlanmamış mikro-batch token'ı yeniden okundu. "
             "Kaynak sürümü, eski/yeni süre ayarı ve checkpoint hash'i training.json içindeki resumes alanında kayıtlıdır.", "",
             "## Veri ve validation", "",
             "FineWeb-Edu sample-10BT'nin sabit revizyonu kullanıldı. Yeni train belgelerinden hem eski L4 train hem de val belgelerinin birebir SHA-256 eşleşmeleri çıkarıldı. "
             "Bu, yakın kopya veya benchmark sızıntısı temizliği yapıldığı anlamına gelmez.", "",
             "Eski 131.072-token validation shard'ı byte düzeyinde korundu. Her kontrolde B=4, T=256 ile aynı ilk 32.768 hedef token ölçüldü. "
             "Son ve en iyi checkpoint 38.061. güncellemedir. Train loss T1024'te ve farklı eğitim metninde ölçülür; train/val çizgileri aynı veri ve context koşullarının ölçümü değildir. "
             "OpenAI GPT-2 için bu FineWeb validation loss'u ölçülmedi.", "",
             "## HellaSwag", "",
             "İlk 1.000 örnek önceki deneyle devamlılık içindir; sonraki 1.000 örnek pilotun önceden ayrılmış ana karşılaştırmasıdır. "
             "Checkpoint seçimi validation loss'una göre yapıldı. İki model aynı örnekleri, completion-only ortalama token NLL, "
             "ortak 50.257-token vocabulary, FP32 ve TF32 kapalı ayarlarıyla değerlendirdi.", "",
             "| Grup | Kendi G4 modelimiz | OpenAI GPT-2 | Fark (kendi − GPT-2) |", "|---|---:|---:|---:|"]
    for cohort, label in (("course1000", "İlk 1.000"), ("heldout1000", "Sonraki ayrı 1.000")):
        own, ref = evaluations[("own", cohort)], evaluations[("openai", cohort)]
        lines.append(f"| {label} | {own['correct_norm']}/1000 = %{own['accuracy_norm']*100:.1f} | "
                     f"{ref['correct_norm']}/1000 = %{ref['accuracy_norm']*100:.1f} | {(own['accuracy_norm']-ref['accuracy_norm'])*100:+.1f} yüzde puan |")
    lines += ["", "L4 checkpoint'i ilk grupta %27,6 almıştı: G4 devam eğitiminin aynı gruptaki artışı **4,7 yüzde puan**. "
              "Bu sonuç bir test grubundaki puan farkıdır; genel kodlama, sohbet veya muhakeme yeteneğinde denkliğe çevrilemez.", "",
              "Eşleştirilmiş karşılaştırma (aynı örnekteki doğruluk sonuçları birlikte yeniden örneklendi):", "",
              "| Grup | Yalnız kendi doğru | Yalnız GPT-2 doğru | Fark için paired bootstrap %95 aralığı |", "|---|---:|---:|---:|"]
    for cohort, label in (("course1000", "İlk 1.000"), ("heldout1000", "Ayrı 1.000")):
        paired = analysis["comparisons"][cohort]
        lo, hi = [value*100 for value in paired["paired_bootstrap95"]]
        lines.append(f"| {label} | {paired['counts']['own_only']} | {paired['counts']['reference_only']} | [{lo:+.2f}, {hi:+.2f}] yüzde puan |")
    lines += ["", "Bootstrap 20.000 tekrar, seed=20261108; örnekler arası bağımsızlık varsayımıyla yaklaşık bir belirsizlik ölçüsüdür. "
              "Her modelin Wilson aralığı ve exact McNemar p değeri analysis.json dosyasındadır; bunlar modelin her görevde daha iyi/kötü olduğunu kanıtlamaz.", "",
              "## Üretilen metin", "", "Aynı prompt: `The children walked into the`. Seed=42, temperature=0.8, top-k=50; en fazla 48 yeni token.", "",
              "### G4 başlangıcı", "", "> " + training["samples"][0]["text"].replace("\n", "\n> "), "",
              "### G4 sonu", "", "> " + training["samples"][-1]["text"].replace("\n", "\n> "), "",
              "Cümle yapısı daha düzenli görünse de tekrarlar ve anlam sorunları sürüyor. Son metin 48-token üretim sınırında kesilir; bu örnek tek başına kapsamlı kalite değerlendirmesi değildir.", "",
              "## Kanıt ve yeniden üretim", "",
              "- `updates.jsonl`: 38.061 güncellemenin loss, norm, LR, süre ve token/s kayıtları.",
              "- `training.json`: örneklenmiş history, 19 validation kontrolü, örnek metinler, optimizer ayarları ve resume geçmişi.",
              "- `data_manifest.json`: belge hash'leri, veri revizyonu ve shard hash'leri; kaynak metin veya token dizileri içermez.",
              "- `evaluations/`: dört değerlendirmede örnek başına etiket, tahmin ve seçenek NLL'leri.",
              "- `sources/`: GPU koşusunda kullanılan kaynak dosyalarının byte düzeyinde kopyaları.",
              "- `checkpoint_manifest.json`: son/en iyi checkpoint boyutu, model ayarı ve SHA-256.",
              "- `runtime_verification.json`: Colab'da shard ve Drive kopyası kontrolleri; inference tekrar koşulmadı.",
              "- `export_manifest.json`: indirilen özgün dosyaların hash ve boyutları; `analysis.json` yerel tutarlılık kontrollerini içerir.", "",
              "Checkpoint'ler ve 201 token shard'ı Drive'da doğrulanarak saklandı; Git'e eklenmedi. Benchmark soru metinleri de tekrar yayımlanmadı; "
              "dataset revizyonu ve seçim hash'leri selection_manifest.json içindedir. Yerelde tüm özgün dosyaların hash'leri, kaynak kopyaları, "
              "eğitim muhasebesi ve kaydedilmiş NLL'lerden argmin/doğru sayıları yeniden kontrol edildi. GPU inference yerelde tekrarlanmadı.", "",
              "Repo kökünden:", "", "```bash", "python experiments/g4/analyze_results.py --check", "python experiments/g4/analyze_results.py", "```", "",
              "G4 çalışma adımları ve checkpoint gereksinimleri için [deney README'si](../../README.md).", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true", help="Verify evidence without regenerating derived outputs")
    args = parser.parse_args()
    training, summary, rows, evaluations, analysis = verify(args.results)
    if not args.check:
        (args.results / "analysis.json").write_text(json.dumps(analysis, indent=2)+"\n", encoding="utf-8")
        plot(args.results, training, summary, rows, evaluations)
        write_report(args.results, training, summary, evaluations, analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
