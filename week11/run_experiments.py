"""Measure incremental optimizations and train on a complete cyclic text stream.

CPU --tiny runs are wiring checks. GPU evidence is never inferred from them.
No pretrained download or Transformer library is used in Week 11.
"""

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

try:
    from .training import (GPT, DataLoaderLite, config_for, configure_optimizer,
                           learning_rate, make_scaler, optimizer_update,
                           set_tf32, synchronize)
except ImportError:
    from training import (GPT, DataLoaderLite, config_for, configure_optimizer,
                          learning_rate, make_scaler, optimizer_update,
                          set_tf32, synchronize)

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / ".cache/tiktoken"))


def save_json(path, value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, list):
            return [clean(val) for val in item]
        return item
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment(device):
    return {"device": str(device), "hardware": torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor() or device.type,
            "torch": torch.__version__, "python": platform.python_version(),
            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda" else None}


def source_hashes():
    files = [HERE / name for name in ("training.py", "run_experiments.py", "requirements.txt")]
    files.append(HERE.parent / "week10/model.py")
    return {str(path.relative_to(HERE.parent)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def load_tokens(path, tiny):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if tiny:
        tokens, tokenizer = list(raw) + [256], "UTF-8 bytes + boundary ID 256 (smoke only)"
    else:
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        tokens, tokenizer = encoding.encode_ordinary(text) + [encoding.eot_token], "gpt2 + EOT"
    return tokens, {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                    "utf8_bytes": len(raw), "tokens_including_boundary": len(tokens),
                    "tokenizer": tokenizer, "stream_policy": "cyclic, EOT-separated, no dropped tail"}


def create_model(tiny, settings, device):
    # Start every variant from identical shared FP32 weights. Padding adds rows
    # only; no later block changes its initialization because of an RNG shift.
    torch.manual_seed(1337)
    model = GPT(config_for(tiny, attention=settings["attention"]))
    if settings["padded"]:
        count = config_for(tiny, padded=True).vocab_size
        embedding = torch.nn.Embedding(count, model.config.n_embd)
        torch.nn.init.normal_(embedding.weight, std=0.02)
        with torch.no_grad():
            embedding.weight[:model.config.vocab_size].copy_(model.transformer.wte.weight)
        model.transformer.wte = embedding
        model.lm_head = torch.nn.Linear(model.config.n_embd, count, bias=False, device="meta")
        model.lm_head.weight = embedding.weight
        model.config = replace(model.config, vocab_size=count)
    return model.to(device).train()


def default_settings():
    return dict(tf32=False, precision="fp32", compile=False,
                attention="eager", force_flash=False, padded=False)


def choose_precision(device):
    return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"


def measure_variant(args, tokens, device, settings):
    set_tf32(settings["tf32"])
    model = create_model(args.tiny, settings, device)
    optimizer, audit = configure_optimizer(model, device)
    loader = DataLoaderLite(tokens, args.batch_size, args.sequence_length)
    scaler = make_scaler(device, settings["precision"])
    if settings["compile"]:
        # fullgraph forbids silent eager graph-break fallbacks.
        model = torch.compile(model, fullgraph=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.warmup):
        optimizer_update(model, optimizer, loader, device, precision=settings["precision"],
                         scaler=scaler, force_flash=settings["force_flash"])
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    durations, losses, skipped = [], [], 0
    for _ in range(args.benchmark_steps):
        synchronize(device)
        start = time.perf_counter()
        result = optimizer_update(model, optimizer, loader, device, precision=settings["precision"],
                                  scaler=scaler, force_flash=settings["force_flash"])
        synchronize(device)
        durations.append(time.perf_counter() - start)
        losses.append(result["loss"])
        skipped += int(result["skipped_update"])
        if not result["finite"]:
            raise FloatingPointError("Nonfinite measured update; timing rejected")
    if skipped:
        raise RuntimeError(f"{skipped} measured optimizer updates skipped; timing rejected")
    median = statistics.median(durations)
    return dict(status="measured", settings=settings.copy(), median_ms=median * 1000,
                tokens_per_second=args.batch_size * args.sequence_length / median,
                step_seconds=durations, losses=losses, measured_skipped_updates=skipped,
                warmup_seconds=setup_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None, optimizer=audit)


def benchmark(args, tokens, device, provenance):
    path = args.output_dir / "benchmark.json"
    evidence = dict(provenance, benchmark_shape=[args.batch_size, args.sequence_length],
                    warmup_steps=args.warmup, measured_steps=args.benchmark_steps,
                    timing_scope="load/transfer + forward + backward + unscale/clip + AdamW; "
                                 "synchronized each step; excludes logging/setup/warmup",
                    stages=[])
    accepted = default_settings()
    stages = ["baseline", "tf32", "mixed_precision", "compile", "flash_attention", "padded_vocab"]
    baseline_ms = previous_ms = None
    for stage in stages:
        candidate = accepted.copy()
        reason = None
        if stage == "tf32":
            if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
                reason = "TF32 requires an Ampere-or-newer CUDA device (T4 is sm75)"
            candidate["tf32"] = True
        elif stage == "mixed_precision":
            if device.type != "cuda":
                reason = "This controlled AMP benchmark is CUDA-only; MPS uses FP32"
            candidate["precision"] = choose_precision(device)
        elif stage == "compile":
            if device.type != "cuda" or platform.system() == "Windows":
                reason = "CUDA Inductor comparison runs on Linux/Colab; skipped on this platform"
            candidate["compile"] = True
        elif stage == "flash_attention":
            if device.type != "cuda" or candidate["precision"] == "fp32":
                reason = "Forced CUDA FlashAttention needs a supported GPU and FP16/BF16"
            candidate.update(attention="sdpa", force_flash=True)
        elif stage == "padded_vocab":
            candidate["padded"] = True
        try:
            if reason:
                row = dict(status="skipped", reason=reason)
            else:
                row = measure_variant(args, tokens, device, candidate)
                accepted = candidate
                if baseline_ms is None:
                    baseline_ms = row["median_ms"]
                row["speedup_vs_baseline"] = baseline_ms / row["median_ms"]
                row["speedup_vs_previous"] = (previous_ms or baseline_ms) / row["median_ms"]
                previous_ms = row["median_ms"]
        except (RuntimeError, NotImplementedError, FloatingPointError) as error:
            # Report failures as failures, never silently fall back and label
            # a math SDPA kernel as FlashAttention or eager as compilation.
            row = dict(status="failed", reason=f"{type(error).__name__}: {error}"[:1600])
        row["stage"] = stage
        evidence["stages"].append(row)
        save_json(path, evidence)
        print(stage, json.dumps(row, ensure_ascii=True), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if stage == "baseline" and row["status"] != "measured":
            raise RuntimeError("Baseline failed; reduce batch size/sequence length and rerun")
    return evidence


def plot_schedule(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [learning_rate(i, args.lr_warmup, args.steps) for i in range(args.steps)]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(range(1, args.steps + 1), rates)
    ax.set(xlabel="Optimizer update", ylabel="Learning rate", title="Linear warmup + cosine decay")
    ax.grid(alpha=0.3)
    fig.savefig(args.output_dir / "learning_rate.png", dpi=150)
    plt.close(fig)
    save_json(args.output_dir / "schedule.json", dict(steps=args.steps, warmup=args.lr_warmup,
                                                     max_lr=6e-4, min_lr=6e-5, rates=rates))


def train(args, tokens, device, provenance):
    settings = default_settings()
    settings.update(precision=choose_precision(device), attention="sdpa", padded=True,
                    tf32=device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8,
                    compile=args.compile)
    if args.compile and (device.type != "cuda" or platform.system() == "Windows"):
        raise ValueError("Use --compile with CUDA on Linux/Colab")
    set_tf32(settings["tf32"])
    model = create_model(args.tiny, settings, device)
    optimizer, audit = configure_optimizer(model, device)
    config = asdict(model.config)
    count = model.num_parameters()
    if args.compile:
        model = torch.compile(model, fullgraph=True)
    effective_tokens = args.batch_size * args.sequence_length * args.accumulation_steps
    loader = DataLoaderLite(tokens, args.batch_size, args.sequence_length)
    scaler = make_scaler(device, settings["precision"])
    evidence = dict(provenance, settings=settings, model_config=config, parameters=count,
                    optimizer=audit, effective_batch_tokens=effective_tokens,
                    batch_size=args.batch_size, sequence_length=args.sequence_length,
                    accumulation_steps=args.accumulation_steps,
                    planned_steps=args.steps, warmup_steps=args.lr_warmup,
                    clip_global_norm=1.0, status="running", history=[])
    path = args.output_dir / "training.json"
    save_json(path, evidence)
    for step in range(args.steps):
        lr = learning_rate(step, args.lr_warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        synchronize(device)
        start = time.perf_counter()
        row = optimizer_update(model, optimizer, loader, device,
                               accumulation_steps=args.accumulation_steps,
                               precision=settings["precision"], scaler=scaler)
        synchronize(device)
        elapsed = time.perf_counter() - start
        row.update(step=step + 1, learning_rate=lr, milliseconds=elapsed * 1000,
                   tokens_per_second=effective_tokens / elapsed,
                   tokens_seen=loader.tokens_seen,
                   corpus_passes=loader.tokens_seen / len(tokens))
        evidence["history"].append(row)
        save_json(path, evidence)
        print(f"step {step+1:4d} | loss {row['loss']:.6f} | norm {row['gradient_norm_before_clip']:.4f} "
              f"| lr {lr:.8f} | {elapsed*1000:.2f} ms | {effective_tokens/elapsed:.1f} tok/s "
              f"| skipped {row['skipped_update']}", flush=True)
    evidence["status"] = "complete"
    evidence["successful_updates"] = sum(not row["skipped_update"] for row in evidence["history"])
    evidence["complete_corpus_passes"] = loader.tokens_seen // len(tokens)
    save_json(path, evidence)
    return evidence


def report(output):
    lines = ["# Hafta 11 — eğitim ve hızlandırma raporu", "",
             "Bu rapor yalnızca aynı klasördeki ölçüm dosyalarını özetler.", "",
             "Weight tying: 50.257 × 768 = **38.597.376** parametre tasarrufu. "
             "Bağlı model 124.439.808; bağımsız model 163.037.184 parametre içerir. "
             "50.304 vocab ile bağlı model 124.475.904 parametredir.", "",
             "Başlangıç: matrislerde std=0.02; her bloktaki iki residual çıkış "
             "projeksiyonunda std=0.02/√(2L). Toplanan katkıların varyansı derinlikle "
             "büyümesin diye her katkının ölçeğini küçültüyoruz. Bu bir başlangıç "
             "yaklaşımıdır; dalların bağımsız olduğunu garanti etmez.", "",
             "Gradient accumulation: eşit sayıda token içeren K mikro batch'in "
             "ortalama loss'unu K'ya bölüyoruz. Böylece toplam gradient, birleşik "
             "batch'in ortalama loss gradient'ine eşit olur. Clipping ve AdamW bir "
             "kez uygulanır; FP16'da önce unscale edilir.", ""]
    bench_path, train_path = output / "benchmark.json", output / "training.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        lines += [f"Donanım: `{bench['environment']['hardware']}`; "
                  f"küçük model kontrolü: **{bench['smoke_only']}**.", "",
                  f"B,T={bench['benchmark_shape']}; warmup={bench['warmup_steps']}; "
                  f"ölçülen adım={bench['measured_steps']}. Süre: veri aktarımı, "
                  "forward, backward, clipping ve optimizer dahil; derleme/warmup hariç.", "",
                  "| Aşama | ms/adım (medyan) | token/s | Önceki ölçüme göre hız | Durum |",
                  "|---|---:|---:|---:|---|"]
        measured = []
        for row in bench["stages"]:
            if row["status"] == "measured":
                lines.append(f"| {row['stage']} | {row['median_ms']:.3f} | "
                             f"{row['tokens_per_second']:.1f} | {row['speedup_vs_previous']:.3f}× | ölçüldü |")
                if row["stage"] != "baseline":
                    measured.append(row)
            else:
                reason = row["reason"].replace("|", "/").replace("\n", " ")
                lines.append(f"| {row['stage']} | — | — | — | {row['status']}: {reason} |")
        lines += ["", "Aşamalar birikimlidir. Atlanan/başarısız bir aşama sonraki "
                  "ayara eklenmez. Padding, tokenizer'ı değiştirmez; softmax'a ek "
                  "sınıflar katar, bu yüzden loss tam olarak aynı problem değildir.", ""]
        if measured:
            best = max(measured, key=lambda row: row["speedup_vs_previous"])
            lines += [f"En yüksek artımlı hız oranı: **{best['stage']} "
                      f"({best['speedup_vs_previous']:.3f}×)**. "
                      "1'in altındaki oran yavaşlama demektir; tek koşu belirsizliğini gidermez.", ""]
        eligible = not bench["smoke_only"] and bench["environment"]["device"].startswith(("cuda", "mps"))
        lines += [f"En az iki GPU iyileştirme ölçümü şartı: "
                  f"**{'sağlandı' if eligible and len(measured) >= 2 else 'henüz sağlanmadı'}**.", ""]
    else:
        lines += ["GPU hız ölçümleri: **bekliyor**.", ""]
    if train_path.exists():
        result = json.loads(train_path.read_text(encoding="utf-8"))
        lines += [f"Eğitim durumu: **{result['status']}**; küçük model kontrolü: "
                  f"**{result['smoke_only']}**; donanım: `{result['environment']['hardware']}`.", "",
                  f"AdamW: β=(0.9,0.95), ε=1e-8, decay=0.1; "
                  f"decay uygulanan parametre: **{result['optimizer']['decay_parameters']:,}** "
                  f"({result['optimizer']['decay_tensors']} matris). "
                  f"Decay dışı: {result['optimizer']['no_decay_parameters']:,}.", "",
                  f"B,T={result['batch_size']},{result['sequence_length']}; "
                  f"etkin batch: **{result['effective_batch_tokens']} token**; "
                  f"accumulation={result['accumulation_steps']}. "
                  f"Başarılı optimizer güncellemesi: {result.get('successful_updates', 'eğitim sürüyor')}. "
                  f"Tam veri turu: {result.get('complete_corpus_passes', 'eğitim sürüyor')}.", "",
                  "Her adımın loss, clipping öncesi norm, LR, süre ve token/s değeri "
                  "`training.json` dosyasındadır. FP16 atlanan güncellemeleri ayrıca kaydedilir.", ""]
    else:
        lines += ["Model eğitimi: **bekliyor**.", ""]
    lines += ["![Warmup ve cosine öğrenme oranı](learning_rate.png)", "",
              "Bu haftaki demo metni eğitim hattını gösterir. Ayrılmış validation "
              "verisi ve genelleme değerlendirmesi gelecek haftanın kapsamıdır. "
              "Küçük CPU koşuları 124M GPU ödevinin tamamlandığı anlamına gelmez.", "",
              "Ek görev: DDP/Türkçe tokenizer eğitim karşılaştırması **yapılmadı**. "
              "Farklı tokenizer loss'ları doğrudan kıyaslanmaz; aynı ayrılmış metindeki "
              "toplam NLL / (ln(2) × UTF-8 byte sayısı) ile bits/byte ölçülebilir. "
              "Eşit süre için işlenen byte ve corpus turu da kaydedilmelidir.", ""]
    (output / "submission_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "benchmark", "train", "schedule", "report"), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--corpus", type=Path, default=HERE / "data/demo.txt")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/local_smoke")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr-warmup", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="Compile training too (benchmark tries it automatically)")
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.accumulation_steps, args.warmup,
           args.benchmark_steps, args.threads) < 1:
        parser.error("Batch, context, accumulation, warmup, measurement and thread counts must be positive")
    if args.sequence_length > config_for(args.tiny).block_size:
        parser.error("Sequence length exceeds model context; use <=32 with --tiny")
    learning_rate(0, args.lr_warmup, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.task == "report":
        report(args.output_dir)
        return
    if args.task == "schedule":
        plot_schedule(args)
        report(args.output_dir)
        return
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    tokens, corpus = load_tokens(args.corpus, args.tiny)
    provenance = dict(environment=environment(device), corpus=corpus, source_sha256=source_hashes(),
                      smoke_only=args.tiny,
                      run_configuration={key: getattr(args, key) for key in (
                          "batch_size", "sequence_length", "accumulation_steps", "steps",
                          "lr_warmup", "warmup", "benchmark_steps", "compile")})
    # One folder must describe one environment/configuration and source snapshot.
    for name in ("benchmark.json", "training.json"):
        existing = args.output_dir / name
        if existing.exists():
            prior = json.loads(existing.read_text(encoding="utf-8"))
            if any(prior.get(key) != provenance[key] for key in provenance):
                raise ValueError("Output folder has different provenance; select a new --output-dir")
    plot_schedule(args)
    try:
        if args.task in ("all", "benchmark"):
            benchmark(args, tokens, device, provenance)
        if args.task in ("all", "train"):
            train(args, tokens, device, provenance)
    finally:
        report(args.output_dir)


if __name__ == "__main__":
    main()
