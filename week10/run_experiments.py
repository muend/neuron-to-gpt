"""Inspect GPT-2, verify imported weights, sample, and overfit one fixed batch.

All metrics describe executed work. Running only one task leaves the other
task explicitly pending in the report; tiny runs are labeled smoke-only.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

# Set cache paths before importing network-backed libraries. These files are
# local dependencies, never part of the exported results directory.
os.environ.setdefault("HF_HOME", str(HERE / ".cache" / "huggingface"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / ".cache" / "tiktoken"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.nn import functional as F
import tiktoken

try:
    from .model import GPT, GPTConfig
except ImportError:
    from model import GPT, GPTConfig

PROMPTS = {
    "en": "Hello, I am a language model and",
    "tr": "Merhaba, ben bir dil modeliyim ve",
}
TRANSLATIONS = [
    {"en": "The children are reading a book in the garden.",
     "tr": "Çocuklar bahçede bir kitap okuyor."},
    {"en": "I will meet my friends at the station tomorrow morning.",
     "tr": "Yarın sabah istasyonda arkadaşlarımla buluşacağım."},
    {"en": "We talked about the things we learned during our journey.",
     "tr": "Yolculuğumuz sırasında öğrendiklerimiz hakkında konuştuk."},
]
TRAINING_TEXT = (
    "The small laboratory opened early on Monday. A researcher placed a blue notebook "
    "on the desk and wrote down the questions for the day. Outside, rain tapped on the "
    "window while the first train crossed the bridge. Each experiment began with a "
    "simple prediction, followed by a measurement and a careful comparison. Nobody "
    "expected the first attempt to work perfectly. The team checked the inputs, the "
    "dimensions of every layer, and the direction of the gradients. When a result looked "
    "surprising, they used a smaller example that could be inspected by hand. A model "
    "that cannot remember one small batch probably has a problem in its training path. "
    "A model that can remember it has passed one useful check, but has not demonstrated "
    "that it can generalize. After lunch, the researcher saved the measurements and "
    "explained the remaining uncertainty to the rest of the team. They agreed to keep "
    "the original data separate from generated examples, and to record the exact settings "
    "of every completed run. At sunset the rain stopped, the room grew quiet, and the "
    "next experiment was ready to begin."
)


def save_json(path: Path, value) -> None:
    """Replace a complete result atomically, including during long training runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def environment(device: torch.device) -> dict:
    versions = {}
    for name in ("torch", "transformers", "tiktoken", "safetensors", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions,
            "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
            "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads()}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def plot_positions(weights: torch.Tensor, destination: Path) -> dict:
    plt = plotting()
    array = weights.detach().float().cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    maximum = float(abs(array).max())
    im = axes[0].imshow(array.T, aspect="auto", origin="lower", cmap="RdBu_r",
                        vmin=-maximum, vmax=maximum, interpolation="nearest")
    axes[0].set(title="GPT-2 learned position embeddings (1024 positions × 768 channels)",
                xlabel="Position", ylabel="Embedding channel")
    fig.colorbar(im, ax=axes[0], label="Learned weight")
    selected = (0, 1, 2, 3, 4, 5)
    for channel in selected:
        axes[1].plot(array[:, channel], lw=0.9, label=f"channel {channel}")
    axes[1].set(title="Six individual channels", xlabel="Position", ylabel="Learned weight")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=3)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return {"shape": list(array.shape), "minimum": float(array.min()),
            "maximum": float(array.max()), "mean": float(array.mean()),
            "std": float(array.std()), "plotted_channels": list(selected)}


@torch.no_grad()
def compare_logits(model: GPT, reference, encoding, device: torch.device) -> dict:
    """Compare the same full FP32 tensors, with both dropout and TF32 disabled."""
    texts = [PROMPTS["en"], PROMPTS["tr"], "A quick check: 123, çığ and 🙂."]
    cases = []
    model.eval()
    reference.eval()
    for text in texts:
        ids = encoding.encode_ordinary(text)
        indices = torch.tensor([ids], dtype=torch.long, device=device)
        expected = reference(indices, use_cache=False).logits
        actual, _ = model(indices)
        error = (actual - expected).abs()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        cases.append({"text": text, "input_ids": ids, "shape": list(actual.shape),
                      "max_abs_difference": float(error.max()),
                      "mean_abs_difference": float(error.mean()),
                      "allclose": True,
                      "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))})
    return {"precision": "fp32", "attention": "eager", "tf32": False,
            "rtol": 1e-4, "atol": 1e-4, "all_passed": True, "cases": cases}


@torch.no_grad()
def sample_text(model: GPT, encoding, device: torch.device, *, seed: int,
                new_tokens: int, top_k: int, temperature: float) -> list[dict]:
    model.eval()
    rows = []
    for offset, (language, prompt) in enumerate(PROMPTS.items()):
        ids = encoding.encode_ordinary(prompt)
        generator = torch.Generator(device=device).manual_seed(seed + offset)
        start = time.perf_counter()
        output = model.generate(torch.tensor([ids], dtype=torch.long, device=device),
                                max_new_tokens=new_tokens, temperature=temperature,
                                top_k=top_k, generator=generator)
        synchronize(device)
        generated_ids = output[0].tolist()
        rows.append({"language": language, "prompt": prompt, "prompt_ids": ids,
                     "prompt_tokens": len(ids), "seed": seed + offset,
                     "top_k": top_k, "temperature": temperature,
                     "new_tokens": len(generated_ids) - len(ids),
                     "generated_ids": generated_ids,
                     "continuation": encoding.decode(generated_ids[len(ids):]),
                     "full_text": encoding.decode(generated_ids),
                     "seconds": time.perf_counter() - start})
    return rows


@torch.no_grad()
def sentence_loss(model: GPT, encoding, text: str, device: torch.device) -> dict:
    """Score every text token after an explicit end-of-text start context.

    GPT-2 has no separate BOS token. The shared end-of-text token supplies the
    same initial context to both languages, so the first actual text token is
    included in NLL and the byte denominator covers precisely the scored text.
    """
    ids = encoding.encode_ordinary(text)
    inputs = [encoding.eot_token] + ids[:-1]
    x = torch.tensor([inputs], dtype=torch.long, device=device)
    y = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
    byte_count = len(text.encode("utf-8"))
    return {"text": text, "tokens": len(ids), "words": len(text.split()),
            "bytes": byte_count, "ids": ids, "total_nll_nats": nll,
            "nats_per_token": nll / len(ids), "bits_per_byte": nll / byte_count / math.log(2),
            "context": "initial endoftext ID 50256; every text token scored"}


def pretrained_experiment(args, device: torch.device) -> dict:
    from transformers import GPT2LMHeadModel

    destination = args.output_dir
    start = time.perf_counter()
    print("Loading official GPT-2 safetensors (124M); FP32 eager attention...", flush=True)
    reference = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", revision=args.revision, cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only, use_safetensors=True,
        attn_implementation="eager", torch_dtype=torch.float32,
    ).eval()
    resolved_revision = getattr(reference.config, "_commit_hash", None)
    parameter_rows = []
    # Print every weight name, including the tied output alias, but count its
    # shared storage only once when calculating the actual model size.
    seen_parameters = {}
    for name, parameter in reference.named_parameters(remove_duplicate=False):
        shared_with = seen_parameters.get(id(parameter))
        row = {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel(),
               "shared_with": shared_with}
        seen_parameters.setdefault(id(parameter), name)
        parameter_rows.append(row)
        alias_note = f" (shared with {shared_with})" if shared_with else ""
        print(f"{name:45s} {str(row['shape']):20s} {row['parameters']:,}{alias_note}", flush=True)
    unique_count = sum(row["parameters"] for row in parameter_rows if row["shared_with"] is None)
    with (destination / "weights.tsv").open("w", encoding="utf-8") as file:
        file.write("name\tshape\tparameters\tshared_with\n")
        for row in parameter_rows:
            file.write(f"{row['name']}\t{json.dumps(row['shape'])}\t{row['parameters']}\t{row['shared_with'] or ''}\n")
    print(f"Unique parameters: {unique_count:,}; token/output embedding is tied", flush=True)
    position_metrics = plot_positions(reference.transformer.wpe.weight, destination / "position_embeddings.png")
    config = GPTConfig()
    model = GPT(config)
    load_result = model.load_hf_weights(reference)
    if model.num_parameters() != unique_count:
        raise AssertionError("Our model's unique parameter count differs from Hugging Face")
    reference = reference.to(device)
    model = model.to(device).eval()
    encoding = tiktoken.get_encoding("gpt2")
    parity = compare_logits(model, reference, encoding, device)
    # Release reference weights before generation/scoring and later training.
    del reference
    cleanup(device)
    samples = sample_text(model, encoding, device, seed=args.seed,
                          new_tokens=args.new_tokens, top_k=args.top_k,
                          temperature=args.temperature)
    cl100k = tiktoken.get_encoding("cl100k_base")
    token_comparison = []
    for language, text in PROMPTS.items():
        ids = encoding.encode_ordinary(text)
        modern_ids = cl100k.encode_ordinary(text)
        token_comparison.append({"language": language, "text": text, "words": len(text.split()),
                                 "gpt2_tokens": len(ids), "gpt2_ids": ids,
                                 "gpt2_bytes": [encoding.decode_single_token_bytes(i).hex(" ") for i in ids],
                                 "cl100k_tokens": len(modern_ids), "cl100k_ids": modern_ids})
    translated_losses = []
    model.eval()
    for pair in TRANSLATIONS:
        translated_losses.append({language: sentence_loss(model, encoding, text, device)
                                  for language, text in pair.items()})
    result = {"schema_version": 1, "status": "completed", "task": "pretrained",
              "model": "openai-community/gpt2", "requested_revision": args.revision,
              "resolved_revision": resolved_revision,
              "environment": environment(device), "config": asdict(config),
              "unique_parameters": unique_count, "tied_embedding_parameters_saved": 50257 * 768,
              "weights": parameter_rows, "position_embeddings": position_metrics,
              "load_info": load_result if isinstance(load_result, dict) else None,
              "logit_parity": parity, "samples": samples,
              "tokenizer_comparison": token_comparison,
              "translated_sentence_losses": translated_losses,
              "elapsed_seconds": time.perf_counter() - start}
    save_json(destination / "pretrained_metrics.json", result)
    for row in samples:
        (destination / f"sample_{row['language']}.txt").write_text(row["full_text"] + "\n", encoding="utf-8")
    del model
    cleanup(device)
    print(f"Pretrained tasks completed; largest logits difference "
          f"{max(row['max_abs_difference'] for row in parity['cases']):.3g}", flush=True)
    return result


def fixed_batch(encoding, batch_size: int, block_size: int, device: torch.device):
    tokens = encoding.encode_ordinary(TRAINING_TEXT)
    # The diagnostic needs a fixed batch, not a corpus/data-loader benchmark.
    # Distinct first tokens avoid contradictory next-token labels at position 0.
    rows, starts, cursor, first_tokens = [], [], 0, set()
    while len(rows) < batch_size and cursor + block_size < len(tokens):
        if tokens[cursor] not in first_tokens:
            rows.append(tokens[cursor:cursor + block_size + 1])
            starts.append(cursor)
            first_tokens.add(tokens[cursor])
            cursor += block_size + 1
        else:
            cursor += 1
    if len(rows) < batch_size:
        raise ValueError("Original diagnostic text is too short for this batch; lower --batch-size or --block-size")
    batch = torch.tensor(rows, dtype=torch.long, device=device)
    metadata = {"source": "original English diagnostic paragraph embedded in run_experiments.py",
                "source_sha256": hashlib.sha256(TRAINING_TEXT.encode("utf-8")).hexdigest(),
                "batch_sha256": hashlib.sha256(json.dumps(rows).encode("ascii")).hexdigest(),
                "row_start_offsets": starts, "input_ids": batch[:, :-1].tolist(),
                "target_ids": batch[:, 1:].tolist(),
                "tokens_per_update": batch_size * block_size}
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), metadata


def amp_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type,
                          dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


@torch.no_grad()
def batch_loss(model: GPT, x, y) -> tuple[float, float]:
    model.eval()
    logits, loss = model(x, y)
    accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
    return loss.item(), accuracy


def plot_overfit(history: list[dict], destination: Path) -> None:
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot([row["step"] for row in history], [row["loss"] for row in history])
    axes[0].axhline(math.log(50257), color="gray", ls="--", label="ln(50257)")
    axes[0].set(title="One fixed batch: FP32 evaluation loss", xlabel="Optimizer updates", ylabel="Nats / token")
    axes[0].legend()
    axes[1].plot([row["step"] for row in history], [max(row["loss"], 1e-8) for row in history])
    axes[1].set(yscale="log", title="Same loss on a log scale", xlabel="Optimizer updates", ylabel="Nats / token (log scale)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def overfit_experiment(args, device: torch.device) -> dict:
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("FP16/BF16 training requires CUDA here; select --precision fp32 on CPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA GPU does not support BF16; use FP16 or FP32")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    encoding = tiktoken.get_encoding("gpt2")
    if args.tiny:
        config = GPTConfig(block_size=max(128, args.block_size), vocab_size=50257,
                           n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    else:
        config = GPTConfig(dropout=0.0)
    model = GPT(config).to(device)
    x, y, batch_metadata = fixed_batch(encoding, args.batch_size, args.block_size, device)
    initial_loss, initial_accuracy = batch_loss(model, x, y)
    uniform_loss = math.log(config.vocab_size)
    if not math.isfinite(initial_loss):
        raise AssertionError("Random initialization produced a non-finite loss")
    print(f"Random model: {model.num_parameters():,} parameters; initial loss={initial_loss:.6f}; "
          f"ln(vocab)={uniform_loss:.6f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    history = [{"step": 0, "loss": initial_loss, "token_accuracy": initial_accuracy}]
    result = {"schema_version": 1, "status": "running", "task": "single_batch_overfit",
              "evidence": "smoke_only" if args.tiny else "full_gpt2_124m",
              "environment": environment(device), "config": asdict(config),
              "parameters": model.num_parameters(), "seed": args.seed,
              "steps_requested": args.steps, "steps_completed": 0, "skipped_updates": 0,
              "precision": args.precision, "evaluation_precision": "fp32",
              "learning_rate": args.learning_rate, "weight_decay": 0.0,
              "batch": batch_metadata, "initial_loss": initial_loss,
              "uniform_baseline_nats": uniform_loss,
              "initial_minus_uniform": initial_loss - uniform_loss,
              "near_uniform_initial_loss": abs(initial_loss - uniform_loss) < 1.0,
              "history": history}
    save_json(args.output_dir / "overfit_metrics.json", result)
    synchronize(device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.precision):
            _, loss = model(x, y)
        if not bool(torch.isfinite(loss)):
            result.update(status="failed_nonfinite_loss", steps_completed=step - 1)
            save_json(args.output_dir / "overfit_metrics.json", result)
            raise FloatingPointError(f"Non-finite training loss at update {step}")
        before_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < before_scale
        result["skipped_updates"] += int(skipped)
        result["steps_completed"] = step
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            score, accuracy = batch_loss(model, x, y)
            history.append({"step": step, "loss": score, "token_accuracy": accuracy,
                            "training_loss_before_update": float(loss.detach()),
                            "gradient_norm_before_clip": grad_norm,
                            "last_update_skipped": skipped})
            synchronize(device)
            result.update(elapsed_seconds=time.perf_counter() - start,
                          final_loss=score, final_token_accuracy=accuracy,
                          optimizer_updates=step - result["skipped_updates"])
            save_json(args.output_dir / "overfit_metrics.json", result)
            print(f"step {step:4d}/{args.steps}: fixed-batch loss={score:.6f}, "
                  f"accuracy={accuracy:.3%}, skipped={result['skipped_updates']}", flush=True)
    synchronize(device)
    result.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                  overfit_passed=result["final_loss"] < 0.1 and result["final_token_accuracy"] > 0.99,
                  near_zero_threshold_nats=0.1,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    save_json(args.output_dir / "overfit_metrics.json", result)
    plot_overfit(history, args.output_dir / "overfit_loss.png")
    del model, optimizer
    cleanup(device)
    return result


def write_report(destination: Path) -> Path:
    pretrained = load_json(destination / "pretrained_metrics.json")
    overfit = load_json(destination / "overfit_metrics.json")
    lines = ["# Week 10 — GPT-2 (124M)", "",
             "Bu rapor `run_experiments.py` tarafından yalnız kaydedilmiş çalışma sonuçlarından üretildi. "
             "Eksik deneyler tamamlanmış gibi gösterilmez.", ""]
    if pretrained:
        lines += ["## 1. Hazır model ve pozisyon embedding'leri", "",
                  f"Model: `{pretrained['model']}`; revision: `{pretrained['requested_revision']}`. "
                  f"Benzersiz parametre: **{pretrained['unique_parameters']:,}**. "
                  "Tüm adlar/boyutlar [weights.tsv](weights.tsv) ve JSON çıktısında bulunur. "
                  "Token embedding'i ile çıktı ağırlığı aynı parametreyi kullandığından iki kez sayılmaz.", "",
                  "![Öğrenilmiş pozisyon embedding'leri](position_embeddings.png)", "",
                  "Wpe, 1024 konum için öğrenilmiş 768 boyutlu vektörlerdir. Harita ve seçili kanallar "
                  "konumlar arasında yapı ve farklı değişim ölçekleri gösterir; bu bir hazır sinüs/kosinüs "
                  "tablosu değildir. Tek kanaldaki şekil, o kanalın belirli bir dilsel kavrama karşılık "
                  "geldiğini veya modelin 1024 konumun dışına genelleyebildiğini kanıtlamaz.", "",
                  "## 2. Kendi GPT-2 koduma ağırlık yükleme", "",
                  "Aynı girdinin tüm logits'leri, dropout kapalıyken iki modelde FP32/eager attention "
                  "ile karşılaştırıldı. TF32 kapalı. Tolerans: rtol=1e-4, atol=1e-4.", "",
                  "| Girdi | Boyut | En büyük mutlak fark | Ortalama fark | Eşleşme |",
                  "|---|---|---:|---:|---|"]
        for row in pretrained["logit_parity"]["cases"]:
            lines.append(f"| {row['text']} | {row['shape']} | {row['max_abs_difference']:.8g} | "
                         f"{row['mean_abs_difference']:.8g} | {row['allclose']} |")
        lines += ["", "Hafta 8'deki koduma göre farklar:", "",
                  "- 12 blok, 12 head, 768 embedding; 1024 konum; GPT-2 byte-BPE vocab'ı 50.257.",
                  "- Head başına ayrı Q/K/V katmanları yerine tüm head'ler için tek `c_attn` projeksiyonu; "
                  "çıkış üçe ayrılır ve head eksenine taşınır. GPT-2 bu projeksiyonlarda bias kullanır.",
                  "- Feedforward aktivasyonu ReLU yerine tanh yaklaşımıyla GELU; ara genişlik yine 4×768.",
                  "- Pre-LayerNorm ve iki residual yol korunur. Son LayerNorm'dan sonra bias içermeyen çıktı katmanı vardır.",
                  "- Token ve çıktı embedding'leri paylaşılır. Bu, standart GPT-2'nin özelliğidir; "
                  f"ayrı ağırlıklara göre {pretrained['tied_embedding_parameters_saved']:,} parametre kazandırır.",
                  "- Yerel örnekleme, nedensel attention ve öğrenilmiş mutlak pozisyon embedding'leri aynı temel fikri korur.", "",
                  "Hugging Face'in tarihsel `Conv1D` ağırlıkları (girdi, çıktı), PyTorch `Linear` "
                  "ağırlıkları (çıktı, girdi) düzenindedir. Bu yüzden attention `c_attn`, attention "
                  "`c_proj`, MLP `c_fc` ve MLP `c_proj` ağırlıkları yüklenirken transpoze edilir; "
                  "embedding ve LayerNorm ağırlıkları doğrudan kopyalanır. Kare projeksiyonlarda da "
                  "transpoz gereklidir: şeklin aynı kalması matrisin aynı olduğu anlamına gelmez.", "",
                  "## 3. Sampling ve Türkçe", "",
                  "Her adımda son konumun logits'i temperature ile ölçeklenir; en yüksek k değer "
                  "dışındakiler elenir, softmax sonrası multinomial ile tek token seçilir ve bağlama eklenir. "
                  "Her iki örnek de kendi model kodumdan üretilmiştir; insan yazısı veya eğitim verisinden "
                  "alıntı olarak sunulmaz.", ""]
        for row in pretrained["samples"]:
            lines += [f"### {row['language']} — seed {row['seed']}, top-k {row['top_k']}, T={row['temperature']}", "",
                      "```text", row["full_text"], "```", ""]
        lines += ["| Başlangıç | GPT-2 token | cl100k_base token | Kelime |", "|---|---:|---:|---:|"]
        for row in pretrained["tokenizer_comparison"]:
            lines.append(f"| {row['text']} | {row['gpt2_tokens']} | {row['cl100k_tokens']} | {row['words']} |")
        lines += ["", "GPT-2 ağırlıklı olarak İngilizce WebText ile eğitildi. Türkçe veri ve sık birleşimler "
                  "daha az temsil edildiğinden Türkçe sözcükler daha çok parçaya bölünebilir; aynı token "
                  "bağlamı daha az Türkçe metin taşıyabilir. Tokenizer bunu görünür kılar, fakat düşük "
                  "Türkçe kaliteyi tek başına açıklamaz: modelin dil dağılımı da önemlidir. Geçen hafta "
                  "cl100k_base için bulunan Türkçe/İngilizce farkıyla aynı ilişki burada GPT-2'nin ayrı, "
                  "50.257 öğeli sözlüğünde ölçülür; iki tokenizer'ın sayıları aynı olmak zorunda değildir.", "",
                  "## 5b. Eşleştirilmiş Türkçe ve İngilizce cümlelerin loss'u", "",
                  "Her cümle aynı başlangıç bağlamı olan `<|endoftext|>` ID'sinden sonra puanlandı; "
                  "metnin ilk token'ı dahil bütün token'lar hesaba katıldı. Karşılaştırma küçük ve "
                  "kontrolsüzdür; bir dil için evrensel kalite sıralaması değildir.", "",
                  "| Dil | Cümle | Token | Nats/token | Bits/byte |", "|---|---|---:|---:|---:|"]
        for pair in pretrained["translated_sentence_losses"]:
            for language, row in pair.items():
                lines.append(f"| {language} | {row['text']} | {row['tokens']} | {row['nats_per_token']:.4f} | {row['bits_per_byte']:.4f} |")
        lines += ["", "Token başına loss hem dil dağılımından hem parçalama biçiminden etkilenir. "
                  "Bits/byte, toplam NLL'yi metnin UTF-8 byte sayısına ve ln(2)'ye böler; ortak birim "
                  "sağlar, ama farklı cümlelerin bilgi içeriğini ve çeviri tercihlerinin etkisini ortadan kaldırmaz. "
                  "Sonuçların tamamı tabloda gösterilir; Türkçe her cümlede mutlaka daha yüksek loss vermek zorunda değildir.", ""]
    else:
        lines += ["## Hazır ağırlık deneyleri: bekliyor", "",
                  "Bu klasörde `pretrained_metrics.json` bulunmuyor; ağırlık incelemesi, gerçek GPT-2 "
                  "logits eşleşmesi, örnek üretimi ve 5b sonuçları henüz kaydedilmedi.", ""]
    if overfit:
        smoke = overfit["evidence"] == "smoke_only"
        lines += ["## 4. Rastgele başlangıç ve tek batch ezberleme", "",
                  f"Durum: **{overfit['status']}**. "
                  + ("**Küçük smoke modeli; 124M GPT-2 eğitim sonucu değildir.**" if smoke else "**Tam 124M GPT-2 mimarisi.**"), "",
                  f"Parametre {overfit['parameters']:,}; donanım `{overfit['environment']['hardware']}`; "
                  f"eğitim `{overfit['precision']}`, değerlendirme FP32. "
                  f"Güncelleme {overfit.get('optimizer_updates', 0)}/{overfit['steps_requested']}; "
                  f"atlanan update {overfit['skipped_updates']}.", "",
                  f"İlk loss **{overfit['initial_loss']:.6f} nats/token**; "
                  f"ln(50257) = **{overfit['uniform_baseline_nats']:.6f}**; "
                  f"fark **{overfit['initial_minus_uniform']:+.6f}**.", "",
                  "Başlangıçta model doğru sınıfa ayrıcalık tanımaz; yaklaşık eşit olasılıkların "
                  "negatif log'u ln(vocab) olur. Rastgele logits tam eşit olmadığından ölçüm birebir "
                  "bu sayı değildir. Yakınlık bir sağlık kontrolüdür; yanlış veri kaydırması gibi "
                  "hataların hepsini tek başına dışlamaz.", ""]
        if "final_loss" in overfit:
            lines += [f"Son sabit-batch loss **{overfit['final_loss']:.6f}**; token doğruluğu "
                      f"**{overfit['final_token_accuracy']:.2%}**. "
                      f"Ezberleme eşiği: loss <0.1 ve doğruluk >%99; geçti: "
                      f"**{overfit.get('overfit_passed', False)}**.", "",
                      "![Aynı batch üzerinde loss](overfit_loss.png)", ""]
        lines += ["Her update aynı girdi ve bir token sağa kaydırılmış aynı hedefleri kullanır. "
                  "Dropout sıfırdır. Eğitim verisi bu deney için yazılmış özgün kısa İngilizce "
                  "paragraftır; kimlikler, kaydırma ve SHA-256 JSON'da kaydedilir. Bu test, forward, "
                  "cross-entropy, backward ve optimizer'ın birlikte öğrenebildiğini gösterir. "
                  "Aynı batch'i ezberlemek gerçek doğrulama başarısı veya yeni metinlerde genelleme değildir.", "",
                  "Başlangıç normal std=0.02; residual çıktı projeksiyonları 0.02/√(2×katman) "
                  "ölçeğindedir. Bunlar GPT-2 uyumlu başlangıcı kurar. Bu haftaki deneyde bütün "
                  "korpusu eğitme, öğrenme oranı planlama, dağıtık eğitim veya hız optimizasyonu iddiası yoktur.", ""]
    else:
        lines += ["## 4. Tek batch eğitimi: bekliyor", "",
                  "Bu klasörde `overfit_metrics.json` bulunmuyor. Rastgele 124M modelin ilk loss'u "
                  "ve birkaç yüz update ezberleme sonucu henüz ölçülmedi; Colab hücresini çalıştırın.", ""]
    lines += ["## Kaynaklar", "",
              "- [Karpathy: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)",
              "- [build-nanogpt](https://github.com/karpathy/build-nanogpt)",
              "- [GPT-2 ağırlıkları ve model kartı](https://huggingface.co/openai-community/gpt2)",
              "- [GPT-2 makalesi](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)",
              "- [Hugging Face GPT-2 uygulaması](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)", ""]
    target = destination / "submission_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pretrained", "overfit", "all", "report"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "local")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache" / "huggingface")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tiny", action="store_true", help="Use a small random model only for overfit smoke checks")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.block_size, args.eval_interval, args.new_tokens, args.top_k, args.threads) < 1:
        parser.error("Step, batch, context, generation and thread counts must be positive")
    if args.block_size > 1024 or args.top_k > 50257:
        parser.error("block-size must be <=1024 and top-k <=50257")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("learning-rate must be finite and positive")
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        parser.error("temperature must be finite and positive")
    if args.tiny and args.task in ("all", "pretrained"):
        parser.error("--tiny applies only to --task overfit; real pretrained evidence always uses full GPT-2")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "report":
        print(write_report(args.output_dir))
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose GPU in Colab or use --device cpu")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(json.dumps(environment(device), ensure_ascii=False, indent=2), flush=True)
    if args.task in ("pretrained", "all"):
        pretrained_experiment(args, device)
        print(write_report(args.output_dir), flush=True)
    if args.task in ("overfit", "all"):
        result = overfit_experiment(args, device)
        print(write_report(args.output_dir), flush=True)
        if not args.tiny and not result["overfit_passed"]:
            raise SystemExit("Full GPT-2 did not reach loss <0.1 and accuracy >99%; "
                             "inspect the saved history before claiming this task passed")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
