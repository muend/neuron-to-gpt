"""Audit saved GPT-2 measurements and assemble the Turkish submission report.

Uses only the standard library. Missing/running experiments remain pending;
inconsistent completed evidence fails rather than being presented as success.
The audit checks saved evidence, not a fresh forward pass or training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PARAMETERS = 124_439_808
SAVED_PARAMETERS = 50_257 * 768
SOURCE_FILES = {"model.py", "run_experiments.py", "test_gpt2.py", "requirements.txt"}
EXPECTED_CONFIG = {"block_size": 1024, "vocab_size": 50257, "n_layer": 12,
                   "n_head": 12, "n_embd": 768, "weight_tying": True}
EXPECTED_RUNS = ("local_pretrained", "a100", "t4")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def finite(value, minimum=None, maximum=None):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def close(first, second):
    return finite(first) and finite(second) and math.isclose(first, second, rel_tol=1e-8, abs_tol=1e-9)


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_json(path):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{path}: invalid JSON: {error}") from error
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def audit_config(config):
    require(isinstance(config, dict), "config must be an object")
    for name, expected in EXPECTED_CONFIG.items():
        require(config.get(name) == expected, f"full GPT-2 config {name} must equal {expected!r}")
    require(config.get("dropout") == 0.0, "recorded diagnostic must have dropout zero")


def expected_weight_shapes():
    shapes = {"transformer.wte.weight": [50257, 768], "transformer.wpe.weight": [1024, 768],
              "transformer.ln_f.weight": [768], "transformer.ln_f.bias": [768],
              "lm_head.weight": [50257, 768]}
    for index in range(12):
        prefix = f"transformer.h.{index}."
        for norm in ("ln_1", "ln_2"):
            shapes[prefix + norm + ".weight"] = [768]
            shapes[prefix + norm + ".bias"] = [768]
        # HF's historical Conv1D stores input, output dimensions.
        for name, shape in (("attn.c_attn", [768, 2304]), ("attn.c_proj", [768, 768]),
                            ("mlp.c_fc", [768, 3072]), ("mlp.c_proj", [3072, 768])):
            shapes[prefix + name + ".weight"] = shape
            shapes[prefix + name + ".bias"] = [shape[1]]
    return shapes


def audit_weights(rows):
    require(isinstance(rows, list), "weights must be a list")
    expected = expected_weight_shapes()
    seen, unique_count = {}, 0
    for row in rows:
        require(isinstance(row, dict), "each weight must be an object")
        name, shape = row.get("name"), row.get("shape")
        require(name in expected and name not in seen, f"unexpected or duplicate weight {name!r}")
        require(shape == expected[name], f"wrong shape for weight {name}")
        require(row.get("parameters") == math.prod(shape), f"wrong parameter count for {name}")
        shared = row.get("shared_with")
        if shared is None:
            require(name != "lm_head.weight", "output embedding must be recorded as a tied alias")
            unique_count += row["parameters"]
        else:
            require(name == "lm_head.weight" and shared == "transformer.wte.weight",
                    f"unexpected shared-weight relationship for {name}")
            require(shared in seen and seen[shared]["shape"] == shape,
                    "shared-weight target is absent or has a different shape")
        seen[name] = row
    require(set(seen) == set(expected), "weight inventory omits GPT-2 tensors")
    require(unique_count == PARAMETERS, f"weight inventory counts {unique_count}, expected {PARAMETERS}")


def require_png(path):
    require(path.is_file() and path.stat().st_size > 8, f"missing figure {path.name}")
    with path.open("rb") as file:
        require(file.read(8) == b"\x89PNG\r\n\x1a\n", f"invalid PNG signature: {path.name}")


def audit_pretrained(data, directory):
    require(data.get("schema_version") == 1 and data.get("task") == "pretrained",
            "unsupported pretrained metrics schema/task")
    require(data.get("status") == "completed", "pretrained metrics are not completed")
    require(data.get("model") == "openai-community/gpt2", "unexpected pretrained model")
    audit_config(data.get("config"))
    require(data.get("unique_parameters") == PARAMETERS, "wrong unique pretrained parameter count")
    require(data.get("tied_embedding_parameters_saved") == SAVED_PARAMETERS, "wrong tying savings")
    audit_weights(data.get("weights"))
    require(data.get("position_embeddings", {}).get("shape") == [1024, 768], "wrong WPE shape")
    parity = data.get("logit_parity", {})
    require(parity.get("all_passed") is True, "pretrained logit parity did not pass")
    require(parity.get("precision") == "fp32" and parity.get("attention") == "eager"
            and parity.get("tf32") is False, "parity must use FP32/eager with TF32 disabled")
    require(close(parity.get("rtol"), 1e-4) and close(parity.get("atol"), 1e-4),
            "unexpected logit comparison tolerances")
    cases = parity.get("cases")
    require(isinstance(cases, list) and len(cases) >= 1, "missing logits comparison cases")
    for case in cases:
        ids = case.get("input_ids")
        require(isinstance(ids, list) and ids and all(integer(i) and 0 <= i < 50257 for i in ids),
                "invalid comparison token IDs")
        require(case.get("shape") == [1, len(ids), 50257], "wrong logits shape")
        require(case.get("allclose") is True, "one logits comparison failed")
        require(finite(case.get("max_abs_difference"), 0) and finite(case.get("mean_abs_difference"), 0)
                and case["mean_abs_difference"] <= case["max_abs_difference"],
                "invalid logits difference measurements")
    require(finite(data.get("elapsed_seconds"), 0), "invalid pretrained elapsed time")
    require(isinstance(data.get("environment"), dict), "missing pretrained environment")
    require((directory / "weights.tsv").is_file(), "weights.tsv is missing")
    require((directory / "submission_report.md").is_file(), "per-run report is missing")
    require_png(directory / "position_embeddings.png")


def audit_batch(batch):
    require(isinstance(batch, dict), "missing fixed-batch metadata")
    inputs, targets = batch.get("input_ids"), batch.get("target_ids")
    require(isinstance(inputs, list) and isinstance(targets, list) and inputs
            and len(inputs) == len(targets), "invalid batch rows")
    width = len(inputs[0])
    require(0 < width <= 1024, "invalid batch context length")
    original_rows = []
    for x, y in zip(inputs, targets):
        require(isinstance(x, list) and isinstance(y, list) and len(x) == len(y) == width,
                "batch rows have inconsistent shapes")
        require(all(integer(i) and 0 <= i < 50257 for i in x + y), "batch contains invalid token IDs")
        require(x[1:] == y[:-1], "targets are not shifted by one token")
        original_rows.append(x + [y[-1]])
    digest = hashlib.sha256(json.dumps(original_rows).encode("ascii")).hexdigest()
    require(batch.get("batch_sha256") == digest, "fixed-batch checksum does not match saved IDs")
    require(sha256(batch.get("source_sha256")), "invalid diagnostic-source checksum")
    require(batch.get("tokens_per_update") == len(inputs) * width, "wrong tokens-per-update count")
    return len(inputs), width


def audit_overfit(data, directory):
    require(data.get("schema_version") == 1 and data.get("task") == "single_batch_overfit",
            "unsupported overfit metrics schema/task")
    require(data.get("evidence") == "full_gpt2_124m", "overfit evidence is not the full 124M GPT-2")
    audit_config(data.get("config"))
    require(data.get("parameters") == PARAMETERS, "wrong overfit parameter count")
    require(data.get("precision") in ("fp32", "fp16", "bf16")
            and data.get("evaluation_precision") == "fp32", "invalid training/evaluation precision")
    audit_batch(data.get("batch"))
    requested, completed, skipped = (data.get(name) for name in
                                     ("steps_requested", "steps_completed", "skipped_updates"))
    require(all(integer(value) for value in (requested, completed, skipped))
            and requested >= 300 and 0 <= skipped <= completed <= requested,
            "invalid training-attempt/skip counters or fewer than 300 requested attempts")
    if "optimizer_updates" in data:
        require(data["optimizer_updates"] == completed - skipped, "optimizer-update count disagrees with skips")
    require(finite(data.get("initial_loss"), 0), "invalid initial loss")
    require(close(data.get("uniform_baseline_nats"), math.log(50257)), "incorrect uniform-loss baseline")
    require(close(data.get("initial_minus_uniform"), data["initial_loss"] - math.log(50257)),
            "initial-loss difference does not match the saved baseline")
    history = data.get("history")
    require(isinstance(history, list) and history and history[0].get("step") == 0,
            "loss history must begin at step zero")
    require(close(history[0].get("loss"), data["initial_loss"]), "history initial loss differs")
    previous = -1
    for row in history:
        require(integer(row.get("step")) and previous < row["step"] <= completed,
                "loss history steps are out of order or exceed completed attempts")
        require(finite(row.get("loss"), 0) and finite(row.get("token_accuracy"), 0, 1),
                "invalid history loss or token accuracy")
        previous = row["step"]
    require(history[-1]["step"] == completed, "history does not reach the saved completed attempt")
    if "final_loss" in data:
        require(close(data["final_loss"], history[-1]["loss"])
                and close(data.get("final_token_accuracy"), history[-1]["token_accuracy"]),
                "final metrics disagree with the last history row")
    require(isinstance(data.get("environment"), dict), "missing overfit environment")
    if data.get("status") == "completed":
        require(completed == requested, "completed run did not finish all requested attempts")
        require("optimizer_updates" in data, "completed run omits actual optimizer-update count")
        require(finite(data.get("final_loss"), 0, 0.1) and data["final_loss"] < 0.1
                and finite(data.get("final_token_accuracy"), 0, 1)
                and data["final_token_accuracy"] > 0.99 and data.get("overfit_passed") is True,
                "completed run did not pass loss <0.1 and accuracy >99%")
        require(finite(data.get("elapsed_seconds"), 0), "invalid completed overfit elapsed time")
        if data["environment"].get("device") == "cuda":
            require(integer(data.get("peak_cuda_memory_bytes")) and data["peak_cuda_memory_bytes"] > 0,
                    "completed CUDA run omits peak allocated memory")
        require_png(directory / "overfit_loss.png")


def audit_manifest(data, run):
    require(set(data.get("source_hashes", {})) == SOURCE_FILES, "source manifest must cover the four bundled files")
    require(sha256(data.get("source_bundle_sha256")), "invalid source-bundle checksum")
    for name, digest in data["source_hashes"].items():
        require(sha256(digest), f"invalid source checksum: {name}")
        local = HERE / name
        require(local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == digest,
                f"exported source differs from current local {name}")
    notebook = read_json(HERE / "colab_gpt2.ipynb")
    require(notebook is not None, "the Colab notebook is needed to verify the source-bundle checksum")
    metadata = notebook.get("metadata", {})
    require(metadata.get("week10_source_bundle_sha256") == data["source_bundle_sha256"]
            and metadata.get("week10_source_hashes") == data["source_hashes"],
            "export source manifest differs from the saved Colab notebook snapshot")
    for kind in ("pretrained", "overfit"):
        metrics = run.get(kind)
        if metrics and metrics.get("environment", {}).get("device") == "cuda":
            require(data.get("gpu") == metrics["environment"].get("hardware"),
                    "source-manifest GPU differs from the run hardware")


def audit_run(directory):
    try:
        pretrained = read_json(directory / "pretrained_metrics.json")
        overfit = read_json(directory / "overfit_metrics.json")
        manifest = read_json(directory / "source_manifest.json")
        run = {"name": directory.name, "directory": directory,
               "pretrained": pretrained, "overfit": overfit, "manifest": manifest}
        if pretrained is not None:
            audit_pretrained(pretrained, directory)
        if overfit is not None:
            audit_overfit(overfit, directory)
        for kind in ("pretrained", "overfit"):
            metrics = run.get(kind)
            if metrics and directory.name in ("a100", "t4"):
                environment = metrics.get("environment", {})
                require(environment.get("device") == "cuda"
                        and directory.name in environment.get("hardware", "").lower(),
                        f"{directory.name} directory does not contain matching CUDA hardware evidence")
        if manifest is not None:
            audit_manifest(manifest, run)
        return run
    except (AuditError, KeyError, TypeError, AttributeError, IndexError) as error:
        raise AuditError(f"{directory}: {error}") from error


def audit_tokenizer_comparison(results_root):
    path = results_root / "tokenizer_comparison.json"
    data = read_json(path)
    if data is None:
        return None
    try:
        require(data.get("schema_version") == 1
                and data.get("source") == "week09/results/metrics.json"
                and data.get("word_definition") == "len(text.split())",
                "unsupported tokenizer-comparison schema or word definition")
        require(sha256(data.get("source_sha256")), "invalid Week 9 source checksum")
        previous_path = HERE.parent / "week09/results/metrics.json"
        previous = read_json(previous_path)
        if previous is not None:
            require(hashlib.sha256(previous_path.read_bytes()).hexdigest() == data["source_sha256"],
                    "tokenizer comparison does not reference the current recorded Week 9 metrics")
        rows = data.get("paragraphs")
        require(isinstance(rows, list) and len(rows) == 2
                and {row.get("language") for row in rows} == {"tr", "en"},
                "tokenizer comparison must contain the Turkish and English paragraphs")
        for row in rows:
            text, words = row.get("text"), row.get("words")
            require(isinstance(text, str) and integer(words) and words > 0
                    and len(text.split()) == words, "paragraph word count does not match the text")
            require(row.get("bytes") == len(text.encode("utf-8")), "paragraph byte count is inconsistent")
            ids = row.get("gpt2_ids")
            require(isinstance(ids, list) and all(integer(i) and 0 <= i < 50257 for i in ids)
                    and row.get("gpt2_tokens") == len(ids), "GPT-2 token count/IDs are inconsistent")
            for name in ("gpt2", "cl100k"):
                count = row.get(name + "_tokens")
                require(integer(count) and count > 0
                        and close(row.get(name + "_tokens_per_word"), count / words),
                        f"invalid {name} tokens-per-word calculation")
            if previous is not None:
                old = previous["paragraph_comparison"]["cl100k_base_" + row["language"]]
                require(old["words"] == words and old["bytes"] == row["bytes"]
                        and old["tokens"] == row["cl100k_tokens"],
                        "paragraph comparison differs from the recorded Week 9 counts")
    except (AuditError, KeyError, TypeError, AttributeError) as error:
        raise AuditError(f"{path}: {error}") from error
    return data


def audit_all(results_root):
    names = set(EXPECTED_RUNS)
    if results_root.is_dir():
        names.update(path.name for path in results_root.iterdir() if path.is_dir()
                     and any((path / name).is_file() for name in
                             ("pretrained_metrics.json", "overfit_metrics.json", "source_manifest.json")))
    ordered = list(EXPECTED_RUNS) + sorted(names - set(EXPECTED_RUNS))
    runs = [audit_run(results_root / name) for name in ordered]
    gpu_training = [run for run in runs if run["name"] in ("a100", "t4") and run["overfit"]]
    if len(gpu_training) == 2:
        first, second = (run["overfit"] for run in gpu_training)
        require(first["batch"]["batch_sha256"] == second["batch"]["batch_sha256"]
                and first["batch"]["source_sha256"] == second["batch"]["source_sha256"],
                "A100 and T4 did not train on the same saved fixed batch")
        require(first["seed"] == second["seed"] and first["learning_rate"] == second["learning_rate"],
                "A100 and T4 seed/learning-rate settings differ")
        first_manifest, second_manifest = (run["manifest"] for run in gpu_training)
        if first_manifest and second_manifest:
            require(first_manifest["source_bundle_sha256"] == second_manifest["source_bundle_sha256"]
                    and first_manifest["source_hashes"] == second_manifest["source_hashes"],
                    "A100 and T4 source snapshots differ")
    revisions = {run["pretrained"].get("resolved_revision") for run in runs if run["pretrained"]}
    require(len(revisions) <= 1, "pretrained runs use different resolved model revisions")
    audit_tokenizer_comparison(results_root)
    return runs


def linked(run, filename):
    return f"[{run['name']}]({run['name']}/{filename})"


def relative_links(text, prefix):
    def rewrite(match):
        label, target = match.groups()
        if re.match(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*:|/|#)", target):
            return match.group(0)
        return f"{label}({prefix}/{target})"
    return re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", rewrite, text)


def render_report(runs, results_root):
    complete_gpu = [r for r in runs if r["name"] in ("a100", "t4") and r["overfit"]
                    and r["overfit"].get("status") == "completed"]
    lines = ["# Week 10 — GPT-2 (124M): doğrulanmış sonuçlar", "",
             "Bu rapor `summarize_results.py` ile kaydedilmiş JSON, ağırlık boyutları, "
             "batch kimlikleri ve mevcut kaynak dosyaları denetlenerek oluşturuldu. Denetim "
             "yeni bir eğitim veya logits hesaplaması değildir; eksik koşular tamamlanmış sayılmaz.", "",
             "## Deneylerin durumu", "",
             "| Kayıt | Hazır ağırlık / logits eşleşmesi | Tam 124M tek-batch eğitimi |",
             "|---|---|---|"]
    for run in runs:
        pretrained, overfit = run["pretrained"], run["overfit"]
        first = f"{linked(run, 'pretrained_metrics.json')}: doğrulandı" if pretrained else "bekliyor"
        second = (f"{linked(run, 'overfit_metrics.json')}: "
                  + ("tamamlandı, ezberleme eşiği geçti" if overfit.get("status") == "completed"
                     else f"{overfit.get('status', 'bilinmiyor')}; henüz tamamlanmadı")) if overfit else "bekliyor"
        if run["name"] == "local_pretrained" and not overfit:
            second = "bu CPU koşusunda çalıştırılmadı"
        lines.append(f"| {run['name']} | {first} | {second} |")
    pretrained_runs = [r for r in runs if r["pretrained"]]
    if pretrained_runs:
        lines += ["", "| Hazır model koşusu | Donanım | Logits kontrolü | En büyük mutlak fark | Süre (sn) |",
                  "|---|---|---|---:|---:|"]
        for run in pretrained_runs:
            data = run["pretrained"]
            maximum = max(case["max_abs_difference"] for case in data["logit_parity"]["cases"])
            lines.append(f"| {linked(run, 'pretrained_metrics.json')} | {data['environment']['hardware']} | "
                         f"FP32/eager, TF32 kapalı | {maximum:.8g} | {data['elapsed_seconds']:.2f} |")
        lines += ["", "Her hazır model koşusunda aynı 124.439.808 benzersiz parametre "
                  "ve 50.257 çıktı sınıfı denetlendi. Paylaşılan token/çıktı embedding'i "
                  "iki kez sayılmadı. Sayısal eşleşme element başına rtol=1e-4, atol=1e-4 "
                  "ile kontrol edilir; en büyük mutlak fark tek başına bu birleşik toleransın sınırı değildir."]
    overfit_runs = [r for r in runs if r["overfit"]]
    if overfit_runs:
        lines += ["", "## İlk loss ve aynı batch'i ezberleme", "",
                  "| Koşu | Eğitim | İlk loss | Son FP32 loss | Son doğruluk | Gerçek update / deneme | Atlanan |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for run in overfit_runs:
            data = run["overfit"]
            final = f"{data['final_loss']:.6f}" if "final_loss" in data else "bekliyor"
            accuracy = f"{data['final_token_accuracy']:.2%}" if "final_token_accuracy" in data else "bekliyor"
            updates = data.get("optimizer_updates", data["steps_completed"] - data["skipped_updates"])
            lines.append(f"| {linked(run, 'overfit_metrics.json')} | {data['precision']} | "
                         f"{data['initial_loss']:.6f} | {final} | {accuracy} | "
                         f"{updates} / {data['steps_completed']} | {data['skipped_updates']} |")
        lines += ["", "| Koşu | Donanım | Batch × bağlam | Geçen süre (sn) | Tepe ayrılmış CUDA belleği (GiB) |",
                  "|---|---|---:|---:|---:|"]
        for run in overfit_runs:
            data = run["overfit"]
            batch, width = audit_batch(data["batch"])
            memory = data.get("peak_cuda_memory_bytes")
            memory_text = f"{memory / 1024 ** 3:.3f}" if memory is not None else "ölçülmedi"
            elapsed = f"{data['elapsed_seconds']:.2f}" if "elapsed_seconds" in data else "bekliyor"
            lines.append(f"| {run['name']} | {data['environment']['hardware']} | {batch} × {width} | "
                         f"{elapsed} | {memory_text} |")
        lines += ["", f"ln(50.257) = **{math.log(50257):.6f} nats/token**. "
                  "Başlangıç loss'u bu yaklaşık eşit olasılık referansına göre değerlendirilir. "
                  "Gerçek optimizer update sayısı, deneme sayısından GradScaler'ın atladığı "
                  "update'ler çıkarılarak hesaplanır. Tamamlanmış koşularda son history satırı "
                  "ile nihai loss/doğruluk ve bütün sayaçlar eşleşir; geçme eşiği loss <0.1, doğruluk >%99'dur.", "",
                  "GPU süreleri bu koşuların uçtan uca sabit-batch eğitim döngüsünü "
                  "(ara FP32 değerlendirmeleri ve sonuç yazımı dahil) ölçer; model yükleme "
                  "süresini içermez. GPU'ların eğitim precision'ları farklı olabilir. "
                  "Bunlar yalnız donanım farkını izole eden bir hız benchmark'ı değildir; "
                  "tepe değer `max_memory_allocated` olup toplam GPU kullanımı veya ayrılmış "
                  "allocator rezervi değildir."]
    if len(complete_gpu) == 2:
        digest = complete_gpu[0]["overfit"]["batch"]["batch_sha256"]
        lines += ["", "A100 ve T4'te giriş/hedef token ID'leri, kaynak metin checksum'u, "
                  f"seed ve öğrenme oranı eşleşti. Ortak sabit-batch SHA-256: `{digest}`."]
    elif complete_gpu:
        lines += ["", "Tamamlanmış yalnız bir GPU eğitim kaydı var; iki GPU arasında "
                  "batch eşitliği ve karşılaştırma henüz doğrulanmış değildir."]
    else:
        lines += ["", "Tamamlanmış A100/T4 ezberleme kaydı henüz yok. Notebook'un "
                  "GPU hücreleri çalıştırılıp sonuçlar içeri alınmadan eğitim tamamlandı denemez."]
    lines += ["", "## Kaynak dosyalarının izlenebilirliği", ""]
    manifests = [r for r in runs if r["manifest"]]
    if manifests:
        for run in manifests:
            digest = run["manifest"]["source_bundle_sha256"]
            lines.append(f"- {linked(run, 'source_manifest.json')}: dört kaynak dosyasının SHA-256 "
                         f"değerleri yerel dosyalar ve Colab notebook'uyla eşleşti. Paket SHA-256: `{digest}`.")
        if len(complete_gpu) == 2 and all(r["manifest"] for r in complete_gpu):
            lines += ["", "İki GPU koşusu aynı kaynak paketini kullandı."]
    else:
        lines += ["Kaynak paketi manifest'i henüz bulunmuyor; GPU kaynak eşitliği "
                  "bu aşamada doğrulanmış değildir."]
    comparison = audit_tokenizer_comparison(results_root)
    if comparison:
        lines += ["", "## Geçen haftaki aynı paragraflarda tokenizer karşılaştırması", "",
                  "[Token kimlikleri ve kaynak checksum'u](tokenizer_comparison.json), "
                  "Hafta 9'da kaydedilmiş aynı Türkçe ve İngilizce paragraflara dayanır. "
                  "Kelime tanımı yine `len(text.split())` olduğundan paydalar değişmedi.", "",
                  "| Dil | Kelime | GPT-2 token | GPT-2 token/kelime | cl100k_base token | cl100k_base token/kelime |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in comparison["paragraphs"]:
            lines.append(f"| {row['language']} | {row['words']} | {row['gpt2_tokens']} | "
                         f"{row['gpt2_tokens_per_word']:.4f} | {row['cl100k_tokens']} | "
                         f"{row['cl100k_tokens_per_word']:.4f} |")
        tr = next(row for row in comparison["paragraphs"] if row["language"] == "tr")
        en = next(row for row in comparison["paragraphs"] if row["language"] == "en")
        lines += ["", f"Bu Türkçe paragraf GPT-2'de {tr['gpt2_tokens']}/{tr['words']} = "
                  f"**{tr['gpt2_tokens_per_word']:.4f}**, cl100k_base'de "
                  f"{tr['cl100k_tokens']}/{tr['words']} = **{tr['cl100k_tokens_per_word']:.4f}** "
                  "token/kelime tuttu. "
                  f"İngilizce paragrafın GPT-2 değeri {en['gpt2_tokens']}/{en['words']} = "
                  f"**{en['gpt2_tokens_per_word']:.4f}**; cl100k_base değeri "
                  f"**{en['cl100k_tokens_per_word']:.4f}** oldu."]
        short_tr = next((row for run in pretrained_runs for row in run["pretrained"].get("tokenizer_comparison", [])
                         if row.get("language") == "tr"), None)
        if short_tr and short_tr["gpt2_tokens"] == short_tr["cl100k_tokens"]:
            lines += ["", f"Örnek üretimindeki kısa Türkçe başlangıç ise iki tokenizer'da da "
                      f"**{short_tr['gpt2_tokens']} token** tuttu. Her Türkçe girdinin GPT-2'de "
                      "daha fazla token gerektirdiği sonucu çıkmaz."]
        lines += ["", "Bu, aynı anlamı taşıyan tek bir paragraf çiftinin ölçümüdür; "
                  "dillerin veya tokenizer'ların genel sıralaması değildir. Eklemeli yapı, "
                  "sözlüğün büyüklüğü ve eğitildiği metinler parçalamayı birlikte etkiler."]
    preferred = next((r for name in ("a100", "t4", "local_pretrained") for r in runs
                      if r["name"] == name and r["pretrained"]
                      and (name == "local_pretrained" or (r["overfit"]
                           and r["overfit"].get("status") == "completed"))), None)
    if preferred:
        lines += ["", "## Ödevin açıklamaları ve örnek çıktılar", "",
                  f"Aşağıdaki ayrıntılar [{preferred['name']} koşusunun raporundan]"
                  f"({preferred['name']}/submission_report.md) alınmıştır. "
                  "Görselde ilk konumda keskin bir değişim, 2 numaralı kanalda konuma "
                  "göre yavaş bir salınım görüyorum. Bunlar öğrenilmiş vektörlerdir; "
                  "sabit sinüs/kosinüs pozisyon kodlaması değildir.", ""]
        body = (preferred["directory"] / "submission_report.md").read_text(encoding="utf-8")
        if body.startswith("# "):
            body = body.split("\n", 1)[1].lstrip()
        lines.append(relative_links(body, preferred["name"]))
    else:
        lines += ["", "Hazır ağırlık raporu bulunmadığından mimari/logits/örnek üretimi "
                  "ayrıntıları henüz eklenmedi."]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--check-only", action="store_true", help="Audit without writing the aggregate report")
    args = parser.parse_args()
    try:
        runs = audit_all(args.results_root)
        if not args.check_only:
            args.results_root.mkdir(parents=True, exist_ok=True)
            destination = args.results_root / "submission_report.md"
            destination.write_text(render_report(runs, args.results_root), encoding="utf-8")
            print(f"Report: {destination}")
        for run in runs:
            status = run["overfit"].get("status") if run["overfit"] else "pending"
            print(f"{run['name']}: pretrained={'verified' if run['pretrained'] else 'pending'}; overfit={status}; "
                  f"source={'verified' if run['manifest'] else 'not available'}")
    except (AuditError, OSError) as error:
        print(f"Result audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
