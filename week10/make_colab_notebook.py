"""Build a standalone Colab notebook with an auditable snapshot of our sources.

The notebook works before a Git commit/push: no remote branch or private corpus
is needed. Rebuild it whenever the four bundled source files change.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("model.py", "run_experiments.py", "test_gpt2.py", "requirements.txt")


def cell(kind, identifier, source):
    result = {"cell_type": kind, "id": identifier, "metadata": {},
              "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build():
    buffer = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in SOURCE_FILES:
            raw = (HERE / filename).read_bytes()
            hashes[filename] = hashlib.sha256(raw).hexdigest()
            entry = zipfile.ZipInfo("week10/" + filename, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    setup = f'''import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path('/content/neuron-to-gpt-week10')
ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE_SHA256 = {digest!r}
SOURCE_HASHES = {hashes!r}
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == BUNDLE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    expected = {{'week10/' + name for name in SOURCE_HASHES}}
    assert set(archive.namelist()) == expected
    for name, checksum in SOURCE_HASHES.items():
        raw = archive.read('week10/' + name)
        assert hashlib.sha256(raw).hexdigest() == checksum
        target = ROOT / 'week10' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
os.environ['HF_HOME'] = str(ROOT / 'week10/.cache/huggingface')
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'week10/.cache/tiktoken')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'week10/requirements.txt')], check=True)
print('Verified source bundle:', BUNDLE_SHA256)
print(json.dumps(SOURCE_HASHES, indent=2))
'''
    gpu = '''import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU, then reconnect.'
GPU_NAME = torch.cuda.get_device_name(0)
if 'A100' in GPU_NAME:
    RUN_NAME, PRECISION = 'a100', 'bf16'
elif 'T4' in GPU_NAME:
    RUN_NAME, PRECISION = 't4', 'fp16'
else:
    raise RuntimeError(f'Select a T4 or A100 for this comparison; connected: {GPU_NAME}')
RESULTS = ROOT / 'week10/results' / RUN_NAME
RESULTS.mkdir(parents=True, exist_ok=True)
print('GPU:', GPU_NAME, '| torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('Training:', PRECISION, '| logits parity and loss evaluation: FP32')
print('Results:', RESULTS)
subprocess.run(['nvidia-smi'], check=True)
'''
    tests = '''subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'week10', '-p', 'test_*.py'], cwd=ROOT, check=True)
'''
    pretrained = '''# Full 124M model. The public weights are pinned to one published revision.
subprocess.run([
    sys.executable, '-u', 'week10/run_experiments.py', '--task', 'pretrained',
    '--device', 'cuda', '--output-dir', str(RESULTS), '--new-tokens', '64',
    '--cache-dir', str(ROOT / 'week10/.cache/huggingface'),
    '--revision', '607a30d783dfa663caf39e06633721c8d4cfcd7e',
], cwd=ROOT, check=True)
'''
    training = '''# Fresh random weights; this is not fine-tuning the pretrained model.
# Same fixed batch and schedule on both GPUs. Precision differs, so timings
# are not a controlled hardware-only benchmark. No compile/FlashAttention/DDP.
subprocess.run([
    sys.executable, '-u', 'week10/run_experiments.py', '--task', 'overfit',
    '--device', 'cuda', '--output-dir', str(RESULTS), '--precision', PRECISION,
    '--steps', '300', '--batch-size', '4', '--block-size', '32',
    '--learning-rate', '0.001', '--eval-interval', '25', '--seed', '1337',
], cwd=ROOT, check=True)
'''
    display = '''from IPython.display import display, Markdown, Image
pretrained = json.loads((RESULTS / 'pretrained_metrics.json').read_text())
overfit = json.loads((RESULTS / 'overfit_metrics.json').read_text())
assert pretrained['logit_parity']['all_passed']
assert overfit['evidence'] == 'full_gpt2_124m'
assert overfit['status'] == 'completed' and overfit['overfit_passed']
display(Markdown((RESULTS / 'submission_report.md').read_text()))
display(Image(filename=str(RESULTS / 'position_embeddings.png')))
display(Image(filename=str(RESULTS / 'overfit_loss.png')))
print('Initial loss:', overfit['initial_loss'], '| final:', overfit['final_loss'])
print('Completed optimizer updates:', overfit['optimizer_updates'])
'''
    export = '''from google.colab import files
export = ROOT / f'week10_{{RUN_NAME}}_results.zip'
manifest = {'gpu': GPU_NAME, 'source_bundle_sha256': BUNDLE_SHA256, 'source_hashes': SOURCE_HASHES}
(RESULTS / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.tsv', '.txt', '.md', '.png'}:
            archive.write(path, arcname=RUN_NAME + '/' + path.name)
print('Export:', export, '| bytes:', export.stat().st_size)
print('Save this ZIP before changing GPU/runtime. Model weights and caches are excluded.')
files.download(str(export))
'''
    # This is an ordinary string, not an outer f-string.
    export = export.replace("f'week10_{{RUN_NAME}}_results.zip'", "f'week10_{RUN_NAME}_results.zip'")
    cells = [
        cell("markdown", "intro", """# Week 10: GPT-2 (124M), first training steps

Run this notebook once on **T4**, download the results, then once on **A100**.
Choose **Runtime → Change runtime type → Hardware accelerator** before starting.
This runs the full 124,439,808-parameter model, including a 300-update fixed-batch
diagnostic. It does not train GPT-2 on an entire corpus. GPU execution uses your
Colab allocation; no subscription change or new purchase is needed.

This notebook embeds the project source snapshot with SHA-256 checks. It works
without pushing code to GitHub. No private Week 8 corpus or Google Drive mount is
needed. The exported ZIP contains metrics, samples, plots and source hashes.
"""),
        cell("markdown", "setup-title", "## 1. Unpack the checked source snapshot and install dependencies\n"),
        cell("code", "setup", setup),
        cell("markdown", "gpu-title", "## 2. Verify the selected GPU\n"),
        cell("code", "gpu", gpu),
        cell("markdown", "test-title", "## 3. Run the independent model checks\nThese tests use tiny models without downloading pretrained weights.\n"),
        cell("code", "tests", tests),
        cell("markdown", "pretrained-title", "## 4. Inspect published weights, compare logits, sample, and score translations\nThe roughly 550 MB public safetensors weights download on first run. FP32/eager parity is checked before samples.\n"),
        cell("code", "pretrained", pretrained),
        cell("markdown", "overfit-title", "## 5. Random initialization and 300 updates on exactly one batch\nT4 uses FP16 + GradScaler; A100 uses BF16. Every recorded evaluation uses FP32. Progress is saved every 25 steps. The command fails if the final loss does not meet the diagnostic threshold.\n"),
        cell("code", "overfit", training),
        cell("markdown", "report-title", "## 6. Inspect the measured results\n"),
        cell("code", "report", display),
        cell("markdown", "export-title", "## 7. Download results before switching GPU\nAfter saving the ZIP, change the GPU, reconnect, and rerun cells 1–7 for the other device. A runtime change can erase `/content` files.\n"),
        cell("code", "export", export),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"name": "Week 10 - GPT-2 124M.ipynb", "provenance": []},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"},
                             "accelerator": "GPU", "week10_source_bundle_sha256": digest,
                             "week10_source_hashes": hashes}, "cells": cells}
    target = HERE / "colab_gpt2.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {target}; bundle SHA-256 {digest}")
    return target


if __name__ == "__main__":
    build()
