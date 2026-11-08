"""Shared paths, provenance and atomic JSON writes for Week 12."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "week10/.cache/tiktoken"))
os.environ.setdefault("HF_HOME", str(HERE / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from week11.training import (GPT, GPTConfig, autocast_context, configure_optimizer,
                             make_scaler, optimizer_update, set_tf32, synchronize)

FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    files = [ROOT / "week10/model.py", ROOT / "week11/training.py"]
    files += [path for path in sorted(HERE.glob("*.py")) if path.name != "colab_bootstrap.py"]
    files.append(HERE / "requirements.txt")
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def environment(device):
    return dict(device=str(device), hardware=torch.cuda.get_device_name(device)
                if device.type == "cuda" else platform.processor(),
                torch=str(torch.__version__), python=platform.python_version(),
                cuda=torch.version.cuda, cpu_threads=torch.get_num_threads())


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def choose_precision(device):
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
