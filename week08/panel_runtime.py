"""Read public evidence and sample only from matching, owner-trusted checkpoints."""

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import threading
import time

import torch

from config import ModelConfig
from data import CharacterTokenizer, HERE
from models import make_model

MODELS = {
    "en": ("Shakespeare", "en_a100_layernorm", "KING RICHARD II:\n"),
    "tr": ("Turkish / Nazim Hikmet", "tr_a100_layernorm", "Bir gün\n"),
}


@dataclass(frozen=True)
class SamplingSettings:
    characters: int = 250
    temperature: float = .8
    top_k: int = 40
    seed: int = 1337

    def __post_init__(self):
        for name, lower, upper in (("characters", 1, 1000), ("top_k", 0, 10000), ("seed", 0, 2**31 - 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not math.isfinite(self.temperature) or not .1 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0.1 and 2.0")


class PanelService:
    def __init__(self, checkpoint_root=None, *, results_root=None, device="auto"):
        self.results_root = Path(results_root or HERE / "results/gpu").resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve() if checkpoint_root else None
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self.device not in ("cpu", "cuda") or self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Select cpu, or cuda on an available GPU")
        self.records = {}
        for key, (_, run, _) in MODELS.items():
            directory = self.results_root / run
            self.records[key] = {
                "result": json.loads((directory / "result.json").read_text(encoding="utf-8")),
                "history": json.loads((directory / "history.json").read_text(encoding="utf-8")),
                "sample": (directory / "sample.txt").read_text(encoding="utf-8"),
                "plot": directory / "loss_curve.png",
            }
        self._cache = {}
        self._cache_lock = threading.Lock()

    def record(self, key):
        if key not in MODELS:
            raise ValueError("Unknown model selection")
        return self.records[key]

    def checkpoint_path(self, key):
        self.record(key)
        if self.checkpoint_root is None:
            return None
        path = (self.checkpoint_root / MODELS[key][1] / "best.pt").resolve()
        if not path.is_relative_to(self.checkpoint_root):
            raise ValueError("Checkpoint must remain inside the configured root")
        return path

    def has_checkpoint(self, key):
        path = self.checkpoint_path(key)
        return path is not None and path.is_file()

    def saved_sample(self, key):
        record = self.record(key)
        step = record["result"]["sample_checkpoint_step"]
        return record["sample"], f"Recorded sample · best checkpoint at step {step:,}. Not live generation."

    def load_model(self, key):
        path = self.checkpoint_path(key)
        if path is None or not path.is_file():
            raise ValueError("Live generation needs the owner's best.pt checkpoint. Recorded samples are available without it.")
        details = path.stat()
        if details.st_size > 256_000_000:
            raise ValueError("Checkpoint exceeds the 256 MB limit for these small models")
        signature = (details.st_mtime_ns, details.st_size)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] == signature:
                return cached[1:]
            # Do not accept browser uploads or use unrestricted pickle loading.
            saved = torch.load(path, map_location="cpu", weights_only=True)
            result = self.record(key)["result"]
            if saved["model_config"] != result["model_config"]:
                raise ValueError("Checkpoint architecture does not match the recorded model")
            if saved["sha256"] != result["corpus"]["sha256"]:
                raise ValueError("Checkpoint corpus checksum does not match the recorded training data")
            if saved["step"] != result["sample_checkpoint_step"]:
                raise ValueError("Checkpoint step does not match the recorded best validation step")
            chars = saved["chars"]
            if not isinstance(chars, list) or any(not isinstance(c, str) or len(c) != 1 for c in chars):
                raise ValueError("Checkpoint must contain a character vocabulary")
            if chars != sorted(set(chars)) or len(chars) != result["corpus"]["vocabulary"]:
                raise ValueError("Checkpoint vocabulary is invalid")
            tokenizer = CharacterTokenizer(chars=chars)
            config = ModelConfig(**saved["model_config"])
            # Model initialization must not perturb any caller's global RNG state.
            with torch.random.fork_rng(devices=[]):
                model = make_model(config)
            if sum(p.numel() for p in model.parameters()) != result["parameters"]:
                raise ValueError("Checkpoint model parameter count differs from the report")
            model.load_state_dict(saved["model"], strict=True)
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError("Checkpoint contains nonfinite model parameters")
            model.to(self.device).eval()
            self._cache[key] = (signature, model, tokenizer)
            return model, tokenizer

    def stream(self, key, prompt, settings):
        if not isinstance(prompt, str) or len(prompt) > 4096:
            raise ValueError("Prompt must be text with at most 4,096 characters")
        model, tokenizer = self.load_model(key)
        prompt = prompt or ("\n" if "\n" in tokenizer.stoi else tokenizer.chars[0])
        unknown = sorted(set(prompt) - set(tokenizer.chars))
        if unknown:
            raise ValueError(f"Characters outside this model's vocabulary: {unknown[:12]!r}")
        # Preserve the full visible prompt, but crop each model input to its context.
        indices = torch.tensor([tokenizer.encode(prompt[-model.block_size:])], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        text, started = prompt, time.perf_counter()
        for step in range(1, settings.characters + 1):
            with torch.inference_mode():
                logits, _ = model(indices[:, -model.block_size:])
                scores = logits[:, -1, :] / settings.temperature
                if settings.top_k:
                    limit = min(settings.top_k, scores.shape[-1])
                    cutoff = torch.topk(scores, limit).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                if not torch.isfinite(probabilities).all().item():
                    raise ValueError("Sampling produced nonfinite probabilities")
                following = torch.multinomial(probabilities, 1, generator=generator)
                indices = torch.cat((indices[:, -model.block_size + 1:] if model.block_size > 1 else indices[:, :0], following), dim=1)
                text += tokenizer.chars[following.item()]
            if step % 16 == 0 or step == settings.characters:
                elapsed = time.perf_counter() - started
                state = "Complete" if step == settings.characters else "Generating"
                yield text, (f"{state} · live {self.device.upper()} inference · {step}/{settings.characters} new characters "
                             f"· {elapsed:.2f}s · seed {settings.seed}. Model output, not an author quotation.")


def architecture_rows(path=None):
    with Path(path or HERE / "results/cpu_ablations.csv").open(encoding="utf-8", newline="") as source:
        return [[row["stage"].replace("_", " ").title(), int(row["parameters"]),
                 round(float(row["val_nll"]), 4), round(float(row["training_seconds"]), 2)]
                for row in csv.DictReader(source)]
