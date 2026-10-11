"""Print dimensions and LayerNorm/BatchNorm axes without training."""

import torch

from config import ModelConfig, PRESETS
from data import describe_data
from models import make_model


def main():
    torch.set_num_threads(1)
    config = ModelConfig(65, n_embd=64, n_head=4, n_layer=2, block_size=32, dropout=0)
    model = make_model(config).eval()
    handles = []
    def report(name):
        def hook(module, inputs, output):
            shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
            print(f"{name:45s}: {shape}")
        return hook
    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(report(name)))
    with torch.no_grad():
        logits, _ = model(torch.zeros((2, 8), dtype=torch.long))
    for handle in handles:
        handle.remove()
    print("Final logits:", tuple(logits.shape))
    print("Week 6 channels-last BatchNorm(B,T,C): reduce axes (0,1), keep per-channel running statistics.")
    print("LayerNorm(B,T,C): reduce axis -1, one mean/variance per token, no running statistics.")
    metadata = describe_data()
    vocabularies = {"en": metadata.get("en", {}).get("vocabulary", 65)}
    if "vocabulary" in metadata.get("tr", {}):
        vocabularies["tr"] = metadata["tr"]["vocabulary"]
    else:
        print("Nazim Hikmet vocabulary: pending user-selected source files.")
    for name in ("t4", "a100"):
        settings = {k: v for k, v in PRESETS[name].items() if k not in ("batch_size", "steps", "learning_rate")}
        for language, vocabulary in vocabularies.items():
            large = make_model(ModelConfig(vocabulary, **settings))
            print(f"{name}, language={language}, vocabulary={vocabulary}: {sum(p.numel() for p in large.parameters()):,} parameters")


if __name__ == "__main__":
    main()
