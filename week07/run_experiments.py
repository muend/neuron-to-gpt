"""Train the bigram and single-head models, then inspect learned attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import get_batch, load_data
from models import BigramLanguageModel, SingleHeadLanguageModel

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, eval_batches, device):
    previous_mode = model.training
    model.eval()
    try:
        results = {}
        for split in ("train", "val"):
            # Reset for every model/evaluation: compare exactly the same windows.
            generator = torch.Generator().manual_seed(2026 if split == "train" else 2027)
            losses = []
            for _ in range(eval_batches):
                x, y = get_batch(splits, split, batch_size, block_size, generator, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            results[split] = sum(losses) / len(losses)
        return results
    finally:
        model.train(previous_mode)


def train_model(model, splits, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    history = []
    for step in range(1, args.steps + 1):
        x, y = get_batch(splits, "train", args.batch_size, args.block_size, generator, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(f"  step {step:5d}/{args.steps}: minibatch NLL={loss.item():.4f}", flush=True)
    return history


def plot_outputs(histories, weights, text, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    window = min(50, min(len(history) for history in histories.values()))
    for label, history in histories.items():
        steps = list(range(window, len(history) + 1, window))
        averages = [sum(history[i:i + window]) / window
                    for i in range(0, len(history) - window + 1, window)]
        ax.plot(steps, averages, label=label)
    ax.set(xlabel="Training step", ylabel="Mean minibatch NLL (nats)", title="Week 7 training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    heatmap = ax.imshow(weights.numpy(), vmin=0, vmax=1, cmap="magma")
    labels = [f"{i}:{repr(char)}" for i, char in enumerate(text)]
    ax.set_xticks(range(len(text)), labels, rotation=90)
    ax.set_yticks(range(len(text)), labels)
    ax.set(xlabel="Key position (looked at)", ylabel="Query position (looking)",
           title="Trained single-head causal attention")
    fig.colorbar(heatmap, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output_dir / "attention_heatmap.png", dpi=150)
    plt.close(fig)


def inspect_attention(model, tokenizer, device, block_size):
    text = "To be, or not to be"[:block_size]
    indices = torch.tensor([tokenizer.encode(text)], device=device)
    weights = model.attention_weights(indices)[0].cpu()
    print(f"\nTASK 3 - Learned attention for {text!r}")
    print("wei shape:", tuple(weights.shape))
    print("wei (rows=query, columns=key):")
    with torch.no_grad():
        for row in weights:
            print(" ".join(f"{value:.3f}" for value in row))
    row_index = min(7, len(text) - 1)
    print(f"Reading row {row_index}, query character {text[row_index]!r}:")
    for i, value in enumerate(weights[row_index]):
        print(f"  key {i:2d} {text[i]!r:5s}: {float(value):.6f}")
    print("Future columns are zero; visible weights sum to one.")
    assert torch.allclose(weights.sum(-1), torch.ones(len(text)), atol=1e-6)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    return text, weights, row_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    positive = (args.steps, args.batch_size, args.block_size, args.embedding_size,
                args.head_size, args.eval_batches, args.sample_tokens, args.threads,
                args.learning_rate)
    if any(value <= 0 for value in positive):
        parser.error("Sizes, steps, learning rate and thread count must be positive")
    torch.set_num_threads(args.threads)
    tokenizer, splits, digest = load_data()
    print("TASK 1 - Tiny Shakespeare and the bigram baseline", flush=True)
    print(f"Vocabulary={len(tokenizer)}, train={len(splits['train'])}, val={len(splits['val'])}", flush=True)
    print(f"Batch X/Y=({args.batch_size}, {args.block_size}); targets are shifted by one", flush=True)
    output_dir = HERE / "outputs"
    output_dir.mkdir(exist_ok=True)
    results, histories = {}, {}
    attention_model = None
    for label, constructor in (
        ("bigram", lambda: BigramLanguageModel(len(tokenizer), args.block_size)),
        ("single_head", lambda: SingleHeadLanguageModel(
            len(tokenizer), args.block_size, args.embedding_size, args.head_size)),
    ):
        torch.manual_seed(args.seed)
        model = constructor().to(args.device)
        print(f"\nTraining {label}", flush=True)
        histories[label] = train_model(model, splits, args)
        scores = estimate_loss(model, splits, args.batch_size, args.block_size, args.eval_batches, args.device)
        scores["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        results[label] = scores
        print(f"{label}: train NLL={scores['train']:.4f}, val NLL={scores['val']:.4f}, parameters={scores['parameters']}", flush=True)
        sample_generator = torch.Generator(device=args.device).manual_seed(args.seed)
        prompt = torch.zeros((1, 1), dtype=torch.long, device=args.device)
        generated = tokenizer.decode(model.generate(prompt, args.sample_tokens, sample_generator)[0].tolist())
        (output_dir / f"{label}_sample.txt").write_text(generated, encoding="utf-8")
        print(f"Generated {label} text:\n{generated}", flush=True)
        if label == "single_head":
            attention_model = model
            torch.save({"state_dict": model.cpu().state_dict(), "chars": tokenizer.chars,
                        "config": vars(args), "sha256": digest}, output_dir / "single_head.pt")
            model.to(args.device)
    print("\nTASK 4 - Same validation windows, same training schedule", flush=True)
    for label, scores in results.items():
        print(f"{label:12s} | parameters={scores['parameters']:6d} | val NLL={scores['val']:.4f}", flush=True)
    improvement = results["bigram"]["val"] - results["single_head"]["val"]
    print(f"Validation NLL reduction: {improvement:.4f} nats/character", flush=True)
    text, weights, row_index = inspect_attention(attention_model, tokenizer, args.device, args.block_size)
    report = {"config": vars(args), "torch_version": torch.__version__, "sha256": digest,
              "vocabulary": len(tokenizer), "split_sizes": {k: len(v) for k, v in splits.items()},
              "results": results, "val_reduction": improvement,
              "attention_text": text, "row_index": row_index, "weights": weights.tolist()}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_outputs(histories, weights, text, output_dir)
    print("Saved results, generated text, checkpoint, loss curves and attention heatmap.", flush=True)


if __name__ == "__main__":
    main()
