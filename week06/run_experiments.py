"""Run the required Week 6 comparisons with identical splits and schedules."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiment import (count_parameters, evaluate, load_splits, make_model,
                        sample, shape_trace, train)

HERE = Path(__file__).resolve().parent


def plot_curves(histories: dict[str, list[float]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, history in histories.items():
        # The notebook plots log10(loss); averaging windows removes noisy
        # single-minibatch spikes without changing the measurements.
        window = max(1, len(history) // 50)
        xs = list(range(window, len(history) + 1, window))
        ys = [sum(history[i:i+window]) / window for i in range(0, len(history)-window+1, window)]
        ax.plot(xs, ys, label=label)
    ax.set(xlabel="Training step", ylabel="Mean log10(minibatch NLL)",
           title="Week 6 training loss")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def fit(label, language, context, embedding, hidden, *, steps, hierarchical=False,
        incorrect_bn=False, histories=None):
    training, dev, itos, word_counts = load_splits(language, context)
    model = make_model(len(itos), context=context, embedding=embedding,
                       hidden=hidden, hierarchical=hierarchical,
                       incorrect_bn=incorrect_bn)
    history = train(model, training, steps=steps)
    if histories is not None:
        histories[label] = history
    score = evaluate(model, dev)
    params = count_parameters(model)
    print(f"{label:28s} | params={params:7d} | dev NLL={score:.4f} | steps={steps}", flush=True)
    return model, score, params, itos, word_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000,
                        help="Steps per English and Turkish model (default: 3000)")
    parser.add_argument("--turkish-steps", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.turkish_steps < 1 or args.threads < 1:
        parser.error("Training steps must be positive")
    torch.set_num_threads(args.threads)
    tr_steps = args.turkish_steps
    histories = {}

    print("TASK 1-2 - Sequential layers and the flat-context baseline", flush=True)
    en3 = fit("EN flat context 3", "en", 3, 10, 200, steps=args.steps, histories=histories)
    en8 = fit("EN flat context 8", "en", 8, 10, 200, steps=args.steps, histories=histories)
    print(f"Context-only parameter change: {en8[2] - en3[2]:+d}")
    print(f"Context-only dev NLL change: {en8[1] - en3[1]:+.4f} (negative is better)")

    print("\nTASK 3 - Hierarchical shape trace (batch=4)")
    shape_model = make_model(27, context=8, embedding=10, hidden=200,
                             hierarchical=True)
    for index, (layer, shape) in enumerate(shape_trace(shape_model, 8)):
        print(f"{index:2d} {layer:14s} {shape}")

    print("\nTASK 4 - Three-dimensional BatchNorm axis comparison", flush=True)
    wrong = fit("EN hierarchy WRONG BN", "en", 8, 10, 200,
                steps=args.steps, hierarchical=True, incorrect_bn=True)
    right = fit("EN hierarchy CORRECT BN", "en", 8, 10, 200,
                steps=args.steps, hierarchical=True, histories=histories)
    bad_mean = next(layer.last_mean for layer in wrong[0].layers
                    if hasattr(layer, "last_mean"))
    good_mean = next(layer.last_mean for layer in right[0].layers
                     if hasattr(layer, "last_mean"))
    print(f"First BN mean shape, wrong: {tuple(bad_mean.shape)}; correct: {tuple(good_mean.shape)}")
    print(f"Dev NLL change after correction: {right[1] - wrong[1]:+.4f}")
    print("Wrong axes=(0,) keep time; correct axes=(0,1) combine batch and time.")

    print("\nTASK 5 - Larger embeddings and hidden layers", flush=True)
    large = fit("EN hierarchy enlarged", "en", 8, 24, 256,
                steps=args.steps, hierarchical=True, histories=histories)
    print("Model                        | Parameters | Dev NLL")
    for label, result in [("Context 3 flat MLP", en3),
                          ("Context 8 flat MLP", en8),
                          ("Context 8 hierarchy", large)]:
        print(f"{label:28s} | {result[2]:10d} | {result[1]:.4f}")

    print("\nTASK 6 - Turkish names", flush=True)
    tr3 = fit("TR Week 4-style MLP 3", "tr", 3, 10, 200,
              steps=tr_steps, histories=histories)
    tr8 = fit("TR flat context 8", "tr", 8, 10, 200,
              steps=tr_steps, histories=histories)
    tr_wave = fit("TR hierarchy context 8", "tr", 8, 24, 256,
                  steps=tr_steps, hierarchical=True, histories=histories)
    print(f"Turkish context-only change (3 to 8 flat): {tr8[1] - tr3[1]:+.4f}")
    print(f"Turkish full hierarchy vs Week 4-style: {tr_wave[1] - tr3[1]:+.4f}")
    print("Generated Turkish names:")
    for name in sample(tr_wave[0], tr_wave[3], 8):
        print(f"  {name}")
    plot_path = HERE / "outputs" / "loss_curves.png"
    plot_curves(histories, plot_path)
    print(f"Log-scale smoothed loss curves: {plot_path}")
    print(f"Word splits EN={en3[4]}, TR={tr3[4]}; seed=42, batch=32, lr=0.1 then 0.01")


if __name__ == "__main__":
    main()
