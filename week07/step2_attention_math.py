"""Verify three causal averages and demonstrate attention-score scaling."""

import torch
from torch.nn import functional as F


def causal_averages(x):
    _, time, _ = x.shape
    loop = torch.zeros_like(x)
    for t in range(time):
        loop[:, t] = x[:, :t + 1].mean(dim=1)
    lower = torch.tril(torch.ones(time, time, device=x.device, dtype=x.dtype))
    matrix = lower / lower.sum(dim=-1, keepdim=True)
    multiplied = matrix @ x
    scores = torch.zeros_like(lower).masked_fill(lower == 0, float("-inf"))
    softmax_weights = F.softmax(scores, dim=-1)
    softmax_average = softmax_weights @ x
    return loop, multiplied, softmax_average, matrix, softmax_weights


def scaling_example():
    generator = torch.Generator().manual_seed(1337)
    head_size = 64
    queries = torch.randn(2048, head_size, generator=generator)
    keys = torch.randn(8, head_size, generator=generator)
    scores = queries @ keys.T
    unscaled = F.softmax(scores[0], dim=-1)
    scaled = F.softmax(scores[0] / head_size ** 0.5, dim=-1)
    entropy = lambda p: float(-(p * p.clamp_min(1e-30).log()).sum())
    print(f"Head size: {head_size}; sqrt(head_size): {head_size ** 0.5:.1f}")
    print(f"Score variance: raw={scores.var().item():.4f}, scaled={(scores / head_size ** 0.5).var().item():.4f}")
    print("Same query, eight visible keys:")
    print("key        raw score     raw softmax   scaled softmax")
    for i in range(8):
        print(f"{i:3d}       {scores[0, i]:10.4f}      {unscaled[i]:.6f}       {scaled[i]:.6f}")
    print(f"Entropy (nats): raw={entropy(unscaled):.4f}, scaled={entropy(scaled):.4f}")


def main():
    torch.set_num_threads(1)
    x = torch.randn(4, 8, 2, generator=torch.Generator().manual_seed(42))
    loop, multiplied, softmax_average, matrix, softmax_weights = causal_averages(x)
    print("TASK 2 - Three ways to average the visible past")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"loop vs matrix:  {torch.allclose(loop, multiplied)}")
    print(f"loop vs softmax: {torch.allclose(loop, softmax_average)}")
    print(f"matrix weights vs softmax weights: {torch.allclose(matrix, softmax_weights)}")
    assert torch.allclose(loop, multiplied) and torch.allclose(loop, softmax_average)
    print(f"Row 3 weights: {matrix[3].tolist()}")
    print("Each matrix row sums to one; matmul adds value vectors with those weights.")
    print("\nTASK 3 - Why divide scores by sqrt(head_size)?")
    scaling_example()


if __name__ == "__main__":
    main()
