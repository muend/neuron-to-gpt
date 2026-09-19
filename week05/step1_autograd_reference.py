from __future__ import annotations

import torch.nn.functional as F

from common import FORWARD_NAMES, forward_in_small_steps, prepare_batch, run_autograd


def main() -> None:
    xb, yb, parameters = prepare_batch()
    graph = forward_in_small_steps(xb, yb, parameters)

    # Verify that the expanded loss matches PyTorch's fused implementation.
    reference_loss = F.cross_entropy(graph["logits"], yb)
    assert graph["loss"].allclose(reference_loss, atol=1e-6)

    run_autograd(graph, parameters)

    print("TASK 1 - Expanded forward pass and autograd reference")
    print(f"Minibatch: {tuple(xb.shape)}")
    print(f"Loss:      {graph['loss'].item():.6f}")
    print("\nIntermediate gradients:")
    print(f"{'variable':16s} {'tensor shape':18s} {'grad shape':18s} {'grad norm'}")
    print("-" * 76)
    for name in reversed(FORWARD_NAMES):
        tensor = graph[name]
        assert tensor.grad is not None, f"{name}.grad was not created"
        print(
            f"{name:16s} {str(tuple(tensor.shape)):18s} "
            f"{str(tuple(tensor.grad.shape)):18s} {tensor.grad.norm().item():.6f}"
        )

    print("\nParameter gradients:")
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"{name}.grad was not created"
        print(
            f"{name:8s} parameter={str(tuple(parameter.shape)):18s} "
            f"grad={str(tuple(parameter.grad.shape)):18s}"
        )
    print("\nAll intermediate and parameter gradients were captured.")


if __name__ == "__main__":
    main()
