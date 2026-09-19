from __future__ import annotations

import torch.nn.functional as F

from ortak import FORWARD_NAMES, forward_in_small_steps, prepare_batch, run_autograd


def main() -> None:
    xb, yb, parameters = prepare_batch()
    graph = forward_in_small_steps(xb, yb, parameters)

    # Açık yazılan loss'un PyTorch'un birleşik fonksiyonuyla aynı olduğunu doğrula.
    reference_loss = F.cross_entropy(graph["logits"], yb)
    assert graph["loss"].allclose(reference_loss, atol=1e-6)

    run_autograd(graph, parameters)

    print("GÖREV 1 — Küçük adımlara bölünmüş forward ve autograd referansı")
    print(f"Minibatch: {tuple(xb.shape)}")
    print(f"Loss:      {graph['loss'].item():.6f}")
    print("\nAra değişken gradient'leri:")
    print(f"{'değişken':16s} {'tensor şekli':18s} {'grad şekli':18s} {'grad normu'}")
    print("-" * 76)
    for name in reversed(FORWARD_NAMES):
        tensor = graph[name]
        assert tensor.grad is not None, f"{name}.grad oluşmadı"
        print(
            f"{name:16s} {str(tuple(tensor.shape)):18s} "
            f"{str(tuple(tensor.grad.shape)):18s} {tensor.grad.norm().item():.6f}"
        )

    print("\nParametre gradient'leri:")
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"{name}.grad oluşmadı"
        print(
            f"{name:8s} parametre={str(tuple(parameter.shape)):18s} "
            f"grad={str(tuple(parameter.grad.shape)):18s}"
        )
    print("\nTüm ara değişken ve parametre gradient'leri alındı.")


if __name__ == "__main__":
    main()
