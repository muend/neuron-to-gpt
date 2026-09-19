from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ortak import forward_in_small_steps, prepare_batch, run_autograd


def cmp(
    name: str, manual: torch.Tensor, target: torch.Tensor
) -> Tuple[bool, bool, float]:
    """Elle bulunan gradient'i PyTorch autograd sonucuyla karşılaştır."""
    if target.grad is None:
        raise RuntimeError(f"{name}.grad bulunamadı")
    exact = torch.equal(manual, target.grad)
    approximate = torch.allclose(manual, target.grad, rtol=1e-5, atol=1e-8)
    max_diff = float((manual - target.grad).abs().max().item())
    print(
        f"{name:16s} | exact: {str(exact):5s} | "
        f"approximate: {str(approximate):5s} | maxdiff: {max_diff:.3e}"
    )
    return exact, approximate, max_diff


def manual_backward(
    xb: torch.Tensor,
    yb: torch.Tensor,
    parameters: Dict[str, torch.Tensor],
    graph: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Forward graph'ındaki her işlemi zincir kuralıyla sondan başa türet."""
    C = parameters["C"]
    W1, W2 = parameters["W1"], parameters["W2"]
    bngain = parameters["bngain"]

    emb, embcat = graph["emb"], graph["embcat"]
    hprebn, bnmeani = graph["hprebn"], graph["bnmeani"]
    bndiff, bndiff2 = graph["bndiff"], graph["bndiff2"]
    bnvar, bnvar_inv = graph["bnvar"], graph["bnvar_inv"]
    bnraw, hpreact = graph["bnraw"], graph["hpreact"]
    h, logits = graph["h"], graph["logits"]
    logit_maxes = graph["logit_maxes"]
    norm_logits, counts = graph["norm_logits"], graph["counts"]
    counts_sum = graph["counts_sum"]
    counts_sum_inv = graph["counts_sum_inv"]
    probs, logprobs = graph["probs"], graph["logprobs"]
    n = xb.shape[0]

    # loss = hedef log-olasılıklarının negatif ortalaması.
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), yb] = -1 / n

    # logprobs = log(probs)
    dprobs = (1 / probs) * dlogprobs

    # probs = counts * counts_sum_inv
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    dcounts = counts_sum_inv * dprobs

    # counts_sum_inv = counts_sum**-1
    dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

    # counts_sum = counts.sum(1, keepdim=True)
    # İleri yönde sütunlar toplandı; geri yönde gradient her sütuna yayılır.
    dcounts = dcounts + torch.ones_like(counts) * dcounts_sum

    # counts = exp(norm_logits)
    dnorm_logits = counts * dcounts

    # norm_logits = logits - logit_maxes
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)

    # logit_maxes = logits.max(...); her satırdaki gradient yalnızca maksimuma gider.
    max_locations = F.one_hot(
        logits.max(1).indices, num_classes=logits.shape[1]
    )
    dlogits = dlogits + max_locations * dlogit_maxes

    # logits = h @ W2 + b2
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    # b2 batch boyunca broadcast edildiği için gradient batch boyutunda toplanır.
    db2 = dlogits.sum(0)

    # h = tanh(hpreact); tanh'ın yerel türevi 1 - tanh(x)^2.
    dhpreact = (1 - h**2) * dh

    # hpreact = bngain * bnraw + bnbias
    # gain ve bias batch boyunca broadcast edildi: dim=0 üzerinde sum gerekir.
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)

    # bnraw = bndiff * bnvar_inv
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

    # bnvar_inv = (bnvar + eps)**-0.5
    dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv

    # bnvar = bndiff2.sum(0)/(n-1)
    # Tek satırlık varyans gradient'i bütün batch satırlarına broadcast edilir.
    dbndiff2 = torch.ones_like(bndiff2) * dbnvar / (n - 1)

    # bndiff2 = bndiff**2; bndiff iki farklı kola gittiği için += mantığı vardır.
    dbndiff = dbndiff + 2 * bndiff * dbndiff2

    # bndiff = hprebn - bnmeani
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0, keepdim=True)

    # bnmeani = hprebn.sum(0)/n
    # Mean batch'i topladı; geri dönüşte gradient yeniden her satıra yayılır.
    dhprebn = dhprebn + torch.ones_like(hprebn) * dbnmeani / n

    # hprebn = embcat @ W1 + b1
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    # embcat = emb.view(...): view yalnızca şekli değiştirir.
    demb = dembcat.view(emb.shape)

    # emb = C[xb]: aynı karakter birden fazla kez kullanıldıysa katkılar toplanır.
    dC = torch.zeros_like(C)
    for row in range(xb.shape[0]):
        for column in range(xb.shape[1]):
            char_index = xb[row, column]
            dC[char_index] += demb[row, column]

    return {
        "logprobs": dlogprobs,
        "probs": dprobs,
        "counts_sum_inv": dcounts_sum_inv,
        "counts_sum": dcounts_sum,
        "counts": dcounts,
        "norm_logits": dnorm_logits,
        "logit_maxes": dlogit_maxes,
        "logits": dlogits,
        "h": dh,
        "W2": dW2,
        "b2": db2,
        "hpreact": dhpreact,
        "bngain": dbngain,
        "bnbias": dbnbias,
        "bnraw": dbnraw,
        "bnvar_inv": dbnvar_inv,
        "bnvar": dbnvar,
        "bndiff2": dbndiff2,
        "bndiff": dbndiff,
        "bnmeani": dbnmeani,
        "hprebn": dhprebn,
        "embcat": dembcat,
        "W1": dW1,
        "b1": db1,
        "emb": demb,
        "C": dC,
    }


def main() -> None:
    xb, yb, parameters = prepare_batch()
    graph = forward_in_small_steps(xb, yb, parameters)
    run_autograd(graph, parameters)
    manual = manual_backward(xb, yb, parameters, graph)

    targets = {**graph, **parameters}
    exact_count = 0
    approximate_count = 0

    print("GÖREV 2 — Elle backpropagation / PyTorch karşılaştırması")
    print(f"Loss: {graph['loss'].item():.6f}\n")
    for name, gradient in manual.items():
        exact, approximate, _ = cmp(name, gradient, targets[name])
        exact_count += int(exact)
        approximate_count += int(approximate)
        assert approximate, f"{name} gradient'i PyTorch ile eşleşmedi"

    total = len(manual)
    print(
        f"\nSONUÇ: {total}/{total} gradient doğrulandı "
        f"({exact_count} exact, {approximate_count} approximate)."
    )
    print("Elle yazılan backward zinciri autograd ile eşleşiyor.")


if __name__ == "__main__":
    main()
