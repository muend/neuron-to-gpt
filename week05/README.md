# Week 5 — Writing backpropagation by hand

This week expands the three-character MLP + BatchNorm model from Week 4 into
small operations and derives every gradient without using `loss.backward()`.
PyTorch autograd is used only as the reference result.

## Files

- `common.py`: data preparation, parameter initialization, and the expanded
  forward pass.
- `step1_autograd_reference.py`: calls `retain_grad()` on every intermediate
  tensor, runs `loss.backward()`, and reports gradient shapes and norms.
- `step2_manual_backprop.py`: applies the chain rule in reverse and compares 26
  manually derived gradients against autograd with `cmp`.
- `requirements.txt`: the PyTorch dependency.

The optional Exercises 2–4—combining cross-entropy and BatchNorm backward into
single expressions and training entirely with manual gradients—are outside the
required scope of this submission.

## Setup and execution

Prepare the data first if it is not already available:

```powershell
.\.venv\Scripts\python.exe week03\veri_hazirla.py
```

Run the two required tasks:

```powershell
.\.venv\Scripts\python.exe week05\step1_autograd_reference.py
.\.venv\Scripts\python.exe week05\step2_manual_backprop.py
```

The second command should finish with `26/26 gradients verified`. `exact` means
bit-for-bit equality; `approximate` means equality within floating-point
tolerance when the operation order differs slightly.

## Forward chain

The forward pass is expanded into operations whose gradients can be inspected:

```text
Xb -> emb -> embcat -> hprebn
   -> bnmeani -> bndiff -> bndiff2 -> bnvar -> bnvar_inv
   -> bnraw -> hpreact -> tanh -> logits
   -> logit_maxes -> norm_logits -> counts -> counts_sum
   -> counts_sum_inv -> probs -> logprobs -> loss
```

Backward follows this chain in reverse. When a tensor feeds more than one
forward branch, the gradient contributions from those branches are added.

## Why does broadcasting require `sum` during backward?

A smaller tensor that is broadcast during the forward pass is virtually reused
multiple times. Backward must sum the contribution from every use back into the
original tensor shape:

- `logits = h @ W2 + b2`: `b2` is reused for every batch row, so
  `db2 = dlogits.sum(0)`.
- `hpreact = bngain * bnraw + bnbias`: gain and bias are reused across the
  batch, so their gradients sum over `dim=0`.
- `bndiff = hprebn - bnmeani`: the one-row mean broadcasts across the batch,
  so `dbnmeani = (-dbndiff).sum(0, keepdim=True)`.
- `counts_sum = counts.sum(1, keepdim=True)`: the gradient of each row sum
  broadcasts back to every column in that row.
- `emb = C[Xb]`: repeated character indices contribute to the same row of `dC`,
  so those contributions accumulate.

## Core local derivatives

- `log(x)` → `1/x`
- `exp(x)` → `exp(x)`
- `x**-1` → `-x**-2`
- `x**2` → `2x`
- `tanh(x)` → `1 - tanh(x)**2`
- `A @ B` → `dA = dOut @ B.T`, `dB = A.T @ dOut`

Each local derivative is multiplied element-wise by the gradient arriving from
the remainder of the chain.

## Sources

- [Andrej Karpathy — Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Karpathy's Part 4 exercise notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
