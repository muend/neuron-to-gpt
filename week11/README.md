# Week 11 — Faster GPT-2 training

Continue our [Week 10 GPT-2](../week10/model.py), using random weights and the
same 12-layer, 768-channel architecture. Weight tying and scaled residual
initialization already exist there. This week adds a complete-stream loader,
incremental speed measurements, GPT-3-style optimizer settings, clipping,
warmup/cosine LR and gradient accumulation.

**Status:** the full 124M NVIDIA L4 run completed all six timing stages and
300 successful training updates with no skipped/nonfinite updates. Its returned
ZIP was checked against local source hashes. See the
[measured GPU report](results/nvidia_l4/submission_report.md). Baseline median
was 53.750 ms; the final cumulative variant took 36.227 ms (about 1.48× total
speedup). Compilation provided the largest incremental gain, 1.606×. This
training used the short demonstration corpus; real-data validation and
HellaSwag are Week 12. The optional DDP/tokenizer comparison has not been run.

## Colab: run the assignment

Open [colab_training.ipynb](colab_training.ipynb) in Google Colab. Select a T4
or A100 GPU and execute the numbered cells. The notebook embeds hashed local
sources, including the exact Week 10 model, so it works without a Git push,
Drive mount, model API key or pretrained weight download. Initial dependency
installation and the public GPT-2 tokenizer download require internet access.

1. Verify sources, inspect the connected GPU, and run correctness tests.
2. Run the six cumulative benchmark stages. Each has five warmup updates and
   twenty measured updates. The table records median step time, token/s,
   incremental speedup, unsupported steps and actual failures.
3. Run 300 training update attempts on the whole demo text, with accumulation.
   FP16 overflow can skip an update; the log and final successful-update count
   make this visible. The short schedule advances per attempted update.
4. Inspect the report/LR graph and download the result ZIP before changing GPU.

The model retains 1,024 learned positions. The T4-friendly input shape is
**B=2, T=128**; accumulation=4 gives **1,024 tokens per update**. This is a
small exercise configuration. The lecture's 524,288-token effective batch would
need 2,048 microbatches at this shape and is deliberately not the short-run
default. If memory runs out, reduce B/T and rerun every benchmark stage with
the same shape in a separate output folder.

The stages are FP32 eager → TF32 → BF16/FP16 → `torch.compile` → forced CUDA
FlashAttention → vocabulary padding to 50,304. T4 skips TF32 and uses FP16
with GradScaler; A100 supports TF32 and BF16. FlashAttention is measured only
when the forced PyTorch flash backend actually succeeds. Unsupported hardware
or kernel errors are recorded; an automatic SDPA fallback is never reported
as FlashAttention. Training uses automatic SDPA backend selection separately.
Compilation is measured on CUDA/Linux (Colab); MPS uses FP32 and skips these
CUDA optimization stages. The minimum two-optimization GPU requirement is
checked in the generated report, excluding the baseline.

## Local checks

Python 3.10+ and PyTorch 2.4+:

```powershell
python -m venv week11/.venv
.\week11\.venv\Scripts\python.exe -m pip install -r week11/requirements.txt
.\week11\.venv\Scripts\python.exe -m unittest discover -s week11 -p "test_*.py"
.\week11\.venv\Scripts\python.exe week11/run_experiments.py --tiny --device cpu --batch-size 2 --sequence-length 16 --accumulation-steps 2 --steps 12 --lr-warmup 2 --warmup 2 --benchmark-steps 5 --output-dir week11/results/local_smoke
```

On Linux/macOS use `week11/.venv/bin/python`. An existing compatible Week 10
environment also works; Week 11 does not require Transformers.

The seven tests verify cyclic tail/target alignment, large-batch versus
accumulated gradients and AdamW updates, unique decay groups, LR boundaries,
full-model parameter counts on the meta device, initialization statistics and
identical shared weights across the vocabulary-padding comparison. No GPU or
tokenizer download is required. Tiny runs use byte IDs and a boundary marker,
**not GPT-2 tokenization**; their timings are not GPU evidence.

For a local CUDA or MPS machine, use `--device cuda` or `--device mps` without
`--tiny`, with a suitable `--output-dir`. `--corpus path/to/text.txt` accepts
your UTF-8 text; default [data/demo.txt](data/demo.txt) is an original small
demonstration corpus. Every token, including the tail, is consumed in a cyclic
stream with an EOT document boundary. No validation claim is made for this
training-only demo.

## What the code measures

[`training.py`](training.py) contains the loader, optimizer, schedule and
single-update training function. [`run_experiments.py`](run_experiments.py)
implements `--task benchmark`, `train`, `schedule`, `report` or `all`.

Timing synchronizes CUDA/MPS before and after each update. It includes loading
and transferring a batch, forward, backward, gradient clipping, scaler work and
AdamW. Disk writes and console logs are excluded. Compilation and warmup are
excluded from median step time and recorded as `warmup_seconds`. Raw per-step
durations and GPU peak allocated memory are saved. Optimizer choice, seed,
batch shape and shared initial weights are held fixed across stages; the
padding stage adds output classes and therefore changes softmax normalization.
One short timing run has noise; repeat controlled runs before generalizing.

Training writes each attempted update to `training.json`: loss, pre-clipping
global gradient norm, LR, elapsed time, token/s, cumulative tokens, corpus
passes and skipped updates. Benchmark and training JSON include device,
PyTorch/CUDA versions, source hashes and the corpus checksum. Output folders
reject mixed source/environment/corpus provenance. Rerunning a task with the
same provenance replaces that task's measurements; keep separate folders for
experiments you want to retain.

[`results/local_smoke/submission_report.md`](results/local_smoke/submission_report.md)
contains the measured local check and explanations in Turkish.

## Training settings and explanations

Weight tying saves **50,257 × 768 = 38,597,376** parameters. The tied model has
124,439,808 parameters; padding the vocabulary to 50,304 gives 124,475,904.
Only the two-dimensional matrices receive decay, with shared weights counted
once. For the unpadded model this is 124,318,464 scalars across 50 matrices;
121,344 bias/LayerNorm scalars are excluded. The runner also records the
actual counts for whichever configuration executes.

Linear/embedding weights start with std=0.02; attention-output and MLP-output
projections use 0.02/√(2L). There are two residual additions per block. Scaling
each branch reduces growth in the summed stream's starting variance, under
an approximate independent-contribution argument.

AdamW uses β=(0.9,0.95), ε=1e-8 and weight decay=0.1. Global gradient norm is
clipped at 1.0 once per accumulated update. FP16 gradients are unscaled before
clipping. For K equally sized microbatches, each mean loss is divided by K:
their summed gradients then match the combined batch's mean gradient.
Accumulation reduces activation memory; it does not remove parameter or
optimizer-state memory.

LR warms up from `max_lr / warmup_steps` to 6e-4, then follows cosine decay
to 6e-5 on the last scheduled update. The LR plot is generated from these
exact values. Appendix B of the GPT-3 paper specifies token-based budgets
(375M-token warmup, 260B-token decay) and batch ramp-up. We use the lecture's
short step-based adaptation; those budgets and batch ramp-up are not reproduced.

Different tokenizers require a common evaluation denominator. The optional
comparison would measure bytes processed in equal elapsed time and bits/byte
on the same held-out text: total NLL / (ln(2) × UTF-8 bytes). Raw loss/token
does not directly rank different vocabularies. That experiment and DDP remain
optional work for a later run.

## Rebuild the notebook

```powershell
python week11/make_colab_notebook.py
```

Rebuild whenever a bundled source changes. Returned GPU evidence retains its
source hashes; changing a local script does not update earlier measurements.

## Sources

- [Karpathy lecture, 1:02:00–2:46:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3720s)
- [build-nanogpt, reference for comparison after writing your code](https://github.com/karpathy/build-nanogpt)
- [GPT-3, Appendix B](https://arxiv.org/html/2005.14165v4#A2)
- [PyTorch AMP accumulation and clipping](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
- [PyTorch SDPA and backend selection](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch CUDA/TF32 semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
