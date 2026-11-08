# Week 12 — Real-data GPT-2 training and HellaSwag

Use our Week 10 GPT-2 with the optimizer, accumulation and initialization from
Week 11. Train fresh random weights on a bounded FineWeb-Edu sample, track
held-out loss and generated samples, and evaluate both our trained checkpoint
and published OpenAI GPT-2 weights on identical HellaSwag examples.

**Status:** the NVIDIA L4 run completed 19,586 successful updates in 60.20
minutes, processing 80,224,256 tokens with no skipped updates. Final train NLL
was 4.742307; fixed-validation NLL fell from 11.013391 to 4.984965. Both
HellaSwag evaluations completed on the same first 1,000 validation examples:
our model scored **27.6%**, OpenAI GPT-2 **33.6%**, a six-percentage-point gap.
The returned ZIP, exact archived sources, token/step accounting and all
per-example prediction argmins/counts were checked locally; this was not a
second GPU evaluation. See the [GPU report](results/nvidia_l4/submission_report.md)
and [import validation record](results/nvidia_l4/import_validation.json).
Nine independent correctness tests also pass. The optional Turkish experiment
was not run.

Submission is through the GitHub code and report links, with a
[short email draft](../deliverables/teslim_maili.txt) and
[summary image](../deliverables/yz50_hafta11_12_sonuclar.png). Local ZIP exports
are recovery artifacts and are excluded from Git. The approximately 40 MB token
corpus and 1.5 GB resume checkpoint are separate files in the user's Drive
folder `MyDrive/YZ50/week12/nvidia_l4/`; they are not uploaded to GitHub. The import preserves the exact executed source snapshot
under `results/nvidia_l4/sources/`, including the notebook generator before
the later notebook stdout-forwarding fix.

## Keep the current Colab L4

Download the Week 11 result ZIP first. For the simplest continuation in the
same connected runtime, open [colab_bootstrap.py](colab_bootstrap.py), copy its
whole contents, add a **new code cell at the end of the existing Week 11
notebook**, paste and run it. This verifies embedded sources, installs the
Week 12 dependencies and defines `ROOT`, `DATA`, `RESULTS` and `execute()`.
It does not begin the one-hour training.

Alternatively upload [colab_training.ipynb](colab_training.ipynb) to Colab and
run its numbered cells. A different notebook should not be assumed to share
the original notebook's runtime. No Git push or model API key is needed.

The notebook's sequence is setup, GPU, tests, optional Google Drive backup,
data preparation, HellaSwag data preparation, training, own-model evaluation,
OpenAI evaluation, report and evidence download. Dependencies, the data and
public model weights require internet access.

## What executes

- FineWeb-Edu `sample-10BT`, pinned to
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, is streamed until **20,000,000
  train tokens** and **131,072 validation tokens** have been saved. All tokens
  use GPT-2 tokenization, not the dataset's reported token counts. Shards are
  one-million-token uint16 NPY arrays; validation fits in one separate file.
- Unique stripped documents fill validation first, then training. A document
  is never divided between the splits. Exact duplicate texts are removed,
  boundary documents are truncated with EOT and their remainder discarded.
  This is not near-duplicate removal or benchmark decontamination. Documents,
  source revisions, token counts and shard checksums are recorded.
- The memory-mapped loader joins windows across shard boundaries. Training
  wraps over its own stream. Validation reads the same first 32 complete
  batches without wrapping; at B=4,T=256 this measures **32,768 held-out target
  tokens per check**, not the entire validation file.
- L4 defaults are B=4,T=256, accumulation=4, BF16, SDPA, `torch.compile`, and a
  4,096-token effective batch. T4 uses FP16 and GradScaler. Learned positions
  still have length 1,024. Training starts from random weights, not the Week
  11 demonstration run or OpenAI pretrained weights.
- Training has a **60-minute wall budget**, including first compilation,
  validation, sampling and checkpoint I/O after initialization. Initial setup,
  data preparation and HellaSwag evaluations are separate. Final validation
  and checkpoint writing can extend past the budget. There is a 100-successful-
  update LR warmup and cosine decay over the remaining wall budget.
- Each 250 successful updates, the model records fixed-validation NLL and a
  64-token sample with the same prompt/seed/temperature/top-k. Every attempted
  update records loss, norm, LR, duration, token/s, cumulative tokens and any
  skipped FP16 update. The plot shows measured train/validation loss and LR.
- Every 1,000 updates and at completion, `last.pt` saves raw model weights,
  AdamW/scaler state, loader position, RNG state and evidence. Resume restores
  those values. Optional Drive backup preserves the latest checkpoint and
  metadata. The resume checkpoint is roughly 1.5 GB; it is separate from the
  small result ZIP. A runtime interruption can lose work since the last backup.

## HellaSwag protocol

The deprecated GitHub JSONL URL returns 404. Use `Rowan/hellaswag` at pinned
revision `218ec52e09a7e7462a5400043bb9a69a41d06b76`. Both evaluations use the
**same first 1,000 validation examples**, verified by their content hash.

Context and each space-prefixed ending are encoded separately, without text
cleanup. Predictions score only the ending target positions after shifting
by one token. The primary metric chooses the lowest mean ending-token NLL
(`acc_norm`); sum-NLL accuracy is also recorded. Padding and context positions
are excluded. If context exceeds model capacity it is left-trimmed equally
for all four candidates and this is recorded; endings are never truncated.

Both models are evaluated in FP32 with TF32 disabled and no AMP/compile. Our
47 extra vocabulary padding classes are excluded when scoring, so both use
the common 50,257-ID vocabulary. The OpenAI weights are pinned to revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and loaded into **our Week 10
implementation**, rather than calling a model API.

The JSON files include every predicted label, four scores, correct counts,
selected indices, elapsed time, source/model hashes and a Wilson interval.
Evaluation progress is saved every 50 examples and can resume if model/data/
scoring provenance matches. The report states the measured gap in percentage
points only after both matching evaluations finish. Different HellaSwag text
preprocessing protocols can produce different scores.

## Local checks and commands

Install `week12/requirements.txt` in a Python 3.10+ environment. From the root:

```powershell
.\week10\.venv\Scripts\python.exe -m unittest discover -s week12 -p "test_*.py"
```

The tests use no downloads. They check cross-shard targets, finite/repeatable
validation, split/dedup behavior, corrupted data rejection, ending masks and
length normalization against hand-computed probabilities, shared context
trimming, evaluation resume provenance, schedule bounds, and checkpoint state
restoration including the next optimizer update.

A local UTF-8 source accepts documents separated by blank lines. This tiny
check uses the Week 11 original demo text and cannot satisfy the GPU task:

```powershell
.\week10\.venv\Scripts\python.exe week12/prepare_data.py --source local --text week11/data/demo.txt --output-dir week12/data/local_smoke --train-tokens 256 --val-tokens 128 --shard-size 128
.\week10\.venv\Scripts\python.exe week12/train.py --data-dir week12/data/local_smoke --output-dir week12/results/local_smoke --tiny --device cpu --minutes 1 --max-steps 6 --batch-size 1 --sequence-length 16 --accumulation-steps 2 --warmup-steps 2 --eval-every 3 --val-batches 2 --sample-tokens 8 --checkpoint-every 3 --log-every 1 --threads 2
```

The recorded [CPU report](results/local_smoke/submission_report.md) is explicitly
marked smoke-only. It continues the six-update run above from its checkpoint
through update nine, preserving optimizer and loader state. Its tiny model
retains the GPT-2 vocabulary but uses two 32-channel blocks. The HellaSwag
runner rejects a tiny smoke checkpoint.

On a GPU, the main commands are:

```text
python week12/prepare_data.py
python week12/evaluate.py --task prepare --limit 1000
python week12/train.py --device cuda --compile --minutes 60
python week12/evaluate.py --task own --device cuda --limit 1000
python week12/evaluate.py --task openai --device cuda --limit 1000
python week12/report.py
```

Use separate output directories for separate experiments. Pass `--resume` to
continue our own `last.pt`; source/data/model/training settings must match.
An output folder with existing training is never silently overwritten.

## Rebuild

```text
python week12/make_colab_notebook.py
```

This regenerates the embedded notebook and the single pasteable bootstrap.
The generated bootstrap is excluded from its own archive/provenance to avoid
recursive embedding. Rebuild after changing a bundled source.

## Sources

- [Karpathy lecture, 3:10:21 onward](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s)
- [FineWeb-Edu dataset and ODC-By attribution](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Hugging Face streaming](https://huggingface.co/docs/datasets/stream)
- [HellaSwag dataset](https://huggingface.co/datasets/Rowan/hellaswag)
- [HellaSwag paper](https://arxiv.org/abs/1905.07830)
- [build-nanogpt reference evaluation protocol](https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py)
