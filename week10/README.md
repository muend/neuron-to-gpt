# Week 10 — GPT-2 (124M)

The module is written in PyTorch and imports the published GPT-2 weights into
our own attention, blocks and sampling loop. Tasks 1–4 and bonus 5b cover weight
inspection, learned positions, Hugging Face logits parity, English/Turkish
generation, random initialization and memorization of one fixed batch.

- [Submission report and recorded GPU evidence](results/submission_report.md)
- [Standalone Colab notebook](colab_gpt2.ipynb)
- [Live Colab notebook](https://colab.research.google.com/drive/1zZ6YV3rrVZsUdmp2k_3ltrI3FpXqyAs5)
- [Local pretrained evidence](results/local_pretrained/submission_report.md)

Both full-size Colab runs completed 300 optimizer updates with no skipped
updates and 100% fixed-batch token accuracy. Initial FP32 loss was 11.056351;
final FP32 loss was 0.000106 on A100 and 0.000102 on T4. The diagnostic loops
took 11.69 and 25.45 seconds respectively, including periodic evaluation and
metric writes, excluding model setup. These runs establish memorization of
one batch, not generalization or full-corpus training. Full per-device evidence:
[A100](results/a100/submission_report.md), [T4](results/t4/submission_report.md).

## Run on Colab: T4 and A100

Upload `colab_gpt2.ipynb` to Colab and execute the numbered cells in order. The
notebook embeds the exact source snapshot with SHA-256 checks, so it does not
depend on unpublished files being available on GitHub. No private corpus,
Google Drive mount, Hugging Face token or model API key is required.

1. Select **Runtime → Change runtime type → T4 GPU** and connect.
2. Run the setup and model tests, then the pretrained experiment. Its full
   logits comparison runs in FP32 with eager attention and TF32 disabled.
3. Run 300 updates on the same fixed batch. T4 uses FP16 and GradScaler;
   all recorded loss evaluations use FP32.
4. Inspect and download `week10_t4_results.zip` before switching the runtime.
5. Select **A100 GPU**, reconnect, and rerun the cells. A100 uses BF16 for
   training. Download `week10_a100_results.zip`.

Both use the full **124,439,808-parameter** architecture, batch size 4, context
32, seed 1337 and AdamW learning rate 0.001. The fixed text was written for this
diagnostic. Training from scratch here means random weights on one batch;
this is not a reproduction of GPT-2's full WebText pretraining. Precision differs
between GPU runs, so timing differences are not a controlled hardware benchmark.
Runtime changes may erase `/content`; download each result ZIP first.

The live notebook combines the experiment cells for execution through the
browser. The local notebook retains separate numbered cells for convenient
reruns. Downloaded results include the exact source hashes used in the run.

Import the exports locally without copying checkpoints or caches:

```powershell
.\week10\.venv\Scripts\python.exe week10/import_colab_results.py --archive "$env:USERPROFILE/Downloads/week10_a100_results.zip"
.\week10\.venv\Scripts\python.exe week10/import_colab_results.py --archive "$env:USERPROFILE/Downloads/week10_t4_results.zip"
python week10/summarize_results.py
```

The importer accepts only the nine expected evidence files, verifies local
source hashes and recorded architecture/hardware, and rejects conflicting
existing results. It does not independently rerun the exported GPU losses.

## Local setup and checks

Python 3.10+ and PyTorch 2.3+ are required. From the repository root:

```powershell
python -m venv week10/.venv
.\week10\.venv\Scripts\python.exe -m pip install -r week10/requirements.txt
.\week10\.venv\Scripts\python.exe -m unittest discover -s week10 -p "test_*.py"
.\week10\.venv\Scripts\python.exe week10/run_experiments.py --task pretrained --device cpu --output-dir week10/results/local_pretrained
```

On Linux/macOS, use `week10/.venv/bin/python`. The public safetensors checkpoint
downloads on first use (about 550 MB), pinned to revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The cache stays in ignored `.cache/`.
`--local-files-only` reuses a populated cache without model downloads. Tiktoken
vocabulary files must also already be cached for a completely offline run.

The 18 unit tests use small random Hugging Face models and no pretrained
download. They check full logits/loss equality, all projection transposes,
causality, gradients, tied parameter storage, SDPA parity and sampling. The
small one-batch test is labeled as a smoke check and proves no full-size run.

For a short CPU wiring check:

```powershell
.\week10\.venv\Scripts\python.exe week10/run_experiments.py --task overfit --tiny --steps 5 --device cpu --output-dir week10/outputs/tiny_smoke
```

For full 124M training, use the Colab commands above. The runner saves metrics
at step 1, every 25 steps and completion. It records skipped FP16 updates and
requires final FP32 loss <0.1 nats/token with token accuracy >99%. A failed
diagnostic preserves its measurements and exits with an error.

## Implementation

[`model.py`](model.py) has fused Q/K/V projections, 12 heads per block,
pre-LayerNorm, residual paths, tanh-approximate GELU, learned absolute position
embeddings and tied token/output weights. Targets passed to `forward()` are
already shifted by one token; the method does not shift them again. The four
Hugging Face `Conv1D` projection weights in each block are transposed before
copying into `nn.Linear`, including square projections. Imported keys/shapes
are validated before copying. Both models run in evaluation mode for parity.

[`run_experiments.py`](run_experiments.py) supports `--task pretrained`,
`--task overfit`, `--task all` and `--task report`. Use separate output folders
for different experiments: a task deliberately refreshes its own artifacts.
The pretrained task lists all named weights, identifies the shared output
alias and counts unique parameters once. Samples come from our own top-k loop.
Translated sentences are scored both in nats/token and bits/byte; neither a
few sentence pairs nor raw token loss establishes a language-wide ranking.

[`make_colab_notebook.py`](make_colab_notebook.py) rebuilds the source snapshot:

```powershell
python week10/make_colab_notebook.py
```

Rebuild after editing any bundled source. Keep source hashes with returned
GPU results so that a later code change cannot silently relabel old evidence.

`compare_tokenizers.py` reconstructs the exact Week 9 paragraphs from their
recorded cl100k token IDs and measures GPT-2 on the same text. Reproduce the
comparison with `python week10/compare_tokenizers.py` in the installed environment.

## Sources

- [Karpathy's GPT-2 lecture](https://www.youtube.com/watch?v=l8pRSuU81PU), first hour
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Published GPT-2 weights and model card](https://huggingface.co/openai-community/gpt2)
- [Hugging Face GPT-2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
- [Attention Is All You Need (optional)](https://arxiv.org/abs/1706.03762)
