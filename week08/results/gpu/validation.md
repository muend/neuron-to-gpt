# GPU export validation

Assessment: **Share with noted caveats**. Reviewed on 2026-10-11.

## Method and spot checks

The owner returned `week08_gpu_results.zip` from Colab. This review validates the export, not an independently repeated GPU evaluation. Archive SHA-256:

```text
94560af2d799e11fbfbba3c07e638eae45e7e3882fd038c7ee19dd1046683f78
```

- ZIP integrity passed. Exactly the ten expected entries were present; only the nine public result assets were imported. The private corpus manifest was excluded from Git.
- Each run records 5,000 optimizer updates with zero skipped updates on `NVIDIA A100-SXM4-40GB`, PyTorch `2.11.0+cu130`, BF16 training and 200 FP32 evaluation batches.
- Architecture and training settings match the A100 preset. Parameter counts recomputed from the repository implementation are 4,833,345 for English and 4,860,534 for Turkish. Architecture settings match across languages except vocabulary size.
- All ten checkpoints at steps 500 through 5,000 have finite nonnegative losses. Final and best losses agree with the histories to absolute tolerance 1e-12. Best-step labels agree with the reported sample checkpoint.
- Local corpus bytes, character counts, vocabulary, 90/10 split sizes and SHA-256 match the returned training metadata. The Turkish private manifest identifies only the selected author's sources and matches the trained corpus hash.
- Both generated samples are 601 characters, including their initial prompt, and contain only characters in their corresponding corpus vocabulary. This does not establish coherence or authenticate checkpoint weights.
- PNG integrity passed. Both figures were also visually inspected: titles, step axes, NLL labels, legends and the plotted trajectories agree with the histories. Neither figure hides the late English plateau. Their different vertical ranges are appropriate for within-run inspection, not cross-language ranking.
- Positive training times are below elapsed times. Total training time is 872.03 seconds; combined recorded elapsed time is 1,077.57 seconds (about 18 minutes). The timing fields do not measure the entire Colab session.

Machine-readable checks are in [validation.json](validation.json). Repeat the import from the repository root with `python week08/import_gpu_results.py --archive <export.zip>`. Without private corpora, use `--skip-local-corpus-check` and a separate destination; that is a weaker check and is recorded as such.

Local verification also passed all 31 tests: 11 export/report regression tests, 12 Week 8 model/pipeline tests and 8 Week 7 attention tests. Notebook JSON and all Python code cells parsed successfully. These checks do not replace the missing independent GPU evaluation.

## Results

| Run | Final train NLL | Final val NLL | Best val NLL | Best step | Training seconds | Elapsed seconds |
|---|---:|---:|---:|---:|---:|---:|
| English | 1.0412 | 1.4723 | 1.4657 | 4,500 | 435.58 | 538.01 |
| Turkish | 1.3817 | 1.6239 | 1.6239 | 5,000 | 436.46 | 539.56 |

The original returned `submission_report.md`, JSON histories/results, PNGs and samples are preserved without rewriting. This review and the repository README provide the additional interpretation.

## Issues and caveats

No internal inconsistency was found in the returned metrics. Publication is qualified by these limitations:

1. No checkpoints were supplied in the ZIP. The reported losses and generated text cannot be independently reproduced from weights in this export; internal consistency and source matching are not proof of a repeated GPU evaluation.
2. English and Turkish losses are not directly comparable quality scores. Their vocabularies (65 versus 118), distributions, genres, cleaning and held-out texts differ.
3. The scaled English model improves on the CPU LayerNorm run, but changes several settings together. This cannot isolate the effect of dropout, scaling, context or GPU hardware. Use the fixed-budget six-stage CPU table for the architecture progression.
4. English validation levels off while training loss continues falling; its best sampled checkpoint is at 4,500, not 5,000. This is consistent with a plateau or mild overfitting, not statistical proof from repeated seeds. Turkish validation improves through 5,000.
5. Turkish OCR substitutions, shorter/OCR-variant duplicates and contiguous genre/period shifts remain possible. The collection was not exhaustively proofread or balanced by genre.
6. Samples exhibit dialogue and line-format patterns, but have grammar, invented-word and coherence errors. They are model-generated examples, not author quotations or verified literary-quality text.
7. Elapsed times exclude preprocessing, benchmarking, generation, plotting and session idle time. No peak-memory measurement or T4 run was returned.

Raw texts and PDFs remain private and ignored by Git. The optional Week 8 extensions were not required and are not claimed as completed.
