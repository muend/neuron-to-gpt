# Week 9 — Byte Pair Encoding

This week replaces character IDs with a byte-level BPE tokenizer. The Python
implementation learns merges on the same private Turkish corpus used in Week 8.
It also reproduces the ordinary-text IDs of the assignment's `cl100k_base`
encoding using a separate Python encoder.

The measured results and Turkish answers to tasks 1–6a are in
[the submission report](results/submission_report.md). Machine-readable counts,
token IDs, corpus hashes and package versions are in [metrics.json](results/metrics.json).
The report contains measurements from an actual run, not example estimates.

## Run on Windows

From the repository root, create a separate environment:

```powershell
python -m venv week09/.venv
.\week09\.venv\Scripts\python.exe -m pip install -r week09/requirements.txt
.\week09\.venv\Scripts\python.exe -m unittest discover -s week09 -p "test_*.py"
.\week09\.venv\Scripts\python.exe week09/run_experiments.py
```

On Linux/macOS, substitute `week09/.venv/bin/python` for the Windows executable.
Python 3.10 or later is required. The first tiktoken use downloads its
hash-checked vocabulary from OpenAI; later runs use `week09/.tiktoken_cache/`.
No API key, OpenAI API call or GPU is needed.

By default the experiment reads `week08/data/turkish.txt`, uses the first 90%
of Unicode code points for tokenizer training and the final 10% for held-out
compression measurement. It does not normalize or clean the text again.
The 512-entry vocabulary includes all 256 individual bytes and 256 learned
merges. Both tokenizers use the same split and vocabulary budget. Expect a
few minutes on CPU for the deliberately simple implementation.

A fresh clone does not include the owner's private literary corpus. Prepare
the Week 8 data locally, or supply your own UTF-8 text:

```powershell
.\week09\.venv\Scripts\python.exe week09/run_experiments.py --corpus path/to/turkish.txt --vocab-size 512
```

The runner writes local tokenizer JSON models under ignored `week09/outputs/`
and refreshes `week09/results/`. For a separate experiment, preserve the recorded
submission with `--output-dir week09/outputs/my_run --results-dir week09/outputs/my_run/results`.
An input too small to support the requested number of merges fails with the
actual achievable vocabulary size; lower `--vocab-size` in that case.

## Code and task map

| Task | Implementation / evidence |
|---|---|
| 1. Code points and UTF-8 | `unicode_rows()` and the report's character table |
| 2. Turkish BPE, first 20 merges, compression | `BasicTokenizer.train()`, both split measurements in the report |
| 3. Encode/decode and invalid UTF-8 | `encode()`, `decode()`, Turkish + emoji example, behavioral tests |
| 4. Regex training and apostrophes | `RegexTokenizer`, merge comparison, ASCII and curly apostrophe examples |
| 5. Turkish/English tokens per word | Matched original paragraphs, `cl100k_base`, both custom tokenizers |
| 6a. Exact GPT-4 tokenizer IDs | `GPT4Tokenizer`, recovered merges and byte-ID permutation, reference checks |

[`tokenizer.py`](tokenizer.py) contains the from-scratch training and inference
code. Training selects the most frequent adjacent pair, resolves ties by first
corpus occurrence and replaces nonoverlapping matches from left to right.
Repeated regex chunks are counted with multiplicity to reduce work. Encoding
uses the saved merge ranks; it does not retrain or apply longest-prefix matching.
Saved models retain the pattern and ordered merges; loading validates parent IDs.

[`gpt4_tokenizer.py`](gpt4_tokenizer.py) loads pretrained rank metadata through
tiktoken, reconstructs merge parents, then runs its own Python merging and byte
decoding. It uses tiktoken's private `_mergeable_ranks` metadata because the
public encoder API does not expose merge parents; incompatible metadata fails
explicitly. Ordinary text, including literal special-token spellings, matches
`tiktoken.encode_ordinary()`. Control-token recognition with `allowed_special`
is outside this task. The historical `cl100k_base` exercise does not describe
every current ChatGPT model's tokenizer.

## Interpreting the results

Byte/token is a sequence compression ratio, not a disk-storage ratio. A token
can contain an incomplete UTF-8 sequence: join all token bytes before decoding.
The default replacement policy supports arbitrary generated token sequences;
strict decoding is useful for detecting invalid data. Every valid UTF-8 input
roundtrips exactly, including characters absent from tokenizer training.

Tokens per word uses `len(text.split())` as an explicit denominator. Morphology,
training data and vocabulary budget all affect it. A 512-entry local vocabulary
and 100,256-entry pretrained vocabulary are not a controlled quality comparison.
Neither a single translated paragraph nor this literary corpus establishes a
language-wide ranking. BPE token loss also has a different unit from Week 8
character loss; see the report's explanation of evaluation in bits per byte.

## Sources

- [Karpathy's tokenizer lecture](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Lecture Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [minbpe exercises](https://github.com/karpathy/minbpe/blob/master/exercise.md)
- [minbpe merge recovery reference](https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)
- [Official encoding definitions](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
- [Tiktokenizer](https://tiktokenizer.vercel.app)
- [GPT-2 paper, §2.2 (optional)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
