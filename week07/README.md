# Week 7: character data and single-head causal self-attention

This implements Tasks 1-4 of the first 1 hour 22 minutes of the GPT lecture.
Optional Task 5(a), a heatmap of trained attention, is included too.
This is **not** a full GPT: there are no multiple heads, Transformer blocks,
feed-forward sublayers, residual connections or LayerNorm yet.
The implementation was written before opening the reference repository;
afterwards, the data shifts, score scaling, mask and context cropping were
checked against the reference. The final GPT file was not copied.

## Run from the repository root

```powershell
.\.venv\Scripts\python.exe -m pip install -r week07\requirements.txt
.\.venv\Scripts\python.exe week07\data.py
.\.venv\Scripts\python.exe week07\step2_attention_math.py
.\.venv\Scripts\python.exe week07\test_attention.py
.\.venv\Scripts\python.exe week07\run_experiments.py
.\.venv\Scripts\python.exe week07\step3_inspect_attention.py
```

The first data command downloads the corpus once; later runs use the local
cache. `run_experiments.py` also downloads it if necessary. CPU is the default;
`--device cuda` is available when CUDA-enabled PyTorch and hardware are installed.
One CPU thread avoids thread-management overhead for these small models.
Use `python` instead of the explicit virtual-environment executable if your
active environment already has the dependencies.

On the development machine, the existing `.venv` launcher points to a missing
Windows Store Python executable (`No Python at ...`). The environment was not
replaced. The recorded run and tests used the installed Anaconda Python 3.12.10
with the existing virtual environment's packages. The verified fallback, from
the repository root, is:

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv\Lib\site-packages').Path
python week07\test_attention.py
python week07\run_experiments.py
python week07\step3_inspect_attention.py
```

The environment variable applies only to the current terminal process.

For a quick wiring check, not a meaningful loss comparison:

```powershell
.\.venv\Scripts\python.exe week07\run_experiments.py --steps 10 --eval-batches 2 --sample-tokens 40
```

Every training run overwrites its own `outputs` results. The default full run
recreates the recorded experiment. `step3_inspect_attention.py` just reads the
local checkpoint, prints its weights, and does not retrain anything.

## Task 1: data and the bigram baseline

`data.py` creates sorted character-to-index and index-to-character mappings.
`encode` and `decode` preserve spaces, punctuation, case and line breaks.
The supplied **Tiny Shakespeare** is a small Shakespeare corpus, not the
complete collection of all plays.

- Characters: 1,115,394; vocabulary: 65.
- Contiguous split: first 1,003,854 characters for training (90% after rounding),
  last 111,540 for validation. A window never crosses the split boundary.
- SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Default batch: `X` and `Y` are `(32, 32)` integer tensors.
  If `X` contains characters at positions `i...i+31`, `Y` contains `i+1...i+32`.
  Each input position predicts the next character, including the last position.

`BigramLanguageModel` is an `nn.Module` with a `65 x 65` embedding table.
Looking up the current character gives the next-character logits directly;
earlier characters in the batch do not influence that prediction.
Flattening logits to `(B*T, 65)` and targets to `(B*T,)` lets cross entropy
average next-character negative log likelihood over all batch positions.
Loss is in **nats per character**, and lower is better.

## Task 2: three ways to average the visible past

`step2_attention_math.py` compares a loop, normalized lower-triangular matrix
multiplication, and softmax of causally masked zero scores with `torch.allclose`.
All three checks print `True`.

My explanation: row `t` says how much to take from each input vector. Matrix
multiplication multiplies each vector by that row's coefficient and adds the
results. Nonnegative coefficients summing to one make this a weighted average.
For the fourth token the row is `[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]`.
Future tokens contribute nothing. Masking equal zero scores with `-inf` before
softmax gives exactly the same uniform weights over visible positions.
`keepdim=True` preserves the column dimension for row-wise normalization.

## Task 3: learned query, key, value and positions

`models.py` contains the single `AttentionHead` and `SingleHeadLanguageModel`.
Token and learned position embeddings are added before the head.
Query describes what a position looks for, key describes what a position offers
for matching, and value carries the information that will be mixed.

| Operation | Default shape | Meaning |
|---|---|---|
| Input indices | `(32, 32)` | Batch and context positions |
| Token + position embeddings | `(32, 32, 64)` | One vector per position |
| Queries, keys, values | `(32, 32, 64)` each | Learned projections |
| `q @ k.transpose(-2, -1)` | `(32, 32, 32)` | Query-key scores |
| Scaled, masked softmax (`wei`) | `(32, 32, 32)` | One distribution per query |
| `wei @ values` | `(32, 32, 64)` | Weighted value vectors |
| Output linear projection | `(32, 32, 65)` | Next-character logits |

My explanation of positions: dot-product attention by itself compares vector
content, not character indices. The causal mask tells it which positions are
visible, but does not explicitly identify each visible character's position or
distance. Learned position embeddings let identical characters at different
positions have different vectors. In this model positions are relative to the
current context window, from zero to `T-1`.

### Why divide by the square root of head size?

For independent unit-variance query/key coordinates, the sum of `head_size`
products has variance roughly `head_size`. Dividing by its square root brings
the score variance back toward one. Without scaling, large scores can make
softmax concentrate almost all weight on one key and make learning harder.
This is a scale-control argument, not a guarantee that learned attention must
always be diffuse.

The deterministic numeric experiment uses 64 channels (`sqrt(64) = 8`), 2,048
queries and eight keys: score variance is **64.7460 unscaled, 1.0117 scaled**.
For the first query, the same eight keys give:

| Key | Raw softmax | Scaled softmax |
|---|---:|---:|
| 0 | 0.055442 | 0.222601 |
| 1 | 0.000001 | 0.054196 |
| 2 | 0.000000 | 0.005970 |
| 3 | 0.937624 | 0.316994 |
| 4 | 0.000024 | 0.084559 |
| 5 | 0.000000 | 0.023294 |
| 6 | 0.006450 | 0.170114 |
| 7 | 0.000459 | 0.122272 |

Entropy rises from 0.2571 to 1.7419 nats in this example.
The script also prints the underlying scores; zeros above are rounded values.

### Reading one trained attention row

The default run prints the entire `wei` matrix for `To be, or not to be`.
It has shape `(1, 19, 19)` before removing the batch dimension for display.
Row 7 belongs to the `o` at position 7 and helps predict position 8, `r`:

| Visible key position | Character | Weight |
|---|---|---:|
| 0 | `T` | 0.000053 |
| 1 | `o` | 0.010031 |
| 2 | space | 0.006021 |
| 3 | `b` | 0.022187 |
| 4 | `e` | 0.018420 |
| 5 | `,` | 0.080777 |
| 6 | space | 0.208152 |
| 7 | `o` | 0.654360 |

My reading: this query puts about 65.44% on its own value, 20.82% on the
preceding space and 8.08% on the comma. All keys at positions 8-18 have exactly
zero weight. The visible weights sum to one. Attention includes the current
character because it is already known when predicting the next character.

## Task 4: trained comparison and text generation

Recorded on CPU with PyTorch 2.14.0+cpu, seed 1337, AdamW (learning rate 0.003,
default weight decay 0.01), 4,000 updates per model, batch 32, context 32,
embedding 64 and head size 64. Both models receive the same sequence of
training windows. Evaluation resets separate generators so both models see
the same 200 train batches and the same 200 validation batches; evaluation
does not change the training generator. These are **sampled** losses, not an
exhaustive pass over every possible validation window. No validation gradient
updates or hyperparameter search were used.

| Model | Parameters | Train NLL | Validation NLL |
|---|---:|---:|---:|
| Bigram | 4,225 | 2.4636 | 2.4942 |
| Single-head causal attention + positions | 22,721 | 2.2931 | 2.3457 |

Validation NLL drops by **0.1484 nats/character**, about **5.95%**, using the
unrounded saved values. My explanation: the bigram table can use only the
current character. Attention can mix information from the preceding context
and distinguish positions, allowing it to learn richer next-character patterns.
The attention model also has more parameters; this is not a parameter-matched
ablation isolating attention alone.

The sampled texts are stored in `outputs/bigram_sample.txt` and
`outputs/single_head_sample.txt`. These are small models and their text still
contains invented words and incoherent sentences; a lower loss does not mean
fluent Shakespeare. In the attention sample, short fragments include
`For`, `And` and `The`, but most longer passages remain nonsensical.

My explanation of cropping: training only taught positions `0...31`, and the
mask buffer is also only `32 x 32`. Generation can grow beyond 32 characters,
but every new prediction must pass only the last `block_size` characters to
the model. The full generated sequence is still kept; only its input window
is cropped. The code uses `indices[:, -self.block_size:]` and a test exercises
a prompt already longer than the context window.

## Optional Task 5(a): attention heatmap

After running training, open `outputs/attention_heatmap.png` in VS Code.
Rows are query characters, columns are key characters; each label includes
its index so repeated letters remain distinguishable. The upper triangle is
black because future positions are masked. Many rows emphasize the diagonal,
while row 7 also looks at the preceding space and comma. This is an observed
weight pattern, not evidence that the model understands the sentence's meaning.
`outputs/loss_curves.png` shows 50-step averages of minibatch NLL.

The downloaded corpus and checkpoint stay local and are ignored by Git.
The PNGs, compact `outputs/results.json` and generated text are committed as evidence.
The mask-removal alternative is not implemented.

## Verification

`test_attention.py` checks tokenizer round trips, shifted batches confined to
their split, all three averaging methods, normalized causal weights, future
token independence, finite nonzero parameter gradients, context cropping and
mode restoration during generation, and oversized-context rejection.
All eight tests pass. The math script separately asserts averaging equivalence;
the training script asserts row sums and zero future weights for the trained head.

## Sources

- [Karpathy's GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY), through 1:21:59.
- [Lecture Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing).
- [Tiny Shakespeare source](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
- [Reference repository](https://github.com/karpathy/ng-video-lecture), consulted after implementation.
