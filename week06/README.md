# Week 6 — Layer classes and a WaveNet-style character model

This week refactors the character MLP into callable `Linear`, `BatchNorm1d`,
`Tanh`, `Embedding`, `Flatten`, and `Sequential` classes. The training loop only
uses `model(x)`, `model.parameters()`, `model.train()`, and `model.eval()`. Layer
computations are written with tensor operations rather than `torch.nn` layers.

The eight-character hierarchical model follows the pairwise tree demonstrated
in Karpathy's Part 5 notebook. It is **WaveNet-inspired**, not a full
implementation of the original dilated causal convolutional WaveNet.

## Run

From the repository root in PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" "week06\test_layers.py"
& ".\.venv\Scripts\python.exe" "week06\run_experiments.py"
```

If the name files are missing, run `python week03/veri_hazirla.py` first. The
default experiment uses 3,000 training steps per English model and 5,000 per
Turkish model. For a quick mechanical check use `--steps 20
--turkish-steps 20`; those short-run losses are not meaningful comparisons.
The script saves a smoothed `log10(minibatch NLL)` plot to
`week06/outputs/loss_curves.png`.

## Layer shapes

For a batch of four English examples, embedding size 10 and hidden width 200:

| Stage | Shape | Reason |
|---|---:|---|
| Input indices | `(4, 8)` | Eight preceding characters per example. |
| Embedding | `(4, 8, 10)` | Each character becomes a 10-number vector. |
| First pair merge | `(4, 4, 20)` | Eight positions become four adjacent pairs, each with 20 values. |
| First linear/BN/tanh | `(4, 4, 200)` | Each pair becomes a 200-channel representation. |
| Second pair merge | `(4, 2, 400)` | Four positions become two pairs of 200-channel vectors. |
| Second linear/BN/tanh | `(4, 2, 200)` | Each pair returns to 200 channels. |
| Third pair merge | `(4, 400)` | The final two positions combine into one vector. |
| Third linear/BN/tanh | `(4, 200)` | One 200-channel vector per example. |
| Output logits | `(4, 27)` | One score for each possible next character. |

## The three-dimensional BatchNorm mistake

Our tensor convention is `(batch, time, channels)`. For a 3D activation,
`x.mean(dim=0)` leaves shape `(time, channels)`, so it normalizes each time
position separately. The corrected expression is
`x.mean(dim=(0, 1))`, which leaves one statistic per channel. Variance must
use the same axes. The `incorrect_3d` switch preserves the mistake solely for
the controlled comparison; the normal model uses the corrected axes. Evaluation
uses running statistics, while training uses minibatch statistics.

The corrected model's dev NLL was 2.1786 versus 2.2037 for the incorrect
version at 3,000 steps. Both use the same initialization, split, minibatch
sequence, and parameter count. This is one seeded experiment, not a guarantee
that every run will improve by exactly this amount.

## Results

Data were split by name, 80/10/10 with seed 42. Dev NLL is average per
character, including the end token. Lower is better. These are the recorded
outputs of `python week06/run_experiments.py` with CPU, batch size 32,
learning rate 0.1 for the first 75% of steps and 0.01 afterward.

| English model | Context | Embedding | Hidden | Parameters | Steps | Dev NLL |
|---|---:|---:|---:|---:|---:|---:|
| Flat MLP | 3 | 10 | 200 | 12,297 | 3,000 | 2.3102 |
| Flat MLP | 8 | 10 | 200 | 22,297 | 3,000 | 2.2727 |
| Enlarged hierarchy | 8 | 24 | 256 | 283,555 | 3,000 | 2.1513 |

Changing only the flat MLP context from 3 to 8 adds 10,000 parameters
and reduces dev NLL by 0.0375. The enlarged hierarchy changes architecture
and capacity as well as context, so its larger improvement cannot be credited
to context alone.

| Turkish model | Context | Parameters | Steps | Dev NLL |
|---|---:|---:|---:|---:|
| Week 4-style flat MLP | 3 | 12,930 | 5,000 | 2.2662 |
| Flat MLP | 8 | 22,930 | 5,000 | 2.2133 |
| Enlarged hierarchy | 8 | 284,398 | 5,000 | 2.0654 |

The flat-model context change alone reduced Turkish dev NLL by 0.0529.
The full hierarchy improved on the matching 3-character baseline by 0.2008,
but that difference also includes architecture and capacity changes. The
original Week 4 Turkish script, run separately at its default 7,000 steps,
returned 2.2677 dev NLL; the enlarged hierarchy is 0.2023 lower. Those
runs have different training schedules, so the comparison is descriptive.

With sampling seed 42, example outputs included `taysavin`, `vaspa`,
`gülsidem`, `atiman`, `ertic`, `yalahan`, `surman`, `selmi`, `ilraz`, and
`neyhut`. These are samples, not claims that each is a novel real name.

## Sources

- [Karpathy, Building makemore Part 5](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- [Part 5 exercise notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
- [WaveNet paper](https://arxiv.org/abs/1609.03499)
