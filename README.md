# neuron-to-gpt

A learning journal that starts with a single neuron, moves through
character-level language models, and builds toward the core ideas behind GPT.
Each week has its own directory with runnable Python files and concept notes.

The first two weeks build derivatives, gradient descent, and backpropagation
without machine-learning libraries. Week 3 moves to PyTorch to show how the
previously implemented `backward()` mechanism scales from scalar values to
tensors.

Weeks 1–4 retain their original filenames for backward compatibility. New
Week 5–7 files and commands use English naming consistently.

## Setup

```bash
git clone https://github.com/muend/neuron-to-gpt
cd neuron-to-gpt
python3 --version
```

Weeks 1–2 use only the standard library. For Weeks 3–7:

```bash
pip install -r week03/requirements.txt
```

## Weeks

| Week | Topic | Status |
|------|-------|--------|
| [01](week01/) | Neurons, layers, loss, gradient descent, and the language-model idea | ✅ |
| [02](week02/) | Backpropagation: `Value`, computation graphs, `backward()`, and MLP training | ✅ |
| [03](week03/) | Bigram character LM: counts, probabilities, sampling, NLL, softmax, and autograd | ✅ |
| [04](week04/) | Embeddings, MLP character LM, initialization, tanh saturation, and BatchNorm | ✅ |
| [05](week05/) | Expanded MLP + BatchNorm graph, broadcasting, and manual backpropagation | ✅ |
| [06](week06/) | Custom layer classes, eight-character context, and a WaveNet-style hierarchy | ✅ |
| [07](week07/) | Tiny Shakespeare, character batches, causal averages and single-head self-attention | ✅ |

## Run Week 1

```bash
cd week01
python3 adim1_tek_noron.py
python3 adim2_katman.py
python3 adim3_loss.py
python3 adim4_loss_egrisi.py
python3 adim5_gradient_descent.py
python3 bonus_bigram_lm.py
```

See [week01/README.md](week01/README.md) for details.

## Run Week 2

```bash
cd week02
python3 adim1_value.py
python3 adim2_elle_gradient.py
python3 adim3_backward.py
python3 adim4_dogrulama.py
python3 adim5_mlp.py
```

See [week02/README.md](week02/README.md) for details.

## Run Week 3

```bash
cd week03
python veri_hazirla.py
python adim1_2_3_bigram_count.py
python adim4_bigram_nn.py
python adim5_turkce.py
python bonus_trigram.py --dataset en
```

See [week03/README.md](week03/README.md) for details.

## Run Week 4

```bash
python week04/adim1_mlp.py
python week04/adim2_init_batchnorm.py
python week04/adim3_turkce.py
python week04/bonus_batchnorm_fold.py
```

See [week04/README.md](week04/README.md) for details.

## Run Week 5

```bash
python week05/step1_autograd_reference.py
python week05/step2_manual_backprop.py
```

See [week05/README.md](week05/README.md) for details.

## Run Week 6

```bash
python week06/test_layers.py
python week06/run_experiments.py
```

See [week06/README.md](week06/README.md) for the model shapes and recorded results.

## Run Week 7

```bash
python week07/data.py
python week07/step2_attention_math.py
python week07/test_attention.py
python week07/run_experiments.py
python week07/step3_inspect_attention.py
```

See [week07/README.md](week07/README.md) for attention weights, scaling examples
and measured bigram/attention validation losses.

## Run Week 8

```bash
python week08/prepare_data.py --english-only
python week08/test_gpt.py
python week08/inspect_model.py
```

See [week08/README.md](week08/README.md) for the six-stage measured comparison,
user-selected Nazim Hikmet data preparation and the Colab GPU training notebook.
Both larger models completed 5,000 updates on an A100 in Colab. Final validation
NLL is 1.4723 for Shakespeare and 1.6239 for the selected Turkish collection;
these cross-language losses are not a direct quality ranking. Reviewed metrics,
loss plots and generated samples are in [week08/results/gpu](week08/results/gpu).

## Sources

- 3Blue1Brown — [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- 3Blue1Brown — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- Andrej Karpathy — [The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd)
- Andrej Karpathy — [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- Andrej Karpathy — [makemore](https://github.com/karpathy/makemore)
- Andrej Karpathy — [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- Andrej Karpathy — [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- Andrej Karpathy — [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- Andrej Karpathy — [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- Andrej Karpathy — [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Bengio et al. — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- PyTorch — [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
