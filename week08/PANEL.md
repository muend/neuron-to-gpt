# Week 8 model lab

An optional English-language Gradio panel adds a writing playground, real training evidence and an architecture walkthrough. It does not retrain the models or replace the assignment report.

## Local Windows: open the evidence panel now

From `C:\Projeler\neuron-to-gpt`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\week08\start_panel.ps1
```

The launcher creates an ignored `.panel_venv` runtime and installs optional Gradio packages into the ignored `.panel_dependencies` folder without modifying the existing Python environment. This avoids mixing Gradio's NumPy with optional compiled packages in a base Anaconda environment. It reuses the existing project's PyTorch packages where available, and opens `http://127.0.0.1:7860`. Keep the terminal running; stop the server with Ctrl+C. If needed, select the Python used to create the runtime with `-Python "C:/Users/Excalibur/anaconda4/python.exe"` or change the port with `-Port 7861`.

Without checkpoints, the panel shows the recorded English/Turkish samples and disables **Generate continuation**. It explicitly labels those samples as recorded, never as live output. Explore all three tabs without consuming Colab GPU time.

In any normal Python environment, install training and UI requirements, then start directly:

```powershell
python -m pip install -r week08/requirements.txt -r week08/ui_requirements.txt
python week08/panel.py --open
```

## Enable real generation without retraining

Download the two `best.pt` files from your existing Drive run. With the notebook's default run name they are under:

```text
MyDrive/neuron-to-gpt/week08_a100_01/
  en_a100_layernorm/best.pt
  tr_a100_layernorm/best.pt
```

Preserve their language folders locally under the ignored `week08/outputs/checkpoints/` directory. Do not rename `last.pt` to `best.pt`: the English best step is 4,500, while its last step is 5,000.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\week08\start_panel.ps1 -CheckpointRoot "C:/Projeler/neuron-to-gpt/week08/outputs/checkpoints"
```

You may connect just one language first. Its Generate button becomes available; the other remains in evidence-only mode. The loader checks architecture, corpus hash, best step, vocabulary, parameter count and finite weights before generation, and uses `torch.load(..., weights_only=True)` for owner-trusted files only. No raw corpus is needed for inference. Checkpoints never enter Git.

CPU generation is supported but can be slower than GPU generation. Start with 250 characters. The panel provides temperature, top-k, seed, streaming text and cancellation. Empty prompts start with a newline; unknown characters are rejected with a readable message. Only the latest context window enters the model, while the complete prompt remains visible.

## Existing Colab runtime: launch against Drive weights

No training cell needs to run again. Use the optional final panel cell added to `colab_training.ipynb`, or run these three cells independently:

1. Mount your Drive and get the updated repository. If `/content/neuron-to-gpt` already exists with your checkout, use `git pull --ff-only` there instead of cloning over it.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import pathlib, subprocess, sys
repo = pathlib.Path('/content/neuron-to-gpt')
if repo.exists():
    subprocess.run(['git', 'pull', '--ff-only'], cwd=repo, check=True)
else:
    subprocess.run(['git', 'clone', 'https://github.com/muend/neuron-to-gpt.git', str(repo)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(repo / 'week08/ui_requirements.txt')], check=True)
```

2. Point at your existing run folder, enter a private password of at least 8 characters and launch. `viewer` is the login name. Change the run folder only if you used a different run name.

```python
import getpass
sys.path.insert(0, str(repo / 'week08'))
from panel import launch_panel
panel_demo = launch_panel(
    checkpoint_root='/content/drive/MyDrive/neuron-to-gpt/week08_a100_01',
    share=True,
    auth=('viewer', getpass.getpass('Private panel password: ')),
)
```

3. Open the generated Gradio link, sign in, select a language and click Generate. The link is public but password-protected; do not share the password or leave the GPU running after the demonstration. Close with `panel_demo.close()`, then disconnect the runtime. Existing model weights are reused; no training is started.

## Safety and interpretation

- Local-only by default, on `127.0.0.1`. Public sharing is opt-in and requires authentication. For CLI sharing set `WEEK08_PANEL_PASSWORD`; the login name is `viewer`.
- No browser checkpoint/corpus upload or filesystem path input. Only two allowlisted models are exposed. Private data, checkpoint folders, Git metadata and installed dependencies are blocked from Gradio file serving. Only copies of public plots enter its cache.
- Analytics is disabled by default. The recorded evidence still reflects export consistency checks, not an independent GPU evaluation. Loading weights for generation does not recompute validation loss.
- English/Turkish NLL values do not directly rank language quality. Samples may contain invented words and incoherent passages; they are not author quotations or chatbot answers.
- Only owner-trusted checkpoints are supported. `weights_only=True` is not a guarantee that arbitrary third-party files are safe to load. Never supply unknown weights.

Run backend/UI regression tests with `python week08/test_panel.py` in the environment containing Gradio and PyTorch. Tiny random checkpoints in tests verify mechanics only, not the trained models' quality.

Verification on the local CPU machine passed 12 panel tests and all 31 existing Week 7/8 regression tests. Browser checks covered language switching, recorded-sample labels, the disabled no-checkpoint Generate button, loss-curve display and the six-row table. Direct HTTP checks returned 403 for private provenance, Git configuration and a non-public source file. Live inference mechanics were tested with tiny random test checkpoints; the actual A100 weights were not supplied locally. The Colab sharing path is documented but was not launched in this local review.

This panel uses Gradio 6's [launch-time theme/CSS API](https://www.gradio.app/guides/gradio-6-migration-guide) and restricted [file access](https://www.gradio.app/guides/file-access). Optional dependencies are kept separate from training requirements.
