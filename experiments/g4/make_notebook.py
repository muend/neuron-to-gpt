"""Build the sequential G4 Colab notebook from plain Python cells."""

import json
from pathlib import Path
import textwrap


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(text).strip().splitlines(True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [cell("markdown", """
    # G4 — 90-minute continuation of our L4 GPT-2

    Run the cells in order on the connected RTX PRO 6000 Blackwell GPU.
    Preparation and final tests are additional to the training cap. The run
    has a **90-minute total budget**, with no clock-time delivery cutoff.
    Final checkpoint saving and evaluation can extend beyond that budget.

    The original L4 artifacts are read only. New results go to a distinct
    `MyDrive/YZ50/g4_pilot/<timestamp>` folder. No GPU outcome is pre-filled.
    """), cell("code", """
    import os, subprocess, sys
    from pathlib import Path
    ROOT = Path('/content/neuron-to-gpt-g4')
    if not ROOT.exists():
        subprocess.check_call(['git', 'clone', '--depth', '1',
                               'https://github.com/muend/neuron-to-gpt.git', str(ROOT)])
    os.chdir(ROOT)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           '-r', 'week12/requirements.txt'])
    import torch
    assert torch.cuda.is_available(), 'Select GPU runtime first'
    print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__,
          '| CUDA:', torch.version.cuda, '| BF16:', torch.cuda.is_bf16_supported())
    print('Source commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    assert torch.cuda.is_bf16_supported()
    def run(*args):
        command = [sys.executable, '-u', 'experiments/g4/pilot.py', *map(str, args)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    subprocess.check_call([sys.executable, '-m', 'unittest', 'experiments.g4.test_pilot'])
    """), cell("code", """
    from google.colab import drive
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import shutil, zipfile
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/YZ50/week12/nvidia_l4')
    assert (BASE / 'last.pt').exists(), f'L4 checkpoint missing: {BASE}'
    candidates = []
    for candidate in BASE.glob('*.zip'):
        with zipfile.ZipFile(candidate) as archive:
            if any(Path(name).name == 'val_00000.npy' for name in archive.namelist()):
                candidates.append(candidate)
    assert len(candidates) == 1, f'Expected one backed-up token archive: {candidates}'
    TOKENS = candidates[0]
    NOW = datetime.now(ZoneInfo('Europe/Istanbul'))
    RUN = NOW.strftime('%Y%m%d_%H%M%S')
    DATA = ROOT / 'experiments/g4/data/fresh200m'
    OUT = ROOT / 'experiments/g4/outputs' / RUN
    BACKUP = Path('/content/drive/MyDrive/YZ50/g4_pilot') / RUN
    LOCAL_CHECKPOINT = ROOT / 'experiments/g4/data/l4_last.pt'
    LOCAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    print('Copying L4 checkpoint to local disk...', flush=True)
    if not LOCAL_CHECKPOINT.exists():
        shutil.copyfile(BASE / 'last.pt', LOCAL_CHECKPOINT)
    print('New backup:', BACKUP, '| training budget: 90 minutes, no delivery cutoff')
    """), cell("code", """
    # Up to 200M fresh tokens, with a soft four-minute preparation budget.
    # The manifest records the actual number prepared; all original docs are excluded.
    run('prepare', '--tokens-zip', TOKENS, '--data', DATA,
        '--train-tokens', 200_000_000, '--prepare-seconds', 240)
    """), cell("code", """
    # 90-minute cap including compile, validation and periodic saves.
    run('train', '--checkpoint', LOCAL_CHECKPOINT, '--data', DATA,
        '--output', OUT, '--backup', BACKUP, '--minutes', 90,
        '--batch-size', 8, '--context', 1024, '--accumulation', 4)
    """), cell("code", """
    # Evaluate validation-selected best.pt and OpenAI GPT-2 on identical examples.
    # First 1000: course continuity; next 1000: untouched pilot comparison.
    run('finish', '--data', DATA, '--output', OUT, '--backup', BACKUP)
    from IPython.display import display, Image
    display(Image(filename=str(OUT / 'summary.png')))
    print((OUT / 'summary.json').read_text())
    print('All results and checkpoints backed up to:', BACKUP)
    """)]


if __name__ == "__main__":
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        accelerator="GPU", colab=dict(name="g4_pilot90.ipynb", provenance=[])),
        nbformat=4, nbformat_minor=5)
    # Validate Python syntax in every generated code cell before publishing.
    for index, item in enumerate(cells):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"cell_{index}", "exec")
    path = Path(__file__).with_name("colab_pilot.ipynb")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
