"""Build a self-contained Week 12 notebook, reusing the same L4 session."""

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def cell(kind, identifier, source):
    result = dict(cell_type=kind, id=identifier, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build():
    files = ["week10/model.py", "week11/training.py", "week12/requirements.txt"]
    files += [path.relative_to(ROOT).as_posix() for path in sorted(HERE.glob("*.py"))
              if path.name != "colab_bootstrap.py"]
    buffer, hashes = io.BytesIO(), {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in files:
            raw = (ROOT / filename).read_bytes()
            hashes[filename] = hashlib.sha256(raw).hexdigest()
            entry = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    payload = buffer.getvalue()
    setup = f'''import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

# Prefer running these cells in the CURRENT Week 11 notebook/runtime.
ROOT = Path('/content/neuron-to-gpt-week12')
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_HASHES = {hashes!r}
payload = base64.b64decode({base64.b64encode(payload).decode('ascii')!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert set(archive.namelist()) == set(SOURCE_HASHES)
    for filename, expected in SOURCE_HASHES.items():
        raw = archive.read(filename)
        assert hashlib.sha256(raw).hexdigest() == expected
        destination = ROOT / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
os.environ['HF_HOME'] = str(ROOT / 'week12/.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# Reuse the tokenizer cache if Week 11 ran in this runtime.
old_cache = Path('/content/neuron-to-gpt-week11/week11/.cache/tiktoken')
os.environ['TIKTOKEN_CACHE_DIR'] = str(old_cache if old_cache.exists() else ROOT / 'week10/.cache/tiktoken')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'week12/requirements.txt')], check=True)
print('Week 12 sources verified:', json.dumps(SOURCE_HASHES, indent=2))
'''
    hardware = '''import re, torch
assert torch.cuda.is_available(), 'GPU runtime required. Keep the current L4 if it is connected.'
GPU_NAME = torch.cuda.get_device_name(0)
RUN_NAME = re.sub(r'[^a-z0-9]+', '_', GPU_NAME.lower()).strip('_')
DATA = ROOT / 'week12/data/fineweb20m'
RESULTS = ROOT / 'week12/outputs' / RUN_NAME
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_TOKENS, VAL_TOKENS = 20_000_000, 131_072
MINUTES, BATCH_SIZE, SEQUENCE_LENGTH, ACCUMULATION = 60, 4, 256, 4
COMPILE = True
HELLASWAG_EXAMPLES = 1000
BACKUP = None  # Set by the optional Drive cell below.
print('GPU:', GPU_NAME, '| torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('Precision:', 'BF16' if torch.cuda.is_bf16_supported() else 'FP16 + GradScaler')
print('Effective batch tokens:', BATCH_SIZE * SEQUENCE_LENGTH * ACCUMULATION)
subprocess.run(['nvidia-smi'], check=True)

def execute(arguments):
    command = [sys.executable, '-u', *arguments]
    # Notebook output streams do not always expose inherited child FDs.
    # Read the pipe and print in the notebook process so progress is visible.
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding='utf-8', errors='replace', bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
'''
    drive = '''# Recommended: preserve the latest checkpoint in your own Google Drive.
# This cell opens Google's normal authorization dialog.
from google.colab import drive
drive.mount('/content/drive')
BACKUP = Path('/content/drive/MyDrive/YZ50/week12') / RUN_NAME
BACKUP.mkdir(parents=True, exist_ok=True)
print('Periodic checkpoint backup:', BACKUP)
'''
    prepare = '''execute([
    'week12/prepare_data.py', '--output-dir', str(DATA),
    '--train-tokens', str(TRAIN_TOKENS), '--val-tokens', str(VAL_TOKENS),
])
manifest = json.loads((DATA / 'manifest.json').read_text())
print({split: {'tokens': row['tokens'], 'shards': len(row['shards']), 'documents': row['documents']}
       for split, row in manifest['splits'].items()})
'''
    hellaprep = '''execute(['week12/evaluate.py', '--task', 'prepare', '--limit', str(HELLASWAG_EXAMPLES)])
'''
    training = '''arguments = [
    'week12/train.py', '--data-dir', str(DATA), '--output-dir', str(RESULTS),
    '--device', 'cuda', '--minutes', str(MINUTES),
    '--batch-size', str(BATCH_SIZE), '--sequence-length', str(SEQUENCE_LENGTH),
    '--accumulation-steps', str(ACCUMULATION), '--warmup-steps', '100',
    '--eval-every', '250', '--checkpoint-every', '1000', '--val-batches', '32',
]
if COMPILE:
    arguments += ['--compile']
if BACKUP is not None:
    arguments += ['--backup-dir', str(BACKUP)]
# Rerunning this cell resumes the last saved optimizer/loader/RNG state.
checkpoint = RESULTS / 'last.pt'
if not checkpoint.exists() and BACKUP is not None and (BACKUP / 'last.pt').exists():
    shutil.copyfile(BACKUP / 'last.pt', checkpoint)
if checkpoint.exists():
    arguments += ['--resume', str(checkpoint)]
execute(arguments)
'''
    own = '''execute([
    'week12/evaluate.py', '--task', 'own', '--limit', str(HELLASWAG_EXAMPLES),
    '--output-dir', str(RESULTS), '--device', 'cuda',
])
'''
    openai = '''execute([
    'week12/evaluate.py', '--task', 'openai', '--limit', str(HELLASWAG_EXAMPLES),
    '--output-dir', str(RESULTS), '--device', 'cuda',
])
'''
    show = '''from IPython.display import display, Markdown, Image
execute(['week12/report.py', '--output-dir', str(RESULTS)])
display(Markdown((RESULTS / 'submission_report.md').read_text(encoding='utf-8')))
display(Image(filename=str(RESULTS / 'training_curves.png')))
'''
    download = '''from google.colab import files
EXPORT = ROOT / 'week12/export' / RUN_NAME
EXPORT.mkdir(parents=True, exist_ok=True)
expected = ['training.json', 'data_manifest.json', 'hellaswag_own.json',
            'hellaswag_openai.json', 'training_curves.png', 'submission_report.md']
for filename in expected:
    source = RESULTS / filename
    assert source.exists(), f'Missing result: {filename}'
    shutil.copyfile(source, EXPORT / filename)
training_result = json.loads((RESULTS / 'training.json').read_text())
assert training_result['status'] == 'complete' and not training_result['smoke_only']
own_result = json.loads((RESULTS / 'hellaswag_own.json').read_text())
openai_result = json.loads((RESULTS / 'hellaswag_openai.json').read_text())
assert own_result['status'] == openai_result['status'] == 'complete'
assert own_result['processed'] == openai_result['processed'] == HELLASWAG_EXAMPLES
assert own_result['selection_sha256'] == openai_result['selection_sha256']
(EXPORT / 'source_manifest.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
archive = shutil.make_archive(str(ROOT / ('week12_' + RUN_NAME + '_results')), 'zip', EXPORT)
if BACKUP is not None:
    shutil.copyfile(archive, BACKUP / Path(archive).name)
files.download(archive)
print('Evidence ZIP:', archive)
print('The checkpoint is separate:', RESULTS / 'last.pt')
print('Drive checkpoint backup:', BACKUP / 'last.pt' if BACKUP is not None else 'not enabled')
'''
    cells = [
        cell("markdown", "intro", """# Hafta 12 — gerçek veride GPT-2 ve HellaSwag

**Hafta 11 sonuç ZIP'ini önce indirin. Mevcut L4 oturumunu koruyabilirsiniz.**
Defteri yükledikten sonra Colab'ın **Düzenle → Tüm hücreleri seç → Hücreleri kopyala**
seçenekleriyle bu hücreleri açık hafta 11 defterinin sonuna yapıştırabilirsiniz.
Alternatif olarak bu defteri kendi GPU oturumunda çalıştırabilirsiniz; yeni defterin
aynı sanal makineyi otomatik paylaştığını varsaymayın.

Akış: kaynaklar → GPU → testler → Drive yedeği → 20M token hazırlığı →
HellaSwag verisini hazırla → yaklaşık 60 dakika eğitim → kendi modelin → OpenAI GPT-2 → sonuç ZIP'i.
Veri hazırlığı ve HellaSwag ölçümleri 60 dakikalık eğitim bütçesine dahil değildir.
İlk derleme, periyodik val/örnekleme/checkpoint süreleri eğitim duvar bütçesine dahildir;
son değerlendirme/kayıt süreleri bütçeyi biraz aşabilir.

124M model sıfırdan başlar. Hafta 11'in kısa demo metninde ezberlenen ağırlıklarını kullanmaz.
"""),
        cell("markdown", "setup-title", "## 1. Kaynakları doğrula ve bağımlılıkları kur"),
        cell("code", "setup", setup),
        cell("markdown", "hardware-title", "## 2. Bağlı GPU ve ayarlar\nL4 için başlangıç: B=4, T=256, accumulation=4. A100 aramak için mevcut oturumu silmek gerekmiyor."),
        cell("code", "hardware", hardware),
        cell("markdown", "tests-title", "## 3. Doğruluk testleri\nShard sınırları, validation ayrımı, HellaSwag maskesi ve uzunluk normalizasyonu."),
        cell("code", "tests", "execute(['-m', 'unittest', 'discover', '-s', 'week12', '-p', 'test_*.py'])\n"),
        cell("markdown", "drive-title", "## 4. Checkpoint için Google Drive yedeği (önerilir)\nHer 1.000 güncellemede ve eğitim sonunda `last.pt` ile metrikler yedeklenir. Resume checkpoint'i yaklaşık 1,5 GB'dir. Drive kullanmak istemiyorsanız bu hücreyi atlayabilirsiniz; yerel `/content` dosyaları oturum silinince kaybolur."),
        cell("code", "drive", drive),
        cell("markdown", "data-title", "## 5. FineWeb-Edu: 20M train + ayrı validation\nVeri kümesinin sabit sürümünden streaming yapılır. GPT-2 tokenizer, 1M token'lık uint16 NPY shard'ları; validation için ayrı belgeler. İlk bağlantı/Parquet okuması zaman alabilir. Tamamlandığında token, shard ve belge sayıları görünür."),
        cell("code", "prepare", prepare),
        cell("markdown", "hellaprep-title", "## 6. HellaSwag'in ilk 1.000 validation örneğini hazırla\nVeri erişimini uzun eğitimden önce kontrol ediyoruz. Eski GitHub URL'si yerine sabit sürümlü Rowan/hellaswag kullanılır."),
        cell("code", "hellaprep", hellaprep),
        cell("markdown", "train-title", "## 7. Yaklaşık bir saat eğit\nİlk `VAL` ve `SAMPLE` rastgele ağırlıklarla ölçülür. İlk compile adımı uzun sürebilir. Her 10 denemede loss/norm/LR/token/s, her 250 başarılı güncellemede val ve aynı prompt'tan örnek yazdırılır. Hücreyi tekrar çalıştırmak son checkpoint'ten devam eder; aynı kaynak/veri ve ayarları koruyun."),
        cell("code", "train", training),
        cell("markdown", "own-title", "## 8. Kendi modelini HellaSwag'de ölç\nİlk 1.000 örnek; ending-only ortalama NLL ana ölçüttür. Her 50 örnekte doğru sayısı yazdırılır. Checkpoint'teki optimizer hali bu hücrede kullanılmaz."),
        cell("code", "own", own),
        cell("markdown", "openai-title", "## 9. OpenAI GPT-2'yi aynı örneklerde ölç\nHerkese açık, sabit sürümlü GPT-2 ağırlıkları indirilir ve kendi hafta 10 modelimize yüklenir. Aynı tokenizer, aynı örnekler, aynı FP32 puanlama; model API anahtarı gerekmez."),
        cell("code", "openai", openai),
        cell("markdown", "show-title", "## 10. Train/val grafiği, örnekler ve karşılaştırma"),
        cell("code", "show", show),
        cell("markdown", "download-title", "## 11. Sonuç ZIP'ini indir\nMetrikler, grafik, örnekler ve kaynak hash'leri ZIP'e eklenir. Checkpoint, veri ve model önbelleği bu küçük ZIP'e eklenmez. Checkpoint Drive yedeğini kontrol edin; Drive kullanmadıysanız ayrıca `files.download(str(RESULTS / 'last.pt'))` ile indirin."),
        cell("code", "download", download),
        cell("markdown", "sources", """## Yorumlama

İngilizce HellaSwag için İngilizce FineWeb-Edu kullanıyoruz. Bir saatlik eğitimin
OpenAI'ın daha uzun ön eğitimini yakalamasını beklemek zorunda değiliz; düşük
sonuç da geçerli bir ölçümdür. Çalıştırılmayan bir aşama tamamlandı sayılmaz.
Ek Türkçe eğitim bu koşuda yoktur. İngilizce HellaSwag Türkçe kalitesini tek başına ölçmez.

- [Karpathy 3:10:21 sonrası](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By-1.0
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
- [HellaSwag makalesi](https://arxiv.org/abs/1905.07830)
"""),
    ]
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                  language_info=dict(name="python", version="3.11")))
    path = HERE / "colab_training.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # One paste cell is a convenient way to preserve the currently connected
    # L4: it creates the same sources/helpers but leaves expensive tasks manual.
    bootstrap = "# WEEK 12 SETUP: paste into a NEW cell below Week 11 results.\n" + setup + "\n" + hardware
    ast.parse(bootstrap)
    (HERE / "colab_bootstrap.py").write_text(bootstrap, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
