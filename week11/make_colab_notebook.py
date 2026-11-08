"""Package local Week 10/11 sources into an offline-source Colab notebook."""

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILES = ("week10/model.py", "week11/training.py", "week11/run_experiments.py",
         "week11/test_training.py", "week11/requirements.txt", "week11/data/demo.txt")


def cell(kind, name, source):
    item = dict(cell_type=kind, id=name, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        ast.parse(source)
        item.update(execution_count=None, outputs=[])
    return item


def build():
    stream, hashes = io.BytesIO(), {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in FILES:
            raw = (ROOT / filename).read_bytes()
            hashes[filename] = hashlib.sha256(raw).hexdigest()
            entry = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    payload = stream.getvalue()
    setup = f'''import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path('/content/neuron-to-gpt-week11')
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
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'week11/.cache/tiktoken')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'week11/requirements.txt')], check=True)
print('Sources verified:', json.dumps(SOURCE_HASHES, indent=2))
'''
    hardware = '''import re, torch
assert torch.cuda.is_available(), 'Runtime > Change runtime type > GPU, ardından yeniden bağlanın.'
GPU_NAME = torch.cuda.get_device_name(0)
RUN_NAME = re.sub(r'[^a-z0-9]+', '_', GPU_NAME.lower()).strip('_')
RESULTS = ROOT / 'week11/results' / RUN_NAME
RESULTS.mkdir(parents=True, exist_ok=True)
BATCH_SIZE, SEQUENCE_LENGTH = 2, 128
STEPS, LR_WARMUP, ACCUMULATION = 300, 30, 4
print('GPU:', GPU_NAME, '| torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('T4: TF32 atlanır; FP16+GradScaler kullanılır. A100: TF32/BF16 desteklenir.')
print('FlashAttention yalnız zorlanan flash backend gerçekten çalışırsa ölçülür.')
subprocess.run(['nvidia-smi'], check=True)

def run_task(task):
    subprocess.run([
        sys.executable, '-u', 'week11/run_experiments.py', '--task', task,
        '--device', 'cuda', '--output-dir', str(RESULTS),
        '--batch-size', str(BATCH_SIZE), '--sequence-length', str(SEQUENCE_LENGTH),
        '--steps', str(STEPS), '--lr-warmup', str(LR_WARMUP),
        '--accumulation-steps', str(ACCUMULATION),
    ], cwd=ROOT, check=True)
'''
    display = '''from IPython.display import display, Markdown, Image
display(Markdown((RESULTS / 'submission_report.md').read_text(encoding='utf-8')))
display(Image(filename=str(RESULTS / 'learning_rate.png')))
'''
    download = '''import shutil
from google.colab import files
archive = shutil.make_archive(str(ROOT / ('week11_' + RUN_NAME + '_results')), 'zip', RESULTS)
files.download(archive)
print('ZIP dosyasını cihaz değiştirmeden önce kaydedin:', archive)
'''
    cells = [
        cell("markdown", "intro", """# Hafta 11 — GPT-2 eğitimini hızlandırma

Bu defter **kendi hafta 10 modelimizi** kullanır; pretrained ağırlık indirmez.
Runtime → Change runtime type → T4/A100 GPU seçin ve hücreleri sırayla çalıştırın.
Kaynaklar deftere gömülüdür; GitHub push, Drive bağlama veya özel corpus gerekmez.

Hızlandırma sırası: FP32 → TF32 → BF16/FP16 → compile → zorlanan FlashAttention → 50.304 vocab.
Desteklenmeyen veya hata veren aşama gerekçesiyle kaydedilir, sonraki ayara eklenmez.
Ölçümde batch/context sabittir. Warmup/derleme ölçüm dışında, ayrıca kayıttadır.
GPU adımları birkaç dakika, ilk derleme daha uzun sürebilir. Sayısal GPU sonuçları çalıştırılana kadar yoktur.

Demo metni yalnız eğitim hattı içindir; büyük veri/genelleme değerlendirmesi gelecek haftadır.
"""),
        cell("markdown", "setup-title", "## 1. Kaynakları doğrula ve bağımlılıkları kur"),
        cell("code", "setup", setup),
        cell("markdown", "hardware-title", "## 2. GPU ve deney ayarları\nBellek yetmezse batch/context'i burada küçültüp **tüm hız aşamalarını aynı ayarla** yeniden çalıştırın."),
        cell("code", "hardware", hardware),
        cell("markdown", "tests-title", "## 3. Loader, accumulation, optimizer, schedule ve başlangıç kontrolleri"),
        cell("code", "tests", "subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'week11', '-p', 'test_*.py'], cwd=ROOT, check=True)\n"),
        cell("markdown", "benchmark-title", "## 4. Birikimli GPU hız ölçümleri\nHer aşamada 5 warmup ve 20 ölçülen güncelleme; ham süreler JSON'a yazılır. Flash backend desteklenmezse hata nedeni tabloda görünür."),
        cell("code", "benchmark", "run_task('benchmark')\n"),
        cell("markdown", "training-title", "## 5. Tam demo metninde 300 güncelleme\nAdamW β=(0.9,0.95), clipping=1, decay=0.1; 30 adım warmup ve cosine.\nEtkin batch: 2×128×4=1.024 token. Videodaki 524.288 token için accumulation=2.048 gerekir; kısa ödev koşusunda bu maliyetli hedef kullanılmaz.\nFP16 atlanan güncellemeleri ayrıca raporlanır. Her satırda loss, norm, LR, süre ve token/s vardır."),
        cell("code", "training", "run_task('train')\n"),
        cell("markdown", "report-title", "## 6. Ölçülen tablo ve LR grafiği\nT4'te en az iki iyileştirme ölçülmesini bekliyoruz: FP16, compile ve vocab padding adayları. Sonuç raporu şartın sağlanıp sağlanmadığını sayar."),
        cell("code", "display", display),
        cell("markdown", "download-title", "## 7. Kanıt dosyalarını indir\nBaşka GPU'ya geçmeden ZIP'i kaydedin. Özel bir metin kullanırsanız kaynak metin ZIP'e eklenmez; adı ve hash'i kaydedilir."),
        cell("code", "download", download),
        cell("markdown", "notes", """## Açıklamalar ve kaynaklar

Weight tying, 50.257×768=38.597.376 parametreyi paylaşır. Residual dallarda
std=0.02/√(2L), art arda toplanan katkıların başlangıç varyansını sınırlar.
K eşit büyüklükte mikro batch için loss/K, gradient toplamını büyük batch'in
ortalama gradient'ine dönüştürür. Clipping accumulation sonunda bir kez uygulanır.

GPT-3 makalesindeki token bütçeli warmup/decay ve batch ramp-up'ın tamamını
yeniden üretmiyoruz; videodaki kısa deney için update sayısına uyarlıyoruz.
Ek DDP/Türkçe tokenizer eğitimi bu defterde yapılmıyor.

- [Karpathy videosu 1:02:00–2:46:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3720s)
- [GPT-3, Appendix B](https://arxiv.org/html/2005.14165v4#A2)
- [PyTorch AMP accumulation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
- [PyTorch SDPA backend seçimi](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
"""),
    ]
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator="GPU", colab=dict(name="Week 11 — Fast GPT-2 training"),
                                  kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                  language_info=dict(name="python", version="3.11")))
    destination = HERE / "colab_training.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    build()
