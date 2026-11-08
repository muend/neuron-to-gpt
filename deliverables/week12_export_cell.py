"""Paste this whole file into a new cell in the connected Week 12 Colab.

Run after BOTH HellaSwag evaluations finish. Uses the existing notebook's
ROOT, DATA, RESULTS, BACKUP and execute variables. Does not start training.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

from google.colab import files


def export_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


training = export_json(RESULTS / 'training.json')
own = export_json(RESULTS / 'hellaswag_own.json')
openai = export_json(RESULTS / 'hellaswag_openai.json')
manifest = export_json(DATA / 'manifest.json')

assert training['status'] == 'complete', 'Training has not completed successfully.'
assert not training['smoke_only'] and training['parameters'] == 124475904
assert training['successful_updates'] > 0
assert len(training['history']) == training['attempted_updates']
assert math.isfinite(training['validation'][-1]['loss'])
assert manifest['status'] == 'complete'
assert export_sha256(DATA / 'manifest.json') == training['provenance']['data_manifest_sha256']

for result in (own, openai):
    assert result['status'] == 'complete', 'A HellaSwag evaluation is incomplete.'
    assert result['processed'] == result['example_count'] == 1000
    assert len(result['predictions']) == 1000
    assert result['correct_norm'] == sum(
        row['predicted_norm'] == row['label'] for row in result['predictions'])
    assert math.isclose(result['accuracy_norm'], result['correct_norm'] / 1000)
for key in ('selection_sha256', 'example_count', 'dataset_revision', 'scoring', 'precision', 'tf32'):
    assert own[key] == openai[key], f'HellaSwag comparison mismatch: {key}'
assert [(row['index'], row['ind'], row['label']) for row in own['predictions']] == [
    (row['index'], row['ind'], row['label']) for row in openai['predictions']]
assert own['training_successful_updates'] == training['successful_updates']
assert own['training_source_sha256'] == training['provenance']['source_sha256']
checkpoint = RESULTS / 'last.pt'
checkpoint_hash = export_sha256(checkpoint)
assert checkpoint_hash == own['checkpoint_sha256'], 'The evaluated checkpoint has changed.'

# Preserve the exact executed source snapshot, even if the local project later changes.
source_hashes = training['provenance']['source_sha256']
for filename, expected in source_hashes.items():
    source = (ROOT / filename).resolve()
    assert source.is_relative_to(ROOT.resolve())
    assert export_sha256(source) == expected, f'Executed source changed: {filename}'

execute(['week12/report.py', '--output-dir', str(RESULTS)])

gap = 100 * (openai['accuracy_norm'] - own['accuracy_norm'])
last_train = training['history'][-1]['loss']
last_val = training['validation'][-1]['loss']
note = f'''Hafta 12 teslimi — gerçek veride GPT-2 ve HellaSwag

124.475.904 parametreli kendi GPT-2 modelimi {training['environment']['hardware']}
üzerinde rastgele ağırlıklardan eğittim. GPT-2 tokenizer'ıyla hazırlanan veri:
{manifest['splits']['train']['tokens']:,} train token'ı,
{manifest['splits']['val']['tokens']:,} validation token'ı.
Belgeler train/val arasında ayrıldı; birebir aynı metin tekrarları çıkarıldı.

Başarılı güncelleme: {training['successful_updates']:,}
İşlenen token (tekrarlar dahil): {training['tokens_seen']:,}
Duvar süresi: {training['elapsed_wall_seconds']/60:.2f} dakika
Update süresine göre hız: {training['tokens_seen']/training['optimizer_seconds']:,.1f} token/s
Toplam duvar süresine göre hız: {training['tokens_seen']/training['elapsed_wall_seconds']:,.1f} token/s
Son train loss: {last_train:.6f} nats/token
İlk/son validation loss: {training['validation'][0]['loss']:.6f} -> {last_val:.6f}
Her validation kontrolünde aynı {training['validation'][-1]['tokens']:,} token ölçüldü.

Aynı ilk 1.000 HellaSwag validation örneğinde, ending-only ortalama token NLL:
Kendi modelim: {own['correct_norm']}/1000 = %{100*own['accuracy_norm']:.2f}
OpenAI GPT-2: {openai['correct_norm']}/1000 = %{100*openai['accuracy_norm']:.2f}
OpenAI eksi kendi modelim: {gap:+.2f} yüzde puan.

Kısa eğitimin veri miktarı, süresi ve metin dağılımı OpenAI ön eğitiminden
farklıdır. Tek bir deneyle farkın nedenlerini birbirinden ayıramam.
Örneklerde kelime ve cümle yapısı oluşsa da tekrarlar ve anlam kopuklukları
devam ediyor. Eğitim başı/ortası/sonu aynı prompt örnekleri rapordadır.
Ek Türkçe eğitim yapılmadı.

Teslim ZIP'i: kaynak kod, ham metrikler, grafikler, örnekler ve açıklama raporu.
Token shard'ları ayrı veri ZIP'inde, eğitilmiş checkpoint ayrı last.pt dosyasında.
FineWeb-Edu kaynağı: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
HellaSwag kaynağı: https://huggingface.co/datasets/Rowan/hellaswag
'''
(RESULTS / 'TESLIM_NOTU.txt').write_text(note, encoding='utf-8')

required = ['training.json', 'data_manifest.json', 'hellaswag_own.json',
            'hellaswag_openai.json', 'training_curves.png', 'submission_report.md',
            'TESLIM_NOTU.txt']
for filename in required:
    assert (RESULTS / filename).is_file(), f'Missing result: {filename}'

archive_path = ROOT / 'hafta12_l4_teslim.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in required:
        archive.write(RESULTS / filename, filename)
    for filename in source_hashes:
        archive.write(ROOT / filename, 'sources/' + filename)
    archive.writestr('source_manifest.json', json.dumps(source_hashes, indent=2) + '\n')
    archive.writestr('checkpoint_manifest.json', json.dumps({
        'filename': 'last.pt', 'sha256': checkpoint_hash,
        'successful_updates': training['successful_updates'],
        'tokens_seen': training['tokens_seen'],
        'included_in_this_zip': False,
    }, indent=2) + '\n')

# The token files are about 40 MB before compression; keep them separate.
data_archive = ROOT / 'hafta12_fineweb20m_tokens.zip'
with zipfile.ZipFile(data_archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    archive.write(DATA / 'manifest.json', 'manifest.json')
    for split in ('train', 'val'):
        for row in manifest['splits'][split]['shards']:
            source = (DATA / row['filename']).resolve()
            assert source.parent == DATA.resolve()
            assert export_sha256(source) == row['sha256']
            archive.write(source, source.name)

if BACKUP is not None:
    BACKUP = Path(BACKUP)
    BACKUP.mkdir(parents=True, exist_ok=True)
    backup_meta = BACKUP / 'training.json'
    backup_checkpoint = BACKUP / 'last.pt'
    checkpoint_current = (
        backup_meta.exists() and backup_checkpoint.exists()
        and export_json(backup_meta).get('successful_updates') == training['successful_updates']
        and backup_checkpoint.stat().st_size == checkpoint.stat().st_size
    )
    if not checkpoint_current:
        print('Copying the final checkpoint to Drive...', flush=True)
        staging = BACKUP / 'last.pt.tmp'
        shutil.copyfile(checkpoint, staging)
        staging.replace(backup_checkpoint)
    for source in [archive_path, data_archive] + [RESULTS / name for name in required]:
        shutil.copyfile(source, BACKUP / source.name)
    print('Drive teslim klasörü:', BACKUP)
    print('Drive checkpoint:', backup_checkpoint)
else:
    print('Drive yedeği yok. GPU değiştirmeden token ZIP ve last.pt dosyasını ayrıca indirin.')

print(note)
print('Teslim paketi hazır:', archive_path)
print('Ayrı token arşivi:', data_archive)
print('Checkpoint SHA-256:', checkpoint_hash)
files.download(str(archive_path))
