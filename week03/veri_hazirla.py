from __future__ import annotations

import argparse
import csv
import io
import unicodedata
import urllib.request
from pathlib import Path

from ortak import TR_LETTERS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

EN_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
TR_URL = "https://raw.githubusercontent.com/niyazikemer/turkce_isimler/main/turkce_isim.csv"

EN_PATH = DATA_DIR / "names_en.txt"
TR_RAW_PATH = DATA_DIR / "turkce_isim.csv"
TR_PATH = DATA_DIR / "names_tr.txt"


def download_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "neuron-to-gpt-week3/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8-sig")


def normalize_turkish_name(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw.strip())
    # Python lower(), Türkçedeki I/İ ayrımını locale'e göre yapmaz.
    text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower().replace("i̇", "i")
    text = (
        text.replace("â", "a")
        .replace("î", "i")
        .replace("û", "ü")
    )
    return unicodedata.normalize("NFC", text)


def prepare_english(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EN_PATH.exists() and not force:
        return EN_PATH

    text = download_text(EN_URL)
    names = [line.strip().lower() for line in text.splitlines() if line.strip()]
    EN_PATH.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"İngilizce veri: {len(names):,} isim -> {EN_PATH}")
    return EN_PATH


def prepare_turkish(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TR_PATH.exists() and TR_RAW_PATH.exists() and not force:
        return TR_PATH

    csv_text = download_text(TR_URL)
    TR_RAW_PATH.write_text(csv_text, encoding="utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))
    seen = set()
    names = []
    dropped = []

    allowed = set(TR_LETTERS)
    for row in reader:
        name = normalize_turkish_name(row["name"])
        if name and all(ch in allowed for ch in name):
            if name not in seen:
                seen.add(name)
                names.append(name)
        else:
            dropped.append(name)

    TR_PATH.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(
        f"Türkçe veri: {len(names):,} temiz benzersiz isim -> {TR_PATH} "
        f"(alfabe dışı olduğu için atılan: {len(dropped):,})"
    )
    if dropped:
        print("Atılan ilk örnekler:", dropped[:10])
    return TR_PATH


def prepare_all(force: bool = False) -> tuple[Path, Path]:
    return prepare_english(force=force), prepare_turkish(force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 3 veri setlerini indir ve temizle.")
    parser.add_argument("--force", action="store_true", help="Dosyalar varsa yeniden indir.")
    args = parser.parse_args()
    prepare_all(force=args.force)


if __name__ == "__main__":
    main()
