"""Prepare Shakespeare and user-selected Nazim Hikmet UTF-8 source files."""

import argparse
import hashlib
import json
import shutil
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from data import HERE, describe_data, validate_turkish_corpus

EN_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SELECTION = "nazim_hikmet"


def normalize_text(text):
    """Keep verse lines, stanza boundaries, case and Turkish characters."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text.replace("\u00ad", "").replace("\xa0", " "))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n") + "\n"


def prepare_english(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "shakespeare.txt"
    if not path.exists():
        previous = HERE.parent / "week07" / "data" / "input.txt"
        if previous.exists():
            shutil.copyfile(previous, path)
        else:
            with urllib.request.urlopen(EN_URL, timeout=60) as response:
                path.write_bytes(response.read())


def compile_turkish(source_dir, directory):
    """Compile only explicitly supplied files; never fetch an author's full texts."""
    source_dir, directory = Path(source_dir).resolve(), Path(directory)
    source_manifest = source_dir / "sources.json"
    if not source_manifest.exists():
        raise ValueError("Provide UTF-8 TXT files and sources.json; see sources.example.json. "
                         "No Turkish texts will be downloaded automatically.")
    selection = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    if selection.get("selection_id") != SELECTION or not selection.get("sources"):
        raise ValueError("sources.json must select nazim_hikmet and contain at least one source")
    texts, sources, seen = [], [], set()
    for item in selection["sources"]:
        required = ("file", "title", "genre", "source_reference", "permission_basis")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError(f"Every source needs nonempty fields: {', '.join(required)}")
        if item.get("author_id") != SELECTION:
            raise ValueError("Every source must identify Nazim Hikmet as its author")
        path = (source_dir / item["file"]).resolve()
        if not path.is_relative_to(source_dir) or path.suffix.lower() != ".txt":
            raise ValueError("Source files must be TXT files inside the selected source folder")
        raw = path.read_bytes()
        text = normalize_text(raw.decode("utf-8-sig"))
        if not text.strip():
            raise ValueError(f"Empty source: {item['file']}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate complete text: {item['file']}; do not repeat works to reach 1 MB")
        seen.add(digest)
        texts.append(text)
        sources.append({**item, "original_sha256": hashlib.sha256(raw).hexdigest(),
                        "normalized_sha256": digest, "characters": len(text)})
    raw = "\n".join(texts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "turkish.txt"
    manifest_path = directory / "corpus_manifest.json"
    if target.exists():
        validate_turkish_corpus(directory)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Existing corpus differs; preserve it and use a new data folder before replacing it")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.exists():
        raise ValueError("An existing manifest without its corpus must be reviewed before compilation")
    manifest = {"selection_id": SELECTION, "compiled_utc": datetime.now(timezone.utc).isoformat(),
                "turkish_sha256": digest, "bytes": len(raw), "sources": sources,
                "near_1mb_target_met": len(raw) >= 900_000,
                "compilation": "UTF-8, NFC, LF; preserve verse and stanza breaks; concatenate whole works; "
                               "reject duplicate complete texts; no automatic scraping or repetition",
                "permission_note": "Permission bases are supplied by the dataset owner, not independently "
                                   "verified by this script. Raw texts are not published to GitHub."}
    target.write_bytes(raw)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=HERE / "data" / "nazim_sources")
    args = parser.parse_args()
    directory = HERE / "data"
    try:
        prepare_english(directory)
        if not args.english_only:
            if (directory / "turkish.txt").exists() and not (args.source_dir / "sources.json").exists():
                validate_turkish_corpus(directory)
            else:
                manifest = compile_turkish(args.source_dir, directory)
                if not manifest["near_1mb_target_met"]:
                    print("WARNING: below 0.9 MB; the assignment's corpus-size target is not yet met.")
        print(json.dumps(describe_data(), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Data preparation stopped: {error}\n")


if __name__ == "__main__":
    main()
