"""Measure exact long overlaps without deleting legitimate poetic repetition."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def fingerprints(text, window=64):
    words = re.findall(r"\w+", text.casefold())
    return {hashlib.blake2b(" ".join(words[i:i+window]).encode("utf-8"), digest_size=16).digest()
            for i in range(len(words) - window + 1)}


def audit(source_dir):
    manifest = json.loads((source_dir / "sources.json").read_text(encoding="utf-8"))
    sets = {item["file"]: fingerprints((source_dir / item["file"]).read_text(encoding="utf-8"))
            for item in manifest["sources"]}
    names = list(sets)
    pairs = [{"sources": [left, right], "shared_64_word_windows": len(sets[left] & sets[right])}
             for index, left in enumerate(names) for right in names[index+1:]]
    result = {"method": "Casefolded exact 64-word sliding windows; no automatic text deletion",
              "pairs": pairs, "limitations": "This detects exact long overlaps, not paraphrases or OCR variants. "
              "Repeated verses within a work are intentionally preserved."}
    (source_dir / "overlap_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    audit(parser.parse_args().source_dir)
