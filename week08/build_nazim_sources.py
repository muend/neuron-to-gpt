"""Build private source TXT files from the user's inspected PDF/OCR candidates."""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

from pypdf import PdfReader


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verse_page(page):
    """Reconstruct body lines by baseline; restore visible stanza gaps."""
    height = float(page.mediabox.height)
    fragments = []
    def visitor(text, cm, tm, font, size):
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        if 3 < y < height - 20 and text.strip():
            fragments.append((y, x, text.strip()))
    page.extract_text(visitor_text=visitor)
    rows = []
    for y, x, text in sorted(fragments, key=lambda item: (-item[0], item[1])):
        if rows and abs(rows[-1][0] - y) <= 2:
            rows[-1][1].append((x, text))
        else:
            rows.append([y, [(x, text)]])
    gaps = [rows[i][0] - rows[i+1][0] for i in range(len(rows)-1)]
    typical = statistics.median([gap for gap in gaps if 5 < gap < 16]) if any(5 < g < 16 for g in gaps) else 11
    lines = []
    for index, (y, chunks) in enumerate(rows):
        if index and rows[index-1][0] - y > typical * 1.6:
            lines.append("")
        lines.append(" ".join(text for _, text in sorted(chunks)))
    return "\n".join(lines).strip()


def clean_ocr_page(page):
    lines = []
    for line in page["lines"]:
        text = line["text"].strip()
        # Remove marginal folios, but keep dates and numbers in body text.
        if line["top"] > page["height"] * .94 and re.fullmatch(r"\d+", text):
            continue
        if re.match(r"^Sene\s+[0-9ıIl]", text):
            continue
        lines.append(text)
    text = "\n".join(lines)
    # Rejoin explicitly hyphenated print wraps in prose only, never poetry.
    return re.sub(r"(?<=\w)-\n(?=\w)", "", text).strip()


def clean_verse_ocr(page):
    rows = [line for line in page["lines"] if line["top"] > page["height"] * .045]
    gaps = [rows[i+1]["top"] - rows[i]["top"] for i in range(len(rows)-1)]
    positive = [gap for gap in gaps if gap > 0]
    typical = statistics.median(positive) if positive else 1
    lines = []
    for index, row in enumerate(rows):
        if row["top"] > page["height"] * .94 and re.fullmatch(r"\d+", row["text"].strip()):
            continue
        if index and row["top"] - rows[index-1]["top"] > typical * 1.6:
            lines.append("")
        lines.append(row["text"])
    return "\n".join(lines).strip()


def collect_pages(pages, extractor):
    accepted, quarantine, empty = [], [], []
    for page in pages:
        text = extractor(page)
        bad = sum(char == "\ufffd" or (unicodedata.category(char) in ("Cc", "Co", "Cn")
                                        and char not in "\n\r\t") for char in text)
        if bad:
            quarantine.append({"page": page["page"], "bad_characters": bad, "text": text})
        elif text.strip():
            accepted.append({"page": page["page"], "text": text})
        else:
            empty.append(page["page"])
    return accepted, quarantine, empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poems-pdf", type=Path, required=True)
    parser.add_argument("--stories-ocr", type=Path, required=True)
    parser.add_argument("--aydinlik-ocr", type=Path, required=True)
    parser.add_argument("--poems-recovery-ocr", type=Path, help="Turkish OCR for quarantined poetry pages")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/nazim_sources/cleaned")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("poems.txt", "stories.txt", "aydinlik_prose.txt", "sources.json", "quality_report.json"):
        if (args.output / name).exists():
            parser.error("Source outputs already exist; select a new output folder instead of overwriting them")
    reader = PdfReader(args.poems_pdf)
    recovery = {p["page"]: clean_verse_ocr(p) for p in read_json(args.poems_recovery_ocr)["pages"]} if args.poems_recovery_ocr else {}
    poem_candidates = []
    for number in range(27, 2084):
        if number > len(reader.pages):
            parser.error("Unexpected poems edition; inspect and adjust the verified page ranges first")
        text = recovery[number] if number in recovery else verse_page(reader.pages[number-1])
        poem_candidates.append({"page": number, "text": text})
        if number % 300 == 0:
            print(f"Poetry body reconstructed through PDF page {number}", flush=True)
    stories = [p for p in read_json(args.stories_ocr)["pages"] if 7 <= p["page"] <= 292]
    articles = [p for p in read_json(args.aydinlik_ocr)["pages"]
                if 47 <= p["page"] <= 152 and p["page"] not in (61, 62)]
    collections = [
        ("poems.txt", "Collected poems and epics", "poetry and epic", poem_candidates,
         lambda p: p["text"], args.poems_pdf.name, "27-2083", "Existing PDF OCR plus Windows Turkish recovery OCR; header removal and stanza reconstruction"),
        ("stories.txt", "Stories", "prose", stories, clean_ocr_page,
         read_json(args.stories_ocr.parent / "stories_pages.json")["file"], "7-292", "Windows Turkish OCR from rendered page images"),
        ("aydinlik_prose.txt", "Aydinlik and newspaper articles", "prose", articles, clean_ocr_page,
         read_json(args.aydinlik_ocr.parent / "aydinlik_pages.json")["file"], "47-60,63-152", "Windows Turkish OCR from rendered page images")]
    sources, report = [], {"status": "candidate_with_known_ocr_limitations", "sources": [],
        "limitations": ["OCR can still substitute valid letters; absence of invalid codepoints is not proof of accuracy.",
                        "Sample pages were visually checked, not every line of all PDFs.",
                        "Stanza gaps are reconstructed from PDF baselines; prose OCR retains print line wraps.",
                        "Cross-source approximate duplication still requires an overlap audit.",
                        "No training has been performed on this corpus yet."],
        "excluded": "Covers, front matter, contents, publishing pages; Aydinlik poetry and photographic appendix"}
    report["poetry_recovery_pages"] = sorted(recovery)
    for name, title, genre, pages, extractor, reference, page_range, method in collections:
        accepted, quarantine, empty = collect_pages(pages, extractor)
        raw = ("\n\n".join(p["text"] for p in accepted) + "\n").encode("utf-8")
        (args.output / name).write_bytes(raw)
        (args.output / (name.removesuffix(".txt") + "_page_map.json")).write_text(
            json.dumps({"accepted": accepted, "quarantined": quarantine, "empty": empty}, ensure_ascii=False), encoding="utf-8")
        sources.append({"file": name, "title": title, "author_id": "nazim_hikmet", "genre": genre,
                        "source_reference": reference, "pdf_pages": page_range, "extraction_method": method,
                        "permission_basis": "User-provided files for local project processing; no redistribution permission asserted",
                        "sha256": hashlib.sha256(raw).hexdigest()})
        report["sources"].append({"file": name, "bytes": len(raw), "characters": len(raw.decode("utf-8")),
                                  "accepted_pages": len(accepted), "quarantined_pages": [p["page"] for p in quarantine],
                                  "empty_pages": empty})
        print(f"{name}: {len(raw):,} bytes, {len(accepted)} accepted pages, {len(quarantine)} quarantined", flush=True)
    (args.output / "sources.json").write_text(json.dumps({"selection_id": "nazim_hikmet", "sources": sources},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
