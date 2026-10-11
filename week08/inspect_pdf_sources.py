"""Inspect user-provided PDFs and cache extraction candidates outside Git."""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from pypdf import PdfReader
import pypdfium2 as pdfium


def inspect(path, directory, slug, render_pages=(), render_only=False):
    directory.mkdir(parents=True, exist_ok=True)
    if render_only:
        render_pdf(path, directory, slug, render_pages)
        return
    reader = PdfReader(path)
    pages = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        invalid = sum(unicodedata.category(char) in ("Cc", "Co", "Cn")
                      and char not in "\n\r\t" for char in text)
        latin = sum(char.isalpha() and "LATIN" in unicodedata.name(char, "") for char in text)
        pages.append({"page": index + 1, "text": text, "characters": len(text),
                      "invalid_characters": invalid, "latin_letters": latin,
                      "replacement_characters": text.count("\ufffd")})
        if (index + 1) % 200 == 0:
            print(f"{slug}: inspected {index + 1}/{len(reader.pages)} pages", flush=True)
    result = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "page_count": len(pages), "pages": pages, "status": "unreviewed_extraction_candidate"}
    (directory / f"{slug}_pages.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source": slug, "pages": len(pages),
                      "characters": sum(p["characters"] for p in pages),
                      "invalid_characters": sum(p["invalid_characters"] for p in pages),
                      "replacement_characters": sum(p["replacement_characters"] for p in pages)}, indent=2), flush=True)
    if render_pages:
        render_pdf(path, directory, slug, render_pages)


def render_pdf(path, directory, slug, render_pages):
    if render_pages:
        doc = pdfium.PdfDocument(path)
        for number in render_pages:
            image_path = directory / f"{slug}_{number:04d}.png"
            if image_path.exists():
                continue
            if not 1 <= number <= len(doc):
                raise ValueError(f"Page out of range: {number}")
            page = doc[number - 1]
            scale = min(2.5, 2500 / max(page.get_size()))
            bitmap = page.render(scale=scale)
            bitmap.to_pil().save(image_path)
            bitmap.close()
            page.close()
            if number % 50 == 0:
                print(f"{slug}: rendered page {number}", flush=True)
        doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/nazim_sources/inspection")
    parser.add_argument("--render-pages", type=int, nargs="*", default=[])
    parser.add_argument("--render-all", action="store_true", help="Render all pages for local OCR")
    parser.add_argument("--render-only", action="store_true", help="Reuse extraction inspection and render selected pages")
    args = parser.parse_args()
    if not args.slug.replace("_", "").isalnum():
        parser.error("Use an alphanumeric source slug")
    pages = range(1, len(PdfReader(args.input).pages) + 1) if args.render_all else args.render_pages
    inspect(args.input, args.output, args.slug, pages, args.render_only)
