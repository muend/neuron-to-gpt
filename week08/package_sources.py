"""Package private selected source files for the owner's Colab upload, not GitHub."""

import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_dir / "sources.json").read_text(encoding="utf-8"))
    if manifest.get("selection_id") != "nazim_hikmet":
        parser.error("Only the user-selected Nazim collection may be packaged")
    if args.output.exists():
        parser.error("Output already exists; select a new filename instead of overwriting it")
    names = ["sources.json"] + [s["file"] for s in manifest["sources"]]
    names += [name for name in ("quality_report.json", "overlap_audit.json") if (args.source_dir / name).exists()]
    for name in names:
        if Path(name).name != name or not (args.source_dir / name).is_file():
            parser.error("Package entries must be existing files directly inside the selected source folder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as package:
        for name in names:
            package.write(args.source_dir / name, name)
    print(f"Private Colab source package: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
