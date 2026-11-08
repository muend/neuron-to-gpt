"""Prepare a bounded FineWeb-Edu stream (or UTF-8 documents) into NPY shards."""

import argparse
from pathlib import Path

try:
    from .common import FINEWEB_REVISION, HERE, read_json, sha256
    from .data import ShardLoader, prepare_documents
except ImportError:
    from common import FINEWEB_REVISION, HERE, read_json, sha256
    from data import ShardLoader, prepare_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fineweb", "local"), default="fineweb")
    parser.add_argument("--text", type=Path, help="Local documents separated by blank lines")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data/fineweb20m")
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-tokens", type=int, default=131_072)
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    args = parser.parse_args()
    if min(args.train_tokens, args.val_tokens, args.shard_size) < 2:
        parser.error("Token budgets and shard size must be at least two")
    if args.source == "local" and not args.text:
        parser.error("--source local requires --text")
    source = (dict(dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT", split="train",
                   revision=FINEWEB_REVISION, license="ODC-By-1.0") if args.source == "fineweb"
              else dict(filename=args.text.name, sha256=sha256(args.text), format="UTF-8 blank-line documents"))
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest["source"] != source or manifest["requested_tokens"] !=
                dict(train=args.train_tokens, val=args.val_tokens) or manifest["shard_size"] != args.shard_size):
            raise ValueError("Existing data has different settings; use a new output directory")
        for split in ("train", "val"):
            ShardLoader(args.output_dir, split, 1, 1)
        print("Reusing fully verified shards:", manifest_path)
        return
    if args.source == "fineweb":
        from datasets import load_dataset
        print("Opening pinned FineWeb-Edu sample-10BT stream; only requested tokens will be saved.", flush=True)
        stream = load_dataset(source["dataset"], name=source["config"], split="train",
                              streaming=True, revision=source["revision"])
        documents = (row["text"] for row in stream)
    else:
        documents = args.text.read_text(encoding="utf-8").split("\n\n")
    import tiktoken
    manifest = prepare_documents(documents, tiktoken.get_encoding("gpt2"), args.output_dir,
                                 train_tokens=args.train_tokens, val_tokens=args.val_tokens,
                                 shard_size=args.shard_size, source=source)
    print("Saved:", manifest_path)
    print({key: (value["tokens"], len(value["shards"])) for key, value in manifest["splits"].items()})


if __name__ == "__main__":
    main()
