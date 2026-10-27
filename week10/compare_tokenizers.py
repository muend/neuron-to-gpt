"""Compare GPT-2 against cl100k_base on the exact measured Week 9 paragraphs."""

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / ".cache/tiktoken"))

import tiktoken


def main():
    source = HERE.parent / "week09/results/metrics.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    cl100k = tiktoken.get_encoding("cl100k_base")
    gpt2 = tiktoken.get_encoding("gpt2")
    rows = []
    for language in ("tr", "en"):
        old = previous["paragraph_comparison"]["cl100k_base_" + language]
        # Decode the recorded ordinary IDs so the comparison cannot silently
        # substitute a different translation or punctuation than Week 9 used.
        text = cl100k.decode(old["ids"], errors="strict")
        assert cl100k.encode_ordinary(text) == old["ids"]
        assert len(text.split()) == old["words"]
        assert len(text.encode("utf-8")) == old["bytes"]
        ids = gpt2.encode_ordinary(text)
        rows.append({"language": language, "text": text, "words": old["words"],
                     "bytes": old["bytes"], "cl100k_tokens": len(old["ids"]),
                     "cl100k_tokens_per_word": len(old["ids"]) / old["words"],
                     "gpt2_tokens": len(ids), "gpt2_tokens_per_word": len(ids) / old["words"],
                     "gpt2_ids": ids})
    result = {"schema_version": 1, "source": "week09/results/metrics.json",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "word_definition": "len(text.split())", "paragraphs": rows}
    output = HERE / "results/tokenizer_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    for row in rows:
        print(f"{row['language']}: GPT-2 {row['gpt2_tokens']}/{row['words']} "
              f"= {row['gpt2_tokens_per_word']:.4f}; cl100k {row['cl100k_tokens']}/{row['words']} "
              f"= {row['cl100k_tokens_per_word']:.4f} tokens/word")


if __name__ == "__main__":
    main()
