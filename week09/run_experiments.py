"""Train byte BPE on the private Week 8 corpus and reproduce the Week 9 report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Keep tiktoken's downloaded, hash-checked vocabulary inside this ignored folder.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / ".tiktoken_cache"))

import regex
import tiktoken

try:
    from .tokenizer import BasicTokenizer, RegexTokenizer, GPT4_SPLIT_PATTERN
    from .gpt4_tokenizer import GPT4Tokenizer
except ImportError:
    from tokenizer import BasicTokenizer, RegexTokenizer, GPT4_SPLIT_PATTERN
    from gpt4_tokenizer import GPT4Tokenizer

UNICODE_TEXT = "Çığ, çay, öğle ışığı, şüphe ve üzüm."
ROUNDTRIP_TEXT = "İstanbul'da çığ, öğle ışığı ve üzüm var. 🧠🤖🙂"
PARAGRAPHS = {
    "tr": (
        "Bu sabah Ankara'dan İstanbul'a trenle geldim. İstasyonda arkadaşlarımla "
        "buluştum; birlikte sahilde yürüyüp küçük bir kafede çay içtik. Yolculuk "
        "boyunca okuduğum kitabı ve gelecek hafta yapacağımız çalışmaları konuştuk. "
        "Akşam eve döndüğümde notlarımı düzenledim ve öğrendiklerimi ailemle paylaştım."
    ),
    "en": (
        "This morning I came from Ankara to Istanbul by train. I met my friends "
        "at the station; together we walked along the shore and drank tea in a "
        "small cafe. We talked about the book I had read during the journey and "
        "the work we would do next week. When I returned home in the evening, "
        "I organized my notes and shared what I had learned with my family."
    ),
}
PARITY_CASES = [
    "", "hello world!!!? (안녕하세요!) lol123 😉", ROUNDTRIP_TEXT,
    *PARAGRAPHS.values(), "Ankara'dan İstanbul'da İzmir'den Türkiye'nin",
    "Ankara’dan İstanbul’da", "I'm you'll we've IT'S I'd", "1234567890 3.14159 -42",
    "a\r\nb\n\n  \t c   ", "e\u0301 I\u0307 ı ç ğ ş ö ü", "中文 日本語 العربية हिन्दी",
    "👨‍👩‍👧‍👦 🇹🇷 👍🏽 \u0000", "<|endoftext|> literal <|fim_prefix|>",
]


def unicode_rows(text):
    return [{"position": i, "character": c, "code_point": f"U+{ord(c):04X}",
             "utf8": list(c.encode("utf-8")), "byte_count": len(c.encode("utf-8"))}
            for i, c in enumerate(text)]


def token_view(data):
    """Use hex as ground truth: one token need not be valid UTF-8 on its own."""
    return {"text": data.decode("utf-8", errors="replace"), "hex": data.hex(" ")}


def first_merges(tokenizer, count=20):
    return [{"id": idx, "left": token_view(tokenizer.vocab[pair[0]]),
             "right": token_view(tokenizer.vocab[pair[1]]),
             "merged": token_view(tokenizer.vocab[idx])}
            for pair, idx in list(tokenizer.merges.items())[:count]]


def compression(tokenizer, text):
    ids = tokenizer.encode(text)
    if tokenizer.decode(ids, errors="strict") != text:
        raise AssertionError("Corpus encode/decode roundtrip failed")
    byte_count = len(text.encode("utf-8"))
    return {"bytes": byte_count, "characters": len(text), "tokens": len(ids),
            "bytes_per_token": byte_count / len(ids) if ids else None,
            "characters_per_token": len(text) / len(ids) if ids else None,
            "roundtrip": True}


def compare_paragraph(text, encode):
    # Explicit, reproducible denominator: whitespace-delimited words, punctuation retained.
    ids = encode(text)
    words = len(text.split())
    return {"words": words, "characters": len(text), "bytes": len(text.encode("utf-8")),
            "tokens": len(ids), "tokens_per_word": len(ids) / words,
            "ids": ids}


def check_parity(adapter, reference):
    cases = list(PARITY_CASES)
    # Deterministic mixtures exercise merge ordering and UTF-8 boundaries beyond examples.
    import random
    rng = random.Random(1337)
    alphabet = "ab c\t\n\r0123456789çğıöşüİ'’!?é\u0301中🙂🧠"
    cases += ["".join(rng.choices(alphabet, k=rng.randrange(1, 160))) for _ in range(100)]
    for text in cases:
        expected = reference.encode_ordinary(text)
        actual = adapter.encode(text)
        if actual != expected:
            raise AssertionError(f"cl100k_base token IDs differ on {text!r}")
        if adapter.decode(actual, errors="strict") != text:
            raise AssertionError(f"cl100k_base roundtrip differs on {text!r}")
    return {"encoding": "cl100k_base", "cases": len(cases), "all_ids_equal": True,
            "all_roundtrips_equal": True, "special_tokens": "ordinary literal text",
            "example": {"text": PARITY_CASES[1], "ids": adapter.encode(PARITY_CASES[1])}}


def quoted(value):
    return json.dumps(value, ensure_ascii=False).replace("|", "\\|").replace("`", "\\`")


def write_report(result, destination):
    bpe = result["tokenizers"]
    lines = [
        "# Week 9 — Byte Pair Encoding", "",
        "Bu dosya `run_experiments.py` tarafından gerçek çalıştırma sonuçlarından üretildi.",
        "Ölçümler tokenizer eğitimine aittir; yeni bir dil modeli eğitilmedi.", "",
        "## 1. Unicode ve UTF-8", "",
        f"Örnek: {UNICODE_TEXT}", "",
        "| Sıra | Karakter | Kod noktası | UTF-8 byte (ondalık) | Byte |",
        "|---:|---|---|---|---:|",
    ]
    for row in result["unicode"]:
        lines.append(f"| {row['position']} | {quoted(row['character'])} | {row['code_point']} | "
                     f"{row['utf8']} | {row['byte_count']} |")
    lines += ["", "ASCII harfleri 1 byte; ç, ğ, ı, ö, ş ve ü bu örnekte 2 byte. "
              "Unicode kod noktası karakterin kimliği, UTF-8 ise bu kimliğin byte dizisi olarak kodlanmasıdır. "
              "Emoji çoğunlukla 4 byte'lık kod noktaları içerir; görünen tek bir emoji birden fazla "
              "kod noktasından oluşabilir. Türkçe bir metindeki her karakterin 2 byte olması gerekmez.", "",
              "Kod noktalarını doğrudan token ID yapmak, en büyük ID'ye göre tablo ayırırsak "
              "çok büyük ve seyrek embedding/çıktı tabloları gerektirir. Yalnız görülen karakterleri "
              "sıkı ID'lere eşlesek bile yeni karakterler için bir çözüm gerekir. Byte tabanı "
              "sabit 256 sembolle her geçerli UTF-8 metni temsil eder; BPE sık dizileri birleştirir.", "",
              "## 2. Kendi Türkçe derlemimde BPE", "",
              f"Kaynak: `{result['corpus']['path']}`; {result['corpus']['bytes']:,} byte, "
              f"{result['corpus']['characters']:,} kod noktası. "
              f"SHA-256: `{result['corpus']['sha256']}`.", "",
              "Metin değiştirilmeden, kod noktası konumundan ilk %90 eğitim ve son %10 "
              "doğrulama olarak ayrıldı. İki tokenizer yalnız eğitim bölümünde, aynı hedef "
              "vocab ile eğitildi. Doğrulama ölçümü bu eğitimden sonra yapıldı. "
              "Her adımda ardışık çiftler sayıldı; en sık çiftin çakışmayan oluşumları "
              "soldan sağa birleştirildi. Eşitlikte metindeki ilk oluşum seçildi.", "",
              "| Tokenizer | Vocab | Bölüm | Byte | Token | Byte/token | Karakter/token |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for name, entry in bpe.items():
        for split in ("train", "validation"):
            m = entry[split]
            lines.append(f"| {name} | {entry['vocab_size']} | {split} | {m['bytes']:,} | "
                         f"{m['tokens']:,} | {m['bytes_per_token']:.4f} | {m['characters_per_token']:.4f} |")
    lines += ["", "Byte/token oranı istenen sıkıştırma ölçüsüdür; ID'lerin diskte kapladığı "
              "alanı veya sözlük maliyetini ölçmez. Karakter/token oranı, harf tabanlı "
              "modele göre aynı block_size içine yaklaşık ne kadar metin sığdığını gösterir."]
    for name, entry in bpe.items():
        lines += ["", f"### {name}: ilk {min(20, entry['merges'])} birleştirme", "",
                  "Tek token tek başına geçerli UTF-8 olmayabilir; `�` görünen satırlarda hex byte'ları esas alın.", "",
                  "| ID | Sol | Sağ | Birleşim | Hex |", "|---:|---|---|---|---|"]
        for m in entry["first_20_merges"]:
            lines.append(f"| {m['id']} | {quoted(m['left']['text'])} | {quoted(m['right']['text'])} | "
                         f"{quoted(m['merged']['text'])} | {m['merged']['hex']} |")
    lines += ["", "Sık Türkçe harflerin UTF-8 byte çiftleri ile kısa, tekrar eden harf "
              "grupları ilk sıralarda görünür. Öğrenilen parçalar dilbilgisel ek veya hece olmak "
              "zorunda değildir; sıklık ve önceki birleşimler belirleyicidir.", "",
              "## 3. Encode ve decode", "",
              f"Örnek: {ROUNDTRIP_TEXT}", "",
              f"İki eğitilmiş tokenizer için roundtrip: `{result['roundtrip']['all_equal']}`. "
              f"Regex token ID'leri: `{result['roundtrip']['regex_ids']}`.", "",
              "Encode yeni metinde çiftleri yeniden eğitmez: kayıtlı birleşim sırasındaki "
              "en erken uygulanabilir çifti seçer. Decode tüm token byte'larını birleştirir, "
              "sonra UTF-8 çözer. Her token'ı ayrı ayrı metne çevirmek Türkçe harfleri ve "
              "emojiyi bozabilir.", "",
              f"Tek başına `[255]` geçersiz UTF-8: varsayılan decode sonucu "
              f"`{result['roundtrip']['invalid_utf8_replacement']}`, strict hata türü "
              f"`{result['roundtrip']['strict_error']}`. Üretim sırasında keyfi token dizileri "
              "geçersiz byte dizileri oluşturabilir. Gösterimde `errors='replace'` okunabilir "
              "çıktı sağlar; veri doğrulamada `errors='strict'` hatayı görünür kılar. "
              "Roundtrip garantisi geçerli girdinin encode çıktısı içindir.", "",
              "## 4. GPT-4 / cl100k_base regex", "", "```python", GPT4_SPLIT_PATTERN, "```", "",
              "Bu desen güncel tiktoken kaynak kodundaki cl100k_base desenidir; dersteki "
              "desenin optimize edilmiş eşdeğeridir. Regex her iki aşamada da kullanılır: "
              "hem eğitimde hem encode sırasında parçaların sınırlarını aşan birleşim yapılmaz.", ""]
    for text, chunks in result["regex_examples"].items():
        lines.append(f"- `{text}` → `{json.dumps(chunks, ensure_ascii=False)}`")
    lines += ["", "ASCII kesme işaretinden sonraki İngilizce `'d` kısaltması Türkçe `'dan` "
              "ve `'da` eklerinin başını yakalar: bunlar kök ve Türkçe ek sınırlarına uygun "
              "bir çözümleme değildir. Unicode sağ kesme işareti `’` aynı İngilizce kısaltma "
              "kuralına girmez. Sayılar en çok üç basamaklı regex parçalarına bölünür; "
              "BPE bunları daha da küçük token'lara ayırabilir. Desen sözcük/anlam çözümleyicisi değildir.", "",
              f"İlk {min(20, bpe['basic']['merges'], bpe['regex']['merges'])} sırada "
              f"aynı byte dizisini üreten birleşim sayısı: "
              f"{result['merge_comparison']['same_result_at_same_position_first_20']}/"
              f"{min(20, bpe['basic']['merges'], bpe['regex']['merges'])}. "
              f"Tüm öğrenilmiş birleşimlerde ortak üretilen byte dizisi: "
              f"{result['merge_comparison']['shared_learned_byte_strings']}. "
              "Regex sınırları aday çiftleri kısıtladığı için sonraki sıralar ve sıkıştırma değişir. "
              "Her iki modelin vocab'ı 256 temel byte'ı da içerir.", "",
              "## 5. Türkçe ve İngilizce", "",
              "Anlamca eşleştirilmiş, bu deney için yazılmış iki paragraf:", "",
              f"**Türkçe:** {PARAGRAPHS['tr']}", "", f"**İngilizce:** {PARAGRAPHS['en']}", "",
              "Kelime sayısı `len(text.split())`: boşlukla ayrılan birimler; noktalama birimin içinde kalır. "
              "Bu seçim diller arasında kusursuz dilbilimsel eşitleme sağlamaz.", "",
              "| Metin | Tokenizer | Kelime | Token | Token/kelime |", "|---|---|---:|---:|---:|"]
    for name, row in result["paragraph_comparison"].items():
        lines.append(f"| {row['language']} | {name} | {row['words']} | {row['tokens']} | {row['tokens_per_word']:.4f} |")
    lines += ["", "Türkçede ekler bir sözcükte çok sayıda anlam birimi biriktirir. Bu, farklı "
              "sözcük biçimleri ve seyrek uzun birleşimler yaratır; BPE bunları birkaç parçaya "
              "ayırabilir. Eklemeli yapı tek başına token sayısını belirlemez: eğitim verisinin "
              "dil dağılımı, byte kodlaması, vocab büyüklüğü ve metin de etkilidir. Küçük özel "
              f"tokenizer ile cl100k_base vocab bütçeleri {result['settings']['vocab_size']} ve 100256 "
              "olağan token'dır; bu tablo yöntemlerin adil bir kalite sıralaması değildir. "
              "Tek paragraf dil genelini temsil etmez; varsayılan Week 8 derlemi de "
              "şiir ağırlıklı edebi metindir.", "",
              "Eklerin ölçülen ayrışması (cl100k_base; parçalar tek başına UTF-8 olmak zorunda değil):", ""]
    for row in result["suffix_examples"]:
        lines.append(f"- `{row['text']}` → `{row['ids']}`; byte parçaları: "
                     f"`{[p['hex'] for p in row['pieces']]}`")
    parity = result["gpt4_parity"]
    lines += ["", "## 6a. cl100k_base ile birebir eşleşme", "",
              "Tiktoken'dan eğitilmiş byte sıraları yüklenir; gerçek encode/decode işlemi "
              "bizim Python kodumuzdadır. Birleşim çiftleri vocab sıralarından geri çıkarılır; "
              "ilk 256 byte'ın token ID permütasyonu korunur. GPT-4'ün tüm eğitim korpusu "
              "yeniden oluşturulmaz; burada hazır tokenizer sözlüğü kullanılır.", "",
              f"{parity['cases']} örnek ve deterministik rastgele girdide tüm token ID'leri "
              f"tiktoken.encode_ordinary ile eşit: `{parity['all_ids_equal']}`. "
              f"Decode roundtrip eşitliği: `{parity['all_roundtrips_equal']}`.", "",
              f"Örnek: `{parity['example']['text']}`", "", f"ID'ler: `{parity['example']['ids']}`", "",
              "Özel token yazımları burada olağan metin olarak kodlanır; "
              "`allowed_special` ile kontrol token'ı işleme bu deneyin kapsamı dışındadır. "
              "cl100k_base, ödevde istenen GPT-4 ailesinin tarihsel tokenizer'ıdır; "
              "tüm güncel ChatGPT modellerinin tokenizer'ı olduğu iddia edilmez.", "",
              "Tokenizer dil modelinden önce ayrı eğitilir ve model eğitimi boyunca sabit kalır. "
              "Tokenizer değişince embedding ve çıkış tablosunun boyutu, bağlamdaki metin "
              "miktarı ve token başına tahmin görevi değişir. Bu nedenle BPE modelinin "
              "nats/token val loss'u geçen haftaki nats/karakter loss ile doğrudan "
              "karşılaştırılamaz. Aynı tutulmuş metinde toplam NLL / UTF-8 byte / ln(2) "
              "gibi bits/byte ölçüsü ortak birim sağlar; veri ve değerlendirme koşulları da "
              "eşleşmelidir. Burada 6a tamamlandığı için ek GPU model eğitimi yapılmadı.", "",
              "Heceleme, basamak işlemleri ve bazı yabancı dil güçlüklerine token sınırları "
              "katkıda bulunabilir; bütün model hatalarını yalnız tokenizer'a bağlamak doğru değildir.", "",
              "## Yeniden üretim ve sınırlar", "",
              f"Python {result['environment']['python']}; regex {result['environment']['regex']}; "
              f"tiktoken {result['environment']['tiktoken']}. Hedef vocab {result['settings']['vocab_size']}. "
              "Eğitilmiş sözlükler `outputs/` altında yerel kalır; özel derlem kopyalanmaz. "
              "Yeni çalıştırmalar donanım/sürüm farkıyla farklı süreler verebilir.", "",
              "- [Karpathy: tokenizer dersi](https://www.youtube.com/watch?v=zduSFxRajkE)",
              "- [Ders notebook'u](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)",
              "- [minbpe alıştırması](https://github.com/karpathy/minbpe/blob/master/exercise.md)",
              "- [minbpe: birleşimleri geri çıkarma referansı](https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py)",
              "- [tiktoken: cl100k_base deseni ve vocab kaynağı](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)",
              "- [Tiktokenizer görselleştiricisi](https://tiktokenizer.vercel.app)", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "week08/data/turkish.txt")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("vocab-size must be at least 256")
    if not args.corpus.is_file():
        parser.error("Corpus missing: prepare the Week 8 private corpus or pass --corpus PATH")
    raw = args.corpus.read_bytes()
    text = raw.decode("utf-8")
    boundary = int(len(text) * 0.9)
    training, validation = text[:boundary], text[boundary:]
    if not training or not validation:
        parser.error("Corpus must contain enough text for nonempty 90/10 splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        corpus_name = args.corpus.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        corpus_name = args.corpus.name
    result = {"schema_version": 1,
              "environment": {"python": platform.python_version(),
                              "regex": importlib.metadata.version("regex"),
                              "tiktoken": importlib.metadata.version("tiktoken")},
              "settings": {"vocab_size": args.vocab_size, "train_fraction": 0.9,
                           "split_unit": "Unicode code point", "normalization": "none"},
              "corpus": {"path": corpus_name, "bytes": len(raw), "characters": len(text),
                         "distinct_characters": len(set(text)),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "training_sha256": hashlib.sha256(training.encode("utf-8")).hexdigest(),
                         "validation_sha256": hashlib.sha256(validation.encode("utf-8")).hexdigest()},
              "unicode": unicode_rows(UNICODE_TEXT), "tokenizers": {}}
    print("Unicode / UTF-8:", flush=True)
    for row in result["unicode"]:
        print(f"  {row['character']!r}: {row['code_point']}, "
              f"UTF-8={row['utf8']}, {row['byte_count']} byte", flush=True)
    tokenizers = {}
    for name, cls in (("basic", BasicTokenizer), ("regex", RegexTokenizer)):
        print(f"Training {name} on {len(training.encode('utf-8')):,} UTF-8 bytes...", flush=True)
        tokenizer = cls()
        start = time.perf_counter()
        tokenizer.train(training, args.vocab_size)
        elapsed = time.perf_counter() - start
        tokenizer.save(args.output_dir / f"{name}_tokenizer.json")
        print(f"{name}: vocabulary learned in {elapsed:.2f}s; measuring both splits...", flush=True)
        entry = {"vocab_size": len(tokenizer.vocab), "merges": len(tokenizer.merges),
                 "training_seconds": elapsed, "first_20_merges": first_merges(tokenizer),
                 "train": compression(tokenizer, training),
                 "validation": compression(tokenizer, validation)}
        result["tokenizers"][name] = entry
        tokenizers[name] = tokenizer
        print(f"{name}: {elapsed:.2f}s; validation byte/token={entry['validation']['bytes_per_token']:.4f}", flush=True)
        for merge in entry["first_20_merges"]:
            print(f"  {merge['id']}: {merge['left']['text']!r} + {merge['right']['text']!r} "
                  f"-> {merge['merged']['text']!r} [{merge['merged']['hex']}]", flush=True)
    basic, tokenizer = tokenizers["basic"], tokenizers["regex"]
    all_equal = all(t.decode(t.encode(ROUNDTRIP_TEXT), errors="strict") == ROUNDTRIP_TEXT
                    for t in tokenizers.values())
    if not all_equal:
        raise AssertionError("Turkish + emoji roundtrip failed")
    try:
        tokenizer.decode([255], errors="strict")
    except UnicodeDecodeError as error:
        strict_error = type(error).__name__
    else:
        raise AssertionError("Invalid UTF-8 must fail under strict decoding")
    result["roundtrip"] = {"text": ROUNDTRIP_TEXT, "all_equal": all_equal,
                           "regex_ids": tokenizer.encode(ROUNDTRIP_TEXT),
                           "invalid_utf8_replacement": tokenizer.decode([255]),
                           "strict_error": strict_error}
    result["regex_examples"] = {s: tokenizer.split(s) for s in
                                ("Ankara'dan", "İstanbul'da", "Ankara’dan", "İstanbul’da", "1234567890")}
    left = [basic.vocab[i] for i in basic.merges.values()]
    right = [tokenizer.vocab[i] for i in tokenizer.merges.values()]
    result["merge_comparison"] = {
        "same_result_at_same_position_first_20": sum(a == b for a, b in zip(left[:20], right[:20])),
        "shared_learned_byte_strings": len(set(left) & set(right))}
    reference = tiktoken.get_encoding("cl100k_base")
    result["paragraph_comparison"] = {}
    for language, paragraph in PARAGRAPHS.items():
        row = compare_paragraph(paragraph, reference.encode_ordinary)
        row["language"] = language
        result["paragraph_comparison"][f"cl100k_base_{language}"] = row
    for name, custom in tokenizers.items():
        row = compare_paragraph(PARAGRAPHS["tr"], custom.encode)
        row["language"] = "tr"
        result["paragraph_comparison"][f"custom_{name}_tr"] = row
    result["suffix_examples"] = [
        {"text": s, "ids": reference.encode_ordinary(s),
         "pieces": [token_view(reference.decode_single_token_bytes(i))
                    for i in reference.encode_ordinary(s)]}
        for s in ("ev", "evler", "evlerimiz", "evlerimizden")]
    print("Recovering cl100k_base merges and checking exact token IDs...", flush=True)
    result["gpt4_parity"] = check_parity(GPT4Tokenizer(), reference)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, args.results_dir / "submission_report.md")
    for name, row in result["paragraph_comparison"].items():
        print(f"{name}: {row['tokens']} tokens / {row['words']} words "
              f"= {row['tokens_per_word']:.4f} tokens/word", flush=True)
    print(f"Report: {args.results_dir / 'submission_report.md'}", flush=True)
    print(f"cl100k_base parity: {result['gpt4_parity']['cases']} cases passed", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
