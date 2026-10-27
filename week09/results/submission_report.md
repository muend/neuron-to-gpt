# Week 9 — Byte Pair Encoding

Bu dosya `run_experiments.py` tarafından gerçek çalıştırma sonuçlarından üretildi.
Ölçümler tokenizer eğitimine aittir; yeni bir dil modeli eğitilmedi.

## 1. Unicode ve UTF-8

Örnek: Çığ, çay, öğle ışığı, şüphe ve üzüm.

| Sıra | Karakter | Kod noktası | UTF-8 byte (ondalık) | Byte |
|---:|---|---|---|---:|
| 0 | "Ç" | U+00C7 | [195, 135] | 2 |
| 1 | "ı" | U+0131 | [196, 177] | 2 |
| 2 | "ğ" | U+011F | [196, 159] | 2 |
| 3 | "," | U+002C | [44] | 1 |
| 4 | " " | U+0020 | [32] | 1 |
| 5 | "ç" | U+00E7 | [195, 167] | 2 |
| 6 | "a" | U+0061 | [97] | 1 |
| 7 | "y" | U+0079 | [121] | 1 |
| 8 | "," | U+002C | [44] | 1 |
| 9 | " " | U+0020 | [32] | 1 |
| 10 | "ö" | U+00F6 | [195, 182] | 2 |
| 11 | "ğ" | U+011F | [196, 159] | 2 |
| 12 | "l" | U+006C | [108] | 1 |
| 13 | "e" | U+0065 | [101] | 1 |
| 14 | " " | U+0020 | [32] | 1 |
| 15 | "ı" | U+0131 | [196, 177] | 2 |
| 16 | "ş" | U+015F | [197, 159] | 2 |
| 17 | "ı" | U+0131 | [196, 177] | 2 |
| 18 | "ğ" | U+011F | [196, 159] | 2 |
| 19 | "ı" | U+0131 | [196, 177] | 2 |
| 20 | "," | U+002C | [44] | 1 |
| 21 | " " | U+0020 | [32] | 1 |
| 22 | "ş" | U+015F | [197, 159] | 2 |
| 23 | "ü" | U+00FC | [195, 188] | 2 |
| 24 | "p" | U+0070 | [112] | 1 |
| 25 | "h" | U+0068 | [104] | 1 |
| 26 | "e" | U+0065 | [101] | 1 |
| 27 | " " | U+0020 | [32] | 1 |
| 28 | "v" | U+0076 | [118] | 1 |
| 29 | "e" | U+0065 | [101] | 1 |
| 30 | " " | U+0020 | [32] | 1 |
| 31 | "ü" | U+00FC | [195, 188] | 2 |
| 32 | "z" | U+007A | [122] | 1 |
| 33 | "ü" | U+00FC | [195, 188] | 2 |
| 34 | "m" | U+006D | [109] | 1 |
| 35 | "." | U+002E | [46] | 1 |

ASCII harfleri 1 byte; ç, ğ, ı, ö, ş ve ü bu örnekte 2 byte. Unicode kod noktası karakterin kimliği, UTF-8 ise bu kimliğin byte dizisi olarak kodlanmasıdır. Emoji çoğunlukla 4 byte'lık kod noktaları içerir; görünen tek bir emoji birden fazla kod noktasından oluşabilir. Türkçe bir metindeki her karakterin 2 byte olması gerekmez.

Kod noktalarını doğrudan token ID yapmak, en büyük ID'ye göre tablo ayırırsak çok büyük ve seyrek embedding/çıktı tabloları gerektirir. Yalnız görülen karakterleri sıkı ID'lere eşlesek bile yeni karakterler için bir çözüm gerekir. Byte tabanı sabit 256 sembolle her geçerli UTF-8 metni temsil eder; BPE sık dizileri birleştirir.

## 2. Kendi Türkçe derlemimde BPE

Kaynak: `week08/data/turkish.txt`; 2,072,197 byte, 1,904,245 kod noktası. SHA-256: `95eed4275502beda5e18590c72abd2e9c301f35a975bc6aadd12ef0942ed3e5a`.

Metin değiştirilmeden, kod noktası konumundan ilk %90 eğitim ve son %10 doğrulama olarak ayrıldı. İki tokenizer yalnız eğitim bölümünde, aynı hedef vocab ile eğitildi. Doğrulama ölçümü bu eğitimden sonra yapıldı. Her adımda ardışık çiftler sayıldı; en sık çiftin çakışmayan oluşumları soldan sağa birleştirildi. Eşitlikte metindeki ilk oluşum seçildi.

| Tokenizer | Vocab | Bölüm | Byte | Token | Byte/token | Karakter/token |
|---|---:|---|---:|---:|---:|---:|
| basic | 512 | train | 1,860,597 | 941,336 | 1.9765 | 1.8206 |
| basic | 512 | validation | 211,600 | 105,098 | 2.0134 | 1.8119 |
| regex | 512 | train | 1,860,597 | 942,644 | 1.9738 | 1.8181 |
| regex | 512 | validation | 211,600 | 105,095 | 2.0134 | 1.8119 |

Byte/token oranı istenen sıkıştırma ölçüsüdür; ID'lerin diskte kapladığı alanı veya sözlük maliyetini ölçmez. Karakter/token oranı, harf tabanlı modele göre aynı block_size içine yaklaşık ne kadar metin sığdığını gösterir.

### basic: ilk 20 birleştirme

Tek token tek başına geçerli UTF-8 olmayabilir; `�` görünen satırlarda hex byte'ları esas alın.

| ID | Sol | Sağ | Birleşim | Hex |
|---:|---|---|---|---|
| 256 | "�" | "�" | "ı" | c4 b1 |
| 257 | "n" | " " | "n " | 6e 20 |
| 258 | "�" | "�" | "ü" | c3 bc |
| 259 | "�" | "�" | "ş" | c5 9f |
| 260 | "e" | " " | "e " | 65 20 |
| 261 | "a" | "r" | "ar" | 61 72 |
| 262 | "e" | "r" | "er" | 65 72 |
| 263 | "a" | "n" | "an" | 61 6e |
| 264 | "a" | " " | "a " | 61 20 |
| 265 | "i" | " " | "i " | 69 20 |
| 266 | "." | "\n" | ".\n" | 2e 0a |
| 267 | "i" | "r" | "ir" | 69 72 |
| 268 | "�" | "�" | "ç" | c3 a7 |
| 269 | "i" | "n" | "in" | 69 6e |
| 270 | "," | " " | ", " | 2c 20 |
| 271 | "a" | "k" | "ak" | 61 6b |
| 272 | "a" | "l" | "al" | 61 6c |
| 273 | "e" | "n" | "en" | 65 6e |
| 274 | "�" | "�" | "ö" | c3 b6 |
| 275 | "ı" | " " | "ı " | c4 b1 20 |

### regex: ilk 20 birleştirme

Tek token tek başına geçerli UTF-8 olmayabilir; `�` görünen satırlarda hex byte'ları esas alın.

| ID | Sol | Sağ | Birleşim | Hex |
|---:|---|---|---|---|
| 256 | "�" | "�" | "ı" | c4 b1 |
| 257 | "�" | "�" | "ü" | c3 bc |
| 258 | "a" | "n" | "an" | 61 6e |
| 259 | "�" | "�" | "ş" | c5 9f |
| 260 | " " | "b" | " b" | 20 62 |
| 261 | "a" | "r" | "ar" | 61 72 |
| 262 | "e" | "r" | "er" | 65 72 |
| 263 | "i" | "n" | "in" | 69 6e |
| 264 | "." | "\n" | ".\n" | 2e 0a |
| 265 | "e" | "n" | "en" | 65 6e |
| 266 | " " | "k" | " k" | 20 6b |
| 267 | "i" | "r" | "ir" | 69 72 |
| 268 | " " | "d" | " d" | 20 64 |
| 269 | "�" | "�" | "ç" | c3 a7 |
| 270 | "ı" | "n" | "ın" | c4 b1 6e |
| 271 | " " | "g" | " g" | 20 67 |
| 272 | "a" | "k" | "ak" | 61 6b |
| 273 | "a" | "l" | "al" | 61 6c |
| 274 | " " | "y" | " y" | 20 79 |
| 275 | "�" | "�" | "ö" | c3 b6 |

Sık Türkçe harflerin UTF-8 byte çiftleri ile kısa, tekrar eden harf grupları ilk sıralarda görünür. Öğrenilen parçalar dilbilgisel ek veya hece olmak zorunda değildir; sıklık ve önceki birleşimler belirleyicidir.

## 3. Encode ve decode

Örnek: İstanbul'da çığ, öğle ışığı ve üzüm var. 🧠🤖🙂

İki eğitilmiş tokenizer için roundtrip: `True`. Regex token ID'leri: `[454, 331, 258, 98, 378, 501, 97, 316, 256, 301, 44, 355, 301, 326, 32, 344, 256, 301, 256, 369, 32, 349, 363, 409, 46, 32, 240, 159, 167, 160, 240, 159, 164, 150, 240, 159, 153, 130]`.

Encode yeni metinde çiftleri yeniden eğitmez: kayıtlı birleşim sırasındaki en erken uygulanabilir çifti seçer. Decode tüm token byte'larını birleştirir, sonra UTF-8 çözer. Her token'ı ayrı ayrı metne çevirmek Türkçe harfleri ve emojiyi bozabilir.

Tek başına `[255]` geçersiz UTF-8: varsayılan decode sonucu `�`, strict hata türü `UnicodeDecodeError`. Üretim sırasında keyfi token dizileri geçersiz byte dizileri oluşturabilir. Gösterimde `errors='replace'` okunabilir çıktı sağlar; veri doğrulamada `errors='strict'` hatayı görünür kılar. Roundtrip garantisi geçerli girdinin encode çıktısı içindir.

## 4. GPT-4 / cl100k_base regex

```python
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
```

Bu desen güncel tiktoken kaynak kodundaki cl100k_base desenidir; dersteki desenin optimize edilmiş eşdeğeridir. Regex her iki aşamada da kullanılır: hem eğitimde hem encode sırasında parçaların sınırlarını aşan birleşim yapılmaz.

- `Ankara'dan` → `["Ankara", "'d", "an"]`
- `İstanbul'da` → `["İstanbul", "'d", "a"]`
- `Ankara’dan` → `["Ankara", "’dan"]`
- `İstanbul’da` → `["İstanbul", "’da"]`
- `1234567890` → `["123", "456", "789", "0"]`

ASCII kesme işaretinden sonraki İngilizce `'d` kısaltması Türkçe `'dan` ve `'da` eklerinin başını yakalar: bunlar kök ve Türkçe ek sınırlarına uygun bir çözümleme değildir. Unicode sağ kesme işareti `’` aynı İngilizce kısaltma kuralına girmez. Sayılar en çok üç basamaklı regex parçalarına bölünür; BPE bunları daha da küçük token'lara ayırabilir. Desen sözcük/anlam çözümleyicisi değildir.

İlk 20 sırada aynı byte dizisini üreten birleşim sayısı: 5/20. Tüm öğrenilmiş birleşimlerde ortak üretilen byte dizisi: 138. Regex sınırları aday çiftleri kısıtladığı için sonraki sıralar ve sıkıştırma değişir. Her iki modelin vocab'ı 256 temel byte'ı da içerir.

## 5. Türkçe ve İngilizce

Anlamca eşleştirilmiş, bu deney için yazılmış iki paragraf:

**Türkçe:** Bu sabah Ankara'dan İstanbul'a trenle geldim. İstasyonda arkadaşlarımla buluştum; birlikte sahilde yürüyüp küçük bir kafede çay içtik. Yolculuk boyunca okuduğum kitabı ve gelecek hafta yapacağımız çalışmaları konuştuk. Akşam eve döndüğümde notlarımı düzenledim ve öğrendiklerimi ailemle paylaştım.

**İngilizce:** This morning I came from Ankara to Istanbul by train. I met my friends at the station; together we walked along the shore and drank tea in a small cafe. We talked about the book I had read during the journey and the work we would do next week. When I returned home in the evening, I organized my notes and shared what I had learned with my family.

Kelime sayısı `len(text.split())`: boşlukla ayrılan birimler; noktalama birimin içinde kalır. Bu seçim diller arasında kusursuz dilbilimsel eşitleme sağlamaz.

| Metin | Tokenizer | Kelime | Token | Token/kelime |
|---|---|---:|---:|---:|
| tr | cl100k_base_tr | 36 | 122 | 3.3889 |
| en | cl100k_base_en | 69 | 75 | 1.0870 |
| tr | custom_basic_tr | 36 | 169 | 4.6944 |
| tr | custom_regex_tr | 36 | 167 | 4.6389 |

Türkçede ekler bir sözcükte çok sayıda anlam birimi biriktirir. Bu, farklı sözcük biçimleri ve seyrek uzun birleşimler yaratır; BPE bunları birkaç parçaya ayırabilir. Eklemeli yapı tek başına token sayısını belirlemez: eğitim verisinin dil dağılımı, byte kodlaması, vocab büyüklüğü ve metin de etkilidir. Küçük özel tokenizer ile cl100k_base vocab bütçeleri 512 ve 100256 olağan token'dır; bu tablo yöntemlerin adil bir kalite sıralaması değildir. Tek paragraf dil genelini temsil etmez; varsayılan Week 8 derlemi de şiir ağırlıklı edebi metindir.

Eklerin ölçülen ayrışması (cl100k_base; parçalar tek başına UTF-8 olmak zorunda değil):

- `ev` → `[5230]`; byte parçaları: `['65 76']`
- `evler` → `[5230, 1565]`; byte parçaları: `['65 76', '6c 65 72']`
- `evlerimiz` → `[5230, 1565, 318, 450]`; byte parçaları: `['65 76', '6c 65 72', '69 6d', '69 7a']`
- `evlerimizden` → `[5230, 1565, 318, 450, 5294]`; byte parçaları: `['65 76', '6c 65 72', '69 6d', '69 7a', '64 65 6e']`

## 6a. cl100k_base ile birebir eşleşme

Tiktoken'dan eğitilmiş byte sıraları yüklenir; gerçek encode/decode işlemi bizim Python kodumuzdadır. Birleşim çiftleri vocab sıralarından geri çıkarılır; ilk 256 byte'ın token ID permütasyonu korunur. GPT-4'ün tüm eğitim korpusu yeniden oluşturulmaz; burada hazır tokenizer sözlüğü kullanılır.

114 örnek ve deterministik rastgele girdide tüm token ID'leri tiktoken.encode_ordinary ile eşit: `True`. Decode roundtrip eşitliği: `True`.

Örnek: `hello world!!!? (안녕하세요!) lol123 😉`

ID'ler: `[15339, 1917, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 28509, 4513, 57037]`

Özel token yazımları burada olağan metin olarak kodlanır; `allowed_special` ile kontrol token'ı işleme bu deneyin kapsamı dışındadır. cl100k_base, ödevde istenen GPT-4 ailesinin tarihsel tokenizer'ıdır; tüm güncel ChatGPT modellerinin tokenizer'ı olduğu iddia edilmez.

Tokenizer dil modelinden önce ayrı eğitilir ve model eğitimi boyunca sabit kalır. Tokenizer değişince embedding ve çıkış tablosunun boyutu, bağlamdaki metin miktarı ve token başına tahmin görevi değişir. Bu nedenle BPE modelinin nats/token val loss'u geçen haftaki nats/karakter loss ile doğrudan karşılaştırılamaz. Aynı tutulmuş metinde toplam NLL / UTF-8 byte / ln(2) gibi bits/byte ölçüsü ortak birim sağlar; veri ve değerlendirme koşulları da eşleşmelidir. Burada 6a tamamlandığı için ek GPU model eğitimi yapılmadı.

Heceleme, basamak işlemleri ve bazı yabancı dil güçlüklerine token sınırları katkıda bulunabilir; bütün model hatalarını yalnız tokenizer'a bağlamak doğru değildir.

## Yeniden üretim ve sınırlar

Python 3.12.10; regex 2026.9.29; tiktoken 0.14.0. Hedef vocab 512. Eğitilmiş sözlükler `outputs/` altında yerel kalır; özel derlem kopyalanmaz. Yeni çalıştırmalar donanım/sürüm farkıyla farklı süreler verebilir.

- [Karpathy: tokenizer dersi](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Ders notebook'u](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [minbpe alıştırması](https://github.com/karpathy/minbpe/blob/master/exercise.md)
- [minbpe: birleşimleri geri çıkarma referansı](https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py)
- [tiktoken: cl100k_base deseni ve vocab kaynağı](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
- [Tiktokenizer görselleştiricisi](https://tiktokenizer.vercel.app)
