# Week 10 — GPT-2 (124M)

Bu rapor `run_experiments.py` tarafından yalnız kaydedilmiş çalışma sonuçlarından üretildi. Eksik deneyler tamamlanmış gibi gösterilmez.

## 1. Hazır model ve pozisyon embedding'leri

Model: `openai-community/gpt2`; revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Benzersiz parametre: **124,439,808**. Tüm adlar/boyutlar [weights.tsv](weights.tsv) ve JSON çıktısında bulunur. Token embedding'i ile çıktı ağırlığı aynı parametreyi kullandığından iki kez sayılmaz.

![Öğrenilmiş pozisyon embedding'leri](position_embeddings.png)

Wpe, 1024 konum için öğrenilmiş 768 boyutlu vektörlerdir. Harita ve seçili kanallar konumlar arasında yapı ve farklı değişim ölçekleri gösterir; bu bir hazır sinüs/kosinüs tablosu değildir. Tek kanaldaki şekil, o kanalın belirli bir dilsel kavrama karşılık geldiğini veya modelin 1024 konumun dışına genelleyebildiğini kanıtlamaz.

## 2. Kendi GPT-2 koduma ağırlık yükleme

Aynı girdinin tüm logits'leri, dropout kapalıyken iki modelde FP32/eager attention ile karşılaştırıldı. TF32 kapalı. Tolerans: rtol=1e-4, atol=1e-4.

| Girdi | Boyut | En büyük mutlak fark | Ortalama fark | Eşleşme |
|---|---|---:|---:|---|
| Hello, I am a language model and | [1, 8, 50257] | 6.1035156e-05 | 1.709995e-05 | True |
| Merhaba, ben bir dil modeliyim ve | [1, 11, 50257] | 5.3405762e-05 | 1.4977265e-05 | True |
| A quick check: 123, çığ and 🙂. | [1, 13, 50257] | 0.00012207031 | 1.7259823e-05 | True |

Hafta 8'deki koduma göre farklar:

- 12 blok, 12 head, 768 embedding; 1024 konum; GPT-2 byte-BPE vocab'ı 50.257.
- Head başına ayrı Q/K/V katmanları yerine tüm head'ler için tek `c_attn` projeksiyonu; çıkış üçe ayrılır ve head eksenine taşınır. GPT-2 bu projeksiyonlarda bias kullanır.
- Feedforward aktivasyonu ReLU yerine tanh yaklaşımıyla GELU; ara genişlik yine 4×768.
- Pre-LayerNorm ve iki residual yol korunur. Son LayerNorm'dan sonra bias içermeyen çıktı katmanı vardır.
- Token ve çıktı embedding'leri paylaşılır. Bu, standart GPT-2'nin özelliğidir; ayrı ağırlıklara göre 38,597,376 parametre kazandırır.
- Yerel örnekleme, nedensel attention ve öğrenilmiş mutlak pozisyon embedding'leri aynı temel fikri korur.

Hugging Face'in tarihsel `Conv1D` ağırlıkları (girdi, çıktı), PyTorch `Linear` ağırlıkları (çıktı, girdi) düzenindedir. Bu yüzden attention `c_attn`, attention `c_proj`, MLP `c_fc` ve MLP `c_proj` ağırlıkları yüklenirken transpoze edilir; embedding ve LayerNorm ağırlıkları doğrudan kopyalanır. Kare projeksiyonlarda da transpoz gereklidir: şeklin aynı kalması matrisin aynı olduğu anlamına gelmez.

## 3. Sampling ve Türkçe

Her adımda son konumun logits'i temperature ile ölçeklenir; en yüksek k değer dışındakiler elenir, softmax sonrası multinomial ile tek token seçilir ve bağlama eklenir. Her iki örnek de kendi model kodumdan üretilmiştir; insan yazısı veya eğitim verisinden alıntı olarak sunulmaz.

### en — seed 1337, top-k 50, T=0.8

```text
Hello, I am a language model and design student and I'm here to help you learn to design and design the most efficient way to design or build a website, for a fee. I'm a native English speaker and I like to create my own language model and design courses.<|endoftext|>I have always been fascinated by the idea of being the first person to be
```

### tr — seed 1338, top-k 50, T=0.8

```text
Merhaba, ben bir dil modeliyim vekiye alameh uda' (The children of their fathers are not of the tribe of Israel);

The Children of Israel: O god's Lord, O king of heaven,

And he who is not of the tribe of Israel, shall be cut off from the kingdom of God.


```

| Başlangıç | GPT-2 token | cl100k_base token | Kelime |
|---|---:|---:|---:|
| Hello, I am a language model and | 8 | 8 | 7 |
| Merhaba, ben bir dil modeliyim ve | 11 | 11 | 6 |

GPT-2 ağırlıklı olarak İngilizce WebText ile eğitildi. Türkçe veri ve sık birleşimler daha az temsil edildiğinden Türkçe sözcükler daha çok parçaya bölünebilir; aynı token bağlamı daha az Türkçe metin taşıyabilir. Tokenizer bunu görünür kılar, fakat düşük Türkçe kaliteyi tek başına açıklamaz: modelin dil dağılımı da önemlidir. Geçen hafta cl100k_base için bulunan Türkçe/İngilizce farkıyla aynı ilişki burada GPT-2'nin ayrı, 50.257 öğeli sözlüğünde ölçülür; iki tokenizer'ın sayıları aynı olmak zorunda değildir.

## 5b. Eşleştirilmiş Türkçe ve İngilizce cümlelerin loss'u

Her cümle aynı başlangıç bağlamı olan `<|endoftext|>` ID'sinden sonra puanlandı; metnin ilk token'ı dahil bütün token'lar hesaba katıldı. Karşılaştırma küçük ve kontrolsüzdür; bir dil için evrensel kalite sıralaması değildir.

| Dil | Cümle | Token | Nats/token | Bits/byte |
|---|---|---:|---:|---:|
| en | The children are reading a book in the garden. | 10 | 3.6127 | 1.1330 |
| tr | Çocuklar bahçede bir kitap okuyor. | 16 | 7.0399 | 4.5140 |
| en | I will meet my friends at the station tomorrow morning. | 11 | 3.8240 | 1.1034 |
| tr | Yarın sabah istasyonda arkadaşlarımla buluşacağım. | 27 | 5.4191 | 3.7694 |
| en | We talked about the things we learned during our journey. | 11 | 3.2495 | 0.9047 |
| tr | Yolculuğumuz sırasında öğrendiklerimiz hakkında konuştuk. | 31 | 5.6917 | 3.9774 |

Token başına loss hem dil dağılımından hem parçalama biçiminden etkilenir. Bits/byte, toplam NLL'yi metnin UTF-8 byte sayısına ve ln(2)'ye böler; ortak birim sağlar, ama farklı cümlelerin bilgi içeriğini ve çeviri tercihlerinin etkisini ortadan kaldırmaz. Sonuçların tamamı tabloda gösterilir; Türkçe her cümlede mutlaka daha yüksek loss vermek zorunda değildir.

## 4. Tek batch eğitimi: bekliyor

Bu klasörde `overfit_metrics.json` bulunmuyor. Rastgele 124M modelin ilk loss'u ve birkaç yüz update ezberleme sonucu henüz ölçülmedi; Colab hücresini çalıştırın.

## Kaynaklar

- [Karpathy: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [GPT-2 ağırlıkları ve model kartı](https://huggingface.co/openai-community/gpt2)
- [GPT-2 makalesi](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face GPT-2 uygulaması](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
