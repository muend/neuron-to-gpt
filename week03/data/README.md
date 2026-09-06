# Veri kaynakları

Bu klasördeki veri dosyaları `python veri_hazirla.py` çalıştırılınca otomatik oluşturulur.
Büyük veri dosyalarını elle kopyalamak yerine kaynak URL'den indirip aynı deneyi tekrar üretilebilir tutuyoruz.

## İngilizce isimler

- Kaynak: Andrej Karpathy — `karpathy/makemore`, `names.txt`
- Repo: https://github.com/karpathy/makemore
- Ham dosya: https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
- Makemore README'sine göre dosya yaklaşık 32 bin yaygın İngilizce isim içerir ve SSA verisinden türetilmiştir.

Oluşan dosya: `names_en.txt`

## Türkçe isimler

- Kaynak: `niyazikemer/turkce_isimler`
- Repo: https://github.com/niyazikemer/turkce_isimler
- Ham dosya: https://raw.githubusercontent.com/niyazikemer/turkce_isimler/main/turkce_isim.csv
- Kaynak repo 14.111 Türkçe isim içerdiğini ve verinin ML projelerinde kullanılmak üzere temizlendiğini belirtiyor.

`veri_hazirla.py`, CSV'deki `name` sütununu okur, Unicode normalizasyonu yapar,
Türkçe `I/İ` küçük harf dönüşümünü açıkça ele alır ve sadece 29 harfli Türkçe
alfabedeki karakterleri tutar. Böylece `ç, ğ, ı, ö, ş, ü` ASCII'ye çevrilmez.

Oluşan dosyalar:

- `turkce_isim.csv` — indirilen ham kaynak
- `names_tr.txt` — model için temizlenmiş, tek isim / satır biçimi
