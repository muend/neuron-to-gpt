# Week 8 measured results

Final losses are sampled NLL in nats per character.

| Run | Status | Parameters | Steps | Final val NLL | Best val NLL | Training seconds | Hardware |
|---|---|---:|---:|---:|---:|---:|---|
| en_a100_layernorm | measured | 4,833,345 | 5000 | 1.4723 | 1.4657 | 435.6 | NVIDIA A100-SXM4-40GB |
| tr_a100_layernorm | measured | 4,860,534 | 5000 | 1.6239 | 1.6239 | 436.5 | NVIDIA A100-SXM4-40GB |

## Data

```json
{
  "en": {
    "sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    "bytes": 1115394,
    "characters": 1115394,
    "vocabulary": 65,
    "split_sizes": {
      "train": 1003854,
      "val": 111540
    }
  },
  "tr": {
    "sha256": "95eed4275502beda5e18590c72abd2e9c301f35a975bc6aadd12ef0942ed3e5a",
    "bytes": 2072197,
    "characters": 1904245,
    "vocabulary": 118,
    "split_sizes": {
      "train": 1713820,
      "val": 190425
    }
  }
}
```

## Generated text (best validation checkpoints)

### en_a100_layernorm (measured)

```text


KING RICHARD II:
Shall I, be made to be exceeding?

QUEEN ELIZABETH:
My hard will I be?

KING RICHARD II:
Ay, after thy head,
You shall find you to the wind,
While enter content Hereford to and him provided;
Now well and peace the male with a crown.

QUEEN ELIZABETH:
Bring this I was thy world that mortal king,
Cominius, and do rive a guilty stood more.
The heart my brother son; and therefore you come,
To find our good, and be accursed in a hopely common
And distinion are no vex to change the crown;
And being my service and myself and several;
Therefore lies Romeo's blood finger to one.

JULI
```

### tr_a100_layernorm (measured)

```text

düşmeden elimizin sesi :
"- Yalım,
ben ne olsün benim için?
Dört her yerinde gelip bir hiç yok,
böyle alıp hanımeliydi . ."
Baktı memleketler :
"- Belki yeryüzünde on yıl var mı?"
Sesleri gözleri mi insanın cebinde . ." .
"- Şükrün perde zavallı kavgasını söylersin.
Öyle mi?"
"- Işte yapacakun benim
ben sen, iki .. "
(Belki dedi.
Eczacı Hikmet Halil Çavuş?"

Memleket Hasan Kucaklıgın üstünden.
Kocaman büyük bir zihnim.
'- Evet," dedi.
Geldi, oglum anamdan.
Dört beş gün yaşında olacaksın. radyomunu gibi çıktı.
Belki kucagı ilaç etti ki :
"- Ben memleketler, başka birbirimiz .. . "

Hepsi havayı
```

## Interpretation limits

Do not rank English vs Turkish losses as language quality: vocabularies, character distributions, genres, cleaning and held-out segments differ. Same-language ablations use fixed windows. Smoke runs verify wiring only; they are not final training evidence. Elapsed time is recorded separately in each result.json; training seconds exclude validation and I/O.
