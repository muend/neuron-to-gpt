# User-selected Turkish corpus: Nazim Hikmet

The owner's three provided PDFs have been converted into private TXT sources under `nazim_sources/cleaned/`. The compiled candidate has 2,072,197 UTF-8 bytes, 1,904,245 characters and 118 distinct characters. This exceeds the approximate 1 MB assignment target because the owner requested a broad collection. No Turkish full texts were fetched from author websites, and no Turkish GPU results are available yet.

The poetry PDF needed Turkish OCR recovery for 243 invalid-character pages. Stories and Aydinlik prose were re-read with Windows Turkish OCR. The final sources contain no replacement/control characters, but valid-letter OCR substitutions may remain. Poetry/header reconstruction was spot-checked against rendered pages, not exhaustively proofread. An exact 64-word cross-source overlap audit found zero shared windows; it does not establish absence of OCR-variant or shorter repetitions.

Use poems, epics, theatre and prose selected by the dataset owner. Preserve original verse lines and stanza boundaries. Do not repeat texts to reach the assignment's approximately 1 MB target. Record the actual size and remaining shortfall instead.

1. Place UTF-8 `.txt` files in `week08/data/nazim_sources/`.
2. Copy `sources.example.json` into that folder as `sources.json`.
3. Add one entry per file with its title, genre, source reference and actual permission basis. The template is not a rights clearance; verify usage permission separately.
4. From the repository root, run `python week08/prepare_data.py --source-dir week08/data/nazim_sources/cleaned` for the prepared collection, or pass your new folder explicitly. `--english-only` prepares Shakespeare without requiring Turkish files. Never mix the earlier unreviewed candidates at the parent folder with the prepared sources.

The compiler normalizes Unicode to NFC and line endings to LF without flattening poetry. It rejects duplicate complete files, unexpected authors and changed cached corpora. Text and manifest checksums identify the exact training inputs.

Raw texts, local provenance manifests and checkpoints are ignored by Git. Rejected earlier automatic selections are kept locally under `excluded/`, outside the active corpus path. They must not appear in submission results or be described as Nazim data.

In Colab, upload `nazim_sources_for_colab.zip` to the `neuron-to-gpt` folder in your own Google Drive, or upload the selected TXT files and completed `sources.json` to its `nazim_sources` subfolder. The notebook prepares Shakespeare independently; the Turkish training cell requires this source collection or a previously validated Nazim corpus backup. Neither the source ZIP nor the full-text inputs enter Git.
