# GLM-OCR — Norwegian Nynorsk Fine-tune

Fine-tuning [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) to read Norwegian Nynorsk from historical newspaper scans, specifically the *Sogns Avis* archive (1946–1957) hosted by the National Library of Norway ([nb.no](https://nb.no)).

## Motivation

Norwegian Nynorsk is a low-resource written standard with distinctive orthography (æ, ø, å). GLM-OCR has no Norwegian in its base weights at all. The goal is to adapt it to newspaper layout and Nynorsk vocabulary so it handles archival scans more accurately than baseline GLM-OCR or Tesseract.

## Source data

Newspaper scans are downloaded from the National Library of Norway's digital archive via `retrieve_data.ipynb`, which automates discovery and high-resolution PDF download from [nb.no](https://nb.no).

The scans are licensed **CC BY-NC-ND** — free to share and copy with attribution, non-commercial use only, no derivatives. This project is for personal/research use.

## Model background

GLM-OCR is a **0.9B-parameter multimodal OCR model** built on the GLM-V / CogViT architecture. It is designed for document *understanding* — not just character recognition — and outputs structured Markdown, JSON, or LaTeX.

| Property | Value |
|---|---|
| OmniDocBench V1.5 | **94.62** (SOTA in its class) |
| Throughput | ~1.86 PDF pages/second |
| Image pixel budget | 12 544 – 9 633 792 total pixels (see [Quadrant splitting](#quadrant-splitting)) |
| Languages | 100+, but no Norwegian in base weights |
| License | Apache-2.0 (open weights) |

The model uses **Multi-Token Prediction (MTP)**: rather than copying glyphs rigidly, it uses surrounding context to infer and self-correct tokens during generation. This is especially valuable on degraded historical scans where individual characters are ambiguous.

## Repository layout

```
generate_training_data.py     # Wrapper: runs both generators below
synthesize_newspaper_pages.py # Synthetic multimodal training data (newspaper images)
generate_text_samples.py      # Text-only Nynorsk completion samples
les-avis-zai.ipynb            # Inference notebook (quadrant split + OCR + stitching)
retrieve_data.ipynb           # nb.no newspaper scraper / downloader
evaluate_single.py            # CLI evaluation script
project.yaml                  # Model paths and training metadata

corpus/                       # Raw source text
  wikipedia-nno.parquet       # Norwegian Nynorsk Wikipedia

finetune-data/                # All training data (symlinked into LLaMA-Factory/data/)
  synthetic/                  # Multimodal: newspaper images + ground truth
    images/
    train.json
  text-only/                  # Text-only: Nynorsk completion pairs
    train.json

scans/                        # Real scans for inference
  raw/                        # Downloaded PDFs from nb.no
  ready-for-ocr/              # Extracted grayscale JPEGs (input to model)
  extract_pages_save_jpeg.py  # PDF → grayscale JPEG extraction

LLaMA-Factory/                # Fine-tuning framework (submodule)
training/config/              # Training YAML configs
evaluation/                   # Evaluation scans and baseline outputs
```

## Pipeline overview

```
                           TRAINING

corpus/wikipedia-nno.parquet
        │
        ├──► synthesize_newspaper_pages.py  ──►  finetune-data/synthetic/
        │                                                │
        └──► generate_text_samples.py       ──►  finetune-data/text-only/
                                                         │
                                                         ▼
                                             LLaMA-Factory  (via symlinks)
                                                         │
                                                         ▼
                                                 fine-tuned GLM-OCR

                          INFERENCE

nb.no  ──►  retrieve_data.ipynb  ──►  scans/raw/*.pdf
                                            │
                                            ▼
                                  extract_pages_save_jpeg.py
                                            │
                                            ▼
                                  scans/ready-for-ocr/*.jpeg
                                            │
                                            ▼
                                  les-avis-zai.ipynb
                                  (quadrant split → OCR → LLM stitch)
```

## Training data generation

```bash
python generate_training_data.py --multimodal 20000 --text 5000
python generate_training_data.py --text-only --text 5000    # regenerate text only
```

### Multimodal samples (synthetic newspaper pages)

`synthesize_newspaper_pages.py` creates fake newspaper images from Nynorsk Wikipedia text:

1. **Text source** — loads `corpus/wikipedia-nno.parquet`, extracts titles and sentences.
2. **Layout** — renders randomised 6-column pages as HTML (separate headline/body fonts, justified text, bylines, masthead, warm cream background, faded ink) using Playwright at 1600 × 2400 px — sized to match real-scan quadrants.
3. **Degradation** — grayscale, coarse paper grain, pixel noise (σ 4–14), Gaussian blur (σ 0.5–1.4), optional vignette, JPEG quality 75–92.
4. **Content budget** — articles are added until ground-truth text reaches ~11 000 characters. At 1600×2400 the image costs ~4 900 tokens, leaving ~3 300 for text within the 8 192-token cutoff.
5. **Output** — writes `finetune-data/synthetic/train.json` in ShareGPT format.

### Text-only samples (Nynorsk completion)

`generate_text_samples.py` builds the LLM backbone's Norwegian language prior without consuming image tokens:

1. **Source** — same Wikipedia nynorsk corpus.
2. **Filtering** — sentences are filtered for prose quality: moderate length (50–200 chars), no junk markup, high Norwegian function-word density or presence of æ/ø/å.
3. **Format** — each sample splits a passage at a sentence boundary; the user prompt asks the model to continue the text, and the assistant provides the continuation.
4. **Rationale** — the vision encoder learns to read glyphs from multimodal samples; text-only samples teach the LLM to *generate* correct Nynorsk independently, strengthening the language prior that resolves ambiguous glyphs during OCR.

### Dependencies

```bash
pip install playwright opencv-python numpy pandas pyarrow pymupdf pillow
playwright install chromium
```

## Quadrant splitting

GLM-OCR's image processor (`Glm46VImageProcessorFast`) dynamically resizes inputs to fit within a **total pixel budget**, not fixed edge lengths. The `smart_resize` function (in `transformers`) rounds dimensions to multiples of 28 (patch_size × merge_size) and then enforces:

| Parameter | Config key | Value | Meaning |
|---|---|---|---|
| min_pixels | `size.shortest_edge` | 12 544 (`112²`) | Images smaller than this are upscaled |
| max_pixels | `size.longest_edge` | 9 633 792 (`28² × 15000`) | Images larger than this are downscaled |

A typical full newspaper page at native scan resolution (e.g. 2862 × 4143 = **11.9M pixels**) exceeds the 9.6M budget and gets **downscaled before the model ever sees it**. This silently destroys fine detail — small body text, column separators, and faded ink all suffer.

Splitting the page into four overlapping quadrants (15% overlap) solves this. Each quadrant (e.g. 1645 × 2381 = **3.9M pixels**) sits at ~41% of the budget and passes through at full resolution. The result is dramatically better OCR quality, especially on dense multi-column layouts.

The notebook (`les-avis-zai.ipynb`) implements this:

1. **Split** — slice the page into 4 quadrants with 15% overlap at the boundaries.
2. **OCR** — run GLM-OCR independently on each quadrant.
3. **Stitch** — feed all four transcriptions to an LLM with instructions to merge them using the overlapping text for alignment, preserving column reading order and eliminating duplicates.

## Scan extraction

`scans/extract_pages_save_jpeg.py` extracts the original embedded images from scanned PDFs using PyMuPDF, converts to grayscale, and saves as JPEG. No rasterisation, no DPI conversion — near-instant.

```bash
cd scans
python extract_pages_save_jpeg.py                          # all PDFs in raw/
python extract_pages_save_jpeg.py scan1.pdf scan2.pdf      # specific files
python extract_pages_save_jpeg.py -o /tmp/out scan.pdf     # custom output dir
```

> **Do not hard-binarize.** GLM-OCR handles grayscale well and MTP uses ink-gradient information to resolve ambiguous strokes. Binarization discards cues the CogViT encoder relies on.

## Fine-tuning with LLaMA-Factory

Fine-tuning uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with the config at `training/config/glm_ocr_nynorsk_full.yaml`.

Training data is symlinked from `finetune-data/` into `LLaMA-Factory/data/` — one source of truth, no copies.

| Method | Min VRAM | Learning rate | Notes |
|--------|----------|---------------|-------|
| Full SFT | 24 GB | 1e-5 | Vision tower and projector frozen, LM only |

Cosine scheduling, warmup ratio 0.1, BF16 precision, cutoff length 8 192.

```bash
cd LLaMA-Factory
llamafactory-cli train ../training/config/glm_ocr_nynorsk_full.yaml
```

## Inference

### Notebook (`les-avis-zai.ipynb`)

1. **Quadrant split** — slices the page into 4 overlapping quadrants (15% overlap) to stay within the model's 9.6M pixel budget.
2. **Run GLM-OCR** — runs inference on each quadrant independently, up to 4 096 tokens per quadrant.
3. **Stitch** — assembles the four transcriptions into a single prompt for an LLM to merge, using overlap text to align and de-duplicate.

### CLI (`evaluate_single.py`)

```bash
python evaluate_single.py scans/ready-for-ocr/mypage.jpg --model finetuned
python evaluate_single.py scans/ready-for-ocr/mypage.jpg --model base
```

Aliases `base` and `finetuned` are resolved from `project.yaml`. Outputs a `.txt` file next to the image.

### Maximising accuracy

| Lever | Recommendation |
|---|---|
| **Quadrant splitting** | Always split pages exceeding ~9.6M pixels; each quadrant should stay well under the budget to avoid silent downscaling |
| **Image quality** | Extract raw embedded JPEGs from the PDF — no re-encoding needed |
| **Skew** | Correct even sub-degree tilt — CogViT is sensitive to it |
| **`repetition_penalty`** | 1.3 default; raise to 1.4–1.5 if the model loops on dense columns |
| **Prompt language** | Name the language explicitly: *"All text is in Norwegian nynorsk"* |
| **Prompt verb** | Use *"Extract all text faithfully"* — avoid "summarise" or "describe" |
| **Post-processing** | A normalisation pass (LLM or rules) fixes occasional wrong Nynorsk forms |

## License

Code: MIT — see [LICENSE](LICENSE).

Newspaper scans: **CC BY-NC-ND** (National Library of Norway). Non-commercial use with attribution, no derivatives.

Model weights: **Apache-2.0** (inherited from GLM-OCR).
