# GLM-OCR — Norwegian Nynorsk Fine-tune

Fine-tuning [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) to read Norwegian Nynorsk from newspaper scans, including historical archives.

## Motivation

GLM-OCR is a strong general-purpose OCR model, but Norwegian Nynorsk is a low-resource written standard with distinctive orthography. The goal here is to adapt the model specifically to newspaper layout and Nynorsk vocabulary so it handles archival scans — such as *Sogningen* from the 1950s — more accurately than baseline GLM-OCR or Tesseract.

## Repository layout

```
generate-newspaper-like.py     # Synthetic training data generator
les-avis-zai.ipynb             # Inference notebook (deskew + run GLM-OCR)

corpus/                        # Raw source text downloaded from HuggingFace
  wikipedia-nno.parquet        # Norwegian Nynorsk Wikipedia
  README.md                    # Notes on data sources and corpus quality

finetune-data/                 # All training data consumed by LLaMA-Factory
  synthetic/                   # Machine-generated fake newspaper images + annotations
    images/
    train.json
  real/                        # Real newspaper scans with ground truth (future)
  dataset_info.json            # LLaMA-Factory dataset registry

LLaMA-Factory/                 # Fine-tuning framework (submodule)

evaluation/                    # Evaluation scans and Tesseract baseline outputs
```

## Pipeline overview

```
corpus/wikipedia-nno.parquet
        │
        ▼
generate-newspaper-like.py  ──►  finetune-data/synthetic/
                                          │
                                          ▼
                              LLaMA-Factory  (reads finetune-data/)
                                          │
                                          ▼
                                  fine-tuned GLM-OCR
```

## Synthetic data generation

`generate-newspaper-like.py` creates fake newspaper images from Nynorsk Wikipedia text:

1. **Text source** — loads `corpus/wikipedia-nno.parquet`, extracts article titles and sentences.
2. **Layout** — renders randomized 6-column newspaper pages as HTML (serif fonts, justified text, bylines, masthead) using Playwright and screenshots them at 1280 × 1800 px.
3. **Degradation** — optionally converts to greyscale, adds mild Gaussian noise and slight blur, and saves as JPEG with realistic compression quality (85–97).
4. **Output** — writes `finetune-data/synthetic/train.json` in ShareGPT format:
   ```json
   {
     "messages": [
       {"role": "user",      "content": "<image>Text Recognition:"},
       {"role": "assistant", "content": "<ground truth text>"}
     ],
     "images": ["images/news_00000.jpg"]
   }
   ```

Run it with:
```bash
python generate-newspaper-like.py   # generates 5 samples by default
```

Adjust `num_samples` inside `generate_dataset()` as needed.

### Dependencies

```bash
pip install playwright opencv-python numpy pandas pyarrow
playwright install chromium
```

## Fine-tuning with LLaMA-Factory

Fine-tuning uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with the YAML configs in `LLaMA-Factory/examples/finetune/`. Set `dataset_dir` in the config to point at `../finetune-data` so LLaMA-Factory reads from this repo's data folder rather than its own internal `data/` directory.

Two training approaches are provided:

| Method | Min VRAM | Learning rate | Notes |
|--------|----------|---------------|-------|
| Full SFT | 24 GB | 1e-5 | Vision tower and projector frozen |
| LoRA | 8 GB | 1e-4 | Applies low-rank adapters to linear layers |

Both use cosine scheduling, warmup ratio 0.1, and BF16 precision.

## Inference

`les-avis-zai.ipynb` shows the full inference pipeline:

1. **Deskew** — detects page skew with HoughLines and corrects it with an affine rotation.
2. **Run GLM-OCR** — loads `zai-org/GLM-OCR` via `transformers` and prompts it to extract all Nynorsk text faithfully.
3. **Decode** — generates up to 8192 tokens with `repetition_penalty=1.3`.

A local HTTP server is expected to serve the images (e.g. `python -m http.server 8080` from `evaluation/`).

### Dependencies

```bash
pip install transformers torch pillow opencv-python requests
```

## Evaluation data

`evaluation/` contains real scans used to benchmark the model:

| File | Description |
|------|-------------|
| `sogningen-1957-04-05-page-1-1.jpg` | *Sogningen* front page, 5 April 1957 |
| `2026-03-18T1711_NB_generated-1.jpg` | NB-style generated page, March 2026 |
| `2025-02-01T1458_NB_generated_00*.jpg` | Earlier generated pages |
| `*-tesseract.txt` | Tesseract 5 baseline output for comparison |

## License

MIT — see [LICENSE](LICENSE).
