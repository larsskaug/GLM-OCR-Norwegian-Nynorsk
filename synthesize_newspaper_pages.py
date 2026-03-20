import json
import multiprocessing as mp
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

# --- 1. CONFIGURATION ---
FONTS = ["'Times New Roman', serif", "'Georgia', serif", "'Palatino Linotype', serif"]

# Populated by load_corpus() in the main process before workers are forked.
# Workers inherit these via Linux copy-on-write fork — no redundant loading.
articles: list = []
long_article_pool: list = []


def load_corpus(path: str = "corpus/wikipedia-nno.parquet") -> None:
    """Loads the Wikipedia corpus into module-level article lists."""
    global articles, long_article_pool

    print("Loading Wikipedia Parquet file...")
    try:
        df = pd.read_parquet(path)
        wiki_texts = df["text"].dropna().astype(str).tolist()
        print(f"Loaded {len(wiki_texts)} raw texts.")
    except Exception as e:
        print(f"Error loading Parquet: {e}")
        wiki_texts = ["Dette er ein reservetekst fordi fila ikkje vart funnen."]

    # re.search (not re.match) handles leading whitespace / infobox prefixes.
    for text in wiki_texts[:50_000]:
        m = re.search(r"'''(.+?)'''", text)
        if not m:
            continue
        title = m.group(1)
        sentences = [s.strip() + "." for s in text.split(".") if len(s.strip()) > 25]
        if sentences:
            articles.append({"title": title, "sentences": sentences})

    articles_by_length = sorted(articles, key=lambda a: len(a["sentences"]), reverse=True)
    long_article_pool = articles_by_length[:500]

    print(f"Parsed {len(articles)} articles with titles and body text.")


# --- 2. ARTICLE SAMPLERS ---
def get_article() -> tuple[str, str]:
    """Returns a random article's title and a short sample of its sentences (3–8)."""
    if not articles:
        return "Tittel", "Data mangler."
    article = random.choice(articles)
    sentences = article["sentences"]
    body = " ".join(random.choices(sentences, k=min(random.randint(3, 8), len(sentences))))
    return article["title"], body


def get_long_article() -> tuple[str, str]:
    """Returns a title and a large body (20–60 sentences) from one of the longest articles."""
    if not long_article_pool:
        return get_article()
    article = random.choice(long_article_pool)
    sentences = article["sentences"]
    k = min(random.randint(20, 60), len(sentences))
    body = " ".join(sentences[:k])
    return article["title"], body


# --- 3. TEMPLATE GENERATOR ---
def generate_newspaper_page() -> tuple[str, str]:
    """Generates randomized HTML and the exact reading-order ground truth.

    Ground truth note: CSS `hyphens: auto` causes the browser to visually break
    long Norwegian compound words with a soft hyphen. The ground truth intentionally
    contains the un-hyphenated words because this dataset trains semantic content
    extraction (what does the article say?), not pixel-level glyph transcription.
    If you later need exact glyph-level OCR, remove `hyphens: auto` from h2 styling.
    """
    font = random.choice(FONTS)
    col_count = 6

    ground_truth: list[str] = []
    html_parts: list[str] = []

    masthead = "Norsk Tidende"
    html_parts.append(
        f"<h1 style='text-align: center; border-bottom: 2px solid black; "
        f"padding-bottom: 10px; font-family: sans-serif; text-transform: uppercase;'>"
        f"{masthead}</h1>"
    )
    ground_truth.append(masthead)

    html_parts.append(f"<div class='newspaper-layout' style='column-count: {col_count};'>")

    num_long = random.choice([0, 0, 0, 1, 1, 2])
    num_short = random.randint(4, 8)
    article_sources = (
        [get_long_article() for _ in range(num_long)]
        + [get_article() for _ in range(num_short)]
    )
    random.shuffle(article_sources)

    for title, body in article_sources:
        headline = title.upper()
        html_parts.append(
            f"<h2 style='font-size: {random.randint(14, 20)}px; margin-top: 14px; "
            f"margin-bottom: 4px; column-span: none; hyphens: auto; "
            f"-webkit-hyphens: auto; overflow-wrap: break-word;'>{headline}</h2>"
        )
        ground_truth.append(headline)

        byline = f"Av {random.choice(['Ola Nordmann', 'Kari Nilsen', 'Ivar Aasen', 'Hulda Garborg'])}"
        html_parts.append(f"<p style='font-style: italic; margin: 2px 0 6px;'>{byline}</p>")
        ground_truth.append(byline)

        html_parts.append(f"<p style='text-align: justify; margin: 0 0 8px;'>{body}</p>")
        ground_truth.append(body)

        html_parts.append("<hr style='margin: 10px 0; border: none; border-top: 1px solid #555;'>")

    html_parts.append("</div>")

    html_content = f"""
    <html lang="nn">
    <head>
    <style>
        body {{ width: 1200px; min-height: 1600px; background: #fdfdfc; padding: 40px; color: #111; }}
        .newspaper-layout {{
            column-gap: 20px;
            column-rule: 1px solid #333;
            font-family: {font};
            font-size: {random.randint(11, 14)}px;
            line-height: 1.4;
        }}
    </style>
    </head>
    <body>
        {"".join(html_parts)}
    </body>
    </html>
    """

    return html_content, "\n\n".join(ground_truth)


# --- 4. DEGRADATION ENGINE ---
def degrade_image(img: np.ndarray) -> np.ndarray:
    """Applies very mild noise and compression artifacts in memory (no disk I/O)."""
    if random.random() > 0.3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    noise_intensity = random.uniform(1, 4)
    noise = np.random.normal(0, noise_intensity, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() > 0.4:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.7))

    return img


# --- 5. WORKER ---
PAGE_RECYCLE_INTERVAL = 500


def _worker(worker_id: int, indices: list[int], output_dir: str, images_dir: str, seed: int | None, dataset_name: str = "glm_ocr_nynorsk") -> None:
    """Runs in a forked child process. Generates samples for the given indices.

    Writes results to a partial JSON file; the main process merges these.
    The corpus globals (articles, long_article_pool) are inherited via fork CoW.
    Each worker gets a distinct RNG seed so outputs are not identical across workers.
    """
    if seed is not None:
        worker_seed = seed + worker_id * 997
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    records = []

    with sync_playwright() as p:
        browser = p.chromium.launch()

        def new_page():
            return browser.new_page(viewport={"width": 1280, "height": 1800})

        page = new_page()

        for count, i in enumerate(indices):
            if count > 0 and count % PAGE_RECYCLE_INTERVAL == 0:
                page.close()
                page = new_page()

            img_filename = f"news_{i:05d}.jpg"
            img_path = os.path.join(images_dir, img_filename)

            html, gt_text = generate_newspaper_page()
            page.set_content(html)

            screenshot_bytes = page.screenshot(full_page=True)
            img_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = degrade_image(img)

            quality = random.randint(85, 97)
            cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            records.append({
                "messages": [
                    {"role": "user", "content": "<image>Text Recognition:"},
                    {"role": "assistant", "content": gt_text},
                ],
                "images": [f"{dataset_name}/images/{img_filename}"],
            })

            if (count + 1) % 100 == 0:
                print(f"[worker {worker_id}] {count + 1}/{len(indices)}", flush=True)

        page.close()
        browser.close()

    partial_path = os.path.join(output_dir, f"_partial_{worker_id:04d}.json")
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"[worker {worker_id}] done ({len(indices)} samples)", flush=True)


# --- 6. ORCHESTRATOR ---
def generate_dataset(
    num_samples: int = 5,
    output_dir: str = "finetune-data/synthetic",
    seed: int | None = None,
    num_workers: int | None = None,
    dataset_name: str = "glm_ocr_nynorsk",
) -> None:
    """Generate `num_samples` synthetic newspaper images with ground-truth text.

    Args:
        num_samples: Total number of images to generate.
        output_dir: Root directory for output files.
        seed: Optional RNG seed. Each worker derives its own seed from this base
              so outputs are distinct. Record this value to reproduce a crashed run.
        num_workers: Parallel Chromium workers. Defaults to half the CPU count,
                     which leaves headroom for the OS and other processes.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) // 2)

    # Load corpus in main process BEFORE forking — children inherit via CoW.
    load_corpus()

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    effective_seed = seed if seed is not None else "random"
    print(f"Generating {num_samples} samples with {num_workers} workers (seed={effective_seed})...")

    # Distribute indices evenly across workers (chunked, not interleaved,
    # so each worker's Playwright page sees a contiguous naming range).
    chunk, remainder = divmod(num_samples, num_workers)
    worker_indices: list[list[int]] = []
    start = 0
    for w in range(num_workers):
        end = start + chunk + (1 if w < remainder else 0)
        worker_indices.append(list(range(start, end)))
        start = end

    # Fork workers. Using "fork" (Linux default) so children inherit the corpus
    # globals without pickling or re-loading them.
    ctx = mp.get_context("fork")
    processes = [
        ctx.Process(
            target=_worker,
            args=(w, worker_indices[w], output_dir, images_dir, seed, dataset_name),
            name=f"synth-worker-{w}",
        )
        for w in range(num_workers)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    failed = [w for w, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"WARNING: workers {failed} exited with errors — their samples may be missing.")

    # Merge partial JSONs in index order, then clean up.
    all_records: list[dict] = []
    for w in range(num_workers):
        partial_path = os.path.join(output_dir, f"_partial_{w:04d}.json")
        if os.path.exists(partial_path):
            with open(partial_path, encoding="utf-8") as f:
                all_records.extend(json.load(f))
            os.remove(partial_path)

    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Done! {len(all_records)} samples saved to {output_dir}/")


if __name__ == "__main__":
    generate_dataset(num_samples=5)
