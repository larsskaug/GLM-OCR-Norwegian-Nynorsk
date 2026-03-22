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
BODY_FONTS = ["'Times New Roman', serif", "'Georgia', serif", "'Palatino Linotype', serif"]

# Condensed/bold headline fonts — distinct from body, matching historical newspaper style
HEADLINE_FONTS = [
    "'Impact', 'Arial Black', sans-serif",
    "'Arial Black', 'Impact', sans-serif",
    "'Arial Narrow', 'Arial', sans-serif",
]

# Target ground-truth character count per page.
# At 1600x2400 the image costs ~4900 tokens, leaving ~3300 for text within
# the 8192 cutoff. At ~4 chars/token that's ~13k chars max. With 13-15px font
# the page holds ~14-18k chars fully packed; targeting 11k leaves natural
# whitespace (not every page is 100% text) while staying within token budget.
GT_CHAR_TARGET = 11_000
GT_CHAR_JITTER = 2_000   # ± random jitter so lengths vary naturally

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
    """Returns a random article's title and a medium sample of its sentences (8–20)."""
    if not articles:
        return "Tittel", "Data mangler."
    article = random.choice(articles)
    sentences = article["sentences"]
    k = min(random.randint(8, 20), len(sentences))
    body = " ".join(random.choices(sentences, k=k))
    return article["title"], body


def get_long_article() -> tuple[str, str]:
    """Returns a title and a large body (30–70 sentences) from one of the longest articles."""
    if not long_article_pool:
        return get_article()
    article = random.choice(long_article_pool)
    sentences = article["sentences"]
    k = min(random.randint(30, 70), len(sentences))
    body = " ".join(sentences[:k])
    return article["title"], body


# --- 3. TEMPLATE GENERATOR ---
def generate_newspaper_page() -> tuple[str, str]:
    body_font = random.choice(BODY_FONTS)
    headline_font = random.choice(HEADLINE_FONTS)
    col_count = 3

    bg_light = random.randint(88, 96)
    bg_color = f"hsl(0, 0%, {bg_light}%)"

    ink_lightness = random.randint(8, 22)
    ink_color = f"hsl(0, 0%, {ink_lightness}%)"

    font_size = random.randint(13, 15)

    ground_truth: list[str] = []
    html_parts: list[str] = []

    # Optional Masthead: Simulates the difference between a top quadrant and a bottom quadrant
    if random.random() < 0.35:
        masthead = "Norsk Tidende"
        html_parts.append(
            f"<h1 style='text-align: center; border-bottom: 2px solid {ink_color}; "
            f"padding-bottom: 10px; font-family: {headline_font}; text-transform: uppercase; "
            f"font-size: 28px; letter-spacing: 2px; margin-top: 0;'>"
            f"{masthead}</h1>"
        )
        ground_truth.append(masthead)

    html_parts.append(f"<div class='newspaper-layout' style='column-count: {col_count};'>")

    target_chars = GT_CHAR_TARGET + random.randint(-GT_CHAR_JITTER, GT_CHAR_JITTER)
    gt_chars = sum(len(s) for s in ground_truth)
    long_prob = 0.20

    while gt_chars < target_chars:
        if random.random() < long_prob:
            title, body = get_long_article()
        else:
            title, body = get_article()

        headline = title.upper()
        html_parts.append(
            f"<h2 style='font-size: {random.randint(16, 22)}px; margin-top: 14px; "
            f"margin-bottom: 4px; column-span: none; hyphens: auto; "
            f"-webkit-hyphens: auto; overflow-wrap: break-word; "
            f"font-family: {headline_font}; font-weight: 900; letter-spacing: -0.5px;'>"
            f"{headline}</h2>"
        )
        ground_truth.append(headline)

        byline = f"Av {random.choice(['Ola Nordmann', 'Kari Nilsen', 'Ivar Aasen', 'Hulda Garborg'])}"
        html_parts.append(f"<p style='font-style: italic; margin: 2px 0 6px; font-size: {font_size - 1}px;'>{byline}</p>")
        ground_truth.append(byline)

        html_parts.append(f"<p style='text-align: justify; margin: 0 0 8px; letter-spacing: -0.2px;'>{body}</p>")
        ground_truth.append(body)

        html_parts.append("<hr style='margin: 10px 0; border: none; border-top: 1px solid #888;'>")

        gt_chars += len(headline) + len(byline) + len(body)

    html_parts.append("</div>")

    # CSS Changes: Width set to 1200px (half of 2400px). Margins/Padding tightened.
    html_content = f"""
    <html lang="nn">
    <head>
    <style>
        body {{ 
            width: 1200px; 
            background: {bg_color}; 
            padding: 12px; 
            margin: 0; 
            color: {ink_color}; 
        }}
        .newspaper-layout {{
            column-gap: 16px;
            column-rule: 1px solid #888;
            font-family: {body_font};
            font-size: {font_size}px;
            line-height: 1.35;
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
def _paper_grain(shape: tuple, scale: int = 4) -> np.ndarray:
    """Generate low-frequency paper grain by upscaling small random noise.

    scale controls the spatial frequency: higher = coarser grain / blotches.
    """
    h, w = shape[:2]
    small = np.random.normal(0, 1, (h // scale + 1, w // scale + 1)).astype(np.float32)
    grain = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return grain


def _vignette(shape: tuple, strength: float) -> np.ndarray:
    """Build a radial darkening mask (1 = centre, 1-strength = corners)."""
    h, w = shape[:2]
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    # Normalise distance to [0, 1] across the diagonal half-length
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Smoothstep-ish fall-off
    mask = 1.0 - strength * np.clip(dist - 0.4, 0, 1) / 0.6
    return mask.astype(np.float32)


def degrade_image(img: np.ndarray) -> np.ndarray:
    """Simulate aged-newspaper scan degradation.

    Pipeline (all parameters randomised per image):
      1. Convert to grayscale (scans are mono; keeps file size small).
      2. Apply paper grain (coarse low-frequency texture).
      3. Apply pixel-level noise (scanner grain / film grain).
      4. Apply Gaussian blur (optical softness of scanner lens + aged paper).
      5. Apply vignette (scanner light fall-off toward edges).
      6. Clip and return uint8.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. Paper grain — subtle low-frequency texture, intensity 1–4
    grain_intensity = random.uniform(1.0, 4.0)
    grain_scale = random.choice([3, 4, 5])
    gray += _paper_grain(gray.shape, scale=grain_scale) * grain_intensity

    # 3. Pixel-level noise — light scanner noise, intensity 1–5
    noise_intensity = random.uniform(1.0, 5.0)
    gray += np.random.normal(0, noise_intensity, gray.shape).astype(np.float32)

    # 4. Gaussian blur — very slight softening; σ 0.2–0.5
    sigma = random.uniform(0.2, 0.5)
    ksize = 3
    gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # 5. Vignette — subtle edge darkening, strength 0.02–0.08
    if random.random() > 0.5:
        v_strength = random.uniform(0.02, 0.08)
        gray *= _vignette(gray.shape, v_strength)

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


# --- 5. WORKER ---
PAGE_RECYCLE_INTERVAL = 500

def _worker(worker_id: int, indices: list[int], output_dir: str, images_dir: str, seed: int | None, dataset_name: str = "glm_ocr_nynorsk") -> None:
    if seed is not None:
        worker_seed = seed + worker_id * 997
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    records = []

    with sync_playwright() as p:
        browser = p.chromium.launch()

        def new_page():
            # Set a smaller initial height so scrollHeight isn't artificially inflated
            return browser.new_page(viewport={"width": 1200, "height": 800})

        page = new_page()

        for count, i in enumerate(indices):
            if count > 0 and count % PAGE_RECYCLE_INTERVAL == 0:
                page.close()
                page = new_page()

            img_filename = f"news_{i:05d}.jpg"
            img_path = os.path.join(images_dir, img_filename)

            html, gt_text = generate_newspaper_page()
            page.set_content(html)

            # THE FIX: Tell Playwright to screenshot the body element directly.
            # This perfectly wraps the exact dimensions of the text block, 
            # cutting off the blank background but keeping every character intact.
            screenshot_bytes = page.locator("body").screenshot()
            
            img_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = degrade_image(img)

            quality = random.randint(88, 97)
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
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic newspaper training data.")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to generate (default: 5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    generate_dataset(num_samples=args.num_samples, seed=args.seed)
