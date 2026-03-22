"""
generate_text_samples.py — Generate text-only completion samples from Wikipedia Nynorsk.

Produces training data that teaches the LLM backbone to generate Norwegian Nynorsk
vocabulary and grammar, complementing the multimodal OCR training data. Each sample
splits a Wikipedia passage at a natural midpoint: the first half is the prompt,
the second half is the completion target.

Usage:
    python generate_text_samples.py                    # 3000 samples (default)
    python generate_text_samples.py --num-samples 5000
    python generate_text_samples.py --seed 42
"""

import argparse
import json
import random
import re

import pandas as pd

CORPUS_PATH = "corpus/wikipedia-nno.parquet"
OUTPUT_DIR = "finetune-data/text-only"

# Target character range for each sample (prompt + completion combined).
# Keeps samples comparable in length to the multimodal GT (~5k-11k chars).
MIN_CHARS = 1_000
MAX_CHARS = 8_000

# Common Norwegian function words — sentences with many of these are natural prose.
_NORSK_WORDS = frozenset(
    "og i er det å på for av den dei som med ein har til ikkje var dei kan "
    "når men eller frå ved seg alle dette dei som han ho dei me eit vart "
    "vore meir etter bli også ut no andre mange nokre alle kvar utan mot "
    "mellom under over sidan gjennom sine deira hans hennar dess".split()
)

# Regex for non-prose junk: too many numbers, parens, pipes, URLs, refs
_JUNK_RE = re.compile(
    r"\d{4,}"          # 4+ digit numbers (years, stats)
    r"|[|()\[\]{}<>]"  # brackets, pipes, angle brackets
    r"|https?://"      # URLs
    r"|\.jpg|\.png"    # image filenames
    r"|&[a-z]+;"       # HTML entities
)


def _is_good_sentence(s: str) -> bool:
    """Filter for natural Norwegian prose sentences."""
    # Length: 50-200 chars (skip fragments and run-on markup)
    if not (50 <= len(s) <= 200):
        return False

    # Reject sentences with too much non-prose junk
    if len(_JUNK_RE.findall(s)) >= 2:
        return False

    # Reject sentences where >40% of characters are uppercase (headers, acronyms)
    alpha = [c for c in s if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.4:
        return False

    # Prefer sentences with Norwegian character — score but don't hard-require
    # (many valid nynorsk sentences have no æøå)
    words = s.lower().split()
    norsk_hits = sum(1 for w in words if w.rstrip(".,;:!?") in _NORSK_WORDS)
    has_aeoa = bool(re.search(r"[æøåÆØÅ]", s))

    # At least 20% Norwegian function words OR contains æøå
    if not has_aeoa and (not words or norsk_hits / len(words) < 0.2):
        return False

    return True


def load_articles(path: str) -> list[list[str]]:
    """Load Wikipedia articles as lists of filtered, prose-quality sentences."""
    print("Loading Wikipedia Parquet file...")
    df = pd.read_parquet(path)
    texts = df["text"].dropna().astype(str).tolist()
    print(f"Loaded {len(texts)} raw texts.")

    total_sentences = 0
    kept_sentences = 0
    articles = []
    for text in texts[:50_000]:
        sentences = [s.strip() + "." for s in text.split(".") if len(s.strip()) > 25]
        sentences = [re.sub(r"'{2,3}", "", s) for s in sentences]
        total_sentences += len(sentences)

        good = [s for s in sentences if _is_good_sentence(s)]
        kept_sentences += len(good)

        if len(good) >= 4:
            articles.append(good)

    print(f"Filtered {total_sentences} → {kept_sentences} sentences ({kept_sentences/max(total_sentences,1):.0%} kept).")
    print(f"Parsed {len(articles)} articles with 6+ good sentences.")
    return articles


def make_sample(articles: list[list[str]]) -> dict | None:
    """Create one completion sample from a random article slice."""
    article = random.choice(articles)

    # Pick a contiguous slice of sentences within our char budget
    start = random.randint(0, max(0, len(article) - 6))
    text = ""
    end = start
    for i in range(start, len(article)):
        candidate = text + (" " if text else "") + article[i]
        if len(candidate) > MAX_CHARS:
            break
        text = candidate
        end = i + 1

    if len(text) < MIN_CHARS:
        return None

    # Split at a sentence boundary near the middle
    sentences = article[start:end]
    mid = len(sentences) // 2
    # Add some jitter so the split point varies
    mid = max(2, min(len(sentences) - 2, mid + random.randint(-2, 2)))

    prompt_text = " ".join(sentences[:mid])
    completion_text = " ".join(sentences[mid:])

    if len(prompt_text) < 200 or len(completion_text) < 200:
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Hald fram med denne teksten:\n{prompt_text}",
            },
            {
                "role": "assistant",
                "content": completion_text,
            },
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text-only Nynorsk completion samples."
    )
    parser.add_argument(
        "--num-samples", type=int, default=3000,
        help="Number of samples to generate (default: 3000)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})."
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    articles = load_articles(CORPUS_PATH)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    attempts = 0
    max_attempts = args.num_samples * 10

    while len(records) < args.num_samples and attempts < max_attempts:
        attempts += 1
        sample = make_sample(articles)
        if sample is not None:
            records.append(sample)

        if len(records) % 500 == 0 and len(records) > 0:
            print(f"  {len(records)}/{args.num_samples} samples generated...")

    output_path = os.path.join(args.output_dir, "train.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Stats
    prompt_lens = [len(r["messages"][0]["content"]) for r in records]
    completion_lens = [len(r["messages"][1]["content"]) for r in records]
    total_lens = [p + c for p, c in zip(prompt_lens, completion_lens)]

    print(f"\nDone. {len(records)} samples saved to {output_path}")
    print(f"Total chars per sample: min={min(total_lens)}, median={sorted(total_lens)[len(total_lens)//2]}, max={max(total_lens)}")
    print(f"Prompt chars: min={min(prompt_lens)}, median={sorted(prompt_lens)[len(prompt_lens)//2]}, max={max(prompt_lens)}")
    print(f"Completion chars: min={min(completion_lens)}, median={sorted(completion_lens)[len(completion_lens)//2]}, max={max(completion_lens)}")


if __name__ == "__main__":
    main()
