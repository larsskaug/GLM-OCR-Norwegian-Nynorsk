"""
generate_training_data.py — Generate all training data for GLM-OCR Nynorsk fine-tuning.

Runs both the multimodal (synthetic newspaper) and text-only (completion) generators.
LLaMA-Factory reads the data via symlinks in LLaMA-Factory/data/ that point to
finetune-data/synthetic/ and finetune-data/text-only/.

Usage:
    python generate_training_data.py                              # defaults: 20k multimodal, 5k text
    python generate_training_data.py --multimodal 20000 --text 5000
    python generate_training_data.py --multimodal 20000 --text 5000 --seed 42
    python generate_training_data.py --text-only --text 3000      # skip multimodal
"""

import argparse
import subprocess
import sys
import os


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
    if result.returncode != 0:
        print(f"\nERROR: {label} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all training data.")
    parser.add_argument("--multimodal", type=int, default=20000,
                        help="Number of synthetic newspaper images (default: 20000).")
    parser.add_argument("--text", type=int, default=5000,
                        help="Number of text-only completion samples (default: 5000).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (applied to both generators).")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip multimodal generation, only generate text samples.")
    args = parser.parse_args()

    seed_args = ["--seed", str(args.seed)] if args.seed is not None else []

    # 1. Multimodal (synthetic newspaper pages)
    if not args.text_only:
        run(
            [sys.executable, "synthesize_newspaper_pages.py",
             "--num-samples", str(args.multimodal)] + seed_args,
            f"Generating {args.multimodal} multimodal samples",
        )

    # 2. Text-only (completion samples)
    run(
        [sys.executable, "generate_text_samples.py",
         "--num-samples", str(args.text)] + seed_args,
        f"Generating {args.text} text-only samples",
    )

    print(f"\n{'='*60}")
    print(f"  Done!")
    if not args.text_only:
        print(f"  Multimodal: {args.multimodal} samples")
    print(f"  Text-only:  {args.text} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
