#!/usr/bin/env python3
"""
Evaluate a GLM-OCR model on the Sogningen 1957 scan.

Resizes the image to ~200 DPI effective (3500 px wide) before inference so the
vision encoder sees a manageable tile count (~165 tiles / ~24k image tokens)
while still having ~23 px character height for legible 8 pt body text.

Default model and metadata are read from project.yaml in the project root.
"""

import argparse
import os
import tempfile

import torch
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = os.path.dirname(__file__)
PROJECT_YAML = os.path.join(PROJECT_ROOT, "project.yaml")

DEFAULT_IMAGE_PATH = os.path.join(PROJECT_ROOT,
                                 "evaluation/2026-03-20T1237_NB_generated-1.jpg")

TARGET_WIDTH = 3500   # ~200 DPI effective for a Norwegian broadsheet (~43 cm wide)
MAX_NEW_TOKENS = 12288
REPETITION_PENALTY = 1.3


def load_project() -> dict:
    with open(PROJECT_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


ALIASES = ("base", "finetuned")


def resolve_model(name: str, project: dict) -> str:
    """Expand 'base' / 'finetuned' aliases to their full paths."""
    if name == "finetuned":
        rel = project["models"]["current_finetuned"]["path"]
        return os.path.join(PROJECT_ROOT, rel)
    if name == "base":
        return project["models"]["base"]["path"]
    return name


def default_model_path(project: dict) -> str:
    return resolve_model("finetuned", project)


def is_finetuned(model_path: str, project: dict) -> bool:
    """Return True when model_path resolves to the current finetuned model."""
    finetuned = os.path.realpath(default_model_path(project))
    return os.path.realpath(model_path) == finetuned


def default_output_path(image_path: str, model_path: str, project: dict) -> str:
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "finetuned" if is_finetuned(model_path, project) else "base"
    filename = f"{image_stem}-{suffix}.txt"
    return os.path.join(os.path.dirname(os.path.abspath(image_path)), filename)


def parse_args(project: dict) -> argparse.Namespace:
    finetuned_path = default_model_path(project)
    parser = argparse.ArgumentParser(
        description="Evaluate a GLM-OCR model on a single newspaper scan."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        metavar="IMAGE",
        help=f"Path to the JPEG to transcribe (default: {DEFAULT_IMAGE_PATH!r})",
    )
    parser.add_argument(
        "--model",
        default="finetuned",
        metavar="PATH|ALIAS",
        help=(
            f"Model to evaluate. Use 'finetuned' (default) or 'base' as shorthands, "
            f"or supply an explicit path. "
            f"finetuned → {finetuned_path}"
        ),
    )
    # --output default depends on both image and --model, so parse twice first.
    partial, _ = parser.parse_known_args()
    resolved_model = resolve_model(partial.model, project)
    output_default = default_output_path(partial.image, resolved_model, project)
    parser.add_argument(
        "--output",
        default=output_default,
        metavar="FILE",
        help=(
            f"Path for the transcription output "
            f"(default: <image-name>-finetuned.txt or <image-name>-base.txt; "
            f"currently {output_default!r})"
        ),
    )
    args = parser.parse_args()
    args.model = resolve_model(args.model, project)
    return args


def maybe_downscale(path: str, target_width: int) -> tuple[str, bool]:
    """
    If the image is wider than target_width, downscale it, save to a temp
    JPEG and return (temp_path, True).  Otherwise return (original_path, False).
    """
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path)
    orig_w, orig_h = img.size
    if orig_w <= target_width:
        print(f"Image is {orig_w}x{orig_h} — no resize needed.")
        return path, False
    scale = target_width / orig_w
    new_size = (target_width, int(orig_h * scale))
    print(f"Resizing {orig_w}x{orig_h} → {new_size[0]}x{new_size[1]} "
          f"(scale {scale:.3f}x)")
    img = img.convert("RGB").resize(new_size, Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    tmp.close()
    print(f"Saved resized image to {tmp.name}")
    return tmp.name, True


def main():
    project = load_project()
    args = parse_args(project)

    print(f"Image : {args.image}")
    print(f"Model : {args.model}")
    print(f"Output: {args.output}")

    # 1. Downscale only if needed
    image_path, is_tmp = maybe_downscale(args.image, TARGET_WIDTH)

    # 2. Load model
    print(f"Loading model from {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    # 3. Build prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant for recognising text in images. "
                        "All text in the image is in Norwegian nynorsk. "
                        "Extract all text faithfully."
                    ),
                },
                {
                    "type": "image",
                    "path": image_path,
                },
                {
                    "type": "text",
                    "text": "Text Recognition:",
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    n_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()
    print(f"Image tokens in prompt: {n_image_tokens:,}")
    print(f"Generating (max {MAX_NEW_TOKENS} tokens) ...")

    # 4. Generate
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # 5. Save
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nSaved {len(output_text):,} chars to {args.output}")
    print(f"\n--- First 500 chars ---\n{output_text[:500]}")

    if is_tmp:
        os.unlink(image_path)


if __name__ == "__main__":
    main()
