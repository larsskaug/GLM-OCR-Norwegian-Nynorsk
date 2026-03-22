"""
extract_pages_save_jpeg.py — Extract embedded images from PDF scans as grayscale JPEG.

Uses PyMuPDF to pull image bytes from the PDF, converts to grayscale
(newspaper scans carry no useful color information for OCR), and saves
as JPEG.

Usage:
    python extract_pages_save_jpeg.py                        # all PDFs in raw/
    python extract_pages_save_jpeg.py scan1.pdf scan2.pdf    # specific files
    python extract_pages_save_jpeg.py -o /tmp/out scan.pdf   # custom output dir
"""

import argparse
import os
import sys
from io import BytesIO

import pymupdf
from PIL import Image

JPEG_QUALITY = 90


def extract_pages(pdf_path: str, output_dir: str) -> list[str]:
    """Extract embedded images from a PDF, convert to grayscale JPEG.

    Returns a list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    doc = pymupdf.open(pdf_path)
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images()

        if not image_list:
            print(f"  Page {page_num}: no images found, skipping.")
            continue

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            pil_img = Image.open(BytesIO(image_bytes)).convert("L")

            filename = f"{pdf_stem}-p{page_num}-{img_index}.jpeg"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath, "JPEG", quality=JPEG_QUALITY)

            size_kb = os.path.getsize(filepath) / 1024
            saved.append(filepath)
            print(f"  {filename} ({pil_img.size[0]}x{pil_img.size[1]}, {size_kb:.0f} KB)")

    doc.close()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract original embedded images from scanned PDFs."
    )
    parser.add_argument(
        "files", nargs="*",
        help="PDF files to process. Defaults to all PDFs in raw/."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "ready-for-ocr"),
        help="Output directory (default: scans/ready-for-ocr/)."
    )
    args = parser.parse_args()

    input_dir = os.path.join(os.path.dirname(__file__), "raw")
    inputs = args.files or sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ) if os.path.isdir(input_dir) else []

    if not inputs:
        print(f"No PDFs found. Put them in {input_dir}/ or pass paths as arguments.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for pdf in inputs:
        print(f"\nExtracting: {os.path.basename(pdf)}")
        saved = extract_pages(pdf, args.output_dir)
        total += len(saved)

    print(f"\nDone. Extracted {total} images.")


if __name__ == "__main__":
    main()
