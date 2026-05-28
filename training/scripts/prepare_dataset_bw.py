"""
prepare_dataset_bw.py
=====================
Processes raw manga panel images into training-ready 1024x1024 PNGs.

Input:  Raw JPEG/PNG manga scans (any resolution)
Output: 1024x1024 RGB-encoded grayscale PNGs + blank caption stubs

Run:
    python prepare_dataset_bw.py --input <raw_folder> --output <dataset_folder>

After running, manually:
  1. Cull bad images (blurry, screentone-heavy, wrong art)
  2. Fill in or edit caption .txt files
  3. Run caption_cleanup.py to finalize captions
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SIZE = 1024


def bilateral_denoise(img_np: np.ndarray) -> np.ndarray:
    """Light bilateral filter — removes scan noise, preserves ink edges."""
    return cv2.bilateralFilter(img_np, d=5, sigmaColor=15, sigmaSpace=15)


def process_image(src: Path, dst_dir: Path, index: int) -> bool:
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size

        if min(w, h) < 400:
            print(f"  SKIP (too small {w}x{h}): {src.name}")
            return False

        # Grayscale
        gray = img.convert("L")
        gray_np = np.array(gray)

        # Bilateral denoise (removes scan grain without softening ink)
        denoised = bilateral_denoise(gray_np)

        # Auto-levels (clip 0.5% to eliminate fog/artifacts)
        pil_denoised = Image.fromarray(denoised, mode="L")
        leveled = ImageOps.autocontrast(pil_denoised, cutoff=0.5)

        # Convert back to RGB (SDXL expects 3-channel input)
        rgb = leveled.convert("RGB")

        # Center-crop to square, then resize to 1024
        short = min(rgb.size)
        left = (rgb.width - short) // 2
        top = (rgb.height - short) // 2
        cropped = rgb.crop((left, top, left + short, top + short))
        final = cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Save
        stem = f"panel_{index:04d}"
        out_img = dst_dir / f"{stem}.png"
        out_txt = dst_dir / f"{stem}.txt"

        final.save(out_img, format="PNG")

        # Default caption stub — edit after reviewing image
        if not out_txt.exists():
            out_txt.write_text(
                "gegeakutami, monochrome manga, ",
                encoding="utf-8"
            )

        print(f"  OK {src.name} -> {stem}.png ({w}x{h} -> {TARGET_SIZE}x{TARGET_SIZE})")
        return True

    except Exception as e:
        print(f"  ERROR {src.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with raw images")
    parser.add_argument("--output", required=True, help="Output folder (kohya_ss ready)")
    args = parser.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Found {len(images)} images in {src_dir}")

    ok = 0
    for i, img_path in enumerate(images):
        if process_image(img_path, dst_dir, i + 1):
            ok += 1

    print(f"\nDone: {ok}/{len(images)} processed -> {dst_dir}")
    print("\nNext steps:")
    print("  1. Open output folder, delete any bad images (and their .txt files)")
    print("  2. Edit .txt files to add structural tags after 'gegeakutami, monochrome manga,'")
    print("  3. Run: python caption_cleanup.py --folder <output_folder>")


if __name__ == "__main__":
    main()
