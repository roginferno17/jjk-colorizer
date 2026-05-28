"""
prepare_dataset_color.py
========================
Processes official JJK colored artwork into training-ready 1024x1024 PNGs.
Automatically REJECTS images with fanart indicators in their filename.
From each source image, generates multiple crops (augmentation).

Run:
    python prepare_dataset_color.py --input <raw_folder> --output <dataset_folder>
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SIZE = 1024
# 700px allows Pinterest-resolution covers (736px) to pass
# Color LoRA learns palette/style — fine detail matters less than for BW linework
MIN_SHORT_SIDE = 700

# Filenames containing these strings are AUTO-REJECTED as fanart
FANART_INDICATORS = [
    "@", "twitter", " on x", "on x.", "credit to",
    "jojoland", "vroom", "sneez", "fanart", "fan art",
    "repost", "edit", "tiktok",
]

def is_fanart(path: Path) -> bool:
    name_lower = path.name.lower()
    return any(indicator in name_lower for indicator in FANART_INDICATORS)


def make_crop(img: Image.Image, box: tuple, size: int) -> Image.Image:
    """Crop box then resize to size×size."""
    cropped = img.crop(box)
    return cropped.resize((size, size), Image.LANCZOS)


def process_image(src: Path, dst_dir: Path, index: int) -> int:
    """Returns number of output images created (0 on skip/error)."""
    if is_fanart(src):
        print(f"  REJECT (fanart): {src.name}")
        return 0

    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size

        if min(w, h) < MIN_SHORT_SIDE:
            print(f"  SKIP (too small {w}x{h}): {src.name}")
            return 0

        created = 0
        stem_base = f"cover_{index:03d}"

        # Crop 1: Center square (primary)
        short = min(w, h)
        l = (w - short) // 2
        t = (h - short) // 2
        full_crop = make_crop(img, (l, t, l + short, t + short), TARGET_SIZE)
        _save(full_crop, dst_dir, f"{stem_base}_a", "gegecover, painterly watercolor cover, ")
        created += 1

        # Crop 2: Upper 60% of HEIGHT, cropped to square at image width
        upper_h = int(h * 0.6)
        upper_side = min(w, upper_h)  # square side = min(width, 60%height)
        ul = (w - upper_side) // 2
        if upper_side >= MIN_SHORT_SIDE:
            upper_crop = make_crop(img, (ul, 0, ul + upper_side, upper_side), TARGET_SIZE)
            _save(upper_crop, dst_dir, f"{stem_base}_b", "gegecover, painterly watercolor cover, upper body, ")
            created += 1

        # Crop 3: Top-center 40% (face region for tall covers)
        face_side = min(w, int(h * 0.4))
        fl = (w - face_side) // 2
        if face_side >= MIN_SHORT_SIDE:
            face_crop = make_crop(img, (fl, 0, fl + face_side, face_side), TARGET_SIZE)
            _save(face_crop, dst_dir, f"{stem_base}_c", "gegecover, painterly watercolor cover, face, closeup, ")
            created += 1

        print(f"  OK {src.name} → {created} crops ({w}x{h})")
        return created

    except Exception as e:
        print(f"  ERROR {src.name}: {e}")
        return 0


def _save(img: Image.Image, dst_dir: Path, stem: str, caption_prefix: str):
    img.save(dst_dir / f"{stem}.png", format="PNG")
    txt = dst_dir / f"{stem}.txt"
    if not txt.exists():
        txt.write_text(caption_prefix, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Found {len(images)} images in {src_dir}\n")

    total_created = 0
    rejected = 0
    for i, p in enumerate(images):
        n = process_image(p, dst_dir, i + 1)
        if n == 0 and is_fanart(p):
            rejected += 1
        total_created += n

    print(f"\nResults:")
    print(f"  Auto-rejected (fanart indicators): {rejected}")
    print(f"  Total output images: {total_created}")
    print(f"  Output: {dst_dir}")
    print("\nNext steps:")
    print("  1. Manually inspect output — delete any remaining non-official art + their .txt")
    print("  2. Complete caption .txt files with character/composition tags")
    print("  3. Run: python caption_cleanup.py --folder <output_folder>")


if __name__ == "__main__":
    main()
