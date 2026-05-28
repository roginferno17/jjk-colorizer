"""
audit_dataset.py
================
Audits raw image folders before processing.
Prints:
  - Total count
  - Resolution distribution
  - Images below minimum resolution (flag for deletion)
  - Fanart indicators in filenames (color dataset)
  - Estimated final dataset size after processing

Run:
    python audit_dataset.py --folder <raw_folder> --mode bw
    python audit_dataset.py --folder <raw_folder> --mode color
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

from PIL import Image

# Force UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RESOLUTION = 400  # pixels on shortest side

FANART_INDICATORS = [
    "@", "twitter", " on x", "on x.", "credit to",
    "jojoland", "vroom", "sneez", "fanart", "fan art",
    "repost", "edit", "tiktok",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--mode", choices=["bw", "color"], default="bw")
    args = parser.parse_args()

    folder = Path(args.folder)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

    print(f"\nAUDIT REPORT — {folder.name} ({args.mode.upper()} mode)")
    print("=" * 60)
    print(f"Total images found: {len(images)}")

    too_small = []
    fanart = []
    ok = []
    res_buckets = Counter()

    for p in images:
        # Fanart check (color mode)
        if args.mode == "color":
            name_lower = p.name.lower()
            if any(ind in name_lower for ind in FANART_INDICATORS):
                fanart.append(p)
                continue

        # Resolution check
        try:
            img = Image.open(p)
            w, h = img.size
            short = min(w, h)

            if short < MIN_RESOLUTION:
                too_small.append((p, w, h))
            else:
                ok.append((p, w, h))
                bucket = f"{(short // 200) * 200}+"
                res_buckets[bucket] += 1
        except Exception as e:
            print(f"  UNREADABLE: {p.name} ({e})")

    print(f"\nUsable images: {len(ok)}")
    print(f"Too small (<{MIN_RESOLUTION}px): {len(too_small)}")

    if args.mode == "color":
        print(f"Auto-fanart rejected: {len(fanart)}")
        if fanart:
            print("\nFanart filenames:")
            for p in fanart:
                print(f"  {p.name}")

    if too_small:
        print("\nToo-small images (delete these):")
        for p, w, h in too_small[:20]:
            print(f"  {p.name} ({w}x{h})")

    print("\nResolution distribution (usable):")
    for bucket in sorted(res_buckets.keys()):
        bar = "#" * res_buckets[bucket]
        print(f"  {bucket:>6}px: {bar} ({res_buckets[bucket]})")

    # Estimate output count
    if args.mode == "color":
        estimated_output = len(ok) * 3  # 3 crops per image
        print(f"\nEstimated output after 3x crop augmentation: {estimated_output} images")
        if estimated_output < 80:
            print("  WARNING: Target is 80–130 images. Need more official covers.")
        elif estimated_output > 200:
            print("  NOTE: Dataset may be larger than needed — prioritize quality over count.")
    else:
        estimated_output = len(ok)
        print(f"\nEstimated output: {estimated_output} images")
        if estimated_output < 300:
            print("  WARNING: Target is 300–500 images. Need more manga panels.")

    print("\nRECOMMENDED ACTIONS:")
    if too_small:
        print(f"  - Delete {len(too_small)} too-small images")
    if args.mode == "color" and fanart:
        print(f"  - Auto-rejected {len(fanart)} fanart files (don't copy to dataset)")
    if len(ok) > 0:
        print(f"  - Process remaining {len(ok)} images with prepare_dataset_{args.mode}.py")


if __name__ == "__main__":
    main()
