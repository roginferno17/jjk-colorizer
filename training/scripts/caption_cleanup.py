"""
caption_cleanup.py
==================
Finalizes caption .txt files after manual editing.

What it does:
  1. Reads every .txt in the target folder
  2. Strips style/quality tags that would confuse LoRA training
  3. Ensures trigger word is at position 0
  4. Removes duplicates and normalizes whitespace
  5. Reports any .png files missing a .txt (you must create them)

Style tags removed (model should LEARN these from images, not from text):
  monochrome, greyscale, grayscale, manga, sketch, lineart, rough,
  hatching, cross-hatching, detailed, masterpiece, best quality,
  high quality, ultra detailed, highres, absurdres, watercolor,
  painterly, illustration, artwork, anime style, cel shading,
  digital art, traditional media, etc.

Run:
    python caption_cleanup.py --folder <dataset_folder> --trigger "gegeakutami, monochrome manga"
    python caption_cleanup.py --folder <color_folder>   --trigger "gegecover, painterly watercolor cover"
"""

import argparse
import re
from pathlib import Path

# Tags to strip from captions — these describe style, not content
STYLE_TAGS_TO_REMOVE = {
    "monochrome", "greyscale", "grayscale", "black and white", "black-and-white",
    "manga", "sketch", "lineart", "line art", "rough", "rough sketch",
    "hatching", "cross hatching", "cross-hatching", "detailed", "highly detailed",
    "masterpiece", "best quality", "high quality", "ultra detailed", "highres",
    "absurdres", "watercolor", "watercolour", "painterly", "painting",
    "illustration", "artwork", "drawn", "hand drawn", "hand-drawn",
    "anime style", "anime", "cel shading", "cel-shading", "cel shaded",
    "digital art", "digital painting", "traditional media", "traditional art",
    "beautiful", "amazing", "perfect", "gorgeous", "stunning",
    "artstation", "pixiv", "deviantart",
    "jjk", "jujutsu kaisen", "gege akutami",  # model learns from trigger, not these
}


def normalize_tags(text: str, trigger: str) -> str:
    tags = [t.strip() for t in text.split(",")]
    trigger_tags = [t.strip() for t in trigger.split(",")]

    # Remove empty tags
    tags = [t for t in tags if t]

    # Remove style tags
    cleaned = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in STYLE_TAGS_TO_REMOVE:
            continue
        # Also remove if tag is a substring match of style tags
        if any(style == tag_lower for style in STYLE_TAGS_TO_REMOVE):
            continue
        cleaned.append(tag)

    # Remove trigger tags from body (will be prepended)
    body = [t for t in cleaned if t.lower() not in [x.lower() for x in trigger_tags]]

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for t in body:
        if t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)

    # Rebuild: trigger first, then content tags
    final_tags = trigger_tags + deduped
    return ", ".join(final_tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Dataset folder with .png and .txt files")
    parser.add_argument("--trigger", default="gegeakutami, monochrome manga",
                        help="Trigger phrase to prepend (default: BW trigger)")
    parser.add_argument("--dry_run", action="store_true", help="Print changes without saving")
    args = parser.parse_args()

    folder = Path(args.folder)
    pngs = sorted(folder.glob("*.png"))
    txts = sorted(folder.glob("*.txt"))

    print(f"Folder: {folder}")
    print(f"  PNGs: {len(pngs)}")
    print(f"  TXTs: {len(txts)}")

    # Check for missing captions
    missing = [p for p in pngs if not p.with_suffix(".txt").exists()]
    if missing:
        print(f"\n  WARNING — {len(missing)} PNGs have no .txt caption:")
        for m in missing:
            print(f"    {m.name}")

    # Process existing captions
    print(f"\nCleaning captions (trigger: '{args.trigger}')...")
    changed = 0
    for txt in txts:
        original = txt.read_text(encoding="utf-8").strip()
        cleaned = normalize_tags(original, args.trigger)

        if cleaned != original:
            changed += 1
            print(f"  {txt.name}:")
            print(f"    BEFORE: {original[:80]}{'...' if len(original)>80 else ''}")
            print(f"    AFTER:  {cleaned[:80]}{'...' if len(cleaned)>80 else ''}")
            if not args.dry_run:
                txt.write_text(cleaned, encoding="utf-8")

    print(f"\nModified: {changed}/{len(txts)} files")
    if args.dry_run:
        print("(dry run — no files written)")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
