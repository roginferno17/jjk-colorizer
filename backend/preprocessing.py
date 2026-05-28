"""
preprocessing.py
================
Image preprocessing utilities for JJK Colorizer.

Handles:
  - Image validation and normalization
  - Resolution adjustment for Forge API (must be 1024x1024 for SDXL)
  - Preview generation (small versions for frontend display)
"""

import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_DIM = 1024
PREVIEW_DIM = 512


def load_and_normalize(path: str | Path) -> Image.Image:
    """
    Load image and normalize to 1024x1024 RGB PNG.
    Center-crops to square, then resizes.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Center crop to square
    short = min(w, h)
    l = (w - short) // 2
    t = (h - short) // 2
    img = img.crop((l, t, l + short, t + short))

    # Resize to 1024x1024
    img = img.resize((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return img


def to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def make_preview(img: Image.Image) -> str:
    """Create small preview image for frontend display."""
    preview = img.copy()
    preview.thumbnail((PREVIEW_DIM, PREVIEW_DIM), Image.LANCZOS)
    return to_base64(preview, "JPEG")


def validate_upload(path: str | Path) -> tuple[bool, str]:
    """
    Validate uploaded image.
    Returns (is_valid, error_message).
    """
    path = Path(path)

    if not path.exists():
        return False, "File not found"

    if path.stat().st_size > 50 * 1024 * 1024:  # 50MB limit
        return False, "File too large (max 50MB)"

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        return False, "Unsupported format (use JPG, PNG, or WebP)"

    try:
        img = Image.open(path)
        w, h = img.size
        if min(w, h) < 256:
            return False, f"Image too small ({w}x{h}, minimum 256px)"
    except Exception as e:
        return False, f"Cannot read image: {e}"

    return True, ""
