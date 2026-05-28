"""
download_models.py
==================
Downloads all required models for JJK Colorizer.

Models:
  1. Illustrious XL base checkpoint (~7GB)
  2. ControlNet Lineart SDXL
  3. ControlNet Depth (T2I-Adapter) SDXL
  4. 4x-Manga109Attempt ESRGAN upscaler

Run (from project root):
    python training/scripts/download_models.py

Requires: pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# Forge model directories
FORGE_ROOT = Path("C:/Users/Vishu/AI/StabilityMatrix/Data/Packages/forge-neo/models")
CHECKPOINT_DIR = FORGE_ROOT / "Stable-diffusion"
CONTROLNET_DIR = FORGE_ROOT / "ControlNet"
ESRGAN_DIR = FORGE_ROOT / "ESRGAN"
LORA_DIR = FORGE_ROOT / "Lora"

# Also keep copies in project models/ for reference
PROJECT_MODELS = Path("C:/Users/Vishu/Desktop/lock-in-jjk/models")


def check_exists(path: Path, label: str) -> bool:
    if path.exists():
        size_gb = path.stat().st_size / 1e9
        print(f"  [SKIP] {label} already exists ({size_gb:.2f} GB): {path.name}")
        return True
    return False


def download_illustrious_xl():
    print("\n[1/4] Illustrious XL v0.1 (~7GB)")
    out_path = CHECKPOINT_DIR / "Illustrious-XL-v0.1.safetensors"
    if check_exists(out_path, "Illustrious XL"):
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading from HuggingFace...")
    hf_hub_download(
        repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
        filename="Illustrious-XL-v0.1.safetensors",
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"  Saved: {out_path}")


def download_controlnet_lineart():
    print("\n[2/4] ControlNet Lineart SDXL")
    out_dir = CONTROLNET_DIR / "controlnet-lineart-sdxl"
    if (out_dir / "diffusion_pytorch_model.safetensors").exists():
        print(f"  [SKIP] already exists: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading from xinsir/controlnet-lineart-sdxl-1.0...")
    snapshot_download(
        repo_id="xinsir/controlnet-lineart-sdxl-1.0",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"],  # prefer safetensors
    )
    print(f"  Saved: {out_dir}")


def download_depth_adapter():
    print("\n[3/4] T2I-Adapter Depth SDXL (for depth ControlNet)")
    out_dir = CONTROLNET_DIR / "t2i-adapter-depth-sdxl"
    if (out_dir / "diffusion_pytorch_model.safetensors").exists():
        print(f"  [SKIP] already exists: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading from TencentARC/t2i-adapter-depth-zoe-sdxl-1.0...")
    snapshot_download(
        repo_id="TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"],
    )
    print(f"  Saved: {out_dir}")


def download_esrgan():
    print("\n[4/4] 4x-Manga109Attempt ESRGAN")
    out_path = ESRGAN_DIR / "4x-Manga109Attempt.pth"
    if check_exists(out_path, "4x-Manga109Attempt"):
        return

    ESRGAN_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading from HuggingFace...")
    hf_hub_download(
        repo_id="Kim2091/4x-Manga109Attempt",
        filename="4x-Manga109Attempt.pth",
        local_dir=str(ESRGAN_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"  Saved: {out_path}")

    # Also get AnimeSharp for color output upscaling
    out_path2 = ESRGAN_DIR / "4x-AnimeSharp.pth"
    if not out_path2.exists():
        print("  Also downloading 4x-AnimeSharp (for color output upscaling)...")
        hf_hub_download(
            repo_id="Kim2091/4x-AnimeSharp",
            filename="4x-AnimeSharp.pth",
            local_dir=str(ESRGAN_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  Saved: {out_path2}")


def main():
    print("JJK Colorizer — Model Download")
    print("=" * 50)
    print(f"Forge models root: {FORGE_ROOT}")

    if not FORGE_ROOT.exists():
        print(f"\nERROR: Forge not found at {FORGE_ROOT}")
        print("Launch StabilityMatrix, install forge-neo, then re-run this script.")
        sys.exit(1)

    download_illustrious_xl()
    download_controlnet_lineart()
    download_depth_adapter()
    download_esrgan()

    print("\n" + "=" * 50)
    print("All models downloaded.")
    print("\nNext steps:")
    print("  1. Launch Forge via StabilityMatrix")
    print("  2. Verify Illustrious XL appears in model list")
    print("  3. Go to Settings → Extensions → install ControlNet extension")
    print("  4. Verify ControlNet models appear in ControlNet panel")
    print("  5. Test img2img with a sample image")


if __name__ == "__main__":
    main()
