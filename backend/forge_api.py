"""
forge_api.py
============
Wrapper for SD WebUI Forge's REST API.

Forge must be running with --api flag.
Launch via StabilityMatrix: add --api to Extra Launch Args.

API base: http://localhost:7860

Uses diffusers_xl_canny_mid for structure preservation
(SDXL Lineart ControlNet not publicly available — Canny is functionally equivalent for clean art)
"""

import base64
import io
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from PIL import Image

FORGE_URL = "http://localhost:7860"

# ControlNet model names (must match filenames in Forge ControlNet folder)
CANNY_SDXL_MODEL = "diffusers_xl_canny_mid"
DEPTH_SDXL_MODEL = "diffusers_xl_depth_mid"

# LoRA names (must match filenames in Forge Lora folder, without extension)
BW_LORA_NAME = "gege_bw_lora"
COLOR_LORA_NAME = "gege_color_lora"

# Base model checkpoint name
BASE_CHECKPOINT = "Illustrious-XL-v0.1.safetensors"


def _encode_image(img_or_path) -> str:
    """Encode PIL Image or file path to base64 string."""
    if isinstance(img_or_path, (str, Path)):
        with open(img_or_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    elif isinstance(img_or_path, Image.Image):
        buf = io.BytesIO()
        img_or_path.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    raise TypeError(f"Expected str, Path, or PIL Image, got {type(img_or_path)}")


def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _post(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    url = f"{FORGE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Forge API error {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Forge at {FORGE_URL}. Is it running with --api?") from e


def check_forge_ready() -> bool:
    """Verify Forge is running and accessible."""
    try:
        with urllib.request.urlopen(f"{FORGE_URL}/sdapi/v1/samplers", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_forge(max_wait: int = 60) -> bool:
    """Wait up to max_wait seconds for Forge to become ready."""
    start = time.time()
    while time.time() - start < max_wait:
        if check_forge_ready():
            return True
        time.sleep(3)
    return False


def set_model(checkpoint: str = BASE_CHECKPOINT):
    """Switch Forge to the specified checkpoint."""
    _post("/sdapi/v1/options", {"sd_model_checkpoint": checkpoint})


def pipeline_a_bw(
    input_image,
    lora_weight: float = 0.85,
    canny_weight: float = 0.90,
    depth_weight: float = 0.55,
    denoise: float = 0.63,
    seed: int = -1,
) -> Image.Image:
    """
    Pipeline A: Convert full artwork to Gege-style B&W manga.

    Args:
        input_image: PIL Image or path to input artwork
        lora_weight: BW LoRA strength (0.80–0.95)
        canny_weight: Structure ControlNet weight (0.85–0.95)
        depth_weight: Depth ControlNet weight (0.45–0.65)
        denoise: img2img denoising strength (0.55–0.70)
        seed: -1 for random

    Returns:
        PIL Image: B&W Gege-style output
    """
    b64_input = _encode_image(input_image)

    payload = {
        "init_images": [b64_input],
        "prompt": (
            "gegeakutami, monochrome manga, rough ink linework, expressive hatching, "
            "cross hatching, high contrast, thick ink outlines, manga panel, "
            "imperfect strokes, raw ink energy"
        ),
        "negative_prompt": (
            "color, colorful, rgb, pastel, anime style, glossy, cel shading, "
            "airbrushed, over-detailed, hyperrealistic, smooth, polished linework, "
            "clean lines, digital art, photorealistic, 3d render, watermark, text, nsfw"
        ),
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "steps": 30,
        "cfg_scale": 7.5,
        "denoising_strength": denoise,
        "width": 1024,
        "height": 1024,
        "seed": seed,
        "override_settings": {
            "sd_model_checkpoint": BASE_CHECKPOINT,
        },
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "image": b64_input,
                        "module": "lineart_anime_denoise",  # preprocessor
                        "model": CANNY_SDXL_MODEL,           # model
                        "weight": canny_weight,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "resize_mode": 1,
                        "control_mode": 0,
                    },
                    {
                        "enabled": True,
                        "image": b64_input,
                        "module": "depth_zoe",
                        "model": DEPTH_SDXL_MODEL,
                        "weight": depth_weight,
                        "guidance_start": 0.0,
                        "guidance_end": 0.8,
                        "resize_mode": 1,
                        "control_mode": 0,
                    },
                ]
            },
            "additional networks for generating": {
                "args": [
                    True,           # enabled
                    "LoRA",         # type
                    BW_LORA_NAME,   # lora name
                    lora_weight,    # model weight
                    lora_weight,    # text encoder weight
                    None, None, None, None, None, None,
                ]
            }
        },
    }

    result = _post("/sdapi/v1/img2img", payload)
    return _decode_image(result["images"][0])


def pipeline_b_color(
    bw_image,
    lora_weight: float = 0.80,
    canny_weight: float = 0.97,
    denoise: float = 0.45,
    color_theme: str = "",
    seed: int = -1,
) -> Image.Image:
    """
    Pipeline B: Colorize B&W Gege manga output with volume cover aesthetics.

    Args:
        bw_image: PIL Image or path to B&W Pipeline A output
        lora_weight: Color LoRA strength (0.75–0.85)
        canny_weight: Lineart lock weight (0.93–1.0) — keep high to prevent repainting
        denoise: Must be LOW (0.40–0.50) to preserve linework
        color_theme: Optional theme hint appended to positive prompt
        seed: -1 for random

    Returns:
        PIL Image: Colorized Gege cover-style output
    """
    b64_bw = _encode_image(bw_image)

    positive = (
        "gegecover, painterly watercolor cover, muted color palette, "
        "soft brush texture, cinematic color grading, volume cover art, "
        "expressive paint strokes, rough paint texture, artistic"
    )
    if color_theme:
        positive += f", {color_theme}"

    payload = {
        "init_images": [b64_bw],
        "prompt": positive,
        "negative_prompt": (
            "monochrome, grayscale, neon colors, oversaturated, vibrant, "
            "cel shading, flat colors, anime coloring, airbrushed, sharp, "
            "digital painting, glossy, hyperrealistic, photorealistic, nsfw"
        ),
        "sampler_name": "DPM++ SDE",
        "scheduler": "Karras",
        "steps": 25,
        "cfg_scale": 6.5,
        "denoising_strength": denoise,
        "width": 1024,
        "height": 1024,
        "seed": seed,
        "override_settings": {
            "sd_model_checkpoint": BASE_CHECKPOINT,
        },
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "image": b64_bw,
                        "module": "lineart_anime_denoise",
                        "model": CANNY_SDXL_MODEL,
                        "weight": canny_weight,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "resize_mode": 1,
                        "control_mode": 0,
                    }
                ]
            },
            "additional networks for generating": {
                "args": [
                    True,
                    "LoRA",
                    COLOR_LORA_NAME,
                    lora_weight,
                    lora_weight,
                    None, None, None, None, None, None,
                ]
            }
        },
    }

    result = _post("/sdapi/v1/img2img", payload)
    return _decode_image(result["images"][0])


def upscale_image(
    image,
    scale: float = 2.0,
    upscaler: str = "4x-UltraSharp",
) -> Image.Image:
    """
    Upscale image using ESRGAN via Forge API.

    Args:
        image: PIL Image or path
        scale: Upscale factor (2.0 = 1024 → 2048)
        upscaler: Model name. "4x-UltraSharp" for BW, "AnimeSharp 4x" for color

    Returns:
        PIL Image: Upscaled output
    """
    b64 = _encode_image(image)
    payload = {
        "image": b64,
        "upscaling_resize": scale,
        "upscaler_1": upscaler,
        "upscaler_2": "None",
    }
    result = _post("/sdapi/v1/extra-single-image", payload)
    return _decode_image(result["image"])


def full_pipeline(
    input_image,
    mode: str = "full",
    color_theme: str = "",
    seed: int = -1,
) -> dict:
    """
    Run complete JJK Colorizer pipeline.

    Args:
        input_image: Path or PIL Image of input artwork
        mode: "bw" (Pipeline A only) or "full" (A + B)
        color_theme: Optional color hint for Pipeline B
        seed: -1 for random

    Returns:
        dict with keys "bw" (PIL Image) and optionally "color" (PIL Image)
    """
    results = {}

    bw = pipeline_a_bw(input_image, seed=seed)
    bw_up = upscale_image(bw, scale=2.0, upscaler="4x-UltraSharp")
    results["bw"] = bw_up

    if mode == "full":
        color = pipeline_b_color(bw, color_theme=color_theme, seed=seed)
        color_up = upscale_image(color, scale=2.0, upscaler="AnimeSharp 4x")
        results["color"] = color_up

    return results
