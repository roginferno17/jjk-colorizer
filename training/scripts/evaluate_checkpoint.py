"""
evaluate_checkpoint.py
======================
Generates test images from a saved LoRA checkpoint to evaluate quality.
Run after each checkpoint save to pick the best epoch.

Uses the Forge API (must be running with --api flag).

Run:
    python evaluate_checkpoint.py --lora gege_bw_lora --checkpoint_step 5000 --mode bw
    python evaluate_checkpoint.py --lora gege_color_lora --checkpoint_step 3000 --mode color

Output: docs/checkpoints/<mode>_step<N>_seed<S>.png for each test
"""

import argparse
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

FORGE_URL = "http://localhost:7860"
DOCS_DIR = Path("docs/checkpoints")

BW_TEST_PROMPTS = [
    "gegeakutami, monochrome manga, 1boy, spiky hair, intense expression, hatching, manga panel",
    "gegeakutami, monochrome manga, closeup face, dramatic shadows, cross hatching, ink",
    "gegeakutami, monochrome manga, 1boy, serious expression, dark background, rough linework",
]

COLOR_TEST_PROMPTS = [
    "gegecover, painterly watercolor cover, 1boy, white hair, volume cover art, muted colors",
    "gegecover, painterly watercolor cover, 1boy, dark background, cinematic, soft brushwork",
    "gegecover, painterly watercolor cover, portrait, dramatic lighting, watercolor texture",
]

BW_NEGATIVE = "color, colorful, anime, glossy, smooth, cel shading, airbrushed, watermark, text, nsfw"
COLOR_NEGATIVE = "monochrome, grayscale, oversaturated, neon, anime, glossy, cel shading, flat colors, nsfw"

TEST_SEEDS = [42, 1234, 9999]


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def save_image(b64: str, path: Path):
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run_txt2img(prompt: str, negative: str, lora_name: str, seed: int) -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "steps": 25,
        "cfg_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "seed": seed,
        "alwayson_scripts": {
            "additional networks for generating": {
                "args": [True, "LoRA", lora_name, 0.85, 0.85, None, None, None, None, None, None]
            }
        }
    }
    resp = requests.post(f"{FORGE_URL}/sdapi/v1/txt2img", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["images"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True, help="LoRA name as loaded in Forge")
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--mode", choices=["bw", "color"], required=True)
    args = parser.parse_args()

    prompts = BW_TEST_PROMPTS if args.mode == "bw" else COLOR_TEST_PROMPTS
    negative = BW_NEGATIVE if args.mode == "bw" else COLOR_NEGATIVE
    tag = f"{args.mode}_step{args.checkpoint_step:07d}"

    print(f"Evaluating {args.lora} at step {args.checkpoint_step}...")
    print(f"Running {len(prompts)} prompts × {len(TEST_SEEDS)} seeds = {len(prompts)*len(TEST_SEEDS)} images\n")

    results = []
    for pi, prompt in enumerate(prompts):
        for seed in TEST_SEEDS:
            print(f"  Prompt {pi+1}/{len(prompts)}, seed {seed}...", end=" ", flush=True)
            try:
                b64 = run_txt2img(prompt, negative, args.lora, seed)
                out_path = DOCS_DIR / f"{tag}_p{pi+1}_s{seed}.png"
                save_image(b64, out_path)
                results.append(str(out_path))
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

    # Save evaluation log
    log = {
        "lora": args.lora,
        "step": args.checkpoint_step,
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "outputs": results,
        "prompts": prompts,
        "seeds": TEST_SEEDS,
    }
    log_path = DOCS_DIR / f"{tag}_eval.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nSaved {len(results)} images → {DOCS_DIR}/")
    print(f"Log: {log_path}")
    print("\nReview images and grade against rubric:")
    print("  Line roughness: smooth (bad) → instable organic (good)")
    print("  Shadow blacks: gray (bad) → crushed aggressive (good)")
    print("  Hatching: none (bad) → dense directional (good)")
    print("  Identity: different char (bad) → preserved (good)")


if __name__ == "__main__":
    main()
