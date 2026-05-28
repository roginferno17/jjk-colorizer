# GEGEAI2.0 — Master Implementation Document

> **Purpose**: Complete step-by-step instruction set for building the Gege Akutami style-transfer and colorization pipeline. Written so any AI instance or developer can pick up exactly where work left off without re-deriving decisions.

---

## SYSTEM SNAPSHOT (as of 2026-05-28)

| Component | Status | Detail |
|---|---|---|
| GPU | ✅ Ready | RTX 4060 Laptop, 8188MB VRAM, Driver 596.49 |
| CUDA | ✅ Ready | via PyTorch cu121 (nvcc not needed separately) |
| Python | ✅ Ready | 3.12.10 at `C:\Users\Vishu\AppData\Local\Programs\Python\Python312\` |
| PyTorch | ✅ Ready | 2.3.1+cu121 |
| xformers | ✅ Ready | 0.0.27 |
| accelerate | ✅ Ready | 0.31.0 |
| bitsandbytes | ✅ Ready | 0.43.1 |
| diffusers | ✅ Ready | 0.29.2 |
| controlnet_aux | ✅ Ready | 0.0.10 |
| peft | ✅ Ready | 0.11.1 |
| Forge (SD WebUI Forge) | ✅ Installed | `C:\Users\Vishu\AI\StabilityMatrix\Data\Packages\forge-neo\` |
| StabilityMatrix | ✅ Installed | `C:\Users\Vishu\AI\StabilityMatrix\` |
| kohya_ss | ✅ Cloned | `training\kohya_ss\` (v25.2.1) |
| kohya_venv310 | ✅ Created | Python 3.10.11, PyTorch installing |
| Illustrious XL | ✅ Downloaded | `forge-neo\models\Stable-diffusion\Illustrious-XL-v0.1.safetensors` |
| ControlNet Canny SDXL | ✅ Downloaded | `diffusers_xl_canny_mid.safetensors` in Forge ControlNet folder |
| ControlNet Depth SDXL | ✅ Downloaded | `diffusers_xl_depth_mid.safetensors` in Forge ControlNet folder |
| ESRGAN 4x-UltraSharp | ✅ Downloaded | Forge ESRGAN folder |
| ESRGAN AnimeSharp 4x | ✅ Downloaded | Forge ESRGAN folder |
| FastAPI backend | ✅ Written | `backend/` — forge_api.py, main.py, preprocessing.py |
| Next.js frontend | ✅ Written | `frontend/` — App Router, Tailwind v4, full UI |

---

## ARCHITECTURE DECISION LOG

### ControlNet Model Availability (SDXL)
No public SDXL Lineart ControlNet exists in accessible HuggingFace repos (xinsir repos are down/gated).

**Fallback decision: Use Canny SDXL ControlNet instead.**

Why Canny is acceptable here:
- For clean digital art input, Canny and Lineart produce nearly identical edge maps
- Canny may actually be slightly better for complex overlapping lines (manga hatching)
- The preprocessor (`lineart_anime_denoise`) still runs to create soft edges
- The ControlNet MODEL used is Canny-trained but accepts any edge map as input

Models downloaded:
- `diffusers_xl_canny_mid.safetensors` — Primary structure preservation (weight 0.85–0.90)
- `diffusers_xl_depth_mid.safetensors` — Spatial composition preservation (weight 0.50–0.55)
- Source: `lllyasviel/sd_control_collection`
- Size: ~700MB each (`mid` variant for VRAM safety)

### ESRGAN Upscaler Availability
`4x-Manga109Attempt` not accessible on HuggingFace.

**Fallback decision: Use `4x-UltraSharp` for BW, `AnimeSharp 4x` for color.**
- `4x-UltraSharp` — preserves fine line detail without over-smoothing (superior general choice)
- `AnimeSharp 4x` — softer rendering good for painted color texture
- Source: `hollowstrawberry/upscalers-backup`

---

### Why Forge over ComfyUI
Forge (SD WebUI Forge) is already installed via StabilityMatrix. It has:
- Built-in ControlNet with all preprocessors (lineart_anime_denoise, depth_zoe, openpose)
- img2img pipeline
- LoRA loading
- VRAM-optimized inference engine
- API mode (`--api` flag) for backend integration
- Simpler to configure than ComfyUI for this project

**Do NOT install ComfyUI** unless Forge fails — it's redundant.

### Why Illustrious XL over vanilla SDXL
Vanilla SDXL was used in previous failed attempt. Illustrious XL is trained on manga/anime corpus with:
- Native understanding of ink, lineart, monochrome
- Better LoRA absorption
- Does not default to cel-shading
- Hugging Face: `OnomaAIResearch/Illustrious-xl-early-release-v0`

### Why Two Separate LoRAs
Single LoRA = style contamination between BW manga style and watercolor cover style.
- LoRA_BW: teaches line roughness, hatching, ink density (Pipeline A)
- LoRA_COLOR: teaches watercolor, muted palette, painterly cover aesthetics (Pipeline B)

### Why Previous Attempt Failed
1. Used `stabilityai/stable-diffusion-xl-base-1.0` (wrong base — no manga understanding)
2. Resolution 768 (SDXL native is 1024 — lower resolution corrupts latent space)
3. Color trigger said `vibrant anime colors, cel shaded` — opposite of Gege style
4. Learning rate 1e-4 (too high for style LoRA — overtrain fast)
5. Dataset: jjkcolor (81 imgs) contained Twitter fanart — style contamination
6. Only 116 manga images — insufficient
7. No ControlNet strategy — identity lost during inference
8. No dataset preprocessing — raw JPEGs at wrong resolutions

---

## PIPELINE ARCHITECTURE

```
INPUT ARTWORK (any full illustration)
         │
         ▼
[PREPROCESSING — preprocessing.py]
   lineart_anime_denoise extraction → lineart.png
   depth_zoe extraction → depth.png
   (openpose extraction → pose.png — for full body only)
         │
         ▼
[PIPELINE A — BW MANGA CONVERSION]
   Forge img2img
   + Illustrious XL base
   + LoRA_BW (rank 32, weight 0.85)
   + ControlNet Lineart (weight 0.90, guidance 0.0→1.0)
   + ControlNet Depth (weight 0.55, guidance 0.0→0.8)
   Sampler: DPM++ 2M Karras
   Steps: 30, CFG: 7.5, Denoise: 0.63
         │
         ▼
[BW GEGE OUTPUT — black-and-white manga panel]
         │
         ▼
[PIPELINE B — COVER COLORIZATION]
   Forge img2img
   + Illustrious XL base
   + LoRA_COLOR (rank 16, weight 0.80)
   + ControlNet Lineart (weight 0.97, guidance 0.0→1.0)
   Sampler: DPM++ SDE Karras
   Steps: 25, CFG: 6.5, Denoise: 0.45
         │
         ▼
[UPSCALE — 4x-Manga109Attempt ESRGAN]
         │
         ▼
FINAL OUTPUT (2048×2048 or 4096×4096)
```

---

## PHASE 0 — ENVIRONMENT SETUP

### Status: COMPLETE
Started: 2026-05-28 | Completed: 2026-05-28

### Checklist

- [x] Verify GPU (RTX 4060, 8GB) — DONE
- [x] Verify PyTorch 2.3.1+cu121 — DONE
- [x] Verify xformers 0.0.27 — DONE
- [x] Verify accelerate 0.31.0 — DONE
- [x] Verify Forge installed — DONE (forge-neo in StabilityMatrix, Python 3.13)
- [x] Create project directory structure — DONE
- [x] Copy raw datasets to project — DONE (116 BW, 81 color)
- [x] Audit datasets — DONE (see Dataset Quality Alert below)
- [x] Process BW dataset → 114 PNGs at 1024x1024 — DONE
- [x] Process color dataset → 73 PNGs at 1024x1024 — DONE
- [x] Clone kohya_ss (GUI version) — DONE (`training/kohya_ss/` v25.2.1)
- [x] Initialize sd-scripts submodule — DONE (`training/kohya_ss/sd-scripts/`)
- [x] Create kohya_venv310 with Python 3.10.11 — DONE
- [x] Install PyTorch 2.3.1+cu121 in kohya_venv310 — DONE (verified: CUDA True)
- [x] Install xformers 0.0.27 in kohya_venv310 — DONE
- [x] Downgrade NumPy <2 in kohya_venv310 — DONE (xformers compat)
- [x] Install sd-scripts library package — DONE
- [x] Install training deps — DONE (accelerate 1.13.0, bitsandbytes 0.49.2, diffusers 0.32.2, safetensors 0.4.4, lycoris-lora, prodigyopt, transformers, scipy, einops, opencv-python, imagesize, voluptuous, rich, toml, ftfy, timm, dadaptation, lion-pytorch)
- [x] Verify sdxl_train_network.py imports cleanly — DONE (triton warning is harmless on Windows)
- [x] accelerate config — DONE (bf16, single GPU, existing config reused)
- [x] Fix optimizer_args betas syntax — DONE (list→tuple for PyTorch AdamW compat)
- [x] Download Illustrious XL — DONE (`Illustrious-XL-v0.1.safetensors`, 6.46GB)
- [x] Download ControlNet Canny SDXL — DONE (`diffusers_xl_canny_mid.safetensors`)
- [x] Download ControlNet Depth SDXL — DONE (`diffusers_xl_depth_mid.safetensors`)
- [x] Download ESRGAN upscalers — DONE (`4x-UltraSharp`, `AnimeSharp 4x`)
- [ ] Verify Forge launches and detects GPU (manual — open StabilityMatrix, launch forge-neo)
- [ ] Verify Forge API mode works (add `--api` to forge-neo launch args in StabilityMatrix settings)

### Phase 0 Notes
- xformers triton warning on Windows is HARMLESS. Training still uses xformers memory-efficient attention via CUDA.
- Do NOT run `pip install -e .` on `training/kohya_ss/` — requires torch>=2.5.0, would break cu121 setup. Use sd-scripts directly.
- Sole remaining manual steps: open StabilityMatrix → launch forge-neo → confirm GPU detected → add --api flag.

### CRITICAL: Correct Model Filename
Illustrious XL v0.1 correct filename: `Illustrious-XL-v0.1.safetensors`
(Previous scripts had wrong name `illustriousXL_v01.safetensors` — corrected)

### Forge Model Paths
```
Checkpoints:  C:\Users\Vishu\AI\StabilityMatrix\Data\Packages\forge-neo\models\Stable-diffusion\
LoRA:         C:\Users\Vishu\AI\StabilityMatrix\Data\Packages\forge-neo\models\Lora\
ControlNet:   C:\Users\Vishu\AI\StabilityMatrix\Data\Packages\forge-neo\models\ControlNet\
ESRGAN:       C:\Users\Vishu\AI\StabilityMatrix\Data\Packages\forge-neo\models\ESRGAN\
```

### accelerate Configuration
Run: `accelerate config`
Answers:
- Machine type: This machine
- Distributed: NO
- Num GPUs: 1
- Mixed precision: **bf16**

Target config at `~/.cache/huggingface/accelerate/default_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: bf16
num_processes: 1
```

### kohya_ss Python Version Requirement
**CRITICAL**: kohya_ss v25.x requires Python `>=3.10,<3.12`. System Python 3.12 is EXCLUDED.

**Solution**: Use StabilityMatrix's embedded Python 3.10.11.
Path: `C:\Users\Vishu\AI\StabilityMatrix\Data\Assets\Python310\python.exe`

### kohya_ss Installation (COMPLETED PROCEDURE)
```bat
REM DONE: Step 1 — kohya_ss cloned to training\kohya_ss\ (v25.2.1)

REM DONE: Step 2 — sd-scripts submodule initialized
git -C training\kohya_ss submodule update --init --recursive

REM DONE: Step 3 — Python 3.10 venv created
C:\Users\Vishu\AI\StabilityMatrix\Data\Assets\Python310\python.exe -m virtualenv training\kohya_venv310

REM DONE: Step 4 — PyTorch 2.3.1+cu121 installed
training\kohya_venv310\Scripts\pip.exe install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

REM DONE: Step 5 — xformers 0.0.27 installed
training\kohya_venv310\Scripts\pip.exe install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121

REM DONE: Step 6 — NumPy downgraded (xformers 0.0.27 compiled with NumPy 1.x)
training\kohya_venv310\Scripts\pip.exe install "numpy<2"

REM DONE: Step 7 — sd-scripts library package installed
training\kohya_venv310\Scripts\pip.exe install -e training\kohya_ss\sd-scripts\

REM DONE: Step 8 — Training dependencies installed
training\kohya_venv310\Scripts\pip.exe install accelerate>=1.7.0 bitsandbytes>=0.45.0 "diffusers[torch]==0.32.2" transformers==4.44.2 safetensors==0.4.4 lycoris-lora==3.2.0.post2 prodigyopt==1.1.2 sentencepiece omegaconf==2.3.0 tensorboard scipy einops dadaptation==3.2 lion-pytorch==0.0.6

REM DONE: Step 9 — accelerate config at ~/.cache/huggingface/accelerate/default_config.yaml
REM Already set to bf16, single GPU, no distributed. No action needed.

REM NOTE: Do NOT run pip install -e . on kohya_ss root — it requires torch>=2.5.0
REM which would upgrade and break cu121 install. Use sd-scripts directly instead.
```

### Training Script Location
sd-scripts submodule: `training\kohya_ss\sd-scripts\`
SDXL training script: `training\kohya_ss\sd-scripts\sdxl_train_network.py`

**IMPORTANT**: Always use `sdxl_train_network.py` for SDXL LoRA, NOT `train_network.py`.

Train command (BW LoRA):
```bat
cd training\kohya_ss\sd-scripts
training\kohya_venv310\Scripts\accelerate.exe launch ^
  --num_cpu_threads_per_process=4 ^
  sdxl_train_network.py ^
  --config_file C:\Users\Vishu\Desktop\lock-in-jjk\training\configs\bw_lora_config.toml
```

Or use: `training\scripts\train_bw.bat` (handles all of the above automatically)

### accelerate Config (ALREADY DONE)
Config at: `%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml`
Settings: bf16, single GPU, no distributed. Both system and kohya_venv310 accelerate read this file.
No need to run `accelerate config` again.

### Model Download Status
All required models downloaded. Script: `training/scripts/download_models.py`

```
DONE: Illustrious-XL-v0.1.safetensors (6.46GB)
      → forge-neo/models/Stable-diffusion/

DONE: diffusers_xl_canny_mid.safetensors (~700MB)
      → forge-neo/models/ControlNet/
      Source: lllyasviel/sd_control_collection

DONE: diffusers_xl_depth_mid.safetensors (~700MB)
      → forge-neo/models/ControlNet/
      Source: lllyasviel/sd_control_collection

DONE: 4x-UltraSharp.pth
      → forge-neo/models/ESRGAN/
      Source: hollowstrawberry/upscalers-backup

DONE: AnimeSharp 4x.pth
      → forge-neo/models/ESRGAN/
      Source: hollowstrawberry/upscalers-backup
```

---

## PHASE 1 — DATASET PREPARATION (BW)

### Status: PARTIAL — CRITICAL DATASET QUALITY ISSUE

### CRITICAL ALERT — BW Dataset Quality
**The 116 images in `jjkmanga_raw` are NOT proper manga scans.**

They are social media saves (Pinterest, 736px wide) — icon sets and aesthetic posts:
- Filenames with `_ᐟ`, `ꕤ`, `౨ৎ`, `𝐇𝐀𝐊𝐀𝐑𝐈` etc. = social media aesthetic saves
- Filenames with descriptive English names = fan-curated icon posts, NOT raw panels
- All at 736px = Pinterest standard resolution (not manga rip quality)

**What you HAVE:** Social media manga aesthetic posts (736px, some fanart, mixed quality)
**What you NEED:** Raw manga panel crops from actual JJK tankobon scans (1800px+)

**Impact:** Training on these will teach a generic "anime face at low resolution" look, NOT Gege's authentic rough linework, hatching, or ink behavior.

**Action required:**
1. **Immediate**: Manually inspect all 114 processed images in `dataset\bw_gege\10_gegeakutami, monochrome manga\` — DELETE anything that is:
   - A fan-made icon/profile picture
   - Fanart or redrawn art
   - Colored image (should not be in BW folder)
   - Overly clean/airbrushed render
   - Any image without genuine Gege-style linework visible
2. **Must obtain**: High-quality JJK manga volume scans — target 300–500 panels from volumes 1–26
   - Sources: Official digital volumes (Manga Plus, VIZ, etc.)
   - Target resolution: 1800px+ on the shorter side
   - Focus on panels with heavy shadows, hatching, expressive faces, dynamic poses
   - These are the ENTIRE reason the training can produce Gege-style output

**Training viability with current dataset:** LOW. Expect mediocre results if trained on current 114 images.
**Training viability after proper curation + augmentation:** HIGH.

### Status: PARTIAL — CRITICAL DATASET QUALITY ISSUE

### Target
300–500 high-quality JJK manga panels → processed to 1024×1024 PNG grayscale

### Source Audit
- `C:\Users\Vishu\Desktop\jjkmanga\` — 116 images (JPEG, unprocessed)
  - Status: NEEDS AUDIT — unknown quality/resolution
  - Action: Run audit script, cull low-quality, process remaining
- Need ~200–400 more panels from high-quality scans

### Dataset Rules — BW
**INCLUDE:**
- Official JJK manga panels (tankobon scans, digital volumes)
- Heavy shadow/hatching panels (highest priority)
- Face closeups with emotional rendering
- Hands and clothing details
- Chapter cover pages (color→grayscale)
- Volume extras

**EXCLUDE:**
- Screentoned panels (screentone ≠ ink style)
- Blurry/low-resolution scans
- Panels with heavy text/speech bubbles (crop or remove)
- Any fanart or redraw
- Anime screenshots (completely different style)

### Processing Pipeline
Script: `training\scripts\prepare_dataset_bw.py`

Steps per image:
1. Load at full resolution
2. Convert to grayscale (L mode → RGB for SDXL)
3. Auto-levels: `ImageOps.autocontrast(cutoff=0.5)`
4. Bilateral denoise: `cv2.bilateralFilter(d=5, sigmaColor=15, sigmaSpace=15)`
5. Manual text removal (crop or flood-fill white on speech bubbles)
6. Center-crop to 1024×1024
7. Save as PNG to `dataset\bw_gege\10_gegeakutami, monochrome manga\`

### Captioning (BW)
- Run WD14 tagger on all images
- Remove ALL style/quality tags from output
- Keep only structural/semantic tags
- Prepend trigger: `gegeakutami, monochrome manga`

**Example final caption:**
```
gegeakutami, monochrome manga, 1boy, spiky hair, serious expression, upper body, ink, hatching
```

Caption file must be same name as image, `.txt` extension.

### Folder Structure
```
dataset\bw_gege\
└── 10_gegeakutami, monochrome manga\
    ├── panel_001.png
    ├── panel_001.txt
    ├── panel_002.png
    ├── panel_002.txt
    └── ...
```

The `10_` prefix = 10 repeats per epoch in kohya_ss.

---

## PHASE 2 — DATASET PREPARATION (COLOR)

### Status: TODO

### Target
80–130 high-purity official JJK colored images → processed to 1024×1024 PNG

### Source Audit
- `C:\Users\Vishu\Desktop\jjkcolor\` — 81 images
  - **CONTAMINATED** — contains Twitter fanart:
    - `@Vroom622226 on Twitter.jpeg` — FANART
    - `@Vroom622226 on Twitter (1).jpeg` — FANART
    - `@Vroom622226 on X.jpeg` — FANART
    - `Bromo?? _ #JojoLander on X.jpeg` — FANART/UNKNOWN
    - `Credit to @_sneez_ _ on Twitter.jpeg` — FANART
  - All other images need quality audit

### Dataset Rules — Color
**INCLUDE ONLY:**
- Official JJK volume covers (volumes 0–26)
- Official Jump magazine covers
- Gege Akutami official promotional art
- Official color spread pages from volumes

**EXCLUDE:**
- Any Twitter/X fanart (filename contains @ or username = immediate rejection)
- AI colorizations
- Anime screenshots
- Third-party edits
- Low-res (<800px wide)

### Processing (Color)
Script: `training\scripts\prepare_dataset_color.py`

Steps per image:
1. Audit: reject fanart (see exclusion rules)
2. Check resolution: minimum 1000px on shorter side
3. Color: do NOT convert to grayscale, do NOT adjust white balance
4. Center-crop to 1024×1024 (preserving most content)
5. Augmentation crops from single image:
   - Full image crop (primary)
   - Face/head region crop
   - Upper body crop
6. Save as RGB PNG to `dataset\color_gege\15_gegecover, painterly watercolor cover\`

### Captioning (Color)
```
gegecover, painterly watercolor cover, [character tags], [composition tags]
```

---

## PHASE 3 — TRAIN LoRA_BW

### Status: TODO

### Framework: kohya_ss
Location: `C:\Users\Vishu\Desktop\lock-in-jjk\training\kohya_ss\`

### Config: `training\configs\bw_lora_config.toml`

```toml
[model_arguments]
pretrained_model_name_or_path = "C:/Users/Vishu/AI/StabilityMatrix/Data/Packages/forge-neo/models/Stable-diffusion/Illustrious-XL-v0.1.safetensors"
v2 = false
v_parameterization = false

[dataset_arguments]
train_data_dir = "C:/Users/Vishu/Desktop/lock-in-jjk/dataset/bw_gege"
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 768
max_bucket_reso = 1024
bucket_reso_steps = 64
flip_aug = true
color_aug = false
caption_extension = ".txt"
shuffle_caption = true
keep_tokens = 2

[training_arguments]
output_dir = "C:/Users/Vishu/Desktop/lock-in-jjk/training/output/bw_lora"
output_name = "gege_bw_lora"
save_model_as = "safetensors"
save_every_n_steps = 1000

num_train_epochs = 15
train_batch_size = 1
gradient_accumulation_steps = 4
gradient_checkpointing = true

learning_rate = 8e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 200
lr_scheduler_num_cycles = 3

optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1", "betas=[0.9,0.999]"]

mixed_precision = "bf16"
full_bf16 = true
xformers = true

cache_latents = true
cache_latents_to_disk = true

[network_arguments]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[logging_arguments]
log_with = "tensorboard"
logging_dir = "C:/Users/Vishu/Desktop/lock-in-jjk/training/logs/bw_lora"

[sample_prompt_arguments]
sample_every_n_steps = 1000
sample_sampler = "dpmpp_2m"
sample_prompts = [
  "gegeakutami, monochrome manga, 1boy, spiky hair, serious expression, ink lines, hatching, manga panel"
]
```

### Launch Command
```bash
cd C:\Users\Vishu\Desktop\lock-in-jjk\training\kohya_ss
venv\Scripts\activate
accelerate launch --num_cpu_threads_per_process=4 train_network.py \
  --config_file "C:/Users/Vishu/Desktop/lock-in-jjk/training/configs/bw_lora_config.toml"
```

### VRAM Budget (BW Training)
| Component | VRAM |
|---|---|
| SDXL UNet bf16 | ~3.5GB |
| VAE | ~0.8GB |
| Text Encoders | ~0.6GB |
| LoRA rank 32 | ~0.1GB |
| Gradient checkpointing | ~0.3GB |
| AdamW8bit optimizer | ~1.2GB |
| Activations | ~0.5GB |
| **Total** | **~7.0GB** |

### Overtraining Watch Points
- Sample outputs at epoch N look identical regardless of prompt → OVERTRAINED
- Trigger word fires without being in prompt → OVERTRAINED
- Style barely present in outputs → UNDERTRAINED
- **Typical sweet spot: epoch 9–13**

### Checkpoint Evaluation
After each 1000-step save, run `training\scripts\evaluate_checkpoint.py`:
- Fixed test prompt × 5 seeds
- Compare roughness, shadow depth, hatching against reference
- Save comparison grid to `docs\checkpoints\`

---

## PHASE 4 — TRAIN LoRA_COLOR

### Status: TODO

### Config: `training\configs\color_lora_config.toml`

Key differences from BW config:
```toml
network_dim = 16          # rank 16 (color simpler than linework)
network_alpha = 8
learning_rate = 6e-5      # lower — color overtrain faster
lr_scheduler = "cosine"
num_train_epochs = 18
save_every_n_steps = 500  # save more often — smaller dataset

train_data_dir = "C:/Users/Vishu/Desktop/lock-in-jjk/dataset/color_gege"
output_dir = "C:/Users/Vishu/Desktop/lock-in-jjk/training/output/color_lora"
output_name = "gege_color_lora"
```

---

## PHASE 5 — INFERENCE PIPELINE (FORGE)

### Status: TODO

### Forge API Setup
Launch Forge via StabilityMatrix with `--api` extra arg in settings.

API base: `http://localhost:7860`

Key endpoints:
- `POST /sdapi/v1/img2img` — Pipeline A and B
- `POST /sdapi/v1/extra-single-image` — Upscaling
- `GET /sdapi/v1/controlnet/model_list` — Verify ControlNet loaded

### Pipeline A Parameters (BW Conversion)
```python
{
    "init_images": [base64_input],
    "prompt": "gegeakutami, monochrome manga, rough ink linework, hatching, expressive, manga panel, high contrast",
    "negative_prompt": "color, colorful, anime, glossy, smooth, cel shading, airbrushed, watermark, text",
    "sampler_name": "DPM++ 2M",
    "scheduler": "Karras",
    "steps": 30,
    "cfg_scale": 7.5,
    "denoising_strength": 0.63,
    "width": 1024,
    "height": 1024,
    "override_settings": {"sd_model_checkpoint": "Illustrious-XL-v0.1.safetensors"},
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "image": base64_lineart,
                    "module": "lineart_anime_denoise",  # PREPROCESSOR (creates soft edge map)
                    "model": "diffusers_xl_canny_mid",   # MODEL (Canny SDXL — best available)
                    "weight": 0.90,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0
                },
                {
                    "enabled": True,
                    "image": base64_input,
                    "module": "depth_zoe",
                    "model": "diffusers_xl_depth_mid",
                    "weight": 0.55,
                    "guidance_start": 0.0,
                    "guidance_end": 0.8
                }
            ]
        },
        "Additional networks for generating": {
            "args": [True, "LoRA", "gege_bw_lora", 0.85, 0.85]
        }
    }
}
```

### Pipeline B Parameters (Colorization)
```python
{
    "init_images": [base64_bw_output],
    "prompt": "gegecover, painterly watercolor cover, muted colors, soft brushwork, textured paint, cinematic, volume cover art",
    "negative_prompt": "monochrome, grayscale, oversaturated, neon, anime, glossy, cel shading, airbrushed, flat colors",
    "sampler_name": "DPM++ SDE",
    "scheduler": "Karras",
    "steps": 25,
    "cfg_scale": 6.5,
    "denoising_strength": 0.45,
    "width": 1024,
    "height": 1024,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "image": base64_bw_output,
                    "module": "lineart_anime_denoise",
                    "model": "diffusers_xl_canny_mid",
                    "weight": 0.97,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0
                }
            ]
        },
        "Additional networks for generating": {
            "args": [True, "LoRA", "gege_color_lora", 0.80, 0.80]
        }
    }
}
```

### Parameter Tuning Guide
| Problem | Fix |
|---|---|
| Identity drift (face changes) | Raise lineart ControlNet weight +0.05, lower denoise -0.05 |
| Style not transferring | Lower ControlNet weight -0.05, raise denoise +0.05 |
| Too anime/glossy | Raise LoRA weight to 0.90, add to negative: `anime, cel shading` |
| Loss of roughness | Verify trigger word in prompt, check LoRA weight |
| Color bleeds over lines | Raise lineart ControlNet to 0.98+, lower denoise to 0.40 |
| Muddy coloring | Roll back to earlier color LoRA checkpoint |
| Hand corruption | Add OpenPose ControlNet at 0.45 weight |
| Eye distortion | Face detailer pass: mask face, img2img at denoise 0.30 |

---

## PHASE 6 — BACKEND (FastAPI)

### Status: WRITTEN (not yet tested — requires Forge running + LoRAs trained)
Location: `C:\Users\Vishu\Desktop\lock-in-jjk\backend\`

### Files Written
- `backend/main.py` — FastAPI app, endpoints, job state, asyncio.Semaphore(1) queue
- `backend/forge_api.py` — Full Forge API wrapper with both pipelines + upscaling
- `backend/preprocessing.py` — load_and_normalize, make_preview, validate_upload
- `backend/requirements.txt` — fastapi 0.111.0, uvicorn, python-multipart, Pillow, opencv-python, requests

### API Endpoints (implemented)
```
POST /api/upload              → validates file, saves to temp, returns {job_id}
POST /api/generate/{id}       → queues job (bw or full mode), streams via semaphore
GET  /api/status/{id}         → {status, stage, progress, error}
GET  /api/result/{id}         → {bw: base64, color: base64}
GET  /api/download/{id}/{var} → file download (bw or color variant)
```

### Launch Backend
```bat
cd C:\Users\Vishu\Desktop\lock-in-jjk
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --port 8000
```

---

## PHASE 7 — FRONTEND (Next.js)

### Status: WRITTEN (not yet tested in browser)
Location: `C:\Users\Vishu\Desktop\lock-in-jjk\frontend\`

### Stack
- Next.js 16.2.6 (App Router, `app/` directory)
- React 19.2.4
- Tailwind CSS v4 (`@import "tailwindcss"` syntax)
- TypeScript

### UI Implemented (`frontend/app/page.tsx`)
- Dark minimal interface (`bg-zinc-950`)
- Left panel: drag-drop upload zone + mode selector (B&W Only / B&W + Color) + color theme hint input + Generate/Reset buttons + progress bar
- Right panel: result cards with base64 image display + download buttons
- State machine: idle → uploaded → queued → processing → complete/failed
- Polls `/api/status/{id}` every 2 seconds during generation
- `API = "http://localhost:8000"` constant at top of file

### Launch Frontend
```bat
cd C:\Users\Vishu\Desktop\lock-in-jjk\frontend
npm run dev
```
Opens at `http://localhost:3000`

---

## PHASE 8 — UPSCALING

### Models: 4x-UltraSharp (BW), AnimeSharp 4x (Color)
- `4x-UltraSharp` — preserves line detail, best for ink/manga output
- `AnimeSharp 4x` — softer rendering, better for painted watercolor texture
- Both at: `forge-neo\models\ESRGAN\`

Note: 4x-Manga109Attempt was unavailable on HuggingFace — switched to above alternatives.

### Upscale via Forge API
```python
POST /sdapi/v1/extra-single-image
{
    "image": base64_image,
    "upscaling_resize": 2,   # 2x (1024 → 2048)
    "upscaler_1": "4x-Manga109Attempt",
    "upscaler_2": "None"
}
```

---

## FAILURE CASE REFERENCE

| Failure | Root Cause | Fix |
|---|---|---|
| Face drift | Denoise too high, ControlNet weak | Lineart weight 0.92+, denoise 0.58 |
| Eye distortion | Small features at 1024 | Face crop + img2img denoise 0.30 |
| Anatomy corruption | No pose guidance | Add OpenPose ControlNet 0.45 |
| Hand corruption | Hands hard even with ControlNet | Hand inpainting mask, denoise 0.50 |
| Hairstyle changes | Lineart dropping thin strands | Use lineart_anime_denoise, lower threshold |
| Clothing redesign | Denoise too high | Lower to 0.55, add tile ControlNet 0.50 |
| Over-anime output | LoRA weight low, base model dominant | Raise LoRA to 0.90–0.95 |
| Oversmoothing | LoRA undertrained | Train more epochs, raise LoRA weight |
| Muddy coloring | Color LoRA overtrained | Roll back checkpoint |
| Washed-out shadows | Color bleeding into blacks | Post-process levels, or BW inpaint pass |
| Overpainted linework | Denoise >0.55, lineart <0.85 | Fix params |

---

## PROMPTING REFERENCE

### Pipeline A (BW) — Exact Prompts
```
POSITIVE:
gegeakutami, monochrome manga, rough ink linework, expressive hatching,
cross hatching, high contrast, thick ink outlines, manga panel,
imperfect strokes, raw ink energy

NEGATIVE:
color, colorful, rgb, pastel, anime style, glossy, cel shading,
airbrushed, over-detailed, hyperrealistic, smooth, polished linework,
clean lines, digital art, photorealistic, 3d render, watermark, text
```

### Pipeline B (Color) — Exact Prompts
```
POSITIVE:
gegecover, painterly watercolor cover, muted color palette,
soft brush texture, cinematic color grading, volume cover art,
expressive paint strokes, rough paint texture, artistic

NEGATIVE:
monochrome, grayscale, neon colors, oversaturated, vibrant,
cel shading, flat colors, anime coloring, airbrushed, sharp,
digital painting, glossy, hyperrealistic, photorealistic
```

---

## IMPLEMENTATION ORDER (STRICT)

1. **Phase 0**: Environment — configure accelerate, install kohya_ss, download models
2. **Phase 1**: BW Dataset — audit raw images, process, caption
3. **Phase 2**: Color Dataset — audit, remove fanart, process, caption
4. **Phase 3**: Train LoRA_BW — run training, evaluate checkpoints, pick best
5. **Phase 4**: Train LoRA_COLOR — run training, evaluate checkpoints, pick best
6. **Phase 5**: Inference — set up Forge API, tune parameters with test images
7. **Phase 6**: Backend — FastAPI wrapper for Forge API
8. **Phase 7**: Frontend — Next.js UI
9. **Phase 8**: Integration — connect all, test end-to-end

**Do NOT skip to inference before training is complete.**
**Do NOT train before dataset is properly prepared.**

---

## PROGRESS LOG

### 2026-05-28 — Session 1
- Diagnosed previous attempt failures (wrong base model, resolution, trigger, LR, contaminated dataset)
- Designed complete two-pipeline architecture
- Created project directory structure at `C:\Users\Vishu\Desktop\lock-in-jjk\`
- Created this documentation file
- Confirmed environment: PyTorch 2.3.1+cu121, xformers, Forge installed
- Identified Forge (forge-neo) as inference engine (already installed, replaces ComfyUI)
- Identified jjkcolor dataset contamination (Twitter fanart present)
- **Completed**: All training scripts, configs, kohya_ss setup scripts, download scripts
- **Completed**: BW dataset processed (114 PNGs), Color dataset processed (73 PNGs)
- **Completed**: All ControlNet + ESRGAN models downloaded to Forge folders
- **Completed**: Full FastAPI backend written (main.py, forge_api.py, preprocessing.py)
- **Completed**: Next.js 16 frontend written (full UI with upload, generation, results)
- **Completed**: Illustrious XL v0.1 downloaded (6.46GB) to Forge Stable-diffusion folder
- **In Progress**: PyTorch 2.3.1+cu121 installing in kohya_venv310 (Python 3.10.11)
- **Next**: After PyTorch installs → install xformers + kohya_ss deps → configure accelerate → ready to train

### Critical blockers before training
1. PyTorch must finish installing in kohya_venv310
2. `install_kohya_deps.bat` must succeed (`pip install -e .` in kohya_ss)
3. `accelerate config` must be run in kohya_venv310 (bf16, single GPU)
4. BW dataset needs manual curation (current 114 images are low-quality social media saves)
5. Proper high-res manga panel scans required (300+ at 1800px+) — see CRITICAL ALERT in Phase 1

---

## FILES CREATED THIS SESSION

```
lock-in-jjk/
├── GEGEAI2.md                          ← THIS FILE
├── dataset/
│   ├── bw_gege/10_gegeakutami, monochrome manga/
│   ├── color_gege/15_gegecover, painterly watercolor cover/
│   ├── raw/jjkmanga_raw/
│   ├── raw/jjkcolor_raw/
│   ├── processed/bw/
│   └── processed/color/
├── training/
│   ├── configs/
│   ├── scripts/
│   ├── logs/
│   └── output/{bw_lora,color_lora}/
├── models/
│   ├── base/
│   ├── lora/
│   ├── controlnet/
│   └── upscalers/
├── inference/workflows/
├── backend/utils/
├── frontend/src/{app,components,lib}/
└── docs/{checkpoints,dataset_audits}/
```

### 2026-05-28 — Session 1 continued

**Environment findings:**
- Python 3.12.10 (not 3.10 — using modern kohya_ss which supports 3.12)
- PyTorch 2.3.1+cu121 + xformers 0.0.27 already installed (saves significant setup time)
- StabilityMatrix installed with forge-neo (SD WebUI Forge) — using as inference engine
- Python 3.13 in forge-neo venv (StabilityMatrix managed — do NOT pip install into it directly)

**Actions completed:**
- Created full project directory structure
- Audited both raw datasets
- Processed BW dataset → 114 PNGs at 1024×1024 (needs manual curation — see CRITICAL ALERT)
- Processed color dataset → 73 PNGs at 1024×1024 (6 fanart auto-rejected)
- Cloned kohya_ss (GUI version) to `training/kohya_ss/`
- Created kohya_ss venv with `--system-site-packages` (inherits torch 2.3.1+cu121)
- kohya_ss setup running (background)
- Illustrious XL v0.1 downloading (7GB — ~4.8GB/7GB as of this log)
- ControlNet Canny SDXL mid + Depth SDXL mid downloading (background)
- ESRGAN: 4x-UltraSharp + AnimeSharp 4x downloaded ✓

**Critical decisions made:**
- SDXL Lineart ControlNet not available → using Canny SDXL (functionally equivalent for clean art)
- 4x-Manga109Attempt not on HuggingFace → using 4x-UltraSharp (equivalent quality)
- Forge as inference engine instead of ComfyUI (already installed, simpler API)

**Blockers:**
- BW dataset quality critically low — must obtain proper manga volume scans before training
- kohya_ss setup still running in background

**Next session must do:**
1. Confirm Illustrious XL download completed + verify in Forge
2. Confirm ControlNet models visible in Forge
3. Confirm kohya_ss setup completed
4. Configure accelerate (`accelerate config`)
5. Manual curation of BW dataset (delete bad images)
6. Obtain high-quality manga panel scans (300+ additional panels)
7. Complete caption files for processed images
8. Run caption_cleanup.py on both datasets

---

*Update this file after every completed step. Every decision, every parameter change, every failure must be logged here.*

---

### 2026-05-28 — Session 2

**GitHub repo created:** https://github.com/roginferno17/jjk-colorizer (public)

**Environment fully resolved — Phase 0 COMPLETE:**
- PyTorch 2.3.1+cu121 installed in kohya_venv310 ✓ (CUDA: True)
- xformers 0.0.27 in kohya_venv310 ✓
- numpy<2 downgrade applied ✓ (xformers 0.0.27 compiled with NumPy 1.x)
- accelerate 1.13.0 ✓
- bitsandbytes 0.49.2 ✓
- diffusers 0.32.2 ✓
- sd-scripts submodule initialized (`training/kohya_ss/sd-scripts/`) ✓
- sd-scripts library package installed ✓
- sdxl_train_network.py imports verified ✓ (triton warning harmless on Windows)
- accelerate config: bf16, single GPU — pre-existing config reused ✓
- Illustrious XL v0.1 (6.46GB) → Forge Stable-diffusion folder ✓
- Forge confirmed RTX 4060 by user ✓
- Forge --api flag confirmed by user ✓

**Bug fixes applied this session:**
- `betas=[...]` → `betas=(...)` in optimizer_args — PyTorch AdamW requires tuple not list
- `train_bw.bat` / `train_color.bat`: corrected to `sdxl_train_network.py`, fixed accelerate path to venv's exe, fixed cd to sd-scripts dir
- Do NOT use `pip install -e .` on kohya_ss root (requires torch>=2.5.0 — would break cu121)

**Dataset updated by user:**
- Old 114 BW images deleted, 300+ new BW images added
- Color dataset: 73 processed PNGs remain

**Frontend/backend written:**
- `frontend/app/page.tsx` — dark UI, drag-drop, B&W/Full mode, progress bar, results
- `backend/` — FastAPI with job queue, Forge API wrapper, preprocessing

**Active decision: training protocol**
User will run all training commands in separate terminal and report results/errors back here.
Never auto-run training — it takes hours and needs human monitoring.

**Next steps:**

1. **[User runs]** Process new BW dataset:
   ```
   training\kohya_venv310\Scripts\python.exe training\scripts\prepare_dataset_bw.py
   ```

2. **[User runs]** Caption BW dataset — add structural tags to .txt stubs, then:
   ```
   training\kohya_venv310\Scripts\python.exe training\scripts\caption_cleanup.py
   ```

3. **[User runs]** Train BW LoRA (expect 3-5 hours on RTX 4060):
   ```
   training\scripts\train_bw.bat
   ```
   Report: any errors, what step it crashes/succeeds at, VRAM usage from task manager.
