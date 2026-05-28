# JJK Colorizer

Identity-preserving Gege Akutami manga style transfer and colorization pipeline.

**Pipeline A** — converts any artwork to Gege's raw B&W manga style (ink, hatching, high contrast)  
**Pipeline B** — colorizes the B&W result in Gege's volume cover watercolor aesthetic

Built on Illustrious XL + two custom LoRAs + ControlNet SDXL + SD WebUI Forge.

---

## Architecture

```
Input Image
    │
    ▼
[PIPELINE A — BW]
Forge img2img + LoRA_BW (rank 32)
+ ControlNet Canny SDXL (0.90) + Depth SDXL (0.55)
Illustrious XL base
    │
    ▼
B&W Manga Output
    │
    ▼
[PIPELINE B — COLOR]  ← optional
Forge img2img + LoRA_COLOR (rank 16)
+ ControlNet Canny SDXL (0.97)
    │
    ▼
Color Cover Output
    │
    ▼
[UPSCALE — ESRGAN]
4x-UltraSharp (BW) / AnimeSharp 4x (Color)
    │
    ▼
Final 2048×2048+ Output
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop 8GB)
- **CUDA**: 12.1+ (cu121)
- **RAM**: 16GB+
- **Disk**: ~30GB free (models + training data)

---

## Prerequisites

Install these before anything else:

1. **Python 3.10.11** — required for kohya_ss training (v25.x requires `>=3.10,<3.12`)
   - Use [StabilityMatrix](https://github.com/LykosAI/StabilityMatrix) → it ships Python 3.10 embedded
   - StabilityMatrix Python 3.10 path: `StabilityMatrix\Data\Assets\Python310\python.exe`

2. **SD WebUI Forge** via StabilityMatrix
   - Install StabilityMatrix → add package `Stable Diffusion WebUI Forge` (forge-neo)
   - In Forge launch settings, add `--api` to extra launch arguments
   - Forge will run at `http://localhost:7860`

3. **Git** + **gh CLI** (for repo operations)

4. **Node.js 18+** (for frontend)

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/jjk-colorizer.git
cd jjk-colorizer
```

### 2. Download required models

Place all models in your Forge installation folders:

```
Checkpoints → forge-neo/models/Stable-diffusion/
ControlNet  → forge-neo/models/ControlNet/
ESRGAN      → forge-neo/models/ESRGAN/
LoRA        → forge-neo/models/Lora/   ← after training
```

Run the download script (uses `huggingface_hub`):

```bash
pip install huggingface_hub
python training/scripts/download_models.py
```

Models downloaded:
- `Illustrious-XL-v0.1.safetensors` (~6.5GB) — base model
- `diffusers_xl_canny_mid.safetensors` (~700MB) — ControlNet structure
- `diffusers_xl_depth_mid.safetensors` (~700MB) — ControlNet depth
- `4x-UltraSharp.pth` — ESRGAN for BW output
- `AnimeSharp 4x.pth` — ESRGAN for color output

### 3. Set up kohya_ss training environment

```bash
# Clone kohya_ss (must be separate from this repo)
git clone https://github.com/bmaltais/kohya_ss.git training/kohya_ss
cd training/kohya_ss
git submodule update --init --recursive
cd ../..
```

Create Python 3.10 venv (use StabilityMatrix Python 3.10 path):

```bash
# Windows — adjust Python310 path to your StabilityMatrix install
"C:\Users\YOUR_USER\AI\StabilityMatrix\Data\Assets\Python310\python.exe" -m virtualenv training/kohya_venv310
```

Install training stack:

```bash
# PyTorch 2.3.1 cu121
training/kohya_venv310/Scripts/pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# xformers (must match torch version)
training/kohya_venv310/Scripts/pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121

# NumPy downgrade (xformers 0.0.27 compiled with NumPy 1.x)
training/kohya_venv310/Scripts/pip install "numpy<2"

# sd-scripts library
training/kohya_venv310/Scripts/pip install -e training/kohya_ss/sd-scripts/

# Training deps
training/kohya_venv310/Scripts/pip install accelerate>=1.7.0 bitsandbytes>=0.45.0 "diffusers[torch]==0.32.2" transformers==4.44.2 safetensors==0.4.4 lycoris-lora==3.2.0.post2 prodigyopt==1.1.2 sentencepiece omegaconf==2.3.0 tensorboard scipy einops opencv-python imagesize voluptuous rich toml ftfy timm dadaptation==3.2 lion-pytorch==0.0.6
```

> ⚠️ **Do NOT** run `pip install -e .` on `training/kohya_ss/` root — that package requires `torch>=2.5.0` and `xformers>=0.0.30`, which would overwrite the cu121 install.

Configure accelerate (run once):

```bash
training/kohya_venv310/Scripts/accelerate config
```
Answer: machine type → This machine, distributed → NO, GPUs → 1, mixed precision → **bf16**

### 4. Prepare datasets

#### BW dataset (manga panels)
Place high-quality JJK manga panel crops (1800px+) in `dataset/raw_bw/`:

```bash
python training/scripts/prepare_dataset_bw.py
```

Output: `dataset/bw_gege/10_gegeakutami, monochrome manga/` — 1024×1024 PNG + `.txt` caption stubs

> ⚠️ **Quality matters critically.** Social media saves (Pinterest 736px) produce poor results. You need proper tankobon scans — 300+ panels minimum.

#### Color dataset (official art only)
Place official JJK volume covers + color spreads in `dataset/raw_color/`:

```bash
python training/scripts/prepare_dataset_color.py
```

Output: `dataset/color_gege/15_gegecover, painterly watercolor cover/`

> Fanart is auto-rejected by filename. Only official Gege Akutami art trains the color LoRA.

#### Fill captions
Edit the generated `.txt` caption files. Add structural tags only — NO style tags (those are the trigger words):

```
BW:    gegeakutami, monochrome manga, [structural tags]
Color: gegecover, painterly watercolor cover, [structural tags]
```

Run cleanup after manual edits:
```bash
python training/scripts/caption_cleanup.py
```

### 5. Train LoRAs

> Run these in a separate terminal. Training takes 3–6 hours per LoRA on RTX 4060 8GB.

**Train BW LoRA first:**
```bash
training/scripts/train_bw.bat
```

Monitor: open TensorBoard in another terminal:
```bash
training/kohya_venv310/Scripts/tensorboard --logdir training/logs/bw_lora
```

**Train Color LoRA after BW is validated:**
```bash
training/scripts/train_color.bat
```

Evaluate checkpoints with:
```bash
python training/scripts/evaluate_checkpoint.py
```

### 6. Copy trained LoRAs to Forge

After training, copy the best checkpoint:
```
training/output/bw_lora/gege_bw_lora-XXXX.safetensors  → forge-neo/models/Lora/gege_bw_lora.safetensors
training/output/color_lora/gege_color_lora-XXXX.safetensors → forge-neo/models/Lora/gege_color_lora.safetensors
```

### 7. Launch backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Forge must be running at `http://localhost:7860` before backend can process images.

### 8. Launch frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

---

## Training Config Reference

### BW LoRA (rank 32)
- Base: Illustrious XL v0.1
- LR: 8e-5, cosine_with_restarts, 3 cycles
- Epochs: 15, batch 1, grad accum 4
- Optimizer: AdamW8bit
- Mixed precision: bf16 + xformers
- Trigger: `gegeakutami, monochrome manga`

### Color LoRA (rank 16)
- Base: Illustrious XL v0.1
- LR: 6e-5, cosine
- Epochs: 18, batch 1, grad accum 4
- Trigger: `gegecover, painterly watercolor cover`

---

## Inference Parameters

### Pipeline A — BW Manga Style
```
Sampler: DPM++ 2M Karras
Steps: 30, CFG: 7.5, Denoise: 0.63
ControlNet Canny: weight 0.90
ControlNet Depth: weight 0.55
LoRA_BW: weight 0.85
```

### Pipeline B — Color Cover Style
```
Sampler: DPM++ SDE Karras
Steps: 25, CFG: 6.5, Denoise: 0.45
ControlNet Canny: weight 0.97
LoRA_COLOR: weight 0.80
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Face drifts during style transfer | Raise Canny ControlNet weight +0.05, lower denoise -0.05 |
| Style not applying | Lower ControlNet, raise denoise +0.05, confirm LoRA weight |
| Output too anime/glossy | Add `anime, cel shading, glossy` to negative prompt |
| Training crashes OOM | Reduce train_batch_size to 1, ensure gradient_checkpointing=true |
| xformers triton warning | Harmless on Windows — triton is Linux-only, xformers still works |
| kohya_ss install fails (torch version) | Use sd-scripts directly, don't `pip install -e .` on kohya_ss root |

---

## Project Structure

```
jjk-colorizer/
├── backend/
│   ├── main.py              # FastAPI app + job queue
│   ├── forge_api.py         # Forge API wrapper (img2img, upscale)
│   ├── preprocessing.py     # Image load/validate/preview
│   └── requirements.txt
├── frontend/
│   └── app/
│       ├── page.tsx          # Main UI (upload, generate, results)
│       └── layout.tsx
├── training/
│   ├── configs/
│   │   ├── bw_lora_config.toml
│   │   └── color_lora_config.toml
│   └── scripts/
│       ├── prepare_dataset_bw.py
│       ├── prepare_dataset_color.py
│       ├── caption_cleanup.py
│       ├── audit_dataset.py
│       ├── evaluate_checkpoint.py
│       ├── download_models.py
│       ├── train_bw.bat
│       └── train_color.bat
├── dataset/                  # gitignored — supply your own
├── DEVLOG.md                # Full implementation diary + decisions
└── README.md
```

---

## Progress Log

- [x] Phase 0 — Environment setup (Forge, kohya_ss, Python 3.10 venv, all models)
- [x] Phase 1 — Dataset processing scripts
- [x] Phase 2 — Training configs (BW rank 32, Color rank 16)
- [ ] Phase 3 — Train LoRA_BW
- [ ] Phase 4 — Train LoRA_COLOR
- [ ] Phase 5 — Inference tuning via Forge API
- [ ] Phase 6 — Backend integration test
- [ ] Phase 7 — Frontend end-to-end test
