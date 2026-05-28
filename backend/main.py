"""
main.py
=======
FastAPI backend for JJK Colorizer.

Endpoints:
    POST /api/upload              → {job_id}
    POST /api/generate/{job_id}   → queues generation
    GET  /api/status/{job_id}     → {status, progress, stage}
    GET  /api/result/{job_id}     → {bw_url, color_url}  (when complete)
    GET  /api/download/{job_id}/{variant}  → file download

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Requires Forge running with --api at localhost:7860
"""

import asyncio
import gc
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import forge_api as forge

app = FastAPI(title="JJK Colorizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
(TEMP_DIR / "uploads").mkdir(exist_ok=True)
(TEMP_DIR / "outputs").mkdir(exist_ok=True)

# Serve generated images statically
app.mount("/outputs", StaticFiles(directory=str(TEMP_DIR / "outputs")), name="outputs")

# In-memory job store (use Redis for production)
jobs: dict = {}

# Global semaphore — one generation at a time (GPU constraint)
_gpu_lock = asyncio.Semaphore(1)


class GenerateRequest(BaseModel):
    mode: str = "full"          # "bw" or "full"
    color_theme: str = ""       # optional color hint
    seed: int = -1


@app.get("/api/health")
def health():
    forge_ok = forge.check_forge_ready()
    return {"status": "ok", "forge": "ready" if forge_ok else "not_running"}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".png"
    upload_path = TEMP_DIR / "uploads" / f"{job_id}_input{ext}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "uploaded",
        "stage": None,
        "progress": 0,
        "input_path": str(upload_path),
        "bw_path": None,
        "color_path": None,
        "error": None,
    }
    return {"job_id": job_id}


@app.post("/api/generate/{job_id}")
async def generate(job_id: str, req: GenerateRequest, background_tasks: BackgroundTasks):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("processing", "queued"):
        raise HTTPException(409, "Job already running")

    jobs[job_id]["status"] = "queued"
    jobs[job_id]["mode"] = req.mode
    background_tasks.add_task(_run_pipeline, job_id, req)
    return {"job_id": job_id, "status": "queued"}


async def _run_pipeline(job_id: str, req: GenerateRequest):
    async with _gpu_lock:
        job = jobs[job_id]
        input_path = job["input_path"]

        try:
            # Stage 1: BW conversion
            jobs[job_id].update({"status": "processing", "stage": "pipeline_a", "progress": 5})

            loop = asyncio.get_event_loop()
            bw_img = await loop.run_in_executor(
                None,
                lambda: forge.pipeline_a_bw(input_path)
            )

            bw_path = TEMP_DIR / "outputs" / f"{job_id}_bw.png"
            bw_img.save(bw_path)
            jobs[job_id].update({"bw_path": str(bw_path), "progress": 55})

            if req.mode == "full":
                # Stage 2: Colorization
                jobs[job_id].update({"stage": "pipeline_b", "progress": 60})

                color_img = await loop.run_in_executor(
                    None,
                    lambda: forge.pipeline_b_color(bw_img, color_theme=req.color_theme)
                )

                color_path = TEMP_DIR / "outputs" / f"{job_id}_color.png"
                color_img.save(color_path)
                jobs[job_id].update({"color_path": str(color_path), "progress": 95})

            jobs[job_id].update({"status": "complete", "progress": 100, "stage": None})

        except Exception as e:
            jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": 0,
                "stage": None,
            })
        finally:
            gc.collect()


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job.get("error"),
        "has_bw": job.get("bw_path") is not None,
        "has_color": job.get("color_path") is not None,
    }


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not complete: {job['status']}")

    result = {}
    if job.get("bw_path"):
        result["bw_url"] = f"/api/download/{job_id}/bw"
    if job.get("color_path"):
        result["color_url"] = f"/api/download/{job_id}/color"
    return result


@app.get("/api/download/{job_id}/{variant}")
def download_result(job_id: str, variant: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    path_key = f"{variant}_path"
    file_path = job.get(path_key)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, f"No {variant} output found")

    filename = f"jjk_{job_id}_{variant}.png"
    return FileResponse(file_path, filename=filename, media_type="image/png")
