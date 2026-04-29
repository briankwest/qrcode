from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qrart import Generator, GenerationRequest, STYLE_PRESETS

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="QR Art Studio")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_generator: Generator | None = None
_gen_lock = threading.Lock()
# Diffusion is single-GPU/MPS — serialize requests so we don't OOM.
_run_lock = threading.Lock()


def get_generator() -> Generator:
    global _generator
    with _gen_lock:
        if _generator is None:
            _generator = Generator()
    return _generator


class GenerateBody(BaseModel):
    data: str = Field(..., description="URL or text to encode")
    prompt: str
    style: str = "photoreal"
    negative_prompt: str | None = None
    candidates: int = 3
    steps: int = 28
    controlnet_scale: float = 1.35
    control_end: float = 1.0
    guidance: float = 7.5
    refine: bool = True
    refine_strength: float = 0.30
    refine_steps: int = 20
    size: int = 768
    seed: int | None = None
    require_scan: bool = True
    fast_mode: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/health")
def health() -> dict[str, Any]:
    g = get_generator()
    return {
        "ok": True,
        "device": g.pipeline.device,
        "loaded": g.pipeline._pipe is not None,
        "base_model": g.pipeline.base_model,
        "styles": list(STYLE_PRESETS.keys()),
    }


@app.post("/api/warm")
def warm() -> dict[str, Any]:
    t0 = time.time()
    get_generator().warm()
    return {"ok": True, "elapsed_s": round(time.time() - t0, 2)}


@app.post("/api/generate")
def generate(body: GenerateBody) -> dict[str, Any]:
    if not body.data.strip() or not body.prompt.strip():
        raise HTTPException(400, "data and prompt are required")

    # Fast mode swaps in LCM-LoRA which needs different sampling defaults.
    # Override step/guidance regardless of what the client sent.
    if body.fast_mode:
        steps = 6
        refine_steps = 12
        guidance = 1.5
        # LCM at low CFG (1.5) leaves ControlNet relatively dominant, so the
        # same scale slider produces a much QR-ier image than in Quality mode.
        # Apply a transparent multiplier so the slider's intent matches output.
        controlnet_scale = body.controlnet_scale * 0.75
    else:
        steps = max(10, min(body.steps, 60))
        refine_steps = max(10, min(body.refine_steps, 50))
        guidance = body.guidance
        controlnet_scale = body.controlnet_scale

    print(
        f"[generate] prompt={body.prompt[:60]!r}... "
        f"scale={body.controlnet_scale} refine={body.refine}/{body.refine_strength} "
        f"candidates={body.candidates} fast={body.fast_mode} seed={body.seed}",
        flush=True,
    )

    req = GenerationRequest(
        data=body.data,
        prompt=body.prompt,
        style=body.style,
        negative_prompt=body.negative_prompt,
        candidates=max(1, min(body.candidates, 6)),
        steps=steps,
        controlnet_scale=controlnet_scale,
        control_end=body.control_end,
        guidance=guidance,
        refine=body.refine,
        refine_strength=max(0.05, min(body.refine_strength, 0.6)),
        refine_steps=refine_steps,
        size=body.size,
        seed=body.seed,
        require_scan=body.require_scan,
        fast_mode=body.fast_mode,
    )

    t0 = time.time()
    with _run_lock:
        result = get_generator().generate(req)
    elapsed = round(time.time() - t0, 2)

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, Any]] = []
    for i, c in enumerate(result.candidates):
        path = job_dir / f"cand{i}.png"
        c.image.save(path)
        entry: dict[str, Any] = {
            "index": i,
            "url": f"/outputs/{job_id}/cand{i}.png",
            "seed": c.seed,
            "scans": c.scans,
            "decoded": c.decoded,
            "controlnet_scale": round(c.controlnet_scale, 3),
            "refine_strength": (
                round(c.refine_strength, 3) if c.refine_strength is not None else None
            ),
        }
        if c.pass1_image is not None:
            pass1_path = job_dir / f"cand{i}.pass1.png"
            c.pass1_image.save(pass1_path)
            entry["pass1_url"] = f"/outputs/{job_id}/cand{i}.pass1.png"
        saved.append(entry)

    qr_path = job_dir / "qr.png"
    result.qr_image.save(qr_path)

    best_idx = next(
        (i for i, c in enumerate(result.candidates) if c.image is result.image),
        0,
    )
    return {
        "job_id": job_id,
        "elapsed_s": elapsed,
        "best_index": best_idx,
        "scans": result.scans,
        "decoded": result.decoded,
        "qr_url": f"/outputs/{job_id}/qr.png",
        "candidates": saved,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
