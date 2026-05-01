from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qrart import COMPOSITIONS, Generator, GenerationRequest, STYLE_PRESETS
from qrart.db import get_db, new_job_id
from qrart.pipeline import MODELS
from qrart.worker import Job, MAX_QUEUED, QueueFull, Worker

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="QR Art Studio")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _startup() -> None:
    # Initialize SQLite + run migrations + mark orphans before the first
    # request arrives. Runs once per process.
    get_db()
    _worker.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _worker.stop()

# One Generator per model, lazily created on first use. Models that haven't
# been used in this session don't consume memory. Switching back to a model
# you used earlier is instant.
_generators: dict[str, Generator] = {}
_gen_lock = threading.Lock()


def get_generator(model: str | None = None) -> Generator:
    key = model if model in MODELS else "photoreal"
    with _gen_lock:
        if key not in _generators:
            _generators[key] = Generator(base_model=key)
    return _generators[key]


class GenerateBody(BaseModel):
    data: str = Field(..., description="URL or text to encode")
    prompt: str
    style: str = "photoreal"
    model: str = "photoreal"
    negative_prompt: str | None = None
    candidates: int = 5
    steps: int = 28
    controlnet_scale: float = 1.10
    tile_scale: float = 0.0
    control_end: float = 1.0
    guidance: float = 7.5
    refine: bool = True
    refine_strength: float = 0.30
    refine_steps: int = 20
    size: int = 768
    composition: str = "standalone"
    seed: int | None = None
    require_scan: bool = True
    fast_mode: bool = False
    hires_fix: bool = False
    hires_target: int = 1024
    hires_strength: float = 0.20
    adetailer: bool = False
    adetailer_strength: float = 0.35
    # Phase 1: link a remix back to the source job. When the UI loads a past
    # job's settings into the form and the user resubmits, it sets this so the
    # history can show "remixed from <id>" lineage.
    parent_job_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/health")
def health() -> dict[str, Any]:
    g = get_generator()
    loaded_models = [k for k, gen in _generators.items() if gen.pipeline._pipe is not None]
    state = _worker.state()
    return {
        "ok": True,
        "device": g.pipeline.device,
        "loaded": g.pipeline._pipe is not None,
        "base_model": g.pipeline.base_model,
        "styles": list(STYLE_PRESETS.keys()),
        "compositions": list(COMPOSITIONS.keys()),
        "models": list(MODELS.keys()),
        "loaded_models": loaded_models,
        **state,  # busy, active_model, active_job_id, active_elapsed_s, queue_depth, queued_ids, max_queued
    }


class WarmBody(BaseModel):
    model: str = "photoreal"


@app.post("/api/warm")
def warm(body: WarmBody | None = None) -> dict[str, Any]:
    t0 = time.time()
    model = body.model if body else "photoreal"
    get_generator(model).warm()
    return {"ok": True, "elapsed_s": round(time.time() - t0, 2), "model": model}


def _run_job(job: Job, cancelled: bool) -> None:
    """Worker callback. Drives the pipeline + persistence for one job.

    Cancelled jobs (cancelled while queued) get marked 'cancelled' in DB and
    never invoke the pipeline. Otherwise we update to 'running', generate,
    save outputs + candidates, and finish to 'completed' (or 'failed').
    """
    db = get_db()
    if cancelled:
        db.finish_job(job.job_id, status="cancelled", elapsed_s=0.0)
        return

    db.mark_running(job.job_id)
    t0 = time.time()
    try:
        result = get_generator(job.model).generate(job.request)
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        db.finish_job(
            job.job_id,
            status="failed",
            elapsed_s=elapsed,
            error=f"{e}\n\n{traceback.format_exc()}",
        )
        print(f"[worker] {job.job_id} FAILED: {e}", flush=True)
        return

    elapsed = round(time.time() - t0, 2)
    job_dir = OUTPUT_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids: list[str] = []
    for i, c in enumerate(result.candidates):
        path = job_dir / f"cand{i}.png"
        c.image.save(path)
        pass1_url: str | None = None
        if c.pass1_image is not None:
            pass1_path = job_dir / f"cand{i}.pass1.png"
            c.pass1_image.save(pass1_path)
            pass1_url = f"/outputs/{job.job_id}/cand{i}.pass1.png"
        cid = db.insert_candidate(
            job_id=job.job_id,
            idx=i,
            seed=c.seed,
            controlnet_scale=c.controlnet_scale,
            refine_strength=c.refine_strength,
            scans=c.scans,
            decoded=c.decoded,
            image_path=f"/outputs/{job.job_id}/cand{i}.png",
            pass1_image_path=pass1_url,
        )
        candidate_ids.append(cid)

    qr_path = job_dir / "qr.png"
    result.qr_image.save(qr_path)

    best_idx = next(
        (i for i, c in enumerate(result.candidates) if c.image is result.image), 0
    )
    db.finish_job(
        job.job_id,
        status="completed",
        elapsed_s=elapsed,
        scans=result.scans,
        decoded=result.decoded,
        qr_image_path=f"/outputs/{job.job_id}/qr.png",
        best_candidate_id=candidate_ids[best_idx] if candidate_ids else None,
    )
    print(f"[worker] {job.job_id} done in {elapsed}s · scans={result.scans}", flush=True)


_worker = Worker(_run_job)


@app.post("/api/generate")
def generate(body: GenerateBody, request: Request) -> dict[str, Any]:
    """Enqueue a generation job. Returns immediately — poll /api/jobs/{id}
    until status flips to completed / failed / cancelled."""
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
        f"[generate] model={body.model} prompt={body.prompt[:60]!r}... "
        f"scale={body.controlnet_scale} tile={body.tile_scale} "
        f"refine={body.refine}/{body.refine_strength} "
        f"composition={body.composition} candidates={body.candidates} "
        f"fast={body.fast_mode} hires={body.hires_fix} adetailer={body.adetailer} "
        f"seed={body.seed}",
        flush=True,
    )

    composition = body.composition if body.composition in COMPOSITIONS else "standalone"

    req = GenerationRequest(
        data=body.data,
        prompt=body.prompt,
        style=body.style,
        negative_prompt=body.negative_prompt,
        candidates=max(1, min(body.candidates, 8)),
        steps=steps,
        controlnet_scale=controlnet_scale,
        tile_scale=max(0.0, min(body.tile_scale, 1.0)),
        control_end=body.control_end,
        guidance=guidance,
        refine=body.refine,
        refine_strength=max(0.05, min(body.refine_strength, 0.6)),
        refine_steps=refine_steps,
        size=body.size,
        composition=composition,
        seed=body.seed,
        require_scan=body.require_scan,
        fast_mode=body.fast_mode,
        hires_fix=body.hires_fix,
        hires_target=max(768, min(body.hires_target, 1536)),
        hires_strength=max(0.05, min(body.hires_strength, 0.45)),
        adetailer=body.adetailer,
        adetailer_strength=max(0.1, min(body.adetailer_strength, 0.6)),
    )

    db = get_db()
    job_id = new_job_id()

    # Snapshot post-clamp values so the DB row reflects what diffused, not
    # the raw request body.
    persisted = {
        **body.model_dump(),
        "candidates": req.candidates,
        "steps": req.steps,
        "controlnet_scale": req.controlnet_scale,
        "tile_scale": req.tile_scale,
        "guidance": req.guidance,
        "refine_strength": req.refine_strength,
        "refine_steps": req.refine_steps,
        "hires_target": req.hires_target,
        "hires_strength": req.hires_strength,
        "adetailer_strength": req.adetailer_strength,
        "composition": composition,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    db.insert_job(job_id, persisted)
    db.touch_prompt(body.prompt)

    job = Job(job_id=job_id, model=body.model, request=req, body=persisted)
    try:
        position = _worker.enqueue(job)
    except QueueFull:
        # Roll the row to 'failed' so the user sees why and the queue stays
        # consistent with what the DB shows.
        db.finish_job(
            job_id,
            status="failed",
            elapsed_s=0.0,
            error=f"queue at capacity ({MAX_QUEUED}); try again in a moment",
        )
        raise HTTPException(
            503,
            f"Queue full ({MAX_QUEUED} jobs). Wait for one to finish, then retry.",
        )

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": position,
        "queue_depth": _worker.state()["queue_depth"],
    }


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued job. Cannot kill a running job in phase 2 — that
    requires the diffusion-step callback hook (phase 3)."""
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")
    state = _worker.cancel(job_id)
    if state == "queued":
        return {"job_id": job_id, "cancelled": True}
    if state == "running":
        raise HTTPException(
            409,
            "Job is already running; mid-diffusion cancel ships in phase 3.",
        )
    # 'unknown' — not in flight; either finished, failed, or never enqueued.
    return {"job_id": job_id, "cancelled": False, "reason": job["status"]}


@app.get("/api/jobs")
def list_jobs(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    model: str | None = None,
    scans: bool | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Browse history. Filters: status, model, scans (bool), q (substring of
    prompt or decoded)."""
    db = get_db()
    rows = db.list_jobs(
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        status=status,
        model=model,
        scans=scans,
        q=q,
    )
    return {"jobs": rows, "count": len(rows)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Full job row + candidates. URL paths in the response point at the
    /outputs static mount so the UI can render them directly."""
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")
    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
