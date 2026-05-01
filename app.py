from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

import asyncio
import json as _json
from dataclasses import replace as _replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qrart import COMPOSITIONS, Generator, GenerationRequest, STYLE_PRESETS
from qrart.db import get_db, new_job_id
from qrart.generator import Progress
from qrart.pipeline import (
    MODELS,
    QR_MONSTER_VERSIONS,
    QR_MONSTER_DEFAULT,
    CancelledByUser,
)
from qrart.worker import Job, MAX_QUEUED, QueueFull, Worker

# A2: auto-escalation tuning. When the user opts in (require_scan=True,
# auto_escalate=True), all-fail jobs spawn a follow-up at scale +0.1, capped.
# best-score floor avoids escalating on hopeless prompts where the QR will
# never resolve regardless of scale (e.g. dark cosmic scenes that fight QR
# luminance fundamentally).
ESCALATE_STEP = 0.10
ESCALATE_CAP = 1.50
ESCALATE_MIN_SCORE = 0.70

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="QR Art Studio")


# Optional shared-password auth. If QRART_AUTH=user:pass is set in the env,
# every /api/* request (and /outputs/* + the index) is gated behind HTTP
# Basic auth using a timing-safe compare. /api/health stays open so external
# probes don't have to know the password.
import base64 as _b64
import os as _os
import secrets as _secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as _Response

_AUTH = _os.environ.get("QRART_AUTH")


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not _AUTH:
            return await call_next(request)
        path = request.url.path
        # Allow open access to the health endpoint so probes don't need creds.
        if path == "/api/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        ok = False
        if header.lower().startswith("basic "):
            try:
                creds = _b64.b64decode(header.split(" ", 1)[1]).decode()
                ok = _secrets.compare_digest(creds, _AUTH)
            except Exception:
                ok = False
        if not ok:
            return _Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="QR Art Studio"'},
            )
        return await call_next(request)


if _AUTH:
    app.add_middleware(_BasicAuthMiddleware)
    print("[auth] basic auth enabled via QRART_AUTH env var", flush=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


RETENTION_KEEP = int(__import__("os").environ.get("QRART_RETENTION_KEEP", "1000"))


def _cleanup_evicted_files(evicted_ids: list[str]) -> int:
    """Best-effort rm -rf of the outputs/{id}/ directories for evicted jobs.
    Returns the number of directories actually removed."""
    import shutil as _shutil
    removed = 0
    for jid in evicted_ids:
        d = OUTPUT_DIR / jid
        if d.exists():
            try:
                _shutil.rmtree(d)
                removed += 1
            except OSError:
                pass
    return removed


@app.on_event("startup")
def _startup() -> None:
    # Initialize SQLite + run migrations + mark orphans before the first
    # request arrives. Runs once per process.
    db = get_db()
    evicted = db.evict_old_jobs(keep=RETENTION_KEEP)
    if evicted:
        removed = _cleanup_evicted_files(evicted)
        print(
            f"[retention] kept newest {RETENTION_KEEP} jobs, "
            f"evicted {len(evicted)} (removed {removed} output dirs)",
            flush=True,
        )
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
    # A2: when require_scan is on and zero candidates pass, auto-resubmit with
    # scale +0.1 (capped at 1.5) until one passes or we hit the cap. Off → user
    # gets the failed result and can manually retry.
    auto_escalate: bool = True
    # QR Monster ControlNet version: 'v1' (default) or 'v2'. Both are loaded
    # at warm time and swapped per-request without a model reload.
    qr_monster_version: str = "v1"
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
        "qr_monster_default": QR_MONSTER_DEFAULT,
        "qr_monster_versions": list(QR_MONSTER_VERSIONS),
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

    progress = Progress(
        publish=lambda type_, payload: db.insert_event(job.job_id, type_, payload),
        is_cancelled=lambda: _worker.is_cancelled(job.job_id),
    )
    progress.emit("started", model=job.model)

    try:
        result = get_generator(job.model).generate(job.request, progress=progress)
    except CancelledByUser:
        elapsed = round(time.time() - t0, 2)
        db.finish_job(job.job_id, status="cancelled", elapsed_s=elapsed)
        progress.emit("cancelled")
        print(f"[worker] {job.job_id} CANCELLED after {elapsed}s", flush=True)
        return
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        db.finish_job(
            job.job_id,
            status="failed",
            elapsed_s=elapsed,
            error=f"{e}\n\n{traceback.format_exc()}",
        )
        progress.emit("failed", error=str(e))
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
            scannability=c.scannability,
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
    progress.emit(
        "completed",
        elapsed_s=elapsed,
        scans=result.scans,
        decoded=result.decoded,
        best_candidate_id=candidate_ids[best_idx] if candidate_ids else None,
    )
    print(f"[worker] {job.job_id} done in {elapsed}s · scans={result.scans}", flush=True)

    _maybe_escalate(job, result, db)


def _maybe_escalate(job: Job, result, db) -> None:
    """A2: when require_scan + auto_escalate are on and zero candidates scanned
    but the best score is salvageable, enqueue a follow-up at scale +0.1.

    The best-score floor avoids burning compute on hopeless prompts where
    QR will never resolve at any scale (cosmic galaxies, plain skies, etc.).
    """
    if not job.body.get("require_scan"):
        return
    if not job.body.get("auto_escalate", True):
        return
    if result.scans:
        return
    # Don't escalate cancelled-mid-run results (the worker would have raised
    # CancelledByUser, so we wouldn't reach here — but defensive).
    if _worker.is_cancelled(job.job_id):
        return

    current_scale = float(job.request.controlnet_scale)
    new_scale = round(current_scale + ESCALATE_STEP, 2)
    if new_scale > ESCALATE_CAP:
        db.insert_event(job.job_id, "escalation_skipped", {
            "reason": f"already at cap ({ESCALATE_CAP})",
            "scale": current_scale,
        })
        return

    best_score = max(
        (float(c.scannability) for c in result.candidates), default=0.0
    )
    if best_score < ESCALATE_MIN_SCORE:
        db.insert_event(job.job_id, "escalation_skipped", {
            "reason": f"best score {best_score:.2f} below floor {ESCALATE_MIN_SCORE}",
            "scale": current_scale,
        })
        return

    new_req = _replace(job.request, controlnet_scale=new_scale, seed=None)
    new_body = {
        **job.body,
        "controlnet_scale": new_scale,
        "seed": None,
        "parent_job_id": job.job_id,
    }
    new_jid = new_job_id()
    try:
        db.insert_job(new_jid, new_body)
        new_job = Job(job_id=new_jid, model=job.model, request=new_req, body=new_body)
        _worker.enqueue(new_job)
        db.insert_event(job.job_id, "auto_escalated", {
            "child_job_id": new_jid,
            "from_scale": current_scale,
            "to_scale": new_scale,
            "best_score": round(best_score, 3),
        })
        print(f"[escalate] {job.job_id} -> {new_jid} (scale {current_scale} -> {new_scale}, score {best_score:.2f})", flush=True)
    except QueueFull:
        db.finish_job(new_jid, status="failed", elapsed_s=0.0,
                      error="queue full during auto-escalation")


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
        auto_escalate=body.auto_escalate,
        qr_monster_version=(
            body.qr_monster_version if body.qr_monster_version in QR_MONSTER_VERSIONS
            else QR_MONSTER_DEFAULT
        ),
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
        "qr_monster_version": req.qr_monster_version,
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


@app.post("/api/jobs/{job_id}/rerun")
def rerun_job(
    job_id: str,
    request: Request,
    keep_seed: bool = False,
) -> dict[str, Any]:
    """One-click "Run again" — clones a past job's settings and enqueues it.

    keep_seed=true reproduces the exact run; default keep_seed=false nulls
    the seed so the worker rolls a fresh one and produces variations on the
    same recipe. The new job links back via parent_job_id for lineage.

    Distinct from "Remix" (a UI flow): Remix loads settings into the form
    so the user can edit before submitting; rerun is fire-and-forget.
    """
    db = get_db()
    src = db.get_job(job_id)
    if not src:
        raise HTTPException(404, f"job {job_id} not found")

    body = GenerateBody(
        data=src["data"],
        prompt=src["prompt"],
        style=src["style"],
        model=src["model"],
        negative_prompt=src.get("negative_prompt"),
        candidates=src["candidates"],
        steps=src["steps"],
        controlnet_scale=src["controlnet_scale"],
        tile_scale=src["tile_scale"],
        control_end=src["control_end"],
        guidance=src["guidance"],
        refine=bool(src["refine"]),
        refine_strength=src["refine_strength"],
        refine_steps=src["refine_steps"],
        size=src["size"],
        composition=src["composition"],
        seed=src["seed"] if keep_seed else None,
        require_scan=bool(src["require_scan"]),
        auto_escalate=bool(src.get("auto_escalate", 1)),
        qr_monster_version=src.get("qr_monster_version") or QR_MONSTER_DEFAULT,
        fast_mode=bool(src["fast_mode"]),
        hires_fix=bool(src["hires_fix"]),
        hires_target=src["hires_target"],
        hires_strength=src["hires_strength"],
        adetailer=bool(src["adetailer"]),
        adetailer_strength=src["adetailer_strength"],
        parent_job_id=job_id,
    )
    return generate(body, request)


@app.delete("/api/jobs/{job_id}")
def cancel_or_delete_job(job_id: str) -> dict[str, Any]:
    """Smart DELETE: cancel if the job is in flight, hard-delete if terminal.

    - queued / running -> cancel (worker's cancel set; running jobs are
      caught at the next diffusion step boundary).
    - completed / failed / cancelled -> remove the row + cascade candidates
      and events, plus rm -rf outputs/{id}/ on disk.
    """
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")

    state = _worker.cancel(job_id)
    if state in ("queued", "running"):
        return {"job_id": job_id, "cancelled": True, "was": state}

    # Terminal state — hard delete row + files.
    db.delete_job(job_id)
    _cleanup_evicted_files([job_id])
    return {"job_id": job_id, "deleted": True, "was": job["status"]}


@app.get("/api/prompts/recent")
def list_recent_prompts(limit: int = 20, favorites_only: bool = False) -> dict[str, Any]:
    db = get_db()
    return {
        "prompts": db.list_prompts(
            limit=max(1, min(limit, 100)),
            favorites_only=favorites_only,
        ),
    }


class FavoriteBody(BaseModel):
    favorited: bool = True


@app.post("/api/prompts/{prompt_id}/favorite")
def set_favorite(prompt_id: int, body: FavoriteBody) -> dict[str, Any]:
    db = get_db()
    ok = db.set_prompt_favorite(prompt_id, body.favorited)
    if not ok:
        raise HTTPException(404, f"prompt {prompt_id} not found")
    return {"prompt_id": prompt_id, "favorited": body.favorited}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    db = get_db()
    return db.stats()


@app.post("/api/admin/cleanup")
def admin_cleanup(keep: int = 1000) -> dict[str, Any]:
    """Manual retention sweep. Defaults to the same retention as startup.
    Returns the evicted job count + how many output dirs were removed."""
    db = get_db()
    evicted = db.evict_old_jobs(keep=max(1, keep))
    removed = _cleanup_evicted_files(evicted)
    return {"evicted": len(evicted), "removed_dirs": removed, "keep": keep}


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE feed of job_events. The worker writes events into SQLite as the
    pipeline ticks; this handler polls the table at ~250 ms and forwards each
    new row as a server-sent event. Stream closes after a terminal event
    (completed/failed/cancelled) or when the client disconnects.
    """
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")

    async def gen():
        last_id = 0
        terminal = {"completed", "failed", "cancelled"}
        # Replay any events already on disk (e.g. UI reconnects mid-job).
        seen_terminal = False
        while not seen_terminal:
            if await request.is_disconnected():
                return
            events = db.events_since(job_id, after_id=last_id, limit=200)
            for ev in events:
                last_id = ev["id"]
                payload = {**ev["payload"], "ts": ev["ts"]}
                yield f"event: {ev['type']}\ndata: {_json.dumps(payload)}\n\n"
                if ev["type"] in terminal:
                    seen_terminal = True
                    break
            if seen_terminal:
                break
            # Also short-circuit if the DB row says the job is already done
            # (e.g. completed before the SSE handler attached).
            row = db.get_job(job_id)
            if row and row["status"] in terminal and not events:
                yield f"event: {row['status']}\ndata: {_json.dumps({'status': row['status']})}\n\n"
                seen_terminal = True
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


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
