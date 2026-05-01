"""Background diffusion worker.

A single thread pulls jobs from a queue and runs them serially. MPS only
admits one diffusion process anyway, so single-worker is the architectural
ceiling — adding more would just cause MPS contention.

The worker owns:
  - The FIFO queue (cap = MAX_QUEUED).
  - The "active job" pointer so /api/health can surface what's running.
  - A cancel set: any queued job whose id is in here gets skipped on dequeue.

Mid-diffusion cancellation is *not* implemented in phase 2. Phase 3 will hook
into the diffusers step callback to honor cancellation while a job runs.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .generator import GenerationRequest

MAX_QUEUED = 5


@dataclass
class Job:
    job_id: str
    model: str
    request: GenerationRequest
    body: dict[str, Any]                # already-clamped persisted dict
    enqueued_at: float = field(default_factory=time.time)


class QueueFull(Exception):
    """Raised when the queue is at MAX_QUEUED and a new job is rejected."""


class Worker:
    """Single-thread diffusion worker.

    The handler in app.py builds a Job and calls enqueue(). The worker thread
    blocks on queue.get(), invokes run_fn(job) which runs diffusion + writes
    DB + saves outputs, then loops.
    """

    def __init__(self, run_fn: Callable[[Job, bool], None]) -> None:
        # run_fn signature: run_fn(job, cancelled). When cancelled=True the
        # job was killed while still queued — run_fn should just mark the DB
        # row 'cancelled' and not invoke the pipeline.
        self._run_fn = run_fn
        self._queue: queue.Queue[Job | object] = queue.Queue()
        self._stop_sentinel: object = object()
        self._thread: threading.Thread | None = None

        self._state_lock = threading.RLock()
        self._active: Job | None = None
        self._queued_ids: list[str] = []
        self._cancelled: set[str] = set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="qrart-worker")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._queue.put(self._stop_sentinel)
        self._thread.join(timeout=timeout)
        self._thread = None

    def enqueue(self, job: Job) -> int:
        """Add a job to the queue. Returns its 1-indexed position
        (1 = will run next, 2 = one ahead of it, …). Raises QueueFull if full.
        """
        with self._state_lock:
            depth = len(self._queued_ids)
            if depth >= MAX_QUEUED:
                raise QueueFull(f"queue at capacity ({MAX_QUEUED})")
            self._queued_ids.append(job.job_id)
            self._queue.put(job)
            return len(self._queued_ids)

    def cancel(self, job_id: str) -> str:
        """Flag a job for cancellation.

        Returns:
          'queued'  — was queued; will be skipped on dequeue.
          'running' — currently running; the diffusion step callback will
                      raise CancelledByUser at the next step boundary.
          'unknown' — not in flight (already finished or never enqueued).
        """
        with self._state_lock:
            if self._active is not None and self._active.job_id == job_id:
                self._cancelled.add(job_id)
                return "running"
            if job_id in self._queued_ids:
                self._cancelled.add(job_id)
                return "queued"
        return "unknown"

    # ── Introspection (for /api/health) ──────────────────────────────────────
    def state(self) -> dict[str, Any]:
        with self._state_lock:
            active = self._active
            queued = list(self._queued_ids)
        return {
            "busy": active is not None,
            "active_model": active.model if active else None,
            "active_job_id": active.job_id if active else None,
            "active_elapsed_s": (
                round(time.time() - active.enqueued_at, 1) if active else None
            ),
            "queue_depth": len(queued),
            "queued_ids": queued,
            "max_queued": MAX_QUEUED,
        }

    def is_cancelled(self, job_id: str) -> bool:
        with self._state_lock:
            return job_id in self._cancelled

    # ── Worker loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop_sentinel:
                return
            assert isinstance(item, Job)
            job: Job = item

            with self._state_lock:
                cancelled = job.job_id in self._cancelled
                self._cancelled.discard(job.job_id)
                if job.job_id in self._queued_ids:
                    self._queued_ids.remove(job.job_id)
                self._active = None if cancelled else job

            try:
                self._run_fn(job, cancelled)
            except Exception:
                traceback.print_exc()
            finally:
                with self._state_lock:
                    self._active = None
                    # Clear stale cancel flag (mid-run cancellation already
                    # consumed by the step callback raising CancelledByUser).
                    self._cancelled.discard(job.job_id)
