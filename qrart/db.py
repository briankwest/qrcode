"""SQLite layer.

Single shared connection with a write lock; sqlite3 is thread-safe for reads
when `check_same_thread=False`. Diffusion is serialized at a higher level by
the run lock, so write contention here is minimal — but we still acquire the
write lock to be safe under concurrent /api/jobs reads from the UI.

Migrations are numbered SQL files in qrart/migrations/. Schema version is
tracked in a `meta` table; applying a migration bumps it transactionally with
the schema change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DB_PATH_ENV = "QRART_DB_PATH"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "qrart.db"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def new_candidate_id() -> str:
    return uuid.uuid4().hex[:16]


class Database:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        # check_same_thread=False lets FastAPI handlers (different threads)
        # share the connection; the lock guards writes.
        self.conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._write_lock = threading.RLock()
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        with self._write_lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            current = int(row["value"]) if row else 0
            for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
                version = int(f.stem.split("_", 1)[0])
                if version <= current:
                    continue
                self.conn.executescript("BEGIN; " + f.read_text() + "; COMMIT;")
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(version),),
                )

    # ── Jobs ─────────────────────────────────────────────────────────────────
    def insert_job(self, job_id: str, body: dict[str, Any]) -> None:
        """Insert a job at status='queued'. body is the raw GenerateBody dict
        plus optional client_ip / user_agent / parent_job_id keys.

        The worker calls mark_running() when it dequeues; finish_job() when it
        completes or fails.
        """
        cols = (
            "id created_at status data prompt negative_prompt style model "
            "composition candidates steps guidance controlnet_scale tile_scale control_end "
            "refine refine_strength refine_steps size seed require_scan fast_mode "
            "hires_fix hires_target hires_strength adetailer adetailer_strength "
            "client_ip user_agent parent_job_id"
        ).split()
        now = _now()
        values = (
            job_id, now, "queued",
            body["data"], body["prompt"], body.get("negative_prompt"),
            body["style"], body["model"], body["composition"],
            body["candidates"], body["steps"], body["guidance"],
            body["controlnet_scale"], body["tile_scale"], body["control_end"],
            int(bool(body["refine"])), body["refine_strength"], body["refine_steps"],
            body["size"], body.get("seed"), int(bool(body["require_scan"])),
            int(bool(body["fast_mode"])),
            int(bool(body["hires_fix"])), body["hires_target"], body["hires_strength"],
            int(bool(body["adetailer"])), body["adetailer_strength"],
            body.get("client_ip"), body.get("user_agent"), body.get("parent_job_id"),
        )
        with self._write_lock:
            self.conn.execute(
                f"INSERT INTO jobs ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                values,
            )

    def mark_running(self, job_id: str) -> None:
        with self._write_lock:
            self.conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=? AND status='queued'",
                (_now(), job_id),
            )

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        elapsed_s: float | None = None,
        error: str | None = None,
        scans: bool | None = None,
        decoded: str | None = None,
        qr_image_path: str | None = None,
        best_candidate_id: str | None = None,
    ) -> None:
        with self._write_lock:
            self.conn.execute(
                """
                UPDATE jobs SET
                  finished_at = ?,
                  status = ?,
                  elapsed_s = ?,
                  error = ?,
                  scans = ?,
                  decoded = ?,
                  qr_image_path = ?,
                  best_candidate_id = ?
                WHERE id = ?
                """,
                (
                    _now(), status, elapsed_s, error,
                    None if scans is None else int(bool(scans)),
                    decoded, qr_image_path, best_candidate_id, job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        # job["candidates"] is the INTEGER count from the request — keep it.
        # The joined candidate rows live under a separate key so a Remix /
        # Rerun can read the count without confusion.
        job["candidate_list"] = [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM candidates WHERE job_id = ? ORDER BY idx", (job_id,)
            ).fetchall()
        ]
        return job

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        model: str | None = None,
        scans: bool | None = None,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if model:
            where.append("model = ?")
            params.append(model)
        if scans is not None:
            where.append("scans = ?")
            params.append(int(bool(scans)))
        if q:
            where.append("(prompt LIKE ? OR decoded LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ── Candidates ───────────────────────────────────────────────────────────
    def insert_candidate(
        self,
        *,
        job_id: str,
        idx: int,
        seed: int,
        controlnet_scale: float,
        refine_strength: float | None,
        scans: bool,
        decoded: str | None,
        image_path: str,
        pass1_image_path: str | None,
        scannability: float | None = None,
    ) -> str:
        cid = new_candidate_id()
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO candidates
                (id, job_id, idx, seed, controlnet_scale, refine_strength,
                 scans, decoded, image_path, pass1_image_path, scannability)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    cid, job_id, idx, seed, controlnet_scale, refine_strength,
                    int(bool(scans)), decoded, image_path, pass1_image_path,
                    scannability,
                ),
            )
        return cid

    # ── Events (used by phase 3 SSE) ─────────────────────────────────────────
    def insert_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO job_events (job_id, ts, type, payload) VALUES (?,?,?,?)",
                (job_id, _now(), event_type, json.dumps(payload)),
            )

    def events_since(self, job_id: str, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch events with id > after_id for a job, oldest first. SSE handler
        polls this every ~250 ms with the last event id it saw."""
        rows = self.conn.execute(
            "SELECT id, ts, type, payload FROM job_events"
            " WHERE job_id = ? AND id > ?"
            " ORDER BY id ASC LIMIT ?",
            (job_id, after_id, limit),
        ).fetchall()
        return [
            {"id": r["id"], "ts": r["ts"], "type": r["type"], "payload": json.loads(r["payload"])}
            for r in rows
        ]

    # ── Prompts (used by phase 5 picker) ─────────────────────────────────────
    def touch_prompt(self, text: str) -> None:
        if not text or not text.strip():
            return
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO prompts (text, used_count, last_used_at)
                VALUES (?, 1, ?)
                ON CONFLICT(text) DO UPDATE SET
                  used_count = used_count + 1,
                  last_used_at = excluded.last_used_at
                """,
                (text.strip(), _now()),
            )

    def list_prompts(self, *, limit: int = 20, favorites_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM prompts"
        params: list[Any] = []
        if favorites_only:
            sql += " WHERE favorited = 1"
        sql += " ORDER BY favorited DESC, last_used_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def set_prompt_favorite(self, prompt_id: int, favorited: bool) -> bool:
        with self._write_lock:
            cur = self.conn.execute(
                "UPDATE prompts SET favorited = ? WHERE id = ?",
                (int(bool(favorited)), prompt_id),
            )
        return cur.rowcount > 0

    # ── Stats ────────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        cur = self.conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
              SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS failed,
              SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled,
              SUM(CASE WHEN status='running'   THEN 1 ELSE 0 END) AS running,
              SUM(CASE WHEN status='queued'    THEN 1 ELSE 0 END) AS queued,
              SUM(CASE WHEN scans=1 THEN 1 ELSE 0 END) AS scanned,
              AVG(CASE WHEN status='completed' THEN elapsed_s END) AS avg_elapsed_completed,
              SUM(CASE WHEN status='completed' THEN elapsed_s ELSE 0 END) AS total_runtime_s
            FROM jobs
            """
        ).fetchone()
        totals = {k: cur[k] for k in cur.keys()}
        scanned = totals.get("scanned") or 0
        completed = totals.get("completed") or 0
        scan_rate = (scanned / completed) if completed else None

        by_model = [
            dict(r) for r in self.conn.execute(
                "SELECT model, COUNT(*) AS n,"
                "  SUM(CASE WHEN scans=1 THEN 1 ELSE 0 END) AS scanned"
                " FROM jobs GROUP BY model ORDER BY n DESC LIMIT 10"
            ).fetchall()
        ]
        top_prompts = [
            dict(r) for r in self.conn.execute(
                "SELECT text, used_count, favorited FROM prompts"
                " ORDER BY used_count DESC LIMIT 5"
            ).fetchall()
        ]
        return {
            **totals,
            "scan_rate": scan_rate,
            "avg_elapsed_completed": (
                round(totals["avg_elapsed_completed"], 1)
                if totals.get("avg_elapsed_completed") is not None
                else None
            ),
            "by_model": by_model,
            "top_prompts": top_prompts,
        }

    def delete_job(self, job_id: str) -> bool:
        """Hard-delete a job. ON DELETE CASCADE removes candidates + events.
        Caller is responsible for rm -rf'ing the outputs/{id}/ directory."""
        with self._write_lock:
            cur = self.conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cur.rowcount > 0

    # ── Retention ────────────────────────────────────────────────────────────
    def evict_old_jobs(self, keep: int = 1000) -> list[str]:
        """Keep the most recent `keep` jobs, delete the rest. Returns the
        list of evicted job ids so the caller can also rm -rf their output
        directories. ON DELETE CASCADE handles candidates + job_events.
        """
        rows = self.conn.execute(
            "SELECT id FROM jobs ORDER BY created_at DESC LIMIT -1 OFFSET ?",
            (keep,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            return []
        with self._write_lock:
            placeholders = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", ids)
        return ids

    # ── Recovery ─────────────────────────────────────────────────────────────
    def mark_orphans_failed(self) -> int:
        """On startup, any job left in 'running' or 'queued' from a prior
        process is dead — mark them failed so they don't haunt the history."""
        with self._write_lock:
            cur = self.conn.execute(
                "UPDATE jobs SET status='failed', error='server restarted',"
                " finished_at=? WHERE status IN ('running','queued')",
                (_now(),),
            )
        return cur.rowcount or 0


# Module-level singleton, lazily initialised so tests can override the path.
_db: Database | None = None
_db_lock = threading.Lock()


def get_db(path: Path | str | None = None) -> Database:
    global _db
    with _db_lock:
        if _db is None:
            import os
            resolved = path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH
            _db = Database(resolved)
            orphaned = _db.mark_orphans_failed()
            if orphaned:
                print(f"[db] marked {orphaned} orphaned job(s) as failed", flush=True)
    return _db
