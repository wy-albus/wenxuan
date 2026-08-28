from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .runtime import database_path, runtime_root


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobService:
    def __init__(self) -> None:
        self.db_path = database_path()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, job_type TEXT, status TEXT, progress INTEGER,
                input_files TEXT, output_files TEXT, started_at TEXT, finished_at TEXT,
                error_message TEXT, log_path TEXT)""")

    def create(self, job_type: str, input_files: list[str]) -> dict:
        job_id = uuid.uuid4().hex
        log_path = runtime_root() / "logs" / f"{job_id}.log"
        record = {"job_id": job_id, "job_type": job_type, "status": "QUEUED", "progress": 0,
                  "input_files": input_files, "output_files": [], "started_at": None, "finished_at": None,
                  "error_message": None, "log_path": str(log_path)}
        with self._connect() as conn:
            stored = dict(record)
            stored["input_files"] = json.dumps(stored["input_files"])
            stored["output_files"] = json.dumps(stored["output_files"])
            conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(stored.values()))
        return record

    def update(self, job_id: str, *, status: str, progress: int, output_files: list[str] | None = None,
               error_message: str | None = None) -> None:
        started = _now() if status == "RUNNING" else None
        finished = _now() if status in {"SUCCESS", "FAILED"} else None
        with self._connect() as conn:
            if started:
                conn.execute("UPDATE jobs SET status=?, progress=?, started_at=? WHERE job_id=?", (status, progress, started, job_id))
            elif finished:
                conn.execute("UPDATE jobs SET status=?, progress=?, output_files=?, error_message=?, finished_at=? WHERE job_id=?", (status, progress, json.dumps(output_files or []), error_message, finished, job_id))
            else:
                conn.execute("UPDATE jobs SET status=?, progress=? WHERE job_id=?", (status, progress, job_id))

    def log(self, job_id: str, message: str) -> None:
        path = Path(self.get(job_id)["log_path"])
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_now()} {message}\n")

    def get(self, job_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        keys = ("job_id", "job_type", "status", "progress", "input_files", "output_files", "started_at", "finished_at", "error_message", "log_path")
        result = dict(zip(keys, row, strict=True))
        result["input_files"], result["output_files"] = json.loads(result["input_files"]), json.loads(result["output_files"])
        return result

    def list(self) -> list[dict]:
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT job_id FROM jobs ORDER BY rowid DESC")]
        return [self.get(job_id) for job_id in ids]
