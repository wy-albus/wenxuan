from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from .runtime import database_path


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PredictionRegistry:
    def __init__(self) -> None:
        self.db_path = database_path()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_run_id TEXT PRIMARY KEY, dataset_id TEXT, model_ids TEXT,
                observation_month TEXT, store_ids TEXT, job_id TEXT, status TEXT,
                prediction_dir TEXT, created_at TEXT, error_message TEXT)""")

    def create(self, record: dict) -> dict:
        record = {"status": "QUEUED", "prediction_dir": None, "error_message": None, "created_at": _now(), **record}
        keys = tuple(record)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO prediction_runs ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                tuple(json.dumps(record[key]) if key in {"model_ids", "store_ids"} else record[key] for key in keys),
            )
        return record

    def update(self, run_id: str, *, status: str, prediction_dir: str | None = None, error_message: str | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE prediction_runs SET status=?, prediction_dir=COALESCE(?, prediction_dir), error_message=? WHERE prediction_run_id=?", (status, prediction_dir, error_message, run_id))

    def get(self, run_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM prediction_runs WHERE prediction_run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        keys = ("prediction_run_id", "dataset_id", "model_ids", "observation_month", "store_ids", "job_id", "status", "prediction_dir", "created_at", "error_message")
        result = dict(zip(keys, row, strict=True))
        result["model_ids"] = json.loads(result["model_ids"])
        result["store_ids"] = json.loads(result["store_ids"]) if result["store_ids"] else None
        return result

    def list(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            ids = [row[0] for row in conn.execute("SELECT prediction_run_id FROM prediction_runs ORDER BY rowid DESC")]
        return [self.get(run_id) for run_id in ids]
