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
            columns = {row[1] for row in conn.execute("PRAGMA table_info(prediction_runs)").fetchall()}
            for name in ("target_month", "data_cutoff_month", "inference_feature_path", "provenance"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE prediction_runs ADD COLUMN {name} TEXT")

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

    def update_provenance(self, run_id: str, *, target_month: str | None, data_cutoff_month: str | None, inference_feature_path: str | None, provenance: dict | None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE prediction_runs
                SET target_month=?, data_cutoff_month=?, inference_feature_path=?, provenance=?
                WHERE prediction_run_id=?""",
                (target_month, data_cutoff_month, inference_feature_path, json.dumps(provenance, ensure_ascii=False) if provenance else None, run_id),
            )

    def get(self, run_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT prediction_run_id,dataset_id,model_ids,observation_month,store_ids,job_id,status,
                prediction_dir,created_at,error_message,target_month,data_cutoff_month,inference_feature_path,provenance
                FROM prediction_runs WHERE prediction_run_id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        keys = ("prediction_run_id", "dataset_id", "model_ids", "observation_month", "store_ids", "job_id", "status", "prediction_dir", "created_at", "error_message", "target_month", "data_cutoff_month", "inference_feature_path", "provenance")
        result = dict(zip(keys, row, strict=True))
        result["model_ids"] = json.loads(result["model_ids"])
        result["store_ids"] = json.loads(result["store_ids"]) if result["store_ids"] else None
        result["provenance"] = json.loads(result["provenance"]) if result["provenance"] else None
        return result

    def list(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            ids = [row[0] for row in conn.execute("SELECT prediction_run_id FROM prediction_runs ORDER BY rowid DESC")]
        return [self.get(run_id) for run_id in ids]
