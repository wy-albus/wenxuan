from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime

from .runtime import database_path


class DatasetRegistry:
    def __init__(self) -> None:
        self.db_path = database_path()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY, dataset_name TEXT, source_type TEXT, source_files TEXT,
                date_range TEXT, store_count INTEGER, item_count INTEGER, row_count INTEGER,
                monthly_parquet_path TEXT, active_store_parquet_path TEXT, feature_parquet_path TEXT,
                has_active_store INTEGER, has_diff_features INTEGER, has_cross_store_features INTEGER,
                created_at TEXT, status TEXT)""")

    def register(self, record: dict) -> dict:
        record = {"dataset_id": uuid.uuid4().hex, "created_at": datetime.now(UTC).isoformat(), "status": "READY", **record}
        keys = tuple(record)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"INSERT INTO datasets ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(json.dumps(record[key]) if key in {"source_files", "date_range"} else record[key] for key in keys))
        return record

    def get(self, dataset_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        keys = ("dataset_id", "dataset_name", "source_type", "source_files", "date_range", "store_count", "item_count", "row_count", "monthly_parquet_path", "active_store_parquet_path", "feature_parquet_path", "has_active_store", "has_diff_features", "has_cross_store_features", "created_at", "status")
        result = dict(zip(keys, row, strict=True))
        result["source_files"], result["date_range"] = json.loads(result["source_files"]), json.loads(result["date_range"])
        for field in ("has_active_store", "has_diff_features", "has_cross_store_features"):
            result[field] = bool(result[field])
        return result

    def list(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            ids = [row[0] for row in conn.execute("SELECT dataset_id FROM datasets ORDER BY rowid DESC")]
        return [self.get(dataset_id) for dataset_id in ids]
