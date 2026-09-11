from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .runtime import database_path


class MonthlyDataService:
    def __init__(self) -> None:
        self.db_path = database_path()

    def read_standard_history(self) -> tuple[pd.DataFrame, dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            dataset_rows = conn.execute("SELECT * FROM datasets ORDER BY rowid ASC").fetchall()
        frames: list[pd.DataFrame] = []
        source_file_ids: list[str] = []
        source_months: set[str] = set()
        datasets: list[dict] = []
        for row in dataset_rows:
            record = dict(row)
            monthly_path = Path(record["monthly_parquet_path"])
            if not monthly_path.is_file():
                continue
            frame = pd.read_parquet(monthly_path)
            frame["source_dataset_id"] = record["dataset_id"]
            frames.append(frame)
            months = frame["month"].dropna().astype(str).unique().tolist()
            source_months.update(months)
            source_file_ids.extend(json.loads(record["source_files"] or "[]"))
            datasets.append(
                {
                    "dataset_id": record["dataset_id"],
                    "dataset_name": record["dataset_name"],
                    "monthly_parquet_path": str(monthly_path),
                    "date_range": json.loads(record["date_range"]),
                    "months": sorted(months),
                }
            )
        if not frames:
            return pd.DataFrame(), {"datasets": [], "source_months": [], "source_file_ids": []}
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["month", "site_no", "item_id"], keep="last")
        return combined, {
            "datasets": datasets,
            "source_months": sorted(source_months),
            "source_file_ids": sorted(set(source_file_ids)),
            "monthly_data_version": "sqlite-datasets-v1",
        }
