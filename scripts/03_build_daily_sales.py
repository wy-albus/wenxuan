from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aggregate_sales import build_daily_sales, combine_daily_parts  # noqa: E402
from src.utils.io_utils import ensure_directories, ensure_pyarrow, load_yaml, project_path  # noqa: E402


def build_daily_table() -> dict:
    ensure_pyarrow()
    paths = load_yaml("config/paths.yaml")
    ensure_directories([paths["processed_dir"]])
    interim_dir = project_path(paths["interim_dir"])
    cleaned_files = sorted(interim_dir.glob("cleaned_sales_*.parquet"))
    if not cleaned_files:
        raise FileNotFoundError("No cleaned parquet files found in data/interim.")

    daily_parts = []
    for file in cleaned_files:
        cleaned = pd.read_parquet(file)
        if cleaned.empty:
            continue
        daily_parts.append(build_daily_sales(cleaned))

    daily = combine_daily_parts(daily_parts)
    output = project_path(paths["daily_sales"])
    daily.to_parquet(output, index=False)
    return {"rows": len(daily), "output": str(output)}


if __name__ == "__main__":
    print(build_daily_table())
