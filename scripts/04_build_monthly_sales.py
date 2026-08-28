from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aggregate_sales import build_monthly_sales  # noqa: E402
from src.utils.io_utils import ensure_directories, ensure_pyarrow, load_yaml, project_path  # noqa: E402


def build_monthly_table() -> dict:
    ensure_pyarrow()
    paths = load_yaml("config/paths.yaml")
    ensure_directories([paths["processed_dir"]])
    daily_path = project_path(paths["daily_sales"])
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily sales file not found: {daily_path}")

    daily = pd.read_parquet(daily_path)
    monthly = build_monthly_sales(daily)
    output = project_path(paths["monthly_sales"])
    monthly.to_parquet(output, index=False)
    return {"rows": len(monthly), "output": str(output)}


if __name__ == "__main__":
    print(build_monthly_table())
