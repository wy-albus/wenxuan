from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io_utils import ensure_directories, load_yaml, project_path  # noqa: E402


SCRIPT_ORDER = [
    ("01_read_and_check_raw.py", "inspect_raw"),
    ("02_clean_sales.py", "clean_all_sales"),
    ("03_build_daily_sales.py", "build_daily_table"),
    ("04_build_monthly_sales.py", "build_monthly_table"),
    ("05_long_tail_analysis.py", "analyze_long_tail"),
]


def _load_script(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline() -> dict:
    paths = load_yaml("config/paths.yaml")
    ensure_directories(
        [
            paths["raw_data_dir"],
            paths["interim_dir"],
            paths["processed_dir"],
            paths["outputs_dir"],
            paths["reports_dir"],
        ]
    )

    results = {}
    for script_name, fn_name in SCRIPT_ORDER:
        print(f"Running {script_name} ...", flush=True)
        module = _load_script(script_name)
        fn = getattr(module, fn_name)
        try:
            results[script_name] = fn()
        except Exception as exc:
            print(f"FAILED at {script_name}: {exc}", file=sys.stderr)
            print("Check the previous traceback and the newest report/output file.", file=sys.stderr)
            raise

    daily = pd.read_parquet(project_path(paths["daily_sales"]))
    monthly = pd.read_parquet(project_path(paths["monthly_sales"]))
    summary = {
        "raw_total_rows": results["01_read_and_check_raw.py"]["total_rows"],
        "zip_count": results["01_read_and_check_raw.py"]["zip_count"],
        "csv_count": results["01_read_and_check_raw.py"]["csv_count"],
        "time_range": (
            str(results["01_read_and_check_raw.py"]["min_period"].date()),
            str(results["01_read_and_check_raw.py"]["max_period"].date()),
        ),
        "site_count": int(monthly["site_no"].nunique()),
        "item_count": int(monthly["item_id"].nunique()),
        "isbn_missing_rate": results["01_read_and_check_raw.py"]["isbn_missing_rate"],
        "qty_negative_rows": results["01_read_and_check_raw.py"]["qty_negative"],
        "rtn_flag_values": results["01_read_and_check_raw.py"]["rtn_values"],
        "channel_values": results["01_read_and_check_raw.py"]["channel_values"],
        "daily_rows": len(daily),
        "monthly_rows": len(monthly),
        "long_tail_findings": results["05_long_tail_analysis.py"]["findings"],
    }
    print("FINAL SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    run_pipeline()
