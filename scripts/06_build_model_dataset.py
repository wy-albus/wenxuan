from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.model_dataset import (  # noqa: E402
    DatasetReportStats,
    OUTPUT_COLUMNS,
    build_site_model_dataset,
    month_to_ord,
)
from src.data.reports import write_markdown  # noqa: E402
from src.utils.io_utils import ensure_directories, ensure_pyarrow, load_yaml, project_path  # noqa: E402


READ_COLUMNS = [
    "month",
    "site_no",
    "blt_site_no",
    "item_id",
    "isbn",
    "gds_no",
    "gds_ctgry_3_lvel",
    "gds_ctgry_4_lvel",
    "gds_ctgry_5_lvel",
    "price",
    "total_qty",
    "offline_qty",
    "online_qty",
    "unknown_channel_qty",
    "total_tlp",
    "total_tsp",
    "avg_real_price",
    "discount_rate",
    "sales_days",
    "sales_count",
    "return_count",
    "return_qty",
]

STRING_COLUMNS = [
    "month",
    "site_no",
    "blt_site_no",
    "item_id",
    "isbn",
    "gds_no",
    "gds_ctgry_3_lvel",
    "gds_ctgry_4_lvel",
    "gds_ctgry_5_lvel",
    "split",
]

INT_COLUMNS = [
    "sales_days",
    "sales_count",
    "return_count",
    "future_has_sales_1m",
    "future_has_sales_2m",
    "active_months_last_3m",
    "active_months_last_6m",
    "zero_sales_months_last_3m",
    "zero_sales_months_last_6m",
    "is_sold_last_1m",
    "is_sold_last_3m",
    "is_sold_last_6m",
    "months_since_last_sale",
]


def coerce_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
    for col in OUTPUT_COLUMNS:
        if col not in STRING_COLUMNS and col not in INT_COLUMNS and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df[OUTPUT_COLUMNS]


def _ratio_table(counts: dict, total: int) -> str:
    lines = ["| value | count | ratio |", "|---|---:|---:|"]
    for key, value in counts.items():
        ratio = value / total if total else 0
        lines.append(f"| {key} | {value} | {ratio:.2%} |")
    return "\n".join(lines)


def _stats_table(stats: dict) -> str:
    lines = ["| metric | value |", "|---|---:|"]
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_report(stats: DatasetReportStats, output_path: Path, elapsed_seconds: float) -> None:
    lines = [
        "# Model Dataset Report",
        "",
        "## Overview",
        "",
        f"- Input monthly rows: {stats.input_rows}",
        f"- Rows after zero-month panel completion: {stats.panel_rows}",
        f"- Final model dataset rows: {stats.final_rows}",
        f"- Field count: {stats.field_count}",
        f"- Time range: {stats.min_month} to {stats.max_month}",
        f"- Output file: `{output_path.as_posix()}`",
        f"- Build elapsed seconds: {elapsed_seconds:.1f}",
        "",
        "## Split Counts",
        "",
        _ratio_table(stats.split_counts, stats.final_rows),
        "",
        "## future_qty_1m Distribution",
        "",
        _stats_table(stats.future_qty_1m.as_dict()),
        "",
        "## future_qty_2m Distribution",
        "",
        _stats_table(stats.future_qty_2m.as_dict()),
        "",
        "## future_has_sales_1m Ratio",
        "",
        _ratio_table(stats.future_has_sales_1m_counts, stats.final_rows),
        "",
        "## future_has_sales_2m Ratio",
        "",
        _ratio_table(stats.future_has_sales_2m_counts, stats.final_rows),
        "",
        "## Current Month total_qty Buckets",
        "",
        _ratio_table(stats.current_qty_buckets, stats.final_rows),
        "",
        "## Leakage Check",
        "",
        "- Lag and rolling features are built from shifted historical values only.",
        "- `future_qty_1m` and `future_qty_2m` are created after feature construction and are not used in input feature windows.",
        "- Last two months per `site_no x item_id` are dropped because `future_qty_2m` cannot be fully observed.",
        "- Splits are time-based: train `2023-01` to `2025-06`, valid `2025-07` to `2025-12`, test `2026-01` to `2026-04`.",
        "",
        "## Next-stage Modeling Suggestions",
        "",
        "- Start with historical mean and weighted moving average baselines using this dataset.",
        "- Use `future_qty_1m` first, then extend to `future_qty_2m` once baseline evaluation is stable.",
        "- For LightGBM, use `log1p(future_qty_1m)` or Poisson/Tweedie objectives because the target remains strongly long-tailed.",
        "- For the later two-stage model, use `future_has_sales_1m` or `future_has_sales_2m` as the classification target and the positive-sales subset for regression.",
        "",
    ]
    report_path = load_yaml("config/paths.yaml")["model_dataset_report"]
    write_markdown(report_path, "\n".join(lines))


def compute_panel_row_count(monthly: pd.DataFrame) -> int:
    max_month_ord = int(month_to_ord(monthly["month"]).max())
    first = monthly.groupby(["site_no", "item_id"], sort=False)["month"].min().reset_index()
    first_ord = month_to_ord(first["month"])
    return int((max_month_ord - first_ord + 1).sum())


def summarize_existing_output() -> dict:
    start = time.time()
    paths = load_yaml("config/paths.yaml")
    monthly_path = project_path(paths["monthly_sales"])
    output_path = project_path(paths["model_dataset_monthly"])
    if not output_path.exists():
        raise FileNotFoundError(f"Existing model dataset not found: {output_path}")

    monthly_keys = pd.read_parquet(monthly_path, columns=["month", "site_no", "item_id"])
    stats = DatasetReportStats(input_rows=len(monthly_keys), panel_rows=compute_panel_row_count(monthly_keys))
    pf = pq.ParquetFile(output_path)
    columns = [
        "month",
        "split",
        "total_qty",
        "future_qty_1m",
        "future_qty_2m",
        "future_has_sales_1m",
        "future_has_sales_2m",
    ]
    for batch in pf.iter_batches(columns=columns, batch_size=500_000):
        chunk = batch.to_pandas()
        stats.update_output(chunk)
    elapsed = time.time() - start
    write_report(stats, output_path, elapsed)
    return {
        "output": str(output_path),
        "exists": output_path.exists(),
        "input_rows": stats.input_rows,
        "panel_rows": stats.panel_rows,
        "final_rows": stats.final_rows,
        "field_count": stats.field_count,
        "split_counts": stats.split_counts,
        "time_range": (stats.min_month, stats.max_month),
        "elapsed_seconds": elapsed,
    }


def build_model_dataset() -> dict:
    start = time.time()
    ensure_pyarrow()
    paths = load_yaml("config/paths.yaml")
    ensure_directories([paths["processed_dir"], paths["reports_dir"]])

    monthly_path = project_path(paths["monthly_sales"])
    output_path = project_path(paths["model_dataset_monthly"])
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly sales file not found: {monthly_path}")
    if output_path.exists():
        output_path.unlink()

    monthly = pd.read_parquet(monthly_path, columns=READ_COLUMNS)
    stats = DatasetReportStats(input_rows=len(monthly))
    max_month_ord = int(month_to_ord(monthly["month"]).max())

    writer: pq.ParquetWriter | None = None
    try:
        site_values = monthly["site_no"].drop_duplicates().tolist()
        for idx, site_no in enumerate(site_values, start=1):
            site_df = monthly[monthly["site_no"].eq(site_no)].copy()
            dataset, panel_rows = build_site_model_dataset(site_df, max_month_ord)
            stats.panel_rows += panel_rows
            if not dataset.empty:
                dataset = coerce_output_dtypes(dataset)
                table = pa.Table.from_pandas(dataset, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)
                stats.update_output(dataset)
            print(
                f"[{idx}/{len(site_values)}] site={site_no} monthly_rows={len(site_df)} "
                f"panel_rows={panel_rows} output_rows={len(dataset)}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start
    write_report(stats, output_path, elapsed)
    summary = {
        "output": str(output_path),
        "exists": output_path.exists(),
        "input_rows": stats.input_rows,
        "panel_rows": stats.panel_rows,
        "final_rows": stats.final_rows,
        "field_count": stats.field_count,
        "split_counts": stats.split_counts,
        "time_range": (stats.min_month, stats.max_month),
        "elapsed_seconds": elapsed,
    }
    print("FINAL MODEL DATASET SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-existing",
        action="store_true",
        help="Regenerate reports/model_dataset_report.md from an existing model_dataset_monthly.parquet.",
    )
    args = parser.parse_args()
    if args.report_existing:
        print(summarize_existing_output())
    else:
        build_model_dataset()
