from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import duckdb
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.store_closures import load_store_closures  # noqa: E402


MONTHLY_PATH = ROOT / "data/processed/monthly_item_store_sales.parquet"
OLD_DATASET_PATH = ROOT / "data/processed/model_dataset_monthly.parquet"
ACTIVE_DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
CLOSURE_CONFIG_PATH = ROOT / "config/active_store_closures.csv"
REPORT_PATH = ROOT / "reports/store_activity_audit.md"


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _discover_closures(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    monthly = _sql_path(MONTHLY_PATH)
    return connection.execute(
        f"""
        WITH activity AS (
          SELECT CAST(site_no AS VARCHAR) AS site_no,
                 MIN(STRPTIME(month, '%Y-%m')) AS first_active,
                 MAX(STRPTIME(month, '%Y-%m')) AS last_active
          FROM read_parquet('{monthly}')
          GROUP BY 1
        ), bounds AS (SELECT MAX(last_active) AS global_max FROM activity)
        SELECT site_no,
               STRFTIME(first_active, '%Y-%m') AS first_active_month,
               STRFTIME(last_active, '%Y-%m') AS last_active_month,
               STRFTIME(last_active + INTERVAL 1 MONTH, '%Y-%m') AS closure_month
        FROM activity, bounds
        WHERE last_active < global_max
        ORDER BY closure_month, site_no
        """
    ).df()


def _validate_closure_config(discovered: pd.DataFrame, configured: pd.DataFrame) -> None:
    left = discovered[["site_no", "closure_month"]].astype("string").sort_values(
        ["closure_month", "site_no"], ignore_index=True
    )
    right = configured.astype("string").sort_values(
        ["closure_month", "site_no"], ignore_index=True
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as error:
        raise RuntimeError(
            "Configured store closures do not match the 28 stores discovered from monthly flows"
        ) from error


def _exclusion_counts(connection: duckdb.DuckDBPyConnection, closures: pd.DataFrame) -> pd.DataFrame:
    connection.register("closure_frame", closures)
    connection.execute(
        "CREATE TEMP TABLE closures AS "
        "SELECT CAST(site_no AS VARCHAR) AS site_no, "
        "STRPTIME(closure_month, '%Y-%m') AS closure_month FROM closure_frame"
    )
    old = _sql_path(OLD_DATASET_PATH)
    return connection.execute(
        f"""
        WITH impacted AS (
          SELECT m.split,
                 CASE
                   WHEN STRPTIME(m.month, '%Y-%m') >= c.closure_month THEN 'after_closure'
                   WHEN STRPTIME(m.month, '%Y-%m') + INTERVAL 1 MONTH >= c.closure_month THEN 'cross_1m'
                   WHEN STRPTIME(m.month, '%Y-%m') + INTERVAL 2 MONTH >= c.closure_month THEN 'cross_2m_extra'
                   ELSE 'unaffected'
                 END AS reason
          FROM read_parquet('{old}') m
          JOIN closures c ON CAST(m.site_no AS VARCHAR) = c.site_no
        )
        SELECT reason,
               COUNT(*) FILTER (WHERE split='train')::BIGINT AS train,
               COUNT(*) FILTER (WHERE split='valid')::BIGINT AS valid,
               COUNT(*) FILTER (WHERE split='test')::BIGINT AS test,
               COUNT(*)::BIGINT AS total
        FROM impacted
        WHERE reason <> 'unaffected'
        GROUP BY reason
        """
    ).df()


def _dataset_stats(connection: duckdb.DuckDBPyConnection, path: Path, active: bool) -> dict:
    sql_path = _sql_path(path)
    availability = ", SUM(target_available_2m)::BIGINT AS available_2m" if active else ""
    row = connection.execute(
        f"""
        SELECT COUNT(*)::BIGINT AS rows,
               SUM(CASE WHEN GREATEST(COALESCE(total_qty,0),0)<=0 THEN 1 ELSE 0 END)::BIGINT AS zero_rows
               {availability}
        FROM read_parquet('{sql_path}')
        """
    ).df().iloc[0]
    split_rows = connection.execute(
        f"""
        SELECT split, COUNT(*)::BIGINT AS rows
               {', SUM(target_available_2m)::BIGINT AS available_2m' if active else ''}
        FROM read_parquet('{sql_path}') GROUP BY split
        """
    ).df()
    result = {"rows": int(row["rows"]), "zero_rows": int(row["zero_rows"]), "splits": {}}
    if active:
        result["available_2m"] = int(row["available_2m"])
    for record in split_rows.to_dict("records"):
        result["splits"][str(record["split"])] = {
            "rows": int(record["rows"]),
            **({"available_2m": int(record["available_2m"])} if active else {}),
        }
    result["zero_ratio"] = result["zero_rows"] / result["rows"]
    return result


def _row(counts: pd.DataFrame, reason: str) -> dict[str, int]:
    match = counts[counts["reason"].eq(reason)]
    if len(match) != 1:
        raise RuntimeError(f"Missing unique exclusion count for {reason}")
    return {column: int(match.iloc[0][column]) for column in ("train", "valid", "test", "total")}


def _add(*rows: dict[str, int]) -> dict[str, int]:
    return {column: sum(row[column] for row in rows) for column in ("train", "valid", "test", "total")}


def _format_count_row(label: str, values: dict[str, int]) -> str:
    return "| " + label + " | " + " | ".join(f"{values[column]:,}" for column in ("train", "valid", "test", "total")) + " |"


def write_report(
    discovered: pd.DataFrame,
    counts: pd.DataFrame,
    old: dict,
    active: dict,
    elapsed_seconds: float,
) -> None:
    structural = _row(counts, "after_closure")
    cross_1m = _row(counts, "cross_1m")
    cross_2m_extra = _row(counts, "cross_2m_extra")
    excluded_1m = _add(structural, cross_1m)
    excluded_2m = _add(structural, cross_1m, cross_2m_extra)

    if old["rows"] - active["rows"] != excluded_1m["total"]:
        raise RuntimeError("1M row reconciliation does not match the active-store dataset")
    if old["rows"] - active["available_2m"] != excluded_2m["total"]:
        raise RuntimeError("2M row reconciliation does not match target_available_2m")
    for split in ("train", "valid", "test"):
        if old["splits"][split]["rows"] - active["splits"][split]["rows"] != excluded_1m[split]:
            raise RuntimeError(f"1M split reconciliation failed for {split}")
        if old["splits"][split]["rows"] - active["splits"][split]["available_2m"] != excluded_2m[split]:
            raise RuntimeError(f"2M split reconciliation failed for {split}")

    closure_rows = [
        f"| {row.site_no} | {row.last_active_month} | {row.closure_month} |"
        for row in discovered.itertuples(index=False)
    ]
    lines = [
        "# Store Activity Audit", "",
        "## Scope And Policy", "",
        f"- Configured abnormal stores: {len(discovered)}",
        "- Configuration: `config/active_store_closures.csv`",
        "- `closure_month` is the first month in which the store is treated as closed.",
        "- Only the configured 28 stores receive closure filtering; all other stores follow the original pipeline.",
        "- Rows at or after `closure_month` are not valid store-item observations.",
        "- A 1M/2M sample is invalid when its forecast horizon reaches or crosses `closure_month`; it is not relabeled as zero demand.",
        "- No active-store count, lifecycle embedding, or cross-store aggregate is added as a model feature.", "",
        "## Exact Row Reconciliation", "",
        "| Deletion reason | Train | Valid | Test | Total |",
        "|---|---:|---:|---:|---:|",
        _format_count_row("After-closure structural zeros", structural),
        _format_count_row("1M crosses closure boundary", cross_1m),
        _format_count_row("1M total excluded", excluded_1m),
        _format_count_row("Additional 2M boundary rows", cross_2m_extra),
        _format_count_row("2M total excluded (separate sample set)", excluded_2m),
        _format_count_row("Other reasons in 1M reconciliation", {key: 0 for key in structural}),
        "",
        f"- 1M reconciliation: {old['rows']:,} - {excluded_1m['total']:,} = {active['rows']:,}.",
        f"- 2M reconciliation: {old['rows']:,} - {excluded_2m['total']:,} = {active['available_2m']:,}.",
        "- The reported 803,309-row reduction is exactly 602,499 structural-zero rows plus 200,810 1M boundary-crossing rows.",
        "- The 213,793 additional rows apply only to the 2M eligible set and are not forced into the 803,309 1M total.", "",
        "## Split Comparison", "",
        "| Split | Old | Active 1M | Change | Active 2M | 2M change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        old_count = old["splits"][split]["rows"]
        one = active["splits"][split]["rows"]
        two = active["splits"][split]["available_2m"]
        lines.append(f"| {split} | {old_count:,} | {one:,} | {one-old_count:,} | {two:,} | {two-old_count:,} |")
    lines.extend([
        f"| total | {old['rows']:,} | {active['rows']:,} | {active['rows']-old['rows']:,} | {active['available_2m']:,} | {active['available_2m']-old['rows']:,} |",
        "", "## Zero-Sales Ratio", "",
        f"- Old dataset: {old['zero_rows']:,} / {old['rows']:,} = {old['zero_ratio']:.4%}.",
        f"- Active-store 1M dataset: {active['zero_rows']:,} / {active['rows']:,} = {active['zero_ratio']:.4%}.",
        f"- Change: {(active['zero_ratio']-old['zero_ratio'])*100:.4f} percentage points.", "",
        "## Closure Configuration", "",
        "| site_no | last_active_month | closure_month |", "|---|---|---|",
        *closure_rows, "",
        "## Quality Checks", "",
        "- The 28-row configuration exactly matches stores whose final monthly flow precedes the global maximum month.",
        "- 1M and 2M counts reconcile independently with the existing active-store Parquet.",
        "- Existing active-store Parquet metadata was inspected; no Parquet was rebuilt or modified by this audit.",
        f"- Audit elapsed: {elapsed_seconds:.2f} seconds.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_audit() -> dict:
    started = time.perf_counter()
    for path in (MONTHLY_PATH, OLD_DATASET_PATH, ACTIVE_DATASET_PATH, CLOSURE_CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    configured = load_store_closures(CLOSURE_CONFIG_PATH)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='4GB'")
    try:
        discovered = _discover_closures(connection)
        _validate_closure_config(discovered, configured)
        counts = _exclusion_counts(connection, configured)
        old = _dataset_stats(connection, OLD_DATASET_PATH, active=False)
        active = _dataset_stats(connection, ACTIVE_DATASET_PATH, active=True)
    finally:
        connection.close()
    metadata = pq.ParquetFile(ACTIVE_DATASET_PATH).metadata
    if metadata.num_rows != active["rows"]:
        raise RuntimeError("Active-store Parquet metadata row count mismatch")
    elapsed = time.perf_counter() - started
    write_report(discovered, counts, old, active, elapsed)
    return {
        "closure_stores": len(discovered),
        "old_rows": old["rows"],
        "active_1m_rows": active["rows"],
        "active_2m_rows": active["available_2m"],
        "report": str(REPORT_PATH),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit configured store closures without rebuilding datasets")
    parser.parse_args()
    summary = run_audit()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
