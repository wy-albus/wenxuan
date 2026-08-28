from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.active_store_dataset import (
    ACTIVE_OUTPUT_COLUMNS,
    build_store_activity,
    filter_existing_model_frame_to_active_store,
)
from src.evaluation.mc_metrics import load_mc_level_config, quantity_to_mc


MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_item_store_sales.parquet"
OLD_DATASET_PATH = ROOT / "data" / "processed" / "model_dataset_monthly.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "model_dataset_monthly_active_store.parquet"
STORE_REPORT_PATH = ROOT / "reports" / "store_activity_audit.md"
DATASET_REPORT_PATH = ROOT / "reports" / "model_dataset_active_store_report.md"
MC_CONFIG_PATH = ROOT / "config" / "mc_sales_levels.yaml"

READ_COLUMNS = [
    "month", "site_no", "blt_site_no", "item_id", "isbn", "gds_no",
    "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel", "price",
    "total_qty", "offline_qty", "online_qty", "unknown_channel_qty", "total_tlp",
    "total_tsp", "avg_real_price", "discount_rate", "sales_days", "sales_count",
    "return_count", "return_qty",
]

ACTIVE_COLUMNS = [
    *[column for column in ACTIVE_OUTPUT_COLUMNS if column != "split"],
    "target_mc_1m", "target_mc_2m", "split",
]

STRING_COLUMNS = {
    "month", "site_no", "blt_site_no", "item_id", "isbn", "gds_no",
    "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel",
    "store_first_active_month", "store_last_active_month", "split",
}

INT_COLUMNS = {
    "sales_days", "sales_count", "return_count", "active_months_last_3m",
    "active_months_last_6m", "zero_sales_months_last_3m", "zero_sales_months_last_6m",
    "is_sold_last_1m", "is_sold_last_3m", "is_sold_last_6m", "months_since_last_sale",
    "target_available_1m", "target_available_2m", "target_mc_1m", "target_mc_2m",
}


def _month_ord(values: pd.Series) -> np.ndarray:
    return pd.PeriodIndex(values.astype(str), freq="M").asi8.astype("int32", copy=False)


def _coerce_output(frame: pd.DataFrame, mc_config) -> pd.DataFrame:
    frame = frame.copy()
    frame["target_mc_1m"] = quantity_to_mc(frame["future_qty_1m"], mc_config, "1m")
    target_mc_2m = np.full(len(frame), -1, dtype="int8")
    available_2m = frame["target_available_2m"].eq(1).to_numpy()
    if available_2m.any():
        target_mc_2m[available_2m] = quantity_to_mc(
            frame.loc[available_2m, "future_qty_2m"], mc_config, "2m"
        ).astype("int8")
    frame["target_mc_2m"] = target_mc_2m
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1 if column == "target_mc_2m" else 0).astype("int32")
    for column in ACTIVE_COLUMNS:
        if column not in STRING_COLUMNS and column not in INT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame[ACTIVE_COLUMNS]


def _replace_or_append(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, values) if index >= 0 else table.append_column(name, values)


def _filter_arrow_batch(
    batch: pa.RecordBatch | pa.Table,
    activity: pd.DataFrame,
    first_month_map: dict[str, str],
    last_month_map: dict[str, str],
    last_ord_map: dict[str, int],
    mc_config,
) -> tuple[pa.Table, pd.DataFrame, np.ndarray]:
    table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
    selection = table.select(["month", "site_no", "item_id", "split", "total_qty", "future_qty_1m", "future_qty_2m"]).to_pandas()
    site = selection["site_no"].astype("string")
    month_ord = _month_ord(selection["month"])
    last_ord = site.map(last_ord_map).to_numpy(dtype="float64")
    if np.isnan(last_ord).any():
        raise ValueError("Model dataset contains a site missing from monthly activity audit")
    keep = month_ord + 1 <= last_ord
    available_2m = month_ord + 2 <= last_ord
    filtered_selection = selection.loc[keep].reset_index(drop=True)
    filtered_available_2m = available_2m[keep]
    filtered_site = site.loc[keep].reset_index(drop=True)
    filtered = table.filter(pa.array(keep))
    row_count = len(filtered_selection)
    available_array = pa.array(filtered_available_2m, type=pa.bool_())
    null_future_2m = pa.nulls(row_count, type=filtered["future_qty_2m"].type)
    filtered = _replace_or_append(
        filtered, "future_qty_2m", pc.if_else(available_array, filtered["future_qty_2m"], null_future_2m)
    )
    if "future_has_sales_2m" in filtered.column_names:
        null_has_sales_2m = pa.nulls(row_count, type=filtered["future_has_sales_2m"].type)
        filtered = _replace_or_append(
            filtered,
            "future_has_sales_2m",
            pc.if_else(available_array, filtered["future_has_sales_2m"], null_has_sales_2m),
        )
    target_1m = np.maximum(filtered["future_qty_1m"].combine_chunks().to_numpy(zero_copy_only=False), 0)
    target_2m = filtered["future_qty_2m"].combine_chunks().to_numpy(zero_copy_only=False)
    mc_1m = quantity_to_mc(target_1m, mc_config, "1m").astype("int32")
    mc_2m = np.full(row_count, -1, dtype="int32")
    if filtered_available_2m.any():
        mc_2m[filtered_available_2m] = quantity_to_mc(target_2m[filtered_available_2m], mc_config, "2m")
    filtered = _replace_or_append(filtered, "target_available_1m", pa.array(np.ones(row_count, dtype="int32")))
    filtered = _replace_or_append(filtered, "target_available_2m", pa.array(filtered_available_2m.astype("int32")))
    filtered = _replace_or_append(
        filtered, "store_first_active_month", pa.array(filtered_site.map(first_month_map).astype(str).tolist())
    )
    filtered = _replace_or_append(
        filtered, "store_last_active_month", pa.array(filtered_site.map(last_month_map).astype(str).tolist())
    )
    filtered = _replace_or_append(filtered, "target_mc_1m", pa.array(mc_1m))
    filtered = _replace_or_append(filtered, "target_mc_2m", pa.array(mc_2m))
    return filtered.select(ACTIVE_COLUMNS), filtered_selection, filtered_available_2m


def audit_old_structural_zeros(store_last_ord: dict[str, int]) -> dict:
    parquet = pq.ParquetFile(OLD_DATASET_PATH)
    rows = 0
    zero_rows = 0
    after_close_rows = 0
    after_close_nonzero = 0
    for batch in parquet.iter_batches(columns=["month", "site_no", "total_qty"], batch_size=500_000):
        frame = batch.to_pandas()
        rows += len(frame)
        qty = np.maximum(pd.to_numeric(frame["total_qty"], errors="coerce").fillna(0).to_numpy(dtype="float64"), 0)
        zero_rows += int((qty <= 0).sum())
        last = frame["site_no"].astype("string").map(store_last_ord).to_numpy(dtype="float64")
        after_close = _month_ord(frame["month"]) > last
        after_close_rows += int(after_close.sum())
        after_close_nonzero += int(((qty > 0) & after_close).sum())
    return {
        "rows": rows,
        "zero_rows": zero_rows,
        "zero_ratio": zero_rows / rows if rows else np.nan,
        "after_close_rows": after_close_rows,
        "after_close_nonzero_rows": after_close_nonzero,
    }


def _ratio(value: int, total: int) -> str:
    return f"{value / total:.2%}" if total else "N/A"


def _write_reports(activity: pd.DataFrame, old: dict, stats: dict, elapsed: float) -> None:
    global_last = str(activity["last_active_month"].max())
    closed_before_end = int((activity["last_active_month"] < global_last).sum())
    internal_gap_stores = int((activity["internal_gap_month_count"] > 0).sum())
    STORE_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Store Activity Audit",
                "",
                f"- Monthly flow rows audited: {stats['input_rows']:,}",
                f"- Stores: {len(activity):,}",
                f"- Global observed range: {activity['first_active_month'].min()} to {global_last}",
                f"- Stores whose last flow precedes global maximum: {closed_before_end:,}",
                f"- Stores with at least one internal no-flow gap and later recovery: {internal_gap_stores:,}",
                f"- Total internal gap months inside active intervals: {int(activity['internal_gap_month_count'].sum()):,}",
                f"- Old model rows after store last active month: {old['after_close_rows']:,}",
                f"- Nonzero old rows after store last active month (sanity expectation 0): {old['after_close_nonzero_rows']:,}",
                "",
                "## Rule",
                "",
                "Store activity is based on the existence of any monthly transaction row, not positive net quantity. Internal gaps are retained when later transactions resume. No item panel rows are generated after the store's final observed transaction month.",
                "",
                "## Sanity Checks",
                "",
                f"- Rows generated after store closure: {stats['rows_after_close']:,}",
                f"- Duplicate month + site_no + item_id keys: {stats['duplicate_keys']:,}",
                "- Result: PASS" if stats["rows_after_close"] == 0 and stats["duplicate_keys"] == 0 else "- Result: FAIL",
                "",
            ]
        ),
        encoding="utf-8",
    )
    DATASET_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Active-Store Model Dataset Report",
                "",
                f"- Input monthly rows: {stats['input_rows']:,}",
                f"- Active-store panel rows before target eligibility: {stats['panel_rows']:,}",
                f"- Final unified rows (1M target available): {stats['output_rows']:,}",
                f"- Rows with complete 2M target: {stats['available_2m_rows']:,}",
                f"- Fields: {len(ACTIVE_COLUMNS)}",
                f"- Output range: {stats['min_month']} to {stats['max_month']}",
                f"- Output size: {OUTPUT_PATH.stat().st_size / 2**30:.3f} GiB",
                f"- Build elapsed: {elapsed / 60:.1f} minutes",
                "",
                "## Structural-Zero Correction",
                "",
                "- Store activity boundaries and horizon eligibility are recomputed from `monthly_item_store_sales.parquet`; leakage-safe lag/rolling columns are streamed from the existing monthly-derived model table because retained pre-closure rows are mathematically unchanged.",
                f"- Old model rows: {old['rows']:,}",
                f"- Old rows after store closure: {old['after_close_rows']:,}",
                f"- Old zero-current-sales ratio: {_ratio(old['zero_rows'], old['rows'])}",
                f"- Active-store zero-current-sales ratio: {_ratio(stats['zero_rows'], stats['output_rows'])}",
                f"- Difference between old and active-store unified row count: {old['rows'] - stats['output_rows']:,}",
                "",
                "## Horizon Eligibility",
                "",
                "- 1M rows require month + 1 <= store_last_active_month.",
                "- 2M rows require month + 2 <= store_last_active_month.",
                "- Rows with a valid 1M target but unavailable 2M target remain in the common Parquet and are excluded by the 2M trainer using target_available_2m.",
                "- A store closing during the forecast horizon is never converted into a zero-sales target.",
                "",
                "## Split Counts",
                "",
                "| split | 1M eligible | 2M eligible |",
                "|---|---:|---:|",
                *[
                    f"| {split} | {stats['split_1m'].get(split, 0):,} | {stats['split_2m'].get(split, 0):,} |"
                    for split in ("train", "valid", "test")
                ],
                "",
                "## Leakage and Quality Checks",
                "",
                "- Lag and rolling features use shifted historical quantities only.",
                "- MC targets are derived only from matching future quantity targets and are excluded from all input feature lists.",
                f"- Duplicate keys: {stats['duplicate_keys']:,}",
                f"- Rows after store closure: {stats['rows_after_close']:,}",
                f"- Invalid 1M horizon rows: {stats['invalid_1m_horizon']:,}",
                f"- Invalid 2M horizon rows marked available: {stats['invalid_2m_horizon']:,}",
                "- Existing model_dataset_monthly.parquet was not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_active_store_dataset_arrow() -> dict:
    start = time.time()
    for path in (MONTHLY_PATH, OLD_DATASET_PATH, MC_CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()

    monthly = pd.read_parquet(MONTHLY_PATH, columns=["month", "site_no", "item_id"])
    activity = build_store_activity(monthly)
    activity_by_site = activity.set_index("site_no")
    last_ord_map = {str(index): int(value) for index, value in activity_by_site["last_active_ord"].items()}
    first_month_map = {str(index): str(value) for index, value in activity_by_site["first_active_month"].items()}
    last_month_map = {str(index): str(value) for index, value in activity_by_site["last_active_month"].items()}
    mc_config = load_mc_level_config(MC_CONFIG_PATH)
    monthly_with_ord = monthly.assign(month_ord=_month_ord(monthly["month"]))
    item_first = monthly_with_ord.groupby(["site_no", "item_id"], sort=False)["month_ord"].min().reset_index()
    item_last = item_first["site_no"].astype("string").map(last_ord_map).to_numpy(dtype="int64")
    active_panel_rows = int((item_last - item_first["month_ord"].to_numpy(dtype="int64") + 1).sum())
    del monthly_with_ord, item_first
    old = {"rows": 0, "zero_rows": 0, "zero_ratio": np.nan, "after_close_rows": 0, "after_close_nonzero_rows": 0}
    stats = {
        "input_rows": len(monthly), "panel_rows": active_panel_rows, "output_rows": 0, "available_2m_rows": 0,
        "zero_rows": 0, "duplicate_keys": 0, "rows_after_close": 0,
        "invalid_1m_horizon": 0, "invalid_2m_horizon": 0,
        "split_1m": Counter(), "split_2m": Counter(), "min_month": None, "max_month": None,
    }
    writer = None
    try:
        old_parquet = pq.ParquetFile(OLD_DATASET_PATH)
        old_columns = old_parquet.schema.names
        row_group_block = 5
        for index, start_group in enumerate(range(0, old_parquet.num_row_groups, row_group_block), start=1):
            groups = list(range(start_group, min(start_group + row_group_block, old_parquet.num_row_groups)))
            source_table = old_parquet.read_row_groups(groups, columns=old_columns)
            selection = source_table.select(["month", "site_no", "total_qty"]).to_pandas()
            old["rows"] += len(selection)
            original_qty = np.maximum(pd.to_numeric(selection["total_qty"], errors="coerce").fillna(0).to_numpy(dtype="float64"), 0)
            old["zero_rows"] += int((original_qty <= 0).sum())
            original_month_ord = _month_ord(selection["month"])
            original_last_ord = selection["site_no"].astype("string").map(last_ord_map).to_numpy(dtype="float64")
            after_close = original_month_ord > original_last_ord
            old["after_close_rows"] += int(after_close.sum())
            old["after_close_nonzero_rows"] += int(((original_qty > 0) & after_close).sum())
            table, filtered_selection, available_2m = _filter_arrow_batch(
                source_table, activity, first_month_map, last_month_map, last_ord_map, mc_config
            )
            if table.num_rows == 0:
                continue
            month_ord = _month_ord(filtered_selection["month"])
            last_ord = filtered_selection["site_no"].astype("string").map(last_ord_map).to_numpy(dtype="int64")
            stats["rows_after_close"] += int((month_ord > last_ord).sum())
            stats["invalid_1m_horizon"] += int((month_ord + 1 > last_ord).sum())
            stats["invalid_2m_horizon"] += int(((month_ord + 2 > last_ord) & available_2m).sum())
            stats["duplicate_keys"] += int(filtered_selection.duplicated(["month", "site_no", "item_id"]).sum())
            stats["output_rows"] += table.num_rows
            stats["available_2m_rows"] += int(available_2m.sum())
            filtered_qty = np.maximum(pd.to_numeric(filtered_selection["total_qty"], errors="coerce").fillna(0).to_numpy(), 0)
            stats["zero_rows"] += int((filtered_qty <= 0).sum())
            stats["split_1m"].update(filtered_selection["split"].dropna().astype(str).tolist())
            stats["split_2m"].update(filtered_selection.loc[available_2m, "split"].dropna().astype(str).tolist())
            current_min, current_max = str(filtered_selection["month"].min()), str(filtered_selection["month"].max())
            stats["min_month"] = current_min if stats["min_month"] is None else min(stats["min_month"], current_min)
            stats["max_month"] = current_max if stats["max_month"] is None else max(stats["max_month"], current_max)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy", use_dictionary=True)
            writer.write_table(table, row_group_size=500_000)
            del source_table, selection, filtered_selection, table
            if index % 10 == 0:
                print(f"[batch {index}] output_rows={stats['output_rows']:,}", flush=True)
            if index % 5 == 0:
                gc.collect()
        old_parquet.close()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No active-store rows were generated")
    old["zero_ratio"] = old["zero_rows"] / old["rows"] if old["rows"] else np.nan
    temporary.replace(OUTPUT_PATH)
    elapsed = time.time() - start
    _write_reports(activity, old, stats, elapsed)
    return {**stats, "old": old, "elapsed_seconds": elapsed, "output": str(OUTPUT_PATH)}


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _mc_case_sql(quantity_expression: str, mc_config, *, horizon: str) -> str:
    quantity = f"(({quantity_expression}) / 2.0)" if horizon == "2m" else f"({quantity_expression})"
    rounded = f"FLOOR(GREATEST(COALESCE({quantity}, 0), 0) + 0.5)"
    clauses = []
    for next_bound, code in zip(mc_config.lower_bounds[1:], mc_config.codes[:-1]):
        clauses.append(f"WHEN {rounded} < {int(next_bound)} THEN {int(code)}")
    return "CASE " + " ".join(clauses) + f" ELSE {int(mc_config.codes[-1])} END"


def build_active_store_dataset() -> dict:
    start = time.time()
    for path in (MONTHLY_PATH, OLD_DATASET_PATH, MC_CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = ROOT / "data" / "temp" / "active_store_duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    mc_config = load_mc_level_config(MC_CONFIG_PATH)

    connection = duckdb.connect()
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=6")
    connection.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
    monthly_sql = _sql_path(MONTHLY_PATH)
    old_sql = _sql_path(OLD_DATASET_PATH)
    output_sql = _sql_path(temporary)
    month_date = "STRPTIME(month || '-01', '%Y-%m-%d')::DATE"
    connection.execute(
        f"CREATE TEMP VIEW monthly_source AS SELECT *, {month_date} AS month_date "
        f"FROM read_parquet('{monthly_sql}')"
    )
    connection.execute(
        "CREATE TEMP TABLE store_activity AS "
        "SELECT site_no, MIN(month) AS first_active_month, MAX(month) AS last_active_month, "
        "MIN(month_date) AS first_active_date, MAX(month_date) AS last_active_date, "
        "COUNT(DISTINCT month)::BIGINT AS active_month_count, "
        "DATE_DIFF('month', MIN(month_date), MAX(month_date)) + 1 AS operating_span_months, "
        "DATE_DIFF('month', MIN(month_date), MAX(month_date)) + 1 - COUNT(DISTINCT month) AS internal_gap_month_count "
        "FROM monthly_source GROUP BY site_no"
    )
    activity = connection.execute(
        "SELECT site_no, first_active_month, last_active_month, active_month_count, "
        "operating_span_months, internal_gap_month_count FROM store_activity ORDER BY site_no"
    ).df()
    input_rows = int(connection.execute("SELECT COUNT(*) FROM monthly_source").fetchone()[0])
    panel_rows = int(
        connection.execute(
            "WITH item_first AS (SELECT site_no,item_id,MIN(month_date) AS first_date "
            "FROM monthly_source GROUP BY site_no,item_id) "
            "SELECT SUM(DATE_DIFF('month', item_first.first_date, store_activity.last_active_date) + 1)::BIGINT "
            "FROM item_first JOIN store_activity USING(site_no)"
        ).fetchone()[0]
    )
    old = connection.execute(
        f"WITH old AS (SELECT *, STRPTIME(month || '-01', '%Y-%m-%d')::DATE AS month_date "
        f"FROM read_parquet('{old_sql}')) "
        "SELECT COUNT(*)::BIGINT AS rows, SUM(CASE WHEN GREATEST(COALESCE(total_qty,0),0)<=0 THEN 1 ELSE 0 END)::BIGINT AS zero_rows, "
        "SUM(CASE WHEN old.month_date > a.last_active_date THEN 1 ELSE 0 END)::BIGINT AS after_close_rows, "
        "SUM(CASE WHEN old.month_date > a.last_active_date AND GREATEST(COALESCE(total_qty,0),0)>0 THEN 1 ELSE 0 END)::BIGINT AS after_close_nonzero_rows "
        "FROM old JOIN store_activity a USING(site_no)"
    ).df().iloc[0].to_dict()
    old = {key: int(value) for key, value in old.items()}
    old["zero_ratio"] = old["zero_rows"] / old["rows"] if old["rows"] else np.nan

    mc_1m = _mc_case_sql("old.future_qty_1m", mc_config, horizon="1m")
    mc_2m = _mc_case_sql("old.future_qty_2m", mc_config, horizon="2m")
    available_2m = "old.month_date + INTERVAL '2 months' <= a.last_active_date"
    select_sql = (
        "SELECT old.* EXCLUDE(month_date,future_qty_2m,future_has_sales_2m,split), "
        f"CASE WHEN {available_2m} THEN old.future_qty_2m ELSE NULL END AS future_qty_2m, "
        f"CASE WHEN {available_2m} THEN old.future_has_sales_2m ELSE NULL END AS future_has_sales_2m, "
        "1::INTEGER AS target_available_1m, "
        f"CASE WHEN {available_2m} THEN 1 ELSE 0 END::INTEGER AS target_available_2m, "
        "a.first_active_month AS store_first_active_month, a.last_active_month AS store_last_active_month, "
        f"({mc_1m})::INTEGER AS target_mc_1m, "
        f"CASE WHEN {available_2m} THEN ({mc_2m})::INTEGER ELSE -1 END AS target_mc_2m, old.split "
        f"FROM (SELECT *, STRPTIME(month || '-01', '%Y-%m-%d')::DATE AS month_date FROM read_parquet('{old_sql}')) old "
        "JOIN store_activity a USING(site_no) "
        "WHERE old.month_date + INTERVAL '1 month' <= a.last_active_date"
    )
    connection.execute(
        f"COPY ({select_sql}) TO '{output_sql}' "
        "(FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500000, PER_THREAD_OUTPUT FALSE)"
    )
    temporary.replace(OUTPUT_PATH)

    output_sql_read = _sql_path(OUTPUT_PATH)
    summary = connection.execute(
        f"SELECT COUNT(*)::BIGINT AS output_rows, SUM(target_available_2m)::BIGINT AS available_2m_rows, "
        "SUM(CASE WHEN GREATEST(COALESCE(total_qty,0),0)<=0 THEN 1 ELSE 0 END)::BIGINT AS zero_rows, "
        "MIN(month) AS min_month, MAX(month) AS max_month, "
        "COUNT(*) - COUNT(DISTINCT (month,site_no,item_id)) AS duplicate_keys, "
        "SUM(CASE WHEN STRPTIME(month || '-01','%Y-%m-%d')::DATE > STRPTIME(store_last_active_month || '-01','%Y-%m-%d')::DATE THEN 1 ELSE 0 END)::BIGINT AS rows_after_close, "
        "SUM(CASE WHEN STRPTIME(month || '-01','%Y-%m-%d')::DATE + INTERVAL '1 month' > STRPTIME(store_last_active_month || '-01','%Y-%m-%d')::DATE AND target_available_1m=1 THEN 1 ELSE 0 END)::BIGINT AS invalid_1m_horizon, "
        "SUM(CASE WHEN STRPTIME(month || '-01','%Y-%m-%d')::DATE + INTERVAL '2 months' > STRPTIME(store_last_active_month || '-01','%Y-%m-%d')::DATE AND target_available_2m=1 THEN 1 ELSE 0 END)::BIGINT AS invalid_2m_horizon "
        f"FROM read_parquet('{output_sql_read}')"
    ).df().iloc[0].to_dict()
    stats = {
        "input_rows": input_rows,
        "panel_rows": panel_rows,
        **{key: (int(value) if key not in {"min_month", "max_month"} else str(value)) for key, value in summary.items()},
        "split_1m": Counter(),
        "split_2m": Counter(),
    }
    split_rows = connection.execute(
        f"SELECT split, COUNT(*)::BIGINT AS count_1m, SUM(target_available_2m)::BIGINT AS count_2m "
        f"FROM read_parquet('{output_sql_read}') GROUP BY split"
    ).fetchall()
    for split, count_1m, count_2m in split_rows:
        if split is not None:
            stats["split_1m"][str(split)] = int(count_1m)
            stats["split_2m"][str(split)] = int(count_2m)
    connection.close()
    elapsed = time.time() - start
    _write_reports(activity, old, stats, elapsed)
    return {**stats, "old": old, "elapsed_seconds": elapsed, "output": str(OUTPUT_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild the active-store dataset if it already exists")
    args = parser.parse_args()
    if OUTPUT_PATH.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing active-store dataset: {OUTPUT_PATH}. Use --force explicitly.")
    summary = build_active_store_dataset()
    print("ACTIVE-STORE DATASET COMPLETE")
    for key, value in summary.items():
        if key not in {"split_1m", "split_2m", "old"}:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
