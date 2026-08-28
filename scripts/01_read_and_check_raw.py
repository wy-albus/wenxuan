from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import CORE_FIELDS, missing_core_fields  # noqa: E402
from src.data.reports import write_markdown  # noqa: E402
from src.utils.io_utils import (  # noqa: E402
    csv_entries,
    detect_encoding,
    ensure_directories,
    iter_zip_csv_chunks,
    load_yaml,
    read_csv_header,
    scan_zip_files,
    safe_to_datetime,
    safe_to_numeric,
)


def _missing_rate(count: int, rows: int) -> str:
    if rows == 0:
        return "NA"
    return f"{count / rows:.2%}"


def _top_values(counter: Counter, limit: int = 20) -> str:
    if not counter:
        return "_无数据_"
    lines = ["| value | count |", "|---|---:|"]
    for value, count in counter.most_common(limit):
        lines.append(f"| {value if value != '' else '(blank)'} | {count} |")
    return "\n".join(lines)


def inspect_raw(chunksize: int = 200_000) -> dict:
    paths = load_yaml("config/paths.yaml")
    ensure_directories([paths["reports_dir"]])
    zip_files = scan_zip_files(paths["raw_data_dir"])

    file_rows: list[dict] = []
    totals = {
        "zip_count": len(zip_files),
        "csv_count": 0,
        "total_rows": 0,
        "min_period": None,
        "max_period": None,
        "qty_negative": 0,
    }
    core_missing_counts = defaultdict(int)
    conversion_bad_counts = defaultdict(int)
    channel_values = Counter()
    rtn_values = Counter()

    for zip_path in zip_files:
        entries = csv_entries(zip_path)
        for entry in entries:
            totals["csv_count"] += 1
            encoding = detect_encoding(zip_path, entry.filename)
            header = read_csv_header(zip_path, entry.filename, encoding)
            missing = missing_core_fields(header)
            row_count = 0
            file_min = None
            file_max = None
            file_core_missing = defaultdict(int)

            for chunk in iter_zip_csv_chunks(zip_path, entry.filename, encoding, chunksize=chunksize):
                row_count += len(chunk)
                for field in CORE_FIELDS:
                    if field in chunk.columns:
                        missing_count = int(chunk[field].isna().sum() + (chunk[field].astype(str).str.strip() == "").sum())
                    else:
                        missing_count = len(chunk)
                    file_core_missing[field] += missing_count
                    core_missing_counts[field] += missing_count

                if "period" in chunk.columns:
                    period = safe_to_datetime(chunk["period"])
                    conversion_bad_counts["period"] += int(period.isna().sum())
                    if period.notna().any():
                        current_min = period.min()
                        current_max = period.max()
                        file_min = current_min if file_min is None else min(file_min, current_min)
                        file_max = current_max if file_max is None else max(file_max, current_max)
                for field in ("qty", "price", "tlp", "tsp"):
                    if field in chunk.columns:
                        numeric = safe_to_numeric(chunk[field])
                        conversion_bad_counts[field] += int(numeric.isna().sum())
                        if field == "qty":
                            totals["qty_negative"] += int((numeric < 0).sum())

                if "oln_or_ofln" in chunk.columns:
                    channel_values.update(chunk["oln_or_ofln"].fillna("").astype(str).str.strip().tolist())
                if "rtn_flag" in chunk.columns:
                    rtn_values.update(chunk["rtn_flag"].fillna("").astype(str).str.strip().tolist())

            totals["total_rows"] += row_count
            if file_min is not None:
                totals["min_period"] = file_min if totals["min_period"] is None else min(totals["min_period"], file_min)
                totals["max_period"] = file_max if totals["max_period"] is None else max(totals["max_period"], file_max)

            file_rows.append(
                {
                    "zip": zip_path.name,
                    "csv": entry.filename,
                    "encoding": encoding,
                    "rows": row_count,
                    "columns": ", ".join(header),
                    "missing_fields": ", ".join(missing) if missing else "None",
                    "period_range": f"{file_min.date() if file_min is not None else 'NA'} - {file_max.date() if file_max is not None else 'NA'}",
                    "isbn_missing_rate": _missing_rate(file_core_missing["isbn"], row_count),
                    "gds_no_missing_rate": _missing_rate(file_core_missing["gds_no"], row_count),
                }
            )

    lines = [
        "# Data Quality Report",
        "",
        "## Overall",
        "",
        f"- Zip files: {totals['zip_count']}",
        f"- CSV files: {totals['csv_count']}",
        f"- Total raw rows: {totals['total_rows']}",
        f"- Time range: {totals['min_period'].date() if totals['min_period'] is not None else 'NA'} to {totals['max_period'].date() if totals['max_period'] is not None else 'NA'}",
        f"- qty < 0 rows: {totals['qty_negative']}",
        "",
        "## File Summary",
        "",
        pd.DataFrame(file_rows).to_markdown(index=False) if file_rows else "_No raw CSV files found._",
        "",
        "## Core Field Missing Rates",
        "",
        "| field | missing_count | missing_rate |",
        "|---|---:|---:|",
    ]
    for field in CORE_FIELDS:
        lines.append(
            f"| {field} | {core_missing_counts[field]} | {_missing_rate(core_missing_counts[field], totals['total_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Conversion Failures",
            "",
            "| field | failed_or_missing_count | rate |",
            "|---|---:|---:|",
        ]
    )
    for field in ("period", "qty", "price", "tlp", "tsp"):
        lines.append(
            f"| {field} | {conversion_bad_counts[field]} | {_missing_rate(conversion_bad_counts[field], totals['total_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## oln_or_ofln Values",
            "",
            _top_values(channel_values),
            "",
            "## rtn_flag Values",
            "",
            _top_values(rtn_values),
            "",
        ]
    )
    write_markdown(paths["data_quality_report"], "\n".join(lines))
    return {
        **totals,
        "channel_values": dict(channel_values.most_common(20)),
        "rtn_values": dict(rtn_values.most_common(20)),
        "isbn_missing_rate": _missing_rate(core_missing_counts["isbn"], totals["total_rows"]),
    }


if __name__ == "__main__":
    summary = inspect_raw()
    print(summary)
