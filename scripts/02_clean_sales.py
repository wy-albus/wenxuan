from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean_sales import clean_sales_chunk  # noqa: E402
from src.data.reports import write_markdown  # noqa: E402
from src.utils.io_utils import (  # noqa: E402
    csv_entries,
    detect_encoding,
    ensure_directories,
    ensure_pyarrow,
    iter_zip_csv_chunks,
    load_yaml,
    project_path,
    scan_zip_files,
)


def clean_all_sales(chunksize: int = 200_000) -> dict:
    ensure_pyarrow()
    paths = load_yaml("config/paths.yaml")
    ensure_directories([paths["interim_dir"], paths["reports_dir"]])
    interim_dir = project_path(paths["interim_dir"])

    for old in interim_dir.glob("cleaned_sales_*.parquet"):
        old.unlink()

    zip_files = scan_zip_files(paths["raw_data_dir"])
    total_in = 0
    total_out = 0
    part_count = 0
    dropped = 0
    outputs = []

    for zip_path in zip_files:
        for entry in csv_entries(zip_path):
            encoding = detect_encoding(zip_path, entry.filename)
            stem = Path(zip_path).stem
            for chunk_no, chunk in enumerate(
                iter_zip_csv_chunks(zip_path, entry.filename, encoding, chunksize=chunksize)
            ):
                cleaned, stats = clean_sales_chunk(chunk)
                total_in += stats["input_rows"]
                total_out += stats["output_rows"]
                dropped += stats["dropped_missing_required"]
                if cleaned.empty:
                    continue
                output = interim_dir / f"cleaned_sales_{stem}_part{chunk_no:04d}.parquet"
                cleaned.to_parquet(output, index=False)
                outputs.append(output.name)
                part_count += 1

    lines = [
        "# Cleaning Summary",
        "",
        f"- Input rows: {total_in}",
        f"- Output rows: {total_out}",
        f"- Dropped rows missing period/item_id/site_no: {dropped}",
        f"- Parquet parts: {part_count}",
        "",
        "## Output Files",
        "",
        "\n".join(f"- {name}" for name in outputs) if outputs else "_No cleaned files created._",
        "",
    ]
    write_markdown(paths["cleaning_summary"], "\n".join(lines))
    return {"input_rows": total_in, "output_rows": total_out, "parts": part_count}


if __name__ == "__main__":
    print(clean_all_sales())
