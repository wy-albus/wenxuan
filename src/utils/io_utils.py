from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODINGS = ("utf-8-sig", "gbk", "gb18030")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for path in paths:
        project_path(path).mkdir(parents=True, exist_ok=True)


def ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet outputs. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def clean_column_name(column: object) -> str:
    text = str(column)
    text = text.strip().strip('"').strip("'")
    text = text.replace("\ufeff", "")
    text = text.replace("\t", "")
    return text.strip().lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def scan_zip_files(raw_dir: str | Path) -> list[Path]:
    raw_path = project_path(raw_dir)
    return sorted(raw_path.glob("*.zip"))


def csv_entries(zip_path: str | Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as zf:
        return [info for info in zf.infolist() if info.filename.lower().endswith(".csv")]


def detect_encoding(zip_path: str | Path, entry_name: str, encodings: Iterable[str] = ENCODINGS) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry_name) as raw:
            sample = raw.read(128 * 1024)
    for encoding in encodings:
        try:
            sample.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            continue
    return "gb18030"


def read_csv_header(zip_path: str | Path, entry_name: str, encoding: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry_name) as raw:
            text = io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline="")
            header = pd.read_csv(text, nrows=0, dtype=str)
    return [clean_column_name(col) for col in header.columns]


def iter_zip_csv_chunks(
    zip_path: str | Path,
    entry_name: str,
    encoding: str,
    chunksize: int = 200_000,
    usecols: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry_name) as raw:
            text = io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline="")
            reader = pd.read_csv(
                text,
                dtype=str,
                chunksize=chunksize,
                low_memory=False,
                usecols=usecols,
            )
            for chunk in reader:
                yield normalize_columns(chunk)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")
