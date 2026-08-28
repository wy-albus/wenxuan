from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULT_FILES = {
    "predictions": "predictions.parquet",
    "top_books": "top_books.parquet",
    "store_summary": "store_summary.parquet",
}


def read_result_rows(
    prediction_dir: str | Path,
    *,
    kind: str = "predictions",
    model_id: str | None = None,
    site_no: str | None = None,
    mc: str | None = None,
) -> pd.DataFrame:
    if kind not in RESULT_FILES:
        raise ValueError(f"Unsupported result kind: {kind}")
    rows = pd.read_parquet(Path(prediction_dir) / RESULT_FILES[kind])
    if model_id:
        rows = rows.loc[rows["model_id"].astype(str).eq(str(model_id))]
    if site_no:
        if "site_no" not in rows.columns:
            raise ValueError(f"{kind} has no site_no column")
        rows = rows.loc[rows["site_no"].astype(str).eq(str(site_no))]
    if mc:
        if "pred_mc" not in rows.columns:
            raise ValueError(f"{kind} cannot be filtered by MC")
        rows = rows.loc[rows["pred_mc"].eq(mc)]
    if "pred_qty_int" in rows.columns:
        rows = rows.sort_values(["pred_qty_int", "pred_qty_raw"], ascending=[False, False], kind="stable")
    return rows.reset_index(drop=True)


def page_rows(rows: pd.DataFrame, *, page: int, page_size: int) -> dict:
    total = len(rows)
    start = (page - 1) * page_size
    return {
        "items": rows.iloc[start:start + page_size].where(rows.notna(), None).to_dict(orient="records"),
        "total": int(total), "page": page, "page_size": page_size,
    }
