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
    sort_by: str = "pred_qty_int",
    sort_order: str = "desc",
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
        rows = sort_result_rows(rows, sort_by=sort_by, sort_order=sort_order)
    return rows.reset_index(drop=True)


def sort_result_rows(rows: pd.DataFrame, *, sort_by: str = "pred_qty_int", sort_order: str = "desc") -> pd.DataFrame:
    allowed = {"pred_qty_int", "p_sale", "item_id"}
    if sort_by not in allowed:
        raise ValueError(f"Unsupported sort_by: {sort_by}")
    if sort_order not in {"asc", "desc"}:
        raise ValueError(f"Unsupported sort_order: {sort_order}")
    ascending = sort_order == "asc"
    if sort_by == "item_id":
        return rows.sort_values(["item_id", "pred_qty_int", "p_sale"], ascending=[ascending, False, False], kind="stable")
    return rows.sort_values([sort_by, "p_sale", "item_id"], ascending=[ascending, False, True], kind="stable")


def page_rows(rows: pd.DataFrame, *, page: int, page_size: int) -> dict:
    total = len(rows)
    start = (page - 1) * page_size
    return {
        "items": rows.iloc[start:start + page_size].where(rows.notna(), None).to_dict(orient="records"),
        "total": int(total), "page": page, "page_size": page_size,
    }


def summarize_prediction_rows(rows: pd.DataFrame, *, model_id: str | None, site_no: str | None, mc: str | None) -> dict:
    """Return a display summary using exactly the rows selected by result filters."""
    return {
        "filters": {"model_id": model_id, "site_no": site_no, "mc": mc},
        "prediction_total": int(rows["pred_qty_int"].sum()),
        "predicted_nonzero_book_count": int((rows["pred_qty_int"] > 0).sum()),
        "mc_counts": {f"MC{level}": int((rows["pred_mc"] == f"MC{level}").sum()) for level in range(5)},
        "predicted_20_plus_book_count": int((rows["pred_qty_int"] >= 20).sum()),
        "store_count": int(rows["site_no"].nunique()),
        "item_count": int(rows["item_id"].nunique()),
        "row_count": int(len(rows)),
    }
