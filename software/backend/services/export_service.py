from __future__ import annotations

import re
import json
from copy import copy
from pathlib import Path

import pandas as pd

from .result_query_service import read_result_rows
from .runtime import runtime_root


MAX_EXCEL_DETAIL_ROWS = 1_000_000
MODEL_INFO = pd.DataFrame([
    {"model_id": "E0", "description": "基础 Two-stage，对照模型"},
    {"model_id": "E2", "description": "Cross-store 跨门店特征模型"},
    {"model_id": "E3", "description": "Diff + Cross-store 模型"},
    {"model_id": "说明", "description": "当前为未来预测结果，不是 WAPE、Recall 或 Macro-F1 等评价指标。"},
])


def _safe_component(value: str | None, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or fallback)


def _summary_rows(run: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    result: list[dict] = []
    for model_id, rows in predictions.groupby("model_id", sort=False):
        result.append({
            "prediction_run_id": run["prediction_run_id"], "dataset_id": run["dataset_id"], "model_id": model_id,
            "observation_month": run["observation_month"], "predicted_total": int(rows["pred_qty_int"].sum()),
            "pred_nonzero_count": int((rows["pred_qty_int"] > 0).sum()),
            **{f"pred_mc{level}_count": int((rows["pred_mc"] == f"MC{level}").sum()) for level in range(5)},
            "store_count": int(rows["site_no"].nunique()), "item_count": int(rows["item_id"].nunique()), "created_at": run["created_at"],
        })
    return pd.DataFrame(result)


def export_prediction_excel(
    run: dict,
    *,
    model_id: str | None = None,
    site_no: str | None = None,
    mc: str | None = None,
    include_predictions: bool = False,
    top_n: int = 100,
) -> Path:
    artifact_summary = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    run = {**run, "observation_month": artifact_summary["observation_month"]}
    predictions = read_result_rows(run["prediction_dir"], kind="predictions", model_id=model_id, site_no=site_no, mc=mc)
    if predictions.empty:
        raise ValueError("No predictions match the requested export filters")
    store_summary = predictions.groupby(["site_no", "model_id"], as_index=False).agg(
        pred_total=("pred_qty_int", "sum"),
        pred_nonzero_count=("pred_qty_int", lambda values: int((values > 0).sum())),
        pred_20_plus_count=("pred_qty_int", lambda values: int((values >= 20).sum())),
    ).sort_values(["model_id", "pred_total"], ascending=[True, False])
    top_columns = [column for column in ("site_no", "item_id", "book_name", "category", "model_id", "pred_qty_int", "pred_mc", "p_sale", "conditional_qty") if column in predictions.columns]
    top_books = predictions.loc[:, top_columns].head(top_n)
    include_detail = include_predictions or site_no is not None
    if include_detail and len(predictions) > MAX_EXCEL_DETAIL_ROWS:
        raise ValueError("Prediction detail exceeds Excel row limit; export by site_no or download Parquet instead")
    export_dir = runtime_root() / "exports" / run["prediction_run_id"]
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"prediction_{run['prediction_run_id']}_{_safe_component(model_id, 'all')}_{_safe_component(site_no, 'all')}.xlsx"
    output_path = export_dir / filename
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _summary_rows(run, predictions).to_excel(writer, sheet_name="Summary", index=False)
        store_summary.to_excel(writer, sheet_name="Store_Summary", index=False)
        top_books.to_excel(writer, sheet_name="Top_Books", index=False)
        if include_detail:
            predictions.to_excel(writer, sheet_name="Predictions", index=False)
        MODEL_INFO.to_excel(writer, sheet_name="Model_Info", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
    return output_path
