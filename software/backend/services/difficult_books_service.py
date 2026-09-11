from __future__ import annotations

from pathlib import Path

import pandas as pd

from .result_query_service import page_rows
from .runtime import PROJECT_ROOT, runtime_root


DIFFICULTY_RULES = {
    "metric": "stable_error",
    "stable_error_denominator": "max(abs(actual_qty), 1)",
    "levels": {
        "困难": {"min_abs_error": 10.0, "min_stable_error": 1.0},
        "较难": {"min_abs_error": 5.0, "min_stable_error": 0.5},
    },
    "note": "当前为系统工程默认筛选规则，最终研究阈值待项目组确认。",
}


DIFFICULT_BOOK_FIELDS = [
    {"field": "site_no", "label": "门店编号", "description": "预测对象所属门店的原始编号。"},
    {"field": "item_id", "label": "图书编号", "description": "预测对象所属图书的原始编号。"},
    {"field": "book_name", "label": "图书名称", "description": "如源数据包含名称字段则返回，否则为空。"},
    {"field": "observation_month", "label": "预测基准月份", "description": "用于发起预测的历史观察月份。"},
    {"field": "target_month", "label": "目标月份", "description": "当前工程按 1M 任务口径展示为基准月份的下一月。"},
    {"field": "actual_qty", "label": "历史真实销量", "description": "来自历史验证样本中的 future_qty_1m。"},
    {"field": "pred_qty", "label": "预测销量", "description": "模型输出经现有 1M 后处理口径得到的整数预测销量。"},
    {"field": "abs_error", "label": "绝对误差", "description": "abs(pred_qty - actual_qty)。"},
    {"field": "stable_error", "label": "稳定比例误差", "description": "abs_error / max(abs(actual_qty), 1)，避免 actual=0 除零。"},
    {"field": "pred_mc", "label": "动销等级", "description": "沿用项目既有 MC 等级定义。"},
    {"field": "model_id", "label": "来源模型", "description": "产生该预测结果的模型编号。"},
    {"field": "difficulty_level", "label": "困难等级", "description": "依据 difficulty_rules 配置得到的工程标记。"},
]


def _next_month(month: str | None) -> str | None:
    if not month:
        return None
    try:
        return str(pd.Period(month, freq="M") + 1)
    except ValueError:
        return None


def _assign_level(abs_error: float, stable_error: float) -> str | None:
    hard = DIFFICULTY_RULES["levels"]["困难"]
    if abs_error >= hard["min_abs_error"] or stable_error >= hard["min_stable_error"]:
        return "困难"
    medium = DIFFICULTY_RULES["levels"]["较难"]
    if abs_error >= medium["min_abs_error"] or stable_error >= medium["min_stable_error"]:
        return "较难"
    return None


def build_difficult_books(
    prediction_dir: str | Path,
    *,
    observation_month: str | None,
    model_id: str | None = None,
    site_no: str | None = None,
    difficulty_level: str | None = None,
) -> pd.DataFrame:
    path = Path(prediction_dir) / "predictions.parquet"
    rows = pd.read_parquet(path)
    if "actual_qty" not in rows.columns:
        return pd.DataFrame(columns=[field["field"] for field in DIFFICULT_BOOK_FIELDS])
    rows = rows.loc[rows["actual_qty"].notna()].copy()
    if model_id:
        rows = rows.loc[rows["model_id"].astype(str).eq(str(model_id))]
    if site_no:
        rows = rows.loc[rows["site_no"].astype(str).eq(str(site_no))]
    if rows.empty:
        return pd.DataFrame(columns=[field["field"] for field in DIFFICULT_BOOK_FIELDS])

    rows["actual_qty"] = pd.to_numeric(rows["actual_qty"], errors="coerce")
    rows["pred_qty"] = pd.to_numeric(rows["pred_qty_int"], errors="coerce").fillna(0.0)
    rows = rows.loc[rows["actual_qty"].notna()].copy()
    rows["abs_error"] = (rows["pred_qty"] - rows["actual_qty"]).abs()
    denominator = rows["actual_qty"].abs().clip(lower=1.0)
    rows["stable_error"] = rows["abs_error"] / denominator
    rows["difficulty_level"] = [
        _assign_level(float(abs_error), float(stable_error))
        for abs_error, stable_error in zip(rows["abs_error"], rows["stable_error"], strict=False)
    ]
    rows = rows.loc[rows["difficulty_level"].notna()].copy()
    if difficulty_level:
        rows = rows.loc[rows["difficulty_level"].eq(difficulty_level)]
    if rows.empty:
        return pd.DataFrame(columns=[field["field"] for field in DIFFICULT_BOOK_FIELDS])

    rows["observation_month"] = observation_month
    rows["target_month"] = _next_month(observation_month)
    result_columns = [
        "site_no", "item_id", "book_name", "observation_month", "target_month",
        "actual_qty", "pred_qty", "abs_error", "stable_error", "pred_mc", "model_id", "difficulty_level",
    ]
    for column in result_columns:
        if column not in rows.columns:
            rows[column] = None
    return rows[result_columns].sort_values(["abs_error", "stable_error"], ascending=[False, False], kind="stable").reset_index(drop=True)


def difficult_books_summary(rows: pd.DataFrame, *, total_prediction_rows: int) -> dict:
    if rows.empty:
        return {
            "difficult_book_count": 0,
            "difficult_ratio": 0.0,
            "store_count": 0,
            "avg_abs_error": None,
            "hard_count": 0,
            "difficulty_rules": DIFFICULTY_RULES,
            "field_schema": DIFFICULT_BOOK_FIELDS,
        }
    return {
        "difficult_book_count": int(len(rows)),
        "difficult_ratio": float(len(rows) / total_prediction_rows) if total_prediction_rows else 0.0,
        "store_count": int(rows["site_no"].nunique()),
        "avg_abs_error": float(rows["abs_error"].mean()),
        "hard_count": int((rows["difficulty_level"] == "困难").sum()),
        "difficulty_rules": DIFFICULTY_RULES,
        "field_schema": DIFFICULT_BOOK_FIELDS,
    }


def page_difficult_books(rows: pd.DataFrame, *, page: int, page_size: int) -> dict:
    return {**page_rows(rows, page=page, page_size=page_size), "field_schema": DIFFICULT_BOOK_FIELDS, "difficulty_rules": DIFFICULTY_RULES}


def export_difficult_books(run_id: str, rows: pd.DataFrame) -> Path:
    export_dir = runtime_root() / "exports" / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"difficult_books_{run_id}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="Difficult_Books", index=False)
        pd.DataFrame(DIFFICULT_BOOK_FIELDS).to_excel(writer, sheet_name="Field_Schema", index=False)
        pd.DataFrame([
            {"key": "metric", "value": DIFFICULTY_RULES["metric"]},
            {"key": "stable_error_denominator", "value": DIFFICULTY_RULES["stable_error_denominator"]},
            {"key": "note", "value": DIFFICULTY_RULES["note"]},
        ]).to_excel(writer, sheet_name="Rules", index=False)
    return path
