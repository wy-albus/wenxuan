from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402


KEY_COLUMNS = ("month", "site_no", "item_id")
TARGET_COLUMNS = ("target_qty_1m", "target_qty_2m")
HORIZONS = ("1m", "2m")
SEGMENTS = ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
TOPK_VALUES = (20, 50, 100)
SALES_LEVELS = ("no_sales", "low", "normal", "medium_high", "high")
SALES_LEVEL_LABELS = {
    "no_sales": "无动销",
    "low": "低动销",
    "normal": "一般动销",
    "medium_high": "较高动销",
    "high": "高动销",
}


@dataclass(frozen=True)
class ModelSpec:
    file_key: str
    pred_1m: str
    pred_2m: str
    display_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "historical_mean": ModelSpec("baseline", "historical_mean_pred_1m", "historical_mean_pred_2m", "历史均值"),
    "weighted_moving_average": ModelSpec("baseline", "weighted_ma_pred_1m", "weighted_ma_pred_2m", "加权移动平均"),
    "random_forest": ModelSpec("random_forest", "random_forest_pred_1m", "random_forest_pred_2m", "随机森林"),
    "lightgbm_v2_logl2": ModelSpec(
        "lightgbm_v2_logl2",
        "lightgbm_v2_logl2_pred_1m",
        "lightgbm_v2_logl2_pred_2m",
        "LightGBM V2 Log-L2",
    ),
    "mlp": ModelSpec("mlp", "mlp_pred_1m", "mlp_pred_2m", "MLP"),
    "two_stage": ModelSpec("two_stage", "two_stage_pred_1m", "two_stage_pred_2m", "两阶段模型"),
}


DEFAULT_PREDICTION_PATHS = {
    "baseline": ROOT / "data" / "outputs" / "baseline_predictions.parquet",
    "lightgbm_v2_logl2": ROOT / "data" / "outputs" / "lightgbm_v2_logl2_test_predictions.parquet",
    "random_forest": ROOT / "data" / "outputs" / "random_forest_test_predictions.parquet",
    "mlp": ROOT / "data" / "outputs" / "mlp_test_predictions.parquet",
    "two_stage": ROOT / "data" / "outputs" / "two_stage_test_predictions.parquet",
}

DEFAULT_REPORT_PATHS = [
    ROOT / "reports" / "baseline_model_report.md",
    ROOT / "reports" / "lightgbm_v2_logl2_model_report.md",
    ROOT / "reports" / "random_forest_model_report.md",
    ROOT / "reports" / "mlp_model_report.md",
    ROOT / "reports" / "two_stage_model_report.md",
    ROOT / "reports" / "storage_manifest.md",
]


def _project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _finite_non_negative(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and (values >= 0).all())


def round_prediction_for_level(values) -> np.ndarray:
    values = clip_target(values)
    return np.floor(values + 0.5).astype("int64")


def sales_level(values, *, already_rounded: bool = False) -> np.ndarray:
    qty = np.asarray(values, dtype="float64")
    if not already_rounded:
        qty = round_prediction_for_level(qty)
    return np.select(
        [qty == 0, qty == 1, (qty >= 2) & (qty < 5), (qty >= 5) & (qty < 20)],
        SALES_LEVELS[:4],
        default=SALES_LEVELS[4],
    )


class LevelAccumulator:
    def __init__(self) -> None:
        self.matrix = np.zeros((len(SALES_LEVELS), len(SALES_LEVELS)), dtype=np.int64)

    def update(self, true_qty, pred_qty, *, average_2m: bool = False) -> None:
        true_values = clip_target(true_qty)
        pred_values = clip_target(pred_qty)
        if average_2m:
            true_values = true_values / 2.0
            pred_values = pred_values / 2.0
        true_levels = sales_level(true_values)
        pred_levels = sales_level(pred_values)
        true_codes = pd.Categorical(true_levels, categories=SALES_LEVELS).codes
        pred_codes = pd.Categorical(pred_levels, categories=SALES_LEVELS).codes
        valid = (true_codes >= 0) & (pred_codes >= 0)
        if valid.any():
            flat = true_codes[valid] * len(SALES_LEVELS) + pred_codes[valid]
            counts = np.bincount(flat, minlength=len(SALES_LEVELS) ** 2)
            self.matrix += counts.reshape((len(SALES_LEVELS), len(SALES_LEVELS)))

    def compute(self) -> dict:
        return compute_level_metrics_from_matrix(self.matrix)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def compute_level_metrics_from_matrix(matrix: np.ndarray) -> dict:
    total = int(matrix.sum())
    true_support = matrix.sum(axis=1)
    pred_support = matrix.sum(axis=0)
    tp = np.diag(matrix).astype("float64")
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support != 0)
    recall = np.divide(tp, true_support, out=np.zeros_like(tp), where=true_support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    active_true = int(true_support[1:].sum())
    medium_high_or_high_true = int(true_support[3:].sum())
    high_true = int(true_support[4])
    no_or_low_true = int(true_support[:2].sum())
    return {
        "total": total,
        "accuracy": _ratio(float(tp.sum()), total),
        "macro_f1": float(f1.mean()) if total else np.nan,
        "weighted_f1": _ratio(float((f1 * true_support).sum()), total),
        "active_recall": _ratio(float(matrix[1:, 1:].sum()), active_true),
        "medium_high_or_high_recall": _ratio(float(matrix[3:, 3:].sum()), medium_high_or_high_true),
        "high_recall": _ratio(float(matrix[4, 4]), high_true),
        "severe_underestimate_rate": _ratio(float(matrix[3:, :2].sum()), medium_high_or_high_true),
        "severe_overestimate_rate": _ratio(float(matrix[:2, 3:].sum()), no_or_low_true),
        "per_level": {
            level: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(true_support[idx]),
                "predicted": int(pred_support[idx]),
            }
            for idx, level in enumerate(SALES_LEVELS)
        },
        "confusion_matrix": matrix.copy(),
    }


def compute_level_metrics(true_qty, pred_qty) -> dict:
    accumulator = LevelAccumulator()
    accumulator.update(true_qty, pred_qty)
    return accumulator.compute()


@dataclass
class TopKAccumulator:
    max_k: int = 100
    heaps: dict[tuple[str, str, str, str], list] = field(default_factory=dict)
    group_target_sum: dict[tuple[str, str, str], float] = field(default_factory=lambda: defaultdict(float))
    group_high_count: dict[tuple[str, str, str], int] = field(default_factory=lambda: defaultdict(int))
    counter: int = 0

    def update_group_totals(self, frame: pd.DataFrame, horizon: str, target_col: str) -> None:
        grouped = frame.groupby(["month", "site_no"], sort=False)[target_col]
        sums = grouped.sum()
        highs = frame.assign(_is_high=frame[target_col].to_numpy(dtype="float64") >= 20).groupby(
            ["month", "site_no"], sort=False
        )["_is_high"].sum()
        for (month, site), value in sums.items():
            self.group_target_sum[(horizon, str(month), str(site))] += float(value)
        for (month, site), value in highs.items():
            self.group_high_count[(horizon, str(month), str(site))] += int(value)

    def update_predictions(self, frame: pd.DataFrame, model: str, horizon: str, target_col: str, pred_col: str) -> None:
        work = frame[["month", "site_no", "item_id", target_col, pred_col]].copy()
        for (month, site), group in work.groupby(["month", "site_no"], sort=False):
            if len(group) > self.max_k:
                group = group.nlargest(self.max_k, pred_col, keep="first")
            heap_key = (model, horizon, str(month), str(site))
            heap = self.heaps.setdefault(heap_key, [])
            for row in group.itertuples(index=False):
                pred = float(getattr(row, pred_col))
                target = float(getattr(row, target_col))
                high = int(target >= 20)
                item = (pred, self.counter, target, high)
                self.counter += 1
                if len(heap) < self.max_k:
                    heapq.heappush(heap, item)
                elif pred > heap[0][0]:
                    heapq.heapreplace(heap, item)

    def compute(self) -> list[dict]:
        rows = []
        keys = {(model, horizon) for model, horizon, _, _ in self.heaps}
        for model, horizon in sorted(keys):
            heaps = {
                (month, site): heap
                for (m, h, month, site), heap in self.heaps.items()
                if m == model and h == horizon
            }
            total_sales = sum(value for (h, _, _), value in self.group_target_sum.items() if h == horizon)
            total_high = sum(value for (h, _, _), value in self.group_high_count.items() if h == horizon)
            for k in TOPK_VALUES:
                selected = 0
                positive = 0
                captured_sales = 0.0
                captured_high = 0
                for heap in heaps.values():
                    top = heapq.nlargest(k, heap, key=lambda item: (item[0], item[1]))
                    selected += len(top)
                    positive += sum(1 for _, _, target, _ in top if target > 0)
                    captured_sales += sum(target for _, _, target, _ in top)
                    captured_high += sum(high for _, _, _, high in top)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "k": k,
                        "selected_count": selected,
                        "precision_at_k": _ratio(positive, selected),
                        "sales_capture_at_k": _ratio(captured_sales, total_sales),
                        "high_sales_coverage_at_k": _ratio(captured_high, total_high),
                        "captured_sales": captured_sales,
                        "total_sales": total_sales,
                        "captured_high_count": captured_high,
                        "total_high_count": total_high,
                    }
                )
        return rows


def _read_row_group(parquet: pq.ParquetFile, row_group: int, columns: Iterable[str]) -> pd.DataFrame:
    return parquet.read_row_group(row_group, columns=list(columns), use_threads=True).to_pandas()


def _hash_keys(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[list(KEY_COLUMNS)].astype("string"), index=False).to_numpy(dtype="uint64")


def _frame_hash_score(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame[list(KEY_COLUMNS)].astype("string"), index=False).astype("uint64")


def _validate_prediction_file(path: Path, required_columns: set[str]) -> pq.ParquetFile:
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    missing = required_columns - set(parquet.schema.names)
    if missing:
        parquet.close()
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return parquet


def _model_columns_by_file() -> dict[str, set[str]]:
    columns = {key: set(KEY_COLUMNS) | set(TARGET_COLUMNS) for key in DEFAULT_PREDICTION_PATHS}
    columns["baseline"].add("split")
    for spec in MODEL_SPECS.values():
        columns[spec.file_key].update([spec.pred_1m, spec.pred_2m])
    return columns


def _make_wide_frame(base: pd.DataFrame, file_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = base[list(KEY_COLUMNS) + list(TARGET_COLUMNS)].copy()
    for model, spec in MODEL_SPECS.items():
        source = file_frames[spec.file_key]
        output[f"{model}_pred_1m"] = source[spec.pred_1m].to_numpy(dtype="float64")
        output[f"{model}_pred_2m"] = source[spec.pred_2m].to_numpy(dtype="float64")
    return output


def _append_sample_candidates(candidates: list[pd.DataFrame], wide: pd.DataFrame) -> None:
    sample = wide.copy()
    sample["_sample_hash"] = _frame_hash_score(sample)
    sample["true_level_1m"] = sales_level(sample["target_qty_1m"].to_numpy())
    sample["true_level_2m_cumulative"] = sales_level(sample["target_qty_2m"].to_numpy())
    sample["true_level_2m_monthly_avg"] = sales_level(sample["target_qty_2m"].to_numpy(dtype="float64") / 2.0)
    error_cols = []
    for model in MODEL_SPECS:
        for horizon in HORIZONS:
            pred_col = f"{model}_pred_{horizon}"
            if horizon == "1m":
                sample[f"{model}_level_1m"] = sales_level(sample[pred_col].to_numpy())
            else:
                sample[f"{model}_level_2m_cumulative"] = sales_level(sample[pred_col].to_numpy())
                sample[f"{model}_level_2m_monthly_avg"] = sales_level(sample[pred_col].to_numpy(dtype="float64") / 2.0)
            err_col = f"_{model}_{horizon}_abs_error"
            sample[err_col] = np.abs(sample[pred_col].to_numpy(dtype="float64") - sample[f"target_qty_{horizon}"].to_numpy(dtype="float64"))
            error_cols.append(err_col)
    sample["_max_abs_error"] = sample[error_cols].max(axis=1)

    parts = [sample.nsmallest(min(20, len(sample)), "_sample_hash")]
    for column in ("true_level_1m", "true_level_2m_cumulative"):
        for level in SALES_LEVELS:
            stratum = sample[sample[column] == level]
            if not stratum.empty:
                parts.append(stratum.nsmallest(min(10, len(stratum)), "_sample_hash"))
    parts.append(sample.nlargest(min(30, len(sample)), "_max_abs_error"))
    candidates.append(pd.concat(parts, ignore_index=True))


def _finalize_sample(candidates: list[pd.DataFrame], sample_path: Path, sample_size: int) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        pd.DataFrame().to_csv(sample_path, index=False, encoding="utf-8-sig")
        return
    sample = pd.concat(candidates, ignore_index=True)
    sample = sample.drop_duplicates(list(KEY_COLUMNS), keep="first")
    sample = sample.sort_values("_sample_hash").head(sample_size)
    keep_columns = list(KEY_COLUMNS) + list(TARGET_COLUMNS)
    keep_columns += [f"{model}_pred_1m" for model in MODEL_SPECS]
    keep_columns += [f"{model}_pred_2m" for model in MODEL_SPECS]
    keep_columns += [
        "true_level_1m",
        "true_level_2m_cumulative",
        "true_level_2m_monthly_avg",
    ]
    keep_columns += [f"{model}_level_1m" for model in MODEL_SPECS]
    keep_columns += [f"{model}_level_2m_cumulative" for model in MODEL_SPECS]
    keep_columns += [f"{model}_level_2m_monthly_avg" for model in MODEL_SPECS]
    sample[keep_columns].to_csv(sample_path, index=False, encoding="utf-8-sig")


def _format_float(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def _metrics_to_rows(metrics: dict[str, dict[str, LongTailStreamingMetrics]]) -> list[dict]:
    rows = []
    for model in MODEL_SPECS:
        for horizon in HORIZONS:
            computed = metrics[model][horizon].compute()
            for segment in SEGMENTS:
                values = computed[segment]
                row = {
                    "model": model,
                    "model_name": MODEL_SPECS[model].display_name,
                    "horizon": horizon,
                    "segment": segment,
                    "count": values["count"],
                    "mae": values["mae"],
                    "rmse": values["rmse"],
                    "smape": values["smape"],
                    "wape": values["wape"],
                    "target_sum": values["target_sum"],
                    "prediction_sum": values["prediction_sum"],
                    "total_bias_rate": values["total_bias_rate"],
                    "zero_mean_prediction": values.get("mean_prediction", np.nan),
                    "zero_prediction_gt_0_5_rate": values.get("prediction_gt_0_5_rate", np.nan),
                    "zero_prediction_gt_1_rate": values.get("prediction_gt_1_rate", np.nan),
                }
                rows.append(row)
    return rows


def _level_rows(level_metrics: dict[str, dict[str, LevelAccumulator]]) -> list[dict]:
    rows = []
    for model in MODEL_SPECS:
        for context, accumulator in level_metrics[model].items():
            computed = accumulator.compute()
            for metric in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "active_recall",
                "medium_high_or_high_recall",
                "high_recall",
                "severe_underestimate_rate",
                "severe_overestimate_rate",
            ):
                rows.append(
                    {
                        "section": "aggregate",
                        "model": model,
                        "horizon_context": context,
                        "metric": metric,
                        "value": computed[metric],
                    }
                )
            for level, values in computed["per_level"].items():
                rows.append(
                    {
                        "section": "per_level",
                        "model": model,
                        "horizon_context": context,
                        "level": level,
                        "level_name": SALES_LEVEL_LABELS[level],
                        **values,
                    }
                )
            matrix = computed["confusion_matrix"]
            for true_idx, true_level in enumerate(SALES_LEVELS):
                for pred_idx, pred_level in enumerate(SALES_LEVELS):
                    rows.append(
                        {
                            "section": "confusion_matrix",
                            "model": model,
                            "horizon_context": context,
                            "true_level": true_level,
                            "true_level_name": SALES_LEVEL_LABELS[true_level],
                            "pred_level": pred_level,
                            "pred_level_name": SALES_LEVEL_LABELS[pred_level],
                            "count": int(matrix[true_idx, pred_idx]),
                        }
                    )
    return rows


def _read_text_status(paths: list[Path]) -> list[dict]:
    statuses = []
    for path in paths:
        statuses.append(
            {
                "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "exists": path.exists(),
                "size_mib": path.stat().st_size / 1024**2 if path.exists() else np.nan,
            }
        )
    return statuses


def _load_feature_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "historical_mean": len(config.get("historical_mean_features", [])),
        "weighted_moving_average": len(config.get("weighted_moving_average_features", [])),
        "random_forest": len(config.get("random_forest_features", [])),
        "lightgbm_v2_logl2": len(config.get("lightgbm_features", [])),
        "mlp": len(config.get("mlp_features", [])),
        "two_stage": len(config.get("two_stage_features", [])),
    }


def _file_size_mib(path: str | Path) -> float:
    path = _project_path(path)
    return path.stat().st_size / 1024**2 if path and path.exists() else np.nan


def _model_size_summary() -> dict[str, float]:
    model_files = {
        "historical_mean": [],
        "weighted_moving_average": [],
        "random_forest": ["models/final/random_forest_1m.joblib", "models/final/random_forest_2m.joblib"],
        "lightgbm_v2_logl2": ["models/final/lightgbm_v2_logl2_1m.txt", "models/final/lightgbm_v2_logl2_2m.txt"],
        "mlp": ["models/final/mlp_1m.pt", "models/final/mlp_2m.pt"],
        "two_stage": [
            "models/final/two_stage_classifier_1m.txt",
            "models/final/two_stage_regressor_1m.txt",
            "models/final/two_stage_classifier_2m.txt",
            "models/final/two_stage_regressor_2m.txt",
        ],
    }
    return {model: sum(_file_size_mib(path) for path in paths) if paths else 0.0 for model, paths in model_files.items()}


def _best_by(comparison: pd.DataFrame, segment: str, metric: str, *, lower: bool = True) -> str:
    part = comparison[(comparison["segment"] == segment) & (comparison["horizon"].isin(HORIZONS))]
    grouped = part.groupby("model", sort=False)[metric].mean(numeric_only=True)
    return str(grouped.idxmin() if lower else grouped.idxmax())


def _safe_best_level(level_df: pd.DataFrame) -> str:
    part = level_df[(level_df["section"] == "aggregate") & (level_df["metric"] == "macro_f1")]
    grouped = part.groupby("model", sort=False)["value"].mean(numeric_only=True)
    return str(grouped.idxmax())


def _safe_best_topk(topk_df: pd.DataFrame) -> str:
    part = topk_df[topk_df["k"] == 100]
    grouped = part.groupby("model", sort=False)["sales_capture_at_k"].mean(numeric_only=True)
    return str(grouped.idxmax())


def _render_summary(
    output_path: Path,
    comparison: pd.DataFrame,
    level_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    summary: dict,
) -> None:
    feature_counts = summary["feature_counts"]
    model_sizes = summary["model_sizes_mib"]
    overall_best = _best_by(comparison, "overall", "mae", lower=True)
    zero_best = _best_by(comparison, "0", "zero_mean_prediction", lower=True)
    nonzero_best = _best_by(comparison, "nonzero", "wape", lower=True)
    head_best = _best_by(comparison, "ge_20", "wape", lower=True)
    level_best = _safe_best_level(level_df)
    topk_best = _safe_best_topk(topk_df)
    scale_best = _best_by(comparison.assign(abs_bias=comparison["total_bias_rate"].abs()), "overall", "abs_bias", lower=True)

    def metric_line(model: str, horizon: str) -> str:
        row = comparison[
            (comparison["model"] == model) & (comparison["horizon"] == horizon) & (comparison["segment"] == "overall")
        ].iloc[0]
        return (
            f"| {MODEL_SPECS[model].display_name} | {horizon} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['wape']:.2f}% | {row['total_bias_rate']:.2f}% |"
        )

    lines = [
        "# 集中式预测阶段总结",
        "",
        "## 项目目标",
        "",
        "本阶段只读取已有集中式模型的 test 预测结果和报告，完成六类算法统一比较、暂定动销等级评价、Top-K 重点图书识别验证与阶段性业务总结。未重新训练、微调或覆盖任何已有模型。",
        "",
        "## 数据与预测口径",
        "",
        f"- 一致性检查：{'通过' if summary['consistency']['passed'] else '未通过'}。",
        f"- test 样本量：{summary['rows']:,}。",
        "- 比较键：`month + site_no + item_id`，并检查目标列一致、重复键、缺失、NaN、Inf 和负预测。",
        "- LightGBM 主比较版本为 V2 Log-L2 原始预测；V1 Log-L1、Tweedie Pilot 与 V2 valid-only 校准只作为补充实验背景，不进入六模型主排名。",
        "",
        "## 六类算法及训练方式",
        "",
        "| 模型 | 核心字段类别 | 特征数量 | 类别处理 | 目标变换 | 损失/目标 | 训练样本与轮次 | 模型大小(MiB) |",
        "|---|---|---:|---|---|---|---|---:|",
        f"| 历史均值 | 近3月历史销量 | {feature_counts.get('historical_mean', 0)} | 不需要 | 无 | 公式法 | 无训练 | {model_sizes['historical_mean']:.2f} |",
        f"| 加权移动平均 | 近3月 lag 销量 | {feature_counts.get('weighted_moving_average', 0)} | 不需要 | 无 | 公式法 | 无训练 | {model_sizes['weighted_moving_average']:.2f} |",
        f"| 随机森林 | 历史/频次/金额/门店/类目/时间 | {feature_counts.get('random_forest', 0)} | train-only 编码 | log1p | squared_error | 约150万样本，200棵树 | {model_sizes['random_forest']:.2f} |",
        f"| LightGBM V2 | 历史/频次/金额/门店/类目/时间 | {feature_counts.get('lightgbm_v2_logl2', 0)} | train-only 编码 | log1p | regression_l2 | 分层正式样本，early stopping | {model_sizes['lightgbm_v2_logl2']:.2f} |",
        f"| MLP | 数值特征 + 类别 embedding | {feature_counts.get('mlp', 0)} | embedding | log1p | weighted MSE | 约150万样本，early stopping | {model_sizes['mlp']:.2f} |",
        f"| 两阶段 | LightGBM V2 同信息范围 | {feature_counts.get('two_stage', 0)} | train-only 编码 | 分类原始标签；回归原始正销量 | binary + Tweedie | 四个 LightGBM 组件 | {model_sizes['two_stage']:.2f} |",
        "",
        "## 六模型统一结果",
        "",
        "| 模型 | Horizon | MAE | RMSE | WAPE | 总量偏差 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_SPECS:
        for horizon in HORIZONS:
            lines.append(metric_line(model, horizon))

    lines.extend(
        [
            "",
            "完整分层指标见 `reports/centralized_model_comparison.csv`。不能只按 overall MAE 选模型：零销量样本占绝对多数，overall MAE 更偏向零销量控制；非零、5-20、20+ 和总量偏差更接近补货业务关注点。",
            "",
            "## 暂定动销等级依据和边界",
            "",
            "以下为“基于当前长尾分层的暂定动销等级，尚待文轩业务方确认”。",
            "",
            "- 0：无动销",
            "- 1：低动销",
            "- 2 <= qty < 5：一般动销",
            "- 5 <= qty < 20：较高动销",
            "- qty >= 20：高动销",
            "",
            "模型预测先执行 `rounded_pred = floor(prediction + 0.5)` 并截断非负后再划分等级。2M 同时评估累计等级和月均等级；跨 1M/2M 做业务解释时，以 2M 月均等级为主。",
            "",
            "## 动销等级评价",
            "",
            f"- Macro-F1 平均最好的模型：{MODEL_SPECS[level_best].display_name}。",
            "- 完整等级准确率、Macro-F1、Weighted-F1、各等级 Precision/Recall/F1、混淆矩阵、严重低估率和严重高估率见 `reports/centralized_sales_level_metrics.csv`。",
            "- 等级准确率不能作为唯一依据，因为无动销样本占比极高。",
            "",
            "## Top-K重点图书识别效果",
            "",
            f"- 按 `Sales Capture@100` 均值看，Top-K 表现最好的模型：{MODEL_SPECS[topk_best].display_name}。",
            "- Top-K 只验证“门店-月份下需要重点关注的图书”识别能力，未模拟库存成本、采购提前期或补货量。",
            "- 2M 月均与 2M 累计的 Top-K 排序一致，因此 Top-K 表中保留 1M 和 2M 累计预测。",
            "",
            "## 模型业务定位",
            "",
            f"- 集中式整体点预测表现最好的模型：{MODEL_SPECS[overall_best].display_name}。",
            f"- 零销量控制最好的模型：{MODEL_SPECS[zero_best].display_name}。",
            f"- 非零需求识别最好的模型：{MODEL_SPECS[nonzero_best].display_name}。",
            f"- 20+ 头部图书误差最好的模型：{MODEL_SPECS[head_best].display_name}。",
            f"- 动销等级识别最好的模型：{MODEL_SPECS[level_best].display_name}。",
            f"- Top-K 重点图书识别最好的模型：{MODEL_SPECS[topk_best].display_name}。",
            f"- 集团总销量规模最接近的模型：{MODEL_SPECS[scale_best].display_name}。",
            "",
            "## 集中式阶段最终结论",
            "",
            "- 集中式主模型建议：以 LightGBM V2 Log-L2 作为默认点预测主模型。它在整体误差、工程稳定性和可解释性之间最均衡。",
            "- 辅助模型建议：保留两阶段模型作为总量规模和头部/非零需求识别的辅助判断，保留随机森林作为树模型结构对照；均值类 baseline 继续作为最低基准。",
            "- 后续联邦学习基线建议：优先迁移 LightGBM V2 的特征口径和时间切分作为结构基线；两阶段模型可作为后续增强方向，而不是第一版联邦学习的默认复杂结构。",
            "",
            "## 当前可以提供的现实输出",
            "",
            "- 门店-图书-月份未来 1M/2M 的集中式销量预测。",
            "- 基于暂定规则的动销等级识别。",
            "- 每个门店-月份 Top-K 重点关注图书清单，用于人工核查和补货关注池。",
            "",
            "## 当前局限",
            "",
            "- 缺少库存、在途库存、采购提前期、最小订量、缺货损失和库存成本，不能声称已经完成自动补货或最优补货量。",
            "- 暂定动销等级仍需文轩业务方确认阈值。",
            "- 两阶段模型改善总量偏差时会增加零销量误报，需要业务规则或库存约束共同使用。",
            "",
            "## 下一阶段联邦学习计划",
            "",
            "下一阶段建议先做联邦学习可行性拆解：按门店/区域的数据划分方式、特征一致性、非 IID 程度、通信与隐私约束、以及与集中式 LightGBM V2 的可比评价口径。当前阶段未启动联邦学习或任何新模型训练。",
            "",
            "## 工程审计",
            "",
            f"- 运行耗时：{summary['runtime_seconds']:.1f} 秒。",
            f"- PyArrow 内存池峰值：{summary['arrow_peak_memory_bytes'] / 1024**3:.2f} GiB。",
            f"- 输出样本文件：`data/outputs/centralized_prediction_sample.csv`，固定种子/哈希抽样，不超过 50,000 条。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    summary["winners"] = {
        "overall_point": overall_best,
        "zero_control": zero_best,
        "nonzero": nonzero_best,
        "head_ge20": head_best,
        "level": level_best,
        "topk": topk_best,
        "scale": scale_best,
    }


def _update_storage_manifest(manifest_path: Path, output_paths: dict[str, Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = manifest_path.read_text(encoding="utf-8", errors="ignore") if manifest_path.exists() else "# Storage Manifest\n"
    marker = "\n## 集中式预测阶段新增文件\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    rows = [
        marker,
        "",
        "| 文件 | 用途 | 大小(MiB) | 可重新生成 | 可安全删除 | 生成脚本 |",
        "|---|---|---:|---|---|---|",
    ]
    purposes = {
        "summary": "集中式阶段总结报告",
        "comparison": "六模型统一点预测指标",
        "level": "暂定动销等级评价与混淆矩阵",
        "topk": "门店-月份 Top-K 重点图书识别指标",
        "sample": "人工检查样本",
    }
    for key, path in output_paths.items():
        if key == "manifest":
            continue
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        size = path.stat().st_size / 1024**2 if path.exists() else 0.0
        rows.append(f"| `{relative}` | {purposes.get(key, key)} | {size:.2f} | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |")
    manifest_path.write_text(existing.rstrip() + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def run_centralized_finalization(
    prediction_paths: dict[str, str | Path] | None = None,
    output_paths: dict[str, str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    feature_config_path: str | Path | None = ROOT / "config" / "model_features.yaml",
    storage_manifest_path: str | Path | None = ROOT / "reports" / "storage_manifest.md",
    sample_size: int = 50_000,
    max_row_groups: int | None = None,
) -> dict:
    started = time.perf_counter()
    prediction_paths = {**DEFAULT_PREDICTION_PATHS, **(prediction_paths or {})}
    prediction_paths = {key: _project_path(path) for key, path in prediction_paths.items()}
    output_paths = {
        "summary": ROOT / "reports" / "centralized_stage_summary.md",
        "comparison": ROOT / "reports" / "centralized_model_comparison.csv",
        "level": ROOT / "reports" / "centralized_sales_level_metrics.csv",
        "topk": ROOT / "reports" / "centralized_topk_metrics.csv",
        "sample": ROOT / "data" / "outputs" / "centralized_prediction_sample.csv",
        **(output_paths or {}),
    }
    output_paths = {key: _project_path(path) for key, path in output_paths.items()}
    storage_manifest_path = _project_path(storage_manifest_path)
    report_paths = [_project_path(path) for path in (report_paths if report_paths is not None else DEFAULT_REPORT_PATHS)]
    feature_config_path = _project_path(feature_config_path)

    required_by_file = _model_columns_by_file()
    parquets = {
        key: _validate_prediction_file(path, required_by_file[key])
        for key, path in prediction_paths.items()
    }
    try:
        row_counts = {key: parquet.metadata.num_rows for key, parquet in parquets.items()}
        row_groups = {key: parquet.num_row_groups for key, parquet in parquets.items()}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Prediction files have different row counts: {row_counts}")
        if len(set(row_groups.values())) != 1:
            raise ValueError(f"Prediction files have different row group counts: {row_groups}")
        row_group_count = next(iter(row_groups.values()))
        if max_row_groups is not None:
            row_group_count = min(row_group_count, max_row_groups)

        metrics = {model: {horizon: LongTailStreamingMetrics() for horizon in HORIZONS} for model in MODEL_SPECS}
        level_metrics = {
            model: {
                "1m": LevelAccumulator(),
                "2m_cumulative": LevelAccumulator(),
                "2m_monthly_avg": LevelAccumulator(),
            }
            for model in MODEL_SPECS
        }
        topk = TopKAccumulator(max_k=max(TOPK_VALUES))
        sample_candidates: list[pd.DataFrame] = []
        key_hash_parts: list[np.ndarray] = []
        rows_processed = 0

        for row_group in range(row_group_count):
            file_frames = {
                key: _read_row_group(parquet, row_group, required_by_file[key])
                for key, parquet in parquets.items()
            }
            base = file_frames["baseline"]
            if "split" in base and not (base["split"].astype("string") == "test").all():
                raise ValueError("baseline predictions contain non-test rows")

            for key, frame in file_frames.items():
                if len(frame) != len(base):
                    raise ValueError(f"Row group {row_group} has mismatched row counts in {key}")
                if not frame[list(KEY_COLUMNS)].astype("string").reset_index(drop=True).equals(
                    base[list(KEY_COLUMNS)].astype("string").reset_index(drop=True)
                ):
                    raise ValueError(f"Row group {row_group} key alignment failed for {key}")
                for target_col in TARGET_COLUMNS:
                    if not np.allclose(
                        clip_target(frame[target_col].to_numpy()),
                        clip_target(base[target_col].to_numpy()),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    ):
                        raise ValueError(f"Row group {row_group} target mismatch in {key}:{target_col}")

            wide = _make_wide_frame(base, file_frames)
            if wide[list(KEY_COLUMNS)].isna().any().any() or wide[list(TARGET_COLUMNS)].isna().any().any():
                raise ValueError(f"Row group {row_group} contains missing keys or targets")
            key_hash_parts.append(_hash_keys(wide))

            for model in MODEL_SPECS:
                for horizon in HORIZONS:
                    target_col = f"target_qty_{horizon}"
                    pred_col = f"{model}_pred_{horizon}"
                    prediction = clip_target(wide[pred_col].to_numpy())
                    if not _finite_non_negative(prediction):
                        raise ValueError(f"{model} {horizon} has NaN/Inf/negative predictions")
                    wide[pred_col] = prediction
                    target = clip_target(wide[target_col].to_numpy())
                    metrics[model][horizon].update(target, prediction)
                    if horizon == "1m":
                        level_metrics[model]["1m"].update(target, prediction)
                    else:
                        level_metrics[model]["2m_cumulative"].update(target, prediction)
                        level_metrics[model]["2m_monthly_avg"].update(target, prediction, average_2m=True)

            for horizon in HORIZONS:
                target_col = f"target_qty_{horizon}"
                topk.update_group_totals(wide, horizon, target_col)
                for model in MODEL_SPECS:
                    topk.update_predictions(wide, model, horizon, target_col, f"{model}_pred_{horizon}")

            _append_sample_candidates(sample_candidates, wide)
            rows_processed += len(wide)
            if (row_group + 1) % 20 == 0 or row_group + 1 == row_group_count:
                print(f"Processed row groups {row_group + 1}/{row_group_count}; rows={rows_processed:,}", flush=True)

        hashes = np.concatenate(key_hash_parts)
        hashes.sort()
        duplicate_count = int((hashes[1:] == hashes[:-1]).sum()) if len(hashes) > 1 else 0
        if duplicate_count:
            raise ValueError(f"Duplicate month/site/item keys detected: {duplicate_count}")

        comparison = pd.DataFrame(_metrics_to_rows(metrics))
        level_df = pd.DataFrame(_level_rows(level_metrics))
        topk_df = pd.DataFrame(topk.compute())

        for key in ("comparison", "level", "topk"):
            output_paths[key].parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_paths["comparison"], index=False, encoding="utf-8-sig")
        level_df.to_csv(output_paths["level"], index=False, encoding="utf-8-sig")
        topk_df.to_csv(output_paths["topk"], index=False, encoding="utf-8-sig")
        _finalize_sample(sample_candidates, output_paths["sample"], sample_size)

        summary = {
            "rows": rows_processed,
            "row_groups": row_group_count,
            "consistency": {
                "passed": True,
                "row_counts": row_counts,
                "row_groups": row_groups,
                "duplicate_count": duplicate_count,
            },
            "report_status": _read_text_status(report_paths),
            "feature_counts": _load_feature_summary(feature_config_path),
            "model_sizes_mib": _model_size_summary(),
            "runtime_seconds": time.perf_counter() - started,
            "arrow_peak_memory_bytes": int(pa.default_memory_pool().max_memory()),
        }
        _render_summary(output_paths["summary"], comparison, level_df, topk_df, summary)
        if storage_manifest_path is not None:
            manifest_outputs = {**output_paths, "manifest": storage_manifest_path}
            _update_storage_manifest(storage_manifest_path, manifest_outputs)
        return summary
    finally:
        for parquet in parquets.values():
            parquet.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize centralized prediction-stage comparison.")
    parser.add_argument("--max-row-groups", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_centralized_finalization(max_row_groups=args.max_row_groups, sample_size=args.sample_size)
    winners = summary.get("winners", {})
    print(
        "Centralized finalization complete: "
        f"rows={summary['rows']:,}, runtime={summary['runtime_seconds']:.1f}s, "
        f"overall={winners.get('overall_point')}, level={winners.get('level')}, topk={winners.get('topk')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
