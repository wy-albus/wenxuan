from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.business_evaluation import BusinessEvaluationAccumulator  # noqa: E402
from src.evaluation.mc_metrics import load_mc_level_config  # noqa: E402


DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
MC_CONFIG_PATH = ROOT / "config/mc_sales_levels.yaml"
REPORT_PATH = ROOT / "reports/active_store_business_evaluation.md"
METRICS_PATH = ROOT / "reports/active_store_business_evaluation_metrics.json"
GATE_PATH = ROOT / "reports/two_stage_gate_1m_tradeoff.csv"
TEMP_DIR = ROOT / "data/temp/active_store_business_evaluation"

PREDICTIONS = {
    "1m": ROOT / "data/outputs/lightgbm_v2_active_store_wape_1m_test_predictions.parquet",
    "2m": ROOT / "data/outputs/lightgbm_v2_active_store_wape_2m_test_predictions.parquet",
}
PREDICTION_COLUMNS = {
    "1m": "lightgbm_v2_active_store_wape_pred_1m",
    "2m": "lightgbm_v2_active_store_wape_pred_2m",
}
EXPECTED_ROWS = {"1m": 12_999_597, "2m": 12_859_664}


def iter_joined_chunks(horizon: str):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='2GB'")
    connection.execute(f"PRAGMA temp_directory='{TEMP_DIR.resolve().as_posix()}'")
    dataset = DATASET_PATH.resolve().as_posix().replace("'", "''")
    prediction_path = PREDICTIONS[horizon].resolve().as_posix().replace("'", "''")
    prediction_column = PREDICTION_COLUMNS[horizon]
    try:
        connection.execute(
            f"""
            SELECT
                p.target_qty_{horizon} AS prediction_target,
                p.{prediction_column} AS prediction,
                d.future_qty_{horizon} AS dataset_target,
                d.total_qty AS current_qty
            FROM read_parquet('{prediction_path}') p
            INNER JOIN read_parquet('{dataset}') d
              USING (month, site_no, item_id)
            WHERE d.split='test' AND d.target_available_{horizon}=1
            """
        )
        while True:
            frame = connection.fetch_df_chunk(vectors_per_chunk=64)
            if frame.empty:
                break
            yield frame
    finally:
        connection.close()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


def evaluate_horizon(horizon: str, mc_config) -> tuple[dict, float]:
    accumulator = BusinessEvaluationAccumulator(mc_config, horizon)
    rows = 0
    maximum_target_difference = 0.0
    started = time.perf_counter()
    for frame in iter_joined_chunks(horizon):
        prediction_target = np.maximum(
            pd.to_numeric(frame["prediction_target"], errors="coerce").fillna(0).to_numpy("float64"), 0.0
        )
        dataset_target = np.maximum(
            pd.to_numeric(frame["dataset_target"], errors="coerce").fillna(0).to_numpy("float64"), 0.0
        )
        prediction = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy("float64")
        current = pd.to_numeric(frame["current_qty"], errors="coerce").fillna(0).to_numpy("float64")
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise RuntimeError(f"Invalid {horizon} prediction found")
        maximum_target_difference = max(
            maximum_target_difference, float(np.max(np.abs(prediction_target - dataset_target), initial=0.0))
        )
        accumulator.update(dataset_target, prediction, current)
        rows += len(frame)
    if rows != EXPECTED_ROWS[horizon]:
        raise RuntimeError(f"Expected {EXPECTED_ROWS[horizon]:,} {horizon} rows, got {rows:,}")
    if maximum_target_difference > 1e-6:
        raise RuntimeError(f"{horizon} prediction and dataset targets differ by {maximum_target_difference}")
    result = accumulator.compute()
    result["runtime"] = {
        "rows": rows,
        "evaluation_seconds": time.perf_counter() - started,
        "maximum_target_difference": maximum_target_difference,
    }
    return result, maximum_target_difference


def fmt(value, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def matrix_table(matrix, names: tuple[str, ...]) -> list[str]:
    lines = ["| True \\ Pred | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for name, row in zip(names, matrix):
        lines.append(f"| {name} | " + " | ".join(str(int(value)) for value in row) + " |")
    return lines


def horizon_lines(horizon: str, result: dict, mc_config) -> list[str]:
    quantity = result["quantity"]
    mc = result["mc"]
    trend = result["trend"]
    overall = quantity["overall"]
    zero = quantity["0"]
    lines = [
        f"## LightGBM Active-Store {horizon.upper()}", "",
        "| 指标 | 结果 |", "|---|---:|",
        f"| Overall WAPE | {fmt(overall['wape'])}% |",
        f"| Nonzero WAPE | {fmt(quantity['nonzero']['wape'])}% |",
        f"| 5-20 WAPE | {fmt(quantity['5-20']['wape'])}% |",
        f"| 20+ WAPE | {fmt(quantity['20+']['wape'])}% |",
        f"| MAE | {fmt(overall['mae'], 6)} |",
        f"| RMSE | {fmt(overall['rmse'], 6)} |",
        f"| True total | {fmt(overall['target_sum'], 2)} |",
        f"| Predicted total | {fmt(overall['prediction_sum'], 2)} |",
        f"| Total Bias | {fmt(overall['total_bias_rate'])}% |",
        f"| Zero sample count | {int(zero['count']):,} |",
        f"| Zero mean prediction | {fmt(zero['mean_prediction'], 6)} |",
        f"| Zero predicted total | {fmt(zero['prediction_sum'], 2)} |",
        f"| Zero > 0.5 | {fmt(100 * zero['prediction_gt_0_5_rate'])}% |",
        f"| Zero > 1 | {fmt(100 * zero['prediction_gt_1_rate'])}% |",
        f"| MC Accuracy | {fmt(100 * mc['accuracy'])}% |",
        f"| MC Macro-F1 | {fmt(100 * mc['macro_f1'])}% |",
        f"| MC Weighted-F1 | {fmt(100 * mc['weighted_f1'])}% |",
        f"| MC High-demand Recall | {fmt(100 * mc['high_level_recall'])}% |",
        f"| Trend Accuracy | {fmt(100 * trend['accuracy'])}% |",
        f"| Trend Macro-F1 | {fmt(100 * trend['macro_f1'])}% |",
        f"| Trend Up Recall | {fmt(100 * trend['up_recall'])}% |",
        f"| Trend Down Recall | {fmt(100 * trend['down_recall'])}% |",
        f"| Trend severe direction error | {fmt(100 * trend['severe_direction_error_rate'])}% |",
        "", "### MC 各等级", "",
        "| 等级 | Precision | Recall | F1 |", "|---|---:|---:|---:|",
    ]
    for index, name in enumerate(mc_config.names):
        lines.append(
            f"| {name} | {100 * mc['precision'][index]:.4f}% | "
            f"{100 * mc['recall'][index]:.4f}% | {100 * mc['f1'][index]:.4f}% |"
        )
    lines.extend(["", "MC confusion matrix:", "", *matrix_table(mc["confusion_matrix"], mc_config.names)])
    lines.extend(["", "Trend confusion matrix (`down`, `flat`, `up`):", "", *matrix_table(trend["confusion_matrix"], ("down", "flat", "up"))])
    return lines


def historical_gate_lines() -> list[str]:
    if not GATE_PATH.exists():
        return ["历史 Gate 诊断明细不存在，未补充。"]
    frame = pd.read_csv(GATE_PATH)
    original = frame[frame["method"].eq("original")].iloc[0]
    best = frame[frame["method"].eq("gate_a")].sort_values(["wape", "threshold"]).iloc[0]
    tau_002 = frame[frame["method"].eq("gate_a") & np.isclose(frame["threshold"], 0.02)].iloc[0]
    return [
        "以下仅为 `Pre-Active-Store model / diagnostic only`，不是新版 Active-Store 正式 Two-stage：", "",
        "| 方案 | tau | Overall WAPE | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | Total Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Original | N/A | {original['wape']:.4f}% | {original['nonzero_wape']:.4f}% | {original['5_20_wape']:.4f}% | {original['20plus_wape']:.4f}% | {original['total_bias']:.4f}% |",
        f"| Gate-A low-cost reference | 0.020 | {tau_002['wape']:.4f}% | {tau_002['nonzero_wape']:.4f}% | {tau_002['5_20_wape']:.4f}% | {tau_002['20plus_wape']:.4f}% | {tau_002['total_bias']:.4f}% |",
        f"| Gate-A unconstrained minimum | {best['threshold']:.3f} | {best['wape']:.4f}% | {best['nonzero_wape']:.4f}% | {best['5_20_wape']:.4f}% | {best['20plus_wape']:.4f}% | {best['total_bias']:.4f}% |",
        "", "无约束最低点会大量牺牲中高销量覆盖，只保留为历史诊断，不作为本轮阈值结论。",
    ]


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    mc_config = load_mc_level_config(MC_CONFIG_PATH)
    results = {horizon: evaluate_horizon(horizon, mc_config)[0] for horizon in ("1m", "2m")}
    METRICS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=serializable), encoding="utf-8"
    )
    lines = [
        "# Active-Store 业务评价", "",
        "本报告复用现有 Active-Store LightGBM 正式 Test 预测，不重新训练或推理模型。Test 只用于已冻结模型的业务效果汇报，不参与轮次或阈值选择。", "",
        "## 口径", "",
        "- MC 使用 `config/mc_sales_levels.yaml` 的五级规则：0、1、2-4、5-19、20+。预测先非负截断并执行 `floor(x + 0.5)`。",
        "- 2M MC 使用未来两个月累计销量除以 2 后的月均值，真实值和预测值顺序完全一致。",
        "- 项目已有正式趋势实现：比较当前月销量与未来销量；2M 比较未来两个月月均。严格大于为上升、严格小于为下降、相等为持平，不使用额外百分比阈值。",
        "- MC 边界当前状态为项目暂定口径，后续业务确认后通过统一配置调整。", "",
        *horizon_lines("1m", results["1m"], mc_config), "",
        *horizon_lines("2m", results["2m"], mc_config), "",
        "## 历史 Two-stage Gate 诊断参考", "",
        *historical_gate_lines(), "",
        "## 数据一致性", "",
        f"- 1M rows: {results['1m']['runtime']['rows']:,}; target max difference: {results['1m']['runtime']['maximum_target_difference']:.6g}.",
        f"- 2M rows: {results['2m']['runtime']['rows']:,}; target max difference: {results['2m']['runtime']['maximum_target_difference']:.6g}.",
        "- 所有预测均为有限非负值，预测文件与统一 Active-Store 数据集目标一致。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
