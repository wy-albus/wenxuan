from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    DEMAND_BUCKETS,
    GroupedStreamingMetrics,
    StreamingMetrics,
    clip_target,
    demand_bucket,
)
from src.features.preprocessing import get_feature_list, load_feature_config, project_path  # noqa: E402
from src.models.historical_mean import predict_historical_mean  # noqa: E402
from src.models.weighted_moving_average import predict_weighted_moving_average  # noqa: E402


SOURCE_COLUMNS = (
    "month",
    "site_no",
    "item_id",
    "split",
    "future_qty_1m",
    "future_qty_2m",
    "qty_lag_1m",
    "qty_lag_2m",
    "qty_lag_3m",
    "qty_mean_last_3m",
    "qty_sum_last_3m",
)

PREDICTION_COLUMNS = (
    "month",
    "site_no",
    "item_id",
    "split",
    "target_qty_1m",
    "target_qty_2m",
    "historical_mean_pred_1m",
    "historical_mean_pred_2m",
    "weighted_ma_pred_1m",
    "weighted_ma_pred_2m",
)

MODEL_LABELS = {
    "historical_mean": "历史均值法",
    "weighted_moving_average": "加权移动平均法",
}


def _baseline_settings(config: dict) -> tuple[float, list[float], float]:
    settings = config.get("baseline_settings", {})
    historical_multiplier = float(settings.get("historical_mean", {}).get("horizon_2_multiplier", 2.0))
    weighted = settings.get("weighted_moving_average", {})
    weight_map = weighted.get("weights", {})
    weights = [float(weight_map.get(f"qty_lag_{lag}m", default)) for lag, default in ((1, 0.5), (2, 0.3), (3, 0.2))]
    weighted_multiplier = float(weighted.get("horizon_2_multiplier", 2.0))
    return historical_multiplier, weights, weighted_multiplier


def _validate_config(config: dict) -> None:
    expected = {
        "historical_mean": {"qty_mean_last_3m"},
        "weighted_moving_average": {"qty_lag_1m", "qty_lag_2m", "qty_lag_3m"},
    }
    for model_name, required in expected.items():
        configured = set(get_feature_list(model_name, config))
        missing = required - configured
        if missing:
            raise ValueError(f"{model_name} feature configuration is missing: {sorted(missing)}")
    _, weights, _ = _baseline_settings(config)
    if len(weights) != 3 or not np.isclose(sum(weights), 1.0):
        raise ValueError("weighted moving average weights must contain three values that sum to 1.0")


def _predict(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    historical_multiplier, weights, weighted_multiplier = _baseline_settings(config)
    historical_1m, historical_2m = predict_historical_mean(
        frame["qty_mean_last_3m"].to_numpy(),
        horizon_2_multiplier=historical_multiplier,
    )
    weighted_1m, weighted_2m = predict_weighted_moving_average(
        frame["qty_lag_1m"].to_numpy(),
        frame["qty_lag_2m"].to_numpy(),
        frame["qty_lag_3m"].to_numpy(),
        weights=weights,
        horizon_2_multiplier=weighted_multiplier,
    )
    return {
        "historical_mean_pred_1m": historical_1m,
        "historical_mean_pred_2m": historical_2m,
        "weighted_ma_pred_1m": weighted_1m,
        "weighted_ma_pred_2m": weighted_2m,
    }


def _new_metric_registry() -> dict:
    return {
        split: {
            model: {horizon: GroupedStreamingMetrics() for horizon in ("1m", "2m")}
            for model in MODEL_LABELS
        }
        for split in ("valid", "test")
    }


def _update_metrics(registry: dict, split: str, targets: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> None:
    for model, prefix in (("historical_mean", "historical_mean"), ("weighted_moving_average", "weighted_ma")):
        for horizon in ("1m", "2m"):
            registry[split][model][horizon].update(targets[horizon], predictions[f"{prefix}_pred_{horizon}"])


def _compute_registry(registry: dict) -> dict:
    return {
        split: {
            model: {horizon: grouped.compute() for horizon, grouped in horizons.items()}
            for model, horizons in models.items()
        }
        for split, models in registry.items()
    }


def _prediction_frame(frame: pd.DataFrame, targets: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    output = frame[["month", "site_no", "item_id", "split"]].copy()
    output["target_qty_1m"] = targets["1m"]
    output["target_qty_2m"] = targets["2m"]
    for column, values in predictions.items():
        output[column] = values
    for column in ("month", "site_no", "item_id", "split"):
        output[column] = output[column].astype("string")
    for column in PREDICTION_COLUMNS[4:]:
        output[column] = output[column].astype("float32")
    return output[list(PREDICTION_COLUMNS)]


def _peak_process_memory_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        get_process_memory_info.restype = wintypes.BOOL

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = get_current_process()
        if get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return 0

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _format_number(value: float | int, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if np.isnan(value):
        return "N/A"
    if np.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def _render_report(summary: dict, report_path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# 均值类 Baseline 模型报告",
        "",
        "## 模型定义",
        "",
        "- 历史均值法：`pred_1m = max(qty_mean_last_3m, 0)`，`pred_2m = pred_1m * 2`。",
        "- 加权移动平均法：`pred_1m = max(0.5*qty_lag_1m + 0.3*qty_lag_2m + 0.2*qty_lag_3m, 0)`，`pred_2m = pred_1m * 2`。",
        "- 真实目标在运行时构造为 `max(future_qty, 0)`；未修改公共 Parquet。",
        "",
        "## 使用字段",
        "",
        "仅读取以下 11 列：`" + "`, `".join(SOURCE_COLUMNS) + "`。模型公式使用 `qty_mean_last_3m` 或三个 lag 字段，其他字段用于识别、划分和评价。",
        "",
        "## 整体指标",
        "",
        "SMAPE 和 WAPE 均以百分比表示。",
        "",
        "| split | 模型 | 目标 | 样本量 | MAE | RMSE | SMAPE(%) | WAPE(%) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        for model in MODEL_LABELS:
            for horizon in ("1m", "2m"):
                values = metrics[split][model][horizon]["overall"]
                lines.append(
                    f"| {split} | {MODEL_LABELS[model]} | {horizon} | {_format_number(values['count'])} | "
                    f"{_format_number(values['mae'])} | {_format_number(values['rmse'])} | "
                    f"{_format_number(values['smape'])} | {_format_number(values['wape'])} |"
                )

    lines.extend([
        "",
        "## 各销量层级指标",
        "",
        "分组依据是对应预测目标的非负真实补货需求。",
        "",
        "| split | 模型 | 目标 | 销量层级 | 样本量 | MAE | RMSE | SMAPE(%) | WAPE(%) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for split in ("valid", "test"):
        for model in MODEL_LABELS:
            for horizon in ("1m", "2m"):
                for bucket in DEMAND_BUCKETS:
                    values = metrics[split][model][horizon][bucket]
                    lines.append(
                        f"| {split} | {MODEL_LABELS[model]} | {horizon} | {bucket} | {_format_number(values['count'])} | "
                        f"{_format_number(values['mae'])} | {_format_number(values['rmse'])} | "
                        f"{_format_number(values['smape'])} | {_format_number(values['wape'])} |"
                    )

    lines.extend(["", "## 效果比较", ""])
    winners = []
    for split in ("valid", "test"):
        for horizon in ("1m", "2m"):
            maes = {model: metrics[split][model][horizon]["overall"]["mae"] for model in MODEL_LABELS}
            winner = min(maes, key=maes.get)
            winners.append(f"- {split} / {horizon}：按 MAE，{MODEL_LABELS[winner]}更好（{maes[winner]:.4f}）。")
    lines.extend(winners)

    lines.extend([
        "",
        "两个月预测直接将单月预测乘以 2，因此会放大单月预测偏差；应结合 MAE、WAPE 和高销量层级结果判断该简化假设是否合适。",
        "",
        "## 性能与输出",
        "",
        f"- 正式 valid/test 评价运行时间（不含可选 smoke test）：{summary['runtime_seconds']:.2f} 秒。",
        f"- 进程生命周期峰值内存（包含导入及可选 smoke test）：{summary['peak_memory_bytes'] / 1024 ** 3:.2f} GiB。",
        f"- PyArrow 内存池峰值：{summary['arrow_peak_memory_bytes'] / 1024 ** 3:.2f} GiB。",
        f"- 输入 row group：{summary['row_groups_processed']:,}。",
        f"- valid 样本：{summary['split_rows']['valid']:,}；test 样本：{summary['split_rows']['test']:,}。",
        f"- test 预测文件大小：{summary['prediction_file_bytes'] / 1024 ** 2:.2f} MiB。",
        "- 预测 Parquet 使用 Zstandard 压缩，仅包含必要的 10 个字段。",
        "",
        "## 下一步 LightGBM 建议",
        "",
        "1. 将本报告中的 valid/test 整体和分层指标作为最低比较基准，重点观察 LightGBM 是否改善 `2-5` 以上的有效需求样本。",
        "2. 保持时间划分不变，并继续使用按列读取；训练时对长尾目标使用 `log1p(target_qty)`，预测后再反变换并截断为非负。",
        "3. 同时报告零销量识别能力与非零需求误差，避免总体 MAE 被大量零销量样本主导。",
        "4. 两个月目标建议直接训练独立模型，不沿用简单的单月预测乘 2。",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_smoke_test(
    dataset_path: str | Path,
    row_groups: int = 2,
    rows_per_group: int = 50_000,
    config_path: str | Path = ROOT / "config" / "model_features.yaml",
) -> dict:
    dataset_path = project_path(dataset_path)
    config = load_feature_config(config_path)
    _validate_config(config)
    parquet = pq.ParquetFile(dataset_path)
    collected_targets = []
    collected_predictions = []
    streamed = StreamingMetrics()
    rows_checked = 0

    try:
        for row_group in range(min(row_groups, parquet.num_row_groups)):
            batches = parquet.iter_batches(
                batch_size=rows_per_group,
                row_groups=[row_group],
                columns=list(SOURCE_COLUMNS),
                use_threads=True,
            )
            batch = next(batches, None)
            if batch is None:
                continue
            frame = pa.Table.from_batches([batch]).to_pandas()
            frame = frame[frame["split"].isin(["valid", "test"])]
            if frame.empty:
                continue
            targets = clip_target(frame["future_qty_1m"].to_numpy())
            predictions = _predict(frame, config)
            historical = predictions["historical_mean_pred_1m"]
            weighted = predictions["weighted_ma_pred_1m"]
            lag_1m = frame["qty_lag_1m"].fillna(0).to_numpy(dtype="float64")
            lag_2m = frame["qty_lag_2m"].fillna(0).to_numpy(dtype="float64")
            lag_3m = frame["qty_lag_3m"].fillna(0).to_numpy(dtype="float64")
            expected_weighted = np.maximum(0.5 * lag_1m + 0.3 * lag_2m + 0.2 * lag_3m, 0)
            expected_historical = np.maximum(
                frame["qty_mean_last_3m"].fillna(0).to_numpy(dtype="float64"),
                0,
            )
            np.testing.assert_allclose(historical, expected_historical)
            np.testing.assert_allclose(weighted, expected_weighted)
            if targets.size and targets.min() < 0:
                raise AssertionError("non-negative target clipping failed")
            streamed.update(targets, weighted)
            collected_targets.append(targets)
            collected_predictions.append(weighted)
            rows_checked += len(frame)
    finally:
        parquet.close()

    if not collected_targets:
        raise AssertionError("smoke test did not find valid/test rows")
    one_shot = StreamingMetrics()
    one_shot.update(np.concatenate(collected_targets), np.concatenate(collected_predictions))
    for metric in ("count", "mae", "rmse", "smape", "wape"):
        if not np.allclose(streamed.compute()[metric], one_shot.compute()[metric], equal_nan=True):
            raise AssertionError(f"streaming metric mismatch: {metric}")

    expected_buckets = ["0", "1", "2-5", "5-20", "5-20", "20+", "20+"]
    if demand_bucket(np.array([0, 1, 2, 5, 6, 20, 21])).tolist() != expected_buckets:
        raise AssertionError("demand bucket boundaries are incorrect")
    zero_metrics = StreamingMetrics()
    zero_metrics.update(np.zeros(3), np.zeros(3))
    if zero_metrics.compute()["smape"] != 0 or zero_metrics.compute()["wape"] != 0:
        raise AssertionError("zero-denominator metric handling failed")
    return {"passed": True, "row_groups": min(row_groups, parquet.num_row_groups), "rows_checked": rows_checked}


def run_baseline_pipeline(
    dataset_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    sample_path: str | Path,
    config_path: str | Path = ROOT / "config" / "model_features.yaml",
    max_row_groups: int | None = None,
    sample_rows_per_split: int = 100,
) -> dict:
    dataset_path = project_path(dataset_path)
    predictions_path = project_path(predictions_path)
    report_path = project_path(report_path)
    sample_path = project_path(sample_path)
    config = load_feature_config(config_path)
    _validate_config(config)

    parquet = pq.ParquetFile(dataset_path)
    missing = set(SOURCE_COLUMNS) - set(parquet.schema.names)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    row_group_count = parquet.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.num_row_groups)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions_path = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    temporary_report_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_sample_path = sample_path.with_suffix(sample_path.suffix + ".tmp")
    for temporary_path in (temporary_predictions_path, temporary_report_path, temporary_sample_path):
        temporary_path.unlink(missing_ok=True)

    registry = _new_metric_registry()
    split_rows = {"valid": 0, "test": 0}
    sample_parts = {"valid": [], "test": []}
    sample_counts = {"valid": 0, "test": 0}
    writer = None
    started = time.perf_counter()

    try:
        for row_group in range(row_group_count):
            table = parquet.read_row_group(row_group, columns=list(SOURCE_COLUMNS), use_threads=True)
            frame = table.to_pandas()
            frame = frame[frame["split"].isin(["valid", "test"])]
            if frame.empty:
                continue

            for split in ("valid", "test"):
                split_frame = frame[frame["split"] == split]
                if split_frame.empty:
                    continue
                targets = {
                    "1m": clip_target(split_frame["future_qty_1m"].to_numpy()),
                    "2m": clip_target(split_frame["future_qty_2m"].to_numpy()),
                }
                predictions = _predict(split_frame, config)
                _update_metrics(registry, split, targets, predictions)
                split_rows[split] += len(split_frame)

                remaining = sample_rows_per_split - sample_counts[split]
                if remaining > 0:
                    sample_part = _prediction_frame(split_frame.iloc[:remaining], {k: v[:remaining] for k, v in targets.items()}, {k: v[:remaining] for k, v in predictions.items()})
                    sample_parts[split].append(sample_part)
                    sample_counts[split] += len(sample_part)

                if split == "test":
                    output = _prediction_frame(split_frame, targets, predictions)
                    output_table = pa.Table.from_pandas(output, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary_predictions_path,
                            output_table.schema,
                            compression="zstd",
                            use_dictionary=["month", "site_no", "split"],
                        )
                    writer.write_table(output_table, row_group_size=250_000)

            if (row_group + 1) % 10 == 0 or row_group + 1 == row_group_count:
                print(
                    f"Processed row groups {row_group + 1}/{row_group_count}; "
                    f"valid={split_rows['valid']:,}, test={split_rows['test']:,}, "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        parquet.close()

    if writer is None:
        empty = pd.DataFrame({column: pd.Series(dtype="string" if column in PREDICTION_COLUMNS[:4] else "float32") for column in PREDICTION_COLUMNS})
        pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), temporary_predictions_path, compression="zstd")

    sample_frames = [part for split in ("valid", "test") for part in sample_parts[split]]
    pd.concat(sample_frames, ignore_index=True).to_csv(temporary_sample_path, index=False, encoding="utf-8-sig")

    summary = {
        "metrics": _compute_registry(registry),
        "split_rows": split_rows,
        "row_groups_processed": row_group_count,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": _peak_process_memory_bytes(),
        "arrow_peak_memory_bytes": int(pa.default_memory_pool().max_memory()),
        "prediction_file_bytes": temporary_predictions_path.stat().st_size,
    }
    _render_report(summary, temporary_report_path)
    temporary_predictions_path.replace(predictions_path)
    temporary_report_path.replace(report_path)
    temporary_sample_path.replace(sample_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate historical-mean and weighted-moving-average baselines.")
    parser.add_argument("--dataset", default="data/processed/model_dataset_monthly.parquet")
    parser.add_argument("--config", default="config/model_features.yaml")
    parser.add_argument("--predictions", default="data/outputs/baseline_predictions.parquet")
    parser.add_argument("--report", default="reports/baseline_model_report.md")
    parser.add_argument("--sample", default="data/outputs/baseline_prediction_sample.csv")
    parser.add_argument("--smoke-row-groups", type=int, default=2)
    parser.add_argument("--smoke-rows-per-group", type=int, default=50_000)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-row-groups", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_smoke:
        smoke = run_smoke_test(
            args.dataset,
            row_groups=args.smoke_row_groups,
            rows_per_group=args.smoke_rows_per_group,
            config_path=args.config,
        )
        print(f"Smoke test passed: {smoke}", flush=True)
    if args.smoke_only:
        return

    summary = run_baseline_pipeline(
        args.dataset,
        args.predictions,
        args.report,
        args.sample,
        config_path=args.config,
        max_row_groups=args.max_row_groups,
    )
    print(
        f"Baseline evaluation complete: valid={summary['split_rows']['valid']:,}, "
        f"test={summary['split_rows']['test']:,}, runtime={summary['runtime_seconds']:.2f}s, "
        f"peak_memory={summary['peak_memory_bytes'] / 1024 ** 3:.2f}GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
