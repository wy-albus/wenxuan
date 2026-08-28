from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # Load the Pandas/PyArrow native stack before LightGBM on Python 3.13.
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import load_feature_config, project_path  # noqa: E402
from src.models.lightgbm_model import load_model_bundle, save_model_bundle, train_booster  # noqa: E402
from src.models.lightgbm_objectives import (  # noqa: E402
    ObjectiveSpec,
    apply_calibration,
    calibration_factor,
    candidate_specs,
    inverse_prediction,
    transform_training_target,
)


HORIZONS = ("1m", "2m")
MODEL_NAMES = ("weighted_moving_average", "v1_log_l1", "v2_log_l2")
SEGMENTS = ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
FINAL_OUTPUTS = {
    "model_1m": ROOT / "models/final/lightgbm_v2_logl2_1m.txt",
    "model_2m": ROOT / "models/final/lightgbm_v2_logl2_2m.txt",
    "report": ROOT / "reports/lightgbm_v2_logl2_model_report.md",
    "importance_1m": ROOT / "reports/lightgbm_v2_logl2_feature_importance_1m.csv",
    "importance_2m": ROOT / "reports/lightgbm_v2_logl2_feature_importance_2m.csv",
    "predictions": ROOT / "data/outputs/lightgbm_v2_logl2_test_predictions.parquet",
}


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    directory = ROOT / "logs/lightgbm/v2_logl2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"formal_{run_id}.log"
    logger = logging.getLogger("wenxuan_lightgbm_v2_logl2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def ensure_outputs_absent() -> None:
    existing = [str(path) for path in FINAL_OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing V2 formal outputs: {existing}")


def recover_training_results(checkpoints: dict[str, Path], log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    v1_summary = json.loads((ROOT / "logs/lightgbm/formal_run_summary.json").read_text(encoding="utf-8"))
    recovered = {}
    pattern = re.compile(
        r"Completed V2 Log-L2 (1m|2m) best_iteration=(\d+) sampling=([0-9.]+)s "
        r"training=([0-9.]+)s peak=([0-9.]+)GiB"
    )
    matches = {match.group(1): match for match in pattern.finditer(text)}
    for horizon in HORIZONS:
        if horizon not in matches or not checkpoints[horizon].exists():
            raise RuntimeError(f"Cannot resume {horizon}: checkpoint or completed training log is missing")
        _, metadata = load_model_bundle(checkpoints[horizon])
        match = matches[horizon]
        v1_stage = v1_summary["stages"]["formal"][horizon]
        recovered[horizon] = {
            "horizon": horizon,
            "train_rows": int(v1_stage["train_rows"]),
            "valid_rows": int(v1_stage["valid_rows"]),
            "feature_count": len(metadata["feature_names"]),
            "best_iteration": int(match.group(2)),
            "sampling_seconds": float(match.group(3)),
            "training_seconds": float(match.group(4)),
            "peak_memory_bytes": float(match.group(5)) * 2**30,
            "memory_limit_bytes": None,
            "checkpoint_path": str(checkpoints[horizon]),
            "checkpoint_size_bytes": checkpoints[horizon].stat().st_size,
            "params": metadata["params"],
            "resumed_from_completed_checkpoint": True,
        }
    return recovered


def control_spec() -> ObjectiveSpec:
    return ObjectiveSpec("v1_log_l1", "V1 Log-L1", "log1p", {})


def train_formal_horizon(
    pipeline,
    dataset_path: Path,
    horizon: str,
    config: dict,
    bucket_counts: dict,
    category_maps: dict,
    objective: ObjectiveSpec,
    checkpoint_path: Path,
    logger: logging.Logger,
) -> dict:
    guard = pipeline.MemoryGuard(f"v2-logl2-formal-{horizon}", logger)
    guard.check("stage_start")
    settings = config["lightgbm_training"]
    features = pipeline.horizon_features(config, horizon)
    base_month = config["time_feature_settings"]["base_month"]
    sample_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        dataset_path,
        horizon,
        {"train": settings["sampling_rates"], "valid": settings["valid_sampling_rates"]},
        category_maps,
        features,
        base_month,
        guard,
        logger,
        {"train": None, "valid": None},
        None,
        True,
    )
    sampling_seconds = time.perf_counter() - sample_started
    train_raw = np.expm1(samples["train"]["y"].astype("float64"))
    valid_raw = np.expm1(samples["valid"]["y"].astype("float64"))
    train_y = transform_training_target(train_raw, objective)
    valid_y = transform_training_target(valid_raw, objective)
    train_rows, valid_rows = len(train_y), len(valid_y)
    estimated = int((train_rows + valid_rows) * (len(features) * 3 + 64))
    guard.ensure_capacity(estimated, "before_dataset_construction")
    logger.info(
        "Training V2 Log-L2 %s train=%s valid=%s features=%d",
        horizon, f"{train_rows:,}", f"{valid_rows:,}", len(features),
    )
    started = time.perf_counter()
    booster, evaluations = train_booster(
        samples["train"]["x"], train_y, samples["train"]["weight"],
        samples["valid"]["x"], valid_y, samples["valid"]["weight"],
        categorical_features=pipeline.CATEGORY_FEATURES,
        params=objective.params,
        num_boost_round=2000,
        early_stopping_rounds=100,
        callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)],
        construction_check=lambda label: guard.check(label),
    )
    training_seconds = time.perf_counter() - started
    check_x = samples["valid"]["x"].iloc[:1000]
    prediction = inverse_prediction(booster.predict(check_x, num_iteration=booster.best_iteration), objective)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError(f"Invalid V2 Log-L2 {horizon} predictions")
    metadata = {
        "experiment": "lightgbm_v2_logl2_formal",
        "candidate": "log_l2",
        "target_transform": objective.target_transform,
        "horizon": horizon,
        "feature_names": features,
        "categorical_features": pipeline.CATEGORY_FEATURES,
        "category_maps": category_maps,
        "time_base": base_month,
        "params": objective.params,
        "best_iteration": int(booster.best_iteration),
        "sampling_rates": settings["sampling_rates"],
        "valid_sampling_rates": settings["valid_sampling_rates"],
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
    }
    save_model_bundle(booster, checkpoint_path, metadata)
    loaded, loaded_metadata = load_model_bundle(checkpoint_path)
    reloaded_prediction = inverse_prediction(
        loaded.predict(check_x, num_iteration=loaded_metadata["best_iteration"]), objective
    )
    np.testing.assert_allclose(prediction, reloaded_prediction, rtol=1e-7, atol=1e-8)
    result = {
        "horizon": horizon,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "feature_count": len(features),
        "best_iteration": int(booster.best_iteration),
        "sampling_seconds": sampling_seconds,
        "training_seconds": training_seconds,
        "peak_memory_bytes": guard.peak,
        "memory_limit_bytes": guard.limit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "params": objective.params,
        "evaluations": evaluations,
        "train_distribution": samples["train"]["distribution"],
        "valid_distribution": samples["valid"]["distribution"],
    }
    del samples, train_raw, valid_raw, train_y, valid_y, booster, loaded, check_x, prediction, reloaded_prediction
    gc.collect()
    guard.check("stage_released")
    result["released_rss_bytes"] = pipeline.process_rss()
    logger.info(
        "Completed V2 Log-L2 %s best_iteration=%d sampling=%.1fs training=%.1fs peak=%.2fGiB",
        horizon, result["best_iteration"], sampling_seconds, training_seconds, result["peak_memory_bytes"] / 2**30,
    )
    return result


def weighted_moving_average(frame: pd.DataFrame, config: dict, horizon: str) -> np.ndarray:
    weights = config["baseline_settings"]["weighted_moving_average"]["weights"]
    prediction = np.zeros(len(frame), dtype="float64")
    for column, weight in weights.items():
        prediction += float(weight) * pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype="float64")
    if horizon == "2m":
        prediction *= float(config["baseline_settings"]["weighted_moving_average"]["horizon_2_multiplier"])
    return np.clip(prediction, 0.0, None)


def evaluation_columns(pipeline, bundles: dict, config: dict) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    columns = set(pipeline.IDENTIFIER_COLUMNS)
    columns.update(config["baseline_settings"]["weighted_moving_average"]["weights"])
    for horizon_models in bundles.values():
        for _, metadata in horizon_models.values():
            columns.update(column for column in metadata["feature_names"] if column not in runtime)
    return list(columns)


def predict_models(pipeline, frame: pd.DataFrame, horizon: str, bundles: dict, config: dict) -> dict[str, np.ndarray]:
    v1_booster, v1_metadata = bundles[horizon]["v1_log_l1"]
    v2_booster, v2_metadata = bundles[horizon]["v2_log_l2"]
    if v1_metadata["feature_names"] != v2_metadata["feature_names"]:
        raise RuntimeError(f"V1/V2 feature order mismatch for {horizon}")
    if v1_metadata["category_maps"] != v2_metadata["category_maps"]:
        raise RuntimeError(f"V1/V2 category map mismatch for {horizon}")
    x = pipeline.prepare_evaluation_frame(frame, v2_metadata)
    predictions = {
        "weighted_moving_average": weighted_moving_average(frame, config, horizon),
        "v1_log_l1": inverse_prediction(
            v1_booster.predict(x, num_iteration=v1_metadata["best_iteration"]), control_spec()
        ),
        "v2_log_l2": inverse_prediction(
            v2_booster.predict(x, num_iteration=v2_metadata["best_iteration"]),
            ObjectiveSpec("log_l2", "V2 Log-L2", "log1p", {}),
        ),
    }
    del x
    return predictions


def evaluate_valid(
    pipeline,
    dataset_path: Path,
    bundles: dict,
    config: dict,
    logger: logging.Logger,
) -> tuple[dict, dict[str, float], float, int]:
    guard = pipeline.MemoryGuard("v2-logl2-full-valid", logger)
    metrics = {h: {name: LongTailStreamingMetrics() for name in MODEL_NAMES} for h in HORIZONS}
    columns = evaluation_columns(pipeline, bundles, config)
    parquet = pq.ParquetFile(dataset_path)
    started = time.perf_counter()
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
            frame = frame[frame["split"] == "valid"]
            if frame.empty:
                continue
            for horizon in HORIZONS:
                target = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                predictions = predict_models(pipeline, frame, horizon, bundles, config)
                for name, prediction in predictions.items():
                    metrics[horizon][name].update(target, prediction)
            if (row_group + 1) % 10 == 0:
                guard.check("valid_evaluation")
                logger.info("Valid evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        parquet.close()
    computed = {h: {name: value.compute() for name, value in models.items()} for h, models in metrics.items()}
    factors = {
        h: calibration_factor(
            computed[h]["v2_log_l2"]["overall"]["target_sum"],
            computed[h]["v2_log_l2"]["overall"]["prediction_sum"],
        )
        for h in HORIZONS
    }
    logger.info("Valid-only calibration factors: %s", factors)
    return computed, factors, time.perf_counter() - started, guard.peak


def evaluate_test(
    pipeline,
    dataset_path: Path,
    bundles: dict,
    config: dict,
    factors: dict[str, float],
    temporary_output: Path,
    logger: logging.Logger,
) -> tuple[dict, float, int]:
    guard = pipeline.MemoryGuard("v2-logl2-full-test", logger)
    names = (*MODEL_NAMES, "v2_log_l2_calibrated")
    metrics = {h: {name: LongTailStreamingMetrics() for name in names} for h in HORIZONS}
    columns = evaluation_columns(pipeline, bundles, config)
    parquet = pq.ParquetFile(dataset_path)
    writer = None
    started = time.perf_counter()
    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.unlink(missing_ok=True)
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
            frame = frame[frame["split"] == "test"]
            if frame.empty:
                continue
            targets, raw_v2, calibrated = {}, {}, {}
            for horizon in HORIZONS:
                target = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                predictions = predict_models(pipeline, frame, horizon, bundles, config)
                calibrated_prediction = apply_calibration(predictions["v2_log_l2"], factors[horizon])
                for name, prediction in predictions.items():
                    metrics[horizon][name].update(target, prediction)
                metrics[horizon]["v2_log_l2_calibrated"].update(target, calibrated_prediction)
                targets[horizon] = target
                raw_v2[horizon] = predictions["v2_log_l2"]
                calibrated[horizon] = calibrated_prediction
            output = frame[["month", "site_no", "item_id"]].copy()
            for horizon in HORIZONS:
                output[f"target_qty_{horizon}"] = targets[horizon].astype("float32")
                output[f"lightgbm_v2_logl2_pred_{horizon}"] = raw_v2[horizon].astype("float32")
                output[f"lightgbm_v2_logl2_calibrated_pred_{horizon}"] = calibrated[horizon].astype("float32")
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output, table.schema, compression="zstd", use_dictionary=["month", "site_no"]
                )
            writer.write_table(table, row_group_size=250_000)
            if (row_group + 1) % 10 == 0:
                guard.check("test_evaluation")
                logger.info("Test evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        if writer is not None:
            writer.close()
        parquet.close()
    return (
        {h: {name: value.compute() for name, value in models.items()} for h, models in metrics.items()},
        time.perf_counter() - started,
        guard.peak,
    )


def write_importance(checkpoints: dict[str, Path], temporary_dir: Path) -> tuple[dict, dict[str, Path]]:
    top, paths = {}, {}
    for horizon, path in checkpoints.items():
        booster, metadata = load_model_bundle(path)
        frame = pd.DataFrame({
            "feature": metadata["feature_names"],
            "gain_importance": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }).sort_values("gain_importance", ascending=False)
        destination = temporary_dir / f"lightgbm_v2_logl2_feature_importance_{horizon}.csv"
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        paths[horizon] = destination
        top[horizon] = frame.head(20).to_dict("records")
    return top, paths


def fmt(value, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def render_report(summary: dict, destination: Path) -> None:
    display = {
        "weighted_moving_average": "加权移动平均",
        "v1_log_l1": "LightGBM V1 Log-L1",
        "v2_log_l2": "LightGBM V2 Log-L2",
        "v2_log_l2_calibrated": "V2 Log-L2 校准诊断",
    }
    lines = [
        "# LightGBM V2 Log-L2 正式训练报告",
        "",
        "V2 使用 `log1p(target)`、`regression_l2`，预测后执行 `expm1` 并裁剪为非负。V1 正式产物保持只读。",
        "",
        "## 正式训练",
        "",
        "| Horizon | Train | Early Valid | 特征数 | 最佳轮次 | 采样耗时(秒) | 训练耗时(秒) | 峰值RAM(GiB) | 模型大小(MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        value = summary["training"][horizon]
        lines.append(
            f"| {horizon} | {value['train_rows']:,} | {value['valid_rows']:,} | {value['feature_count']} | "
            f"{value['best_iteration']} | {value['sampling_seconds']:.1f} | {value['training_seconds']:.1f} | "
            f"{value['peak_memory_bytes'] / 2**30:.2f} | {summary['final_model_sizes'][horizon] / 2**20:.2f} |"
        )
    lines.extend([
        "",
        "## Valid 整体对比",
        "",
        "| Horizon | 模型 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        for name in MODEL_NAMES:
            value = summary["valid_metrics"][horizon][name]["overall"]
            lines.append(metric_row(horizon, display[name], value))
    lines.extend([
        "",
        "## Test 整体对比",
        "",
        "| Horizon | 模型 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        for name in (*MODEL_NAMES, "v2_log_l2_calibrated"):
            value = summary["test_metrics"][horizon][name]["overall"]
            lines.append(metric_row(horizon, display[name], value))
    lines.extend([
        "",
        "## 零销量误报诊断",
        "",
        "| 数据集 | Horizon | 模型 | 零销量样本量 | 平均预测 | 预测>0.5 | 预测>1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for split in ("valid", "test"):
        source = summary[f"{split}_metrics"]
        names = MODEL_NAMES if split == "valid" else (*MODEL_NAMES, "v2_log_l2_calibrated")
        for horizon in HORIZONS:
            for name in names:
                zero = source[horizon][name]["0"]
                lines.append(
                    f"| {split} | {horizon} | {display[name]} | {zero['count']:,} | "
                    f"{fmt(zero['mean_prediction'])} | {fmt(100 * zero['prediction_gt_0_5_rate'], 2)}% | "
                    f"{fmt(100 * zero['prediction_gt_1_rate'], 2)}% |"
                )
    lines.extend(["", "## 分层指标", ""])
    for split in ("valid", "test"):
        source = summary[f"{split}_metrics"]
        lines.extend([
            f"### {split.upper()}",
            "",
            "| Horizon | 模型 | 层级 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        names = MODEL_NAMES if split == "valid" else (*MODEL_NAMES, "v2_log_l2_calibrated")
        for horizon in HORIZONS:
            for name in names:
                for segment in SEGMENTS:
                    value = source[horizon][name][segment]
                    lines.append(segment_row(horizon, display[name], segment, value))
        lines.append("")
    lines.extend(["## Valid-only 总量校准诊断", ""])
    for horizon in HORIZONS:
        lines.append(
            f"- `{horizon}` calibration factor = **{summary['calibration_factors'][horizon]:.6f}**，"
            "仅由完整 valid 的真实总量 / V2 原始预测总量计算。"
        )
    lines.extend(["", "## 业务判断", ""])
    for horizon in HORIZONS:
        v1 = summary["test_metrics"][horizon]["v1_log_l1"]
        v2 = summary["test_metrics"][horizon]["v2_log_l2"]
        baseline = summary["test_metrics"][horizon]["weighted_moving_average"]
        lines.append(
            f"- **{horizon}**：总量偏差由 V1 的 {v1['overall']['total_bias_rate']:.2f}% 变为 "
            f"{v2['overall']['total_bias_rate']:.2f}%；nonzero WAPE 由 {v1['nonzero']['wape']:.2f}% 变为 "
            f"{v2['nonzero']['wape']:.2f}%；5-20 WAPE 由 {v1['5-20']['wape']:.2f}% 变为 "
            f"{v2['5-20']['wape']:.2f}%；20+ WAPE 由 {v1['20+']['wape']:.2f}% 变为 {v2['20+']['wape']:.2f}%。"
        )
        lines.append(
            f"  相对加权移动平均，V2 的整体 MAE 为 {v2['overall']['mae']:.4f} 对 "
            f"{baseline['overall']['mae']:.4f}，总量偏差为 {v2['overall']['total_bias_rate']:.2f}% 对 "
            f"{baseline['overall']['total_bias_rate']:.2f}%。"
        )
    lines.extend(["", "## 前 20 个重要特征", ""])
    for horizon in HORIZONS:
        lines.extend([f"### {horizon}", "", "| 排名 | 特征 | Gain | Split |", "|---:|---|---:|---:|"])
        for rank, row in enumerate(summary["feature_importance"][horizon], 1):
            lines.append(f"| {rank} | {row['feature']} | {row['gain_importance']:.2f} | {int(row['split_importance'])} |")
        lines.append("")
    lines.extend([
        "## 运行与结论",
        "",
        f"- 完整 valid 评价耗时：{summary['valid_evaluation_seconds']:.1f} 秒；峰值 RAM：{summary['valid_peak_memory_bytes'] / 2**30:.2f} GiB。",
        f"- 完整 test 评价及预测写入耗时：{summary['test_evaluation_seconds']:.1f} 秒；峰值 RAM：{summary['test_peak_memory_bytes'] / 2**30:.2f} GiB。",
        f"- Python {summary['python_version']}；LightGBM {summary['lightgbm_version']}。",
        "- 原始 Log-L2 是主结果；乘法校准仅用于诊断，没有覆盖模型原始预测。",
        "- V1 正式模型、报告、预测和特征重要性在运行前后哈希一致。",
        "- 本阶段没有启动随机森林、MLP 或两阶段模型。",
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(horizon: str, name: str, value: dict) -> str:
    return (
        f"| {horizon} | {name} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
        f"{fmt(value['smape'], 2)}% | {fmt(value['wape'], 2)}% | {value['target_sum']:.0f} | "
        f"{value['prediction_sum']:.0f} | {fmt(value['total_bias_rate'], 2)}% |"
    )


def segment_row(horizon: str, name: str, segment: str, value: dict) -> str:
    return (
        f"| {horizon} | {name} | {segment} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
        f"{fmt(value['smape'], 2)}% | {fmt(value['wape'], 2)}% | {value['target_sum']:.0f} | "
        f"{value['prediction_sum']:.0f} | {fmt(value['total_bias_rate'], 2)}% |"
    )


def promote_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".promoting")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formal LightGBM V2 Log-L2 models.")
    parser.add_argument("--dataset", default="data/processed/model_dataset_monthly.parquet")
    parser.add_argument("--resume-run-id", help="Resume evaluation from completed formal checkpoints without retraining.")
    args = parser.parse_args()
    ensure_outputs_absent()
    run_id = args.resume_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logging(run_id)
    pipeline = load_script_module("wenxuan_v1_pipeline_for_v2", ROOT / "scripts/10_train_lightgbm.py")
    pilot = load_script_module("wenxuan_v2_pilot_helpers", ROOT / "scripts/11_lightgbm_objective_pilot.py")
    protected_before = pilot.protected_snapshot()
    dataset_path = project_path(args.dataset)
    public_signature = {"size": dataset_path.stat().st_size, "mtime_ns": dataset_path.stat().st_mtime_ns}
    config = load_feature_config()
    objective = candidate_specs(pipeline.training_params(config))["log_l2"]
    checkpoint_dir = ROOT / "models/checkpoints/lightgbm_v2_logl2_formal" / run_id
    temporary_dir = ROOT / "data/temp" / f"lightgbm_v2_logl2_{run_id}"
    if args.resume_run_id:
        if not checkpoint_dir.is_dir() or not temporary_dir.is_dir():
            raise FileNotFoundError(f"Resume directories are missing for run {run_id}")
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        temporary_dir.mkdir(parents=True, exist_ok=False)
    summary_path = ROOT / "logs/lightgbm/v2_logl2" / f"formal_{run_id}.json"
    logger.info("Starting formal V2 Log-L2 Python=%s LightGBM=%s", platform.python_version(), lgb.__version__)
    bucket_counts, category_maps = pipeline.load_or_build_scan_metadata(dataset_path, logger)
    checkpoints = {h: checkpoint_dir / f"lightgbm_v2_logl2_{h}.txt" for h in HORIZONS}
    training = recover_training_results(checkpoints, log_path) if args.resume_run_id else {}
    try:
        if args.resume_run_id:
            logger.info("Resuming from completed formal checkpoints for run %s; training will not run", run_id)
        else:
            for horizon in HORIZONS:
                training[horizon] = train_formal_horizon(
                    pipeline, dataset_path, horizon, config, bucket_counts, category_maps,
                    objective, checkpoints[horizon], logger,
                )
                gc.collect()
                pipeline.MemoryGuard(f"between-horizons-{horizon}", logger).check("released")
        bundles = {
            h: {
                "v1_log_l1": load_model_bundle(ROOT / f"models/final/lightgbm_{h}.txt"),
                "v2_log_l2": load_model_bundle(checkpoints[h]),
            }
            for h in HORIZONS
        }
        valid_metrics, factors, valid_seconds, valid_peak = evaluate_valid(
            pipeline, dataset_path, bundles, config, logger
        )
        temporary_predictions = temporary_dir / "lightgbm_v2_logl2_test_predictions.parquet"
        test_metrics, test_seconds, test_peak = evaluate_test(
            pipeline, dataset_path, bundles, config, factors, temporary_predictions, logger
        )
        expected_valid = int(bucket_counts["valid"]["1m"].total())
        expected_test = int(bucket_counts["test"]["1m"].total())
        if any(valid_metrics[h][name]["overall"]["count"] != expected_valid for h in HORIZONS for name in MODEL_NAMES):
            raise RuntimeError("Complete valid count mismatch")
        if any(test_metrics[h][name]["overall"]["count"] != expected_test for h in HORIZONS for name in (*MODEL_NAMES, "v2_log_l2_calibrated")):
            raise RuntimeError("Complete test count mismatch")
        importance, importance_paths = write_importance(checkpoints, temporary_dir)
        pilot.assert_protected_unchanged(protected_before)
        if dataset_path.stat().st_size != public_signature["size"] or dataset_path.stat().st_mtime_ns != public_signature["mtime_ns"]:
            raise RuntimeError("Public model dataset changed")
        final_sizes = {h: checkpoints[h].stat().st_size for h in HORIZONS}
        summary = {
            "run_id": run_id,
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__,
            "log_path": str(log_path),
            "dataset_signature": public_signature,
            "training": training,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "calibration_factors": factors,
            "valid_evaluation_seconds": valid_seconds,
            "test_evaluation_seconds": test_seconds,
            "valid_peak_memory_bytes": valid_peak,
            "test_peak_memory_bytes": test_peak,
            "feature_importance": importance,
            "final_model_sizes": final_sizes,
            "prediction_size_bytes": temporary_predictions.stat().st_size,
            "v1_protected_artifacts": protected_before,
            "v1_unchanged": True,
            "formal_complete": True,
        }
        temporary_report = temporary_dir / "lightgbm_v2_logl2_model_report.md"
        render_report(summary, temporary_report)
        for horizon in HORIZONS:
            promote_file(checkpoints[horizon], FINAL_OUTPUTS[f"model_{horizon}"])
            promote_file(importance_paths[horizon], FINAL_OUTPUTS[f"importance_{horizon}"])
        promote_file(temporary_predictions, FINAL_OUTPUTS["predictions"])
        promote_file(temporary_report, FINAL_OUTPUTS["report"])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        pilot.assert_protected_unchanged(protected_before)
        logger.info("Formal V2 Log-L2 completed; report=%s", FINAL_OUTPUTS["report"])
    except Exception:
        logger.exception("Formal V2 Log-L2 failed; V1 artifacts remain protected")
        pilot.assert_protected_unchanged(protected_before)
        raise


if __name__ == "__main__":
    main()
