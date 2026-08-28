from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # Load the Pandas/PyArrow native stack before LightGBM on Python 3.13.
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import get_feature_list, load_feature_config, project_path  # noqa: E402
from src.models.lightgbm_model import (  # noqa: E402
    MODEL_METADATA_SENTINEL,
    load_model_bundle,
    save_model_bundle,
    train_booster,
)
from src.models.two_stage_model import StreamingBinaryMetrics, binary_target, combine_predictions  # noqa: E402


HORIZONS = ("1m", "2m")
STAGES = ("classifier", "regressor")
SEGMENTS = ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
REGRESSION_SEGMENTS = ("overall", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
FINAL_OUTPUTS = {
    "classifier_1m": ROOT / "models/final/two_stage_classifier_1m.txt",
    "regressor_1m": ROOT / "models/final/two_stage_regressor_1m.txt",
    "classifier_2m": ROOT / "models/final/two_stage_classifier_2m.txt",
    "regressor_2m": ROOT / "models/final/two_stage_regressor_2m.txt",
    "report": ROOT / "reports/two_stage_model_report.md",
    "importance": ROOT / "reports/two_stage_feature_importance.csv",
    "predictions": ROOT / "data/outputs/two_stage_test_predictions.parquet",
}


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_module():
    return load_script_module("wenxuan_lightgbm_pipeline_for_two_stage", ROOT / "scripts/10_train_lightgbm.py")


def horizon_features(config: dict, horizon: str) -> list[str]:
    return get_feature_list("two_stage", config) + list(config["lightgbm_horizon_features"][horizon])


def lightgbm_horizon_features(config: dict, horizon: str) -> list[str]:
    return get_feature_list("lightgbm", config) + list(config["lightgbm_horizon_features"][horizon])


def regressor_sampling_rates() -> dict[str, float]:
    return {"0": 0.0, "1": 0.20, "2-5": 0.40, "5-20": 0.75, "20+": 1.0}


def _base_params(config: dict) -> dict[str, Any]:
    params = dict(config["lightgbm_training"]["params"])
    for key in ("objective", "metric", "tweedie_variance_power", "scale_pos_weight", "is_unbalance"):
        params.pop(key, None)
    params["num_threads"] = max(1, (os.cpu_count() or 2) - 2)
    params["first_metric_only"] = True
    return params


def classifier_params(config: dict) -> dict[str, Any]:
    return {**_base_params(config), "objective": "binary", "metric": ["binary_logloss", "auc", "average_precision"]}


def regressor_params(config: dict) -> dict[str, Any]:
    return {
        **_base_params(config), "objective": "tweedie", "metric": ["tweedie", "l1"],
        "tweedie_variance_power": 1.4,
    }


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    directory = ROOT / "logs/two_stage"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"training_{run_id}.log"
    logger = logging.getLogger("wenxuan_two_stage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def read_model_metadata(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if MODEL_METADATA_SENTINEL not in text:
        raise ValueError(f"Model metadata not found in {path}")
    return json.loads(text.rsplit(MODEL_METADATA_SENTINEL, maxsplit=1)[1].strip())


def is_resumable_metadata(metadata: dict, component: str, horizon: str) -> bool:
    return (
        metadata.get("run_stage") == "formal"
        and metadata.get("component") == component
        and metadata.get("horizon") == horizon
    )


def component_summary_path(component: str, horizon: str) -> Path:
    return ROOT / f"logs/two_stage/formal_{component}_{horizon}_summary.json"


def recover_component_result(model_path: Path, component: str, horizon: str) -> dict:
    summary_path = component_summary_path(component, horizon)
    if summary_path.exists():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result.get("component") == component and result.get("horizon") == horizon:
            return result

    metadata = read_model_metadata(model_path)
    if not is_resumable_metadata(metadata, component, horizon):
        raise RuntimeError(f"Existing model cannot be resumed: {model_path}")
    training_pattern = re.compile(
        rf"Training formal {component} {horizon} train=([\d,]+) valid=([\d,]+) features=(\d+)"
    )
    completed_pattern = re.compile(
        rf"Completed formal {component} {horizon} best_iteration=(\d+) sampling=([\d.]+)s "
        rf"training=([\d.]+)s peak=([\d.]+)GiB"
    )
    training_match = completed_match = None
    for log_path in sorted((ROOT / "logs/two_stage").glob("training_*.log"), reverse=True):
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        training_match = training_pattern.search(log_text)
        completed_match = completed_pattern.search(log_text)
        if training_match and completed_match:
            break
    if not training_match or not completed_match:
        raise RuntimeError(f"Cannot recover completed training statistics for {component} {horizon}")
    result = {
        "run_stage": "formal", "component": component, "horizon": horizon,
        "train_rows": int(training_match.group(1).replace(",", "")),
        "valid_rows": int(training_match.group(2).replace(",", "")),
        "feature_count": int(training_match.group(3)),
        "best_iteration": int(completed_match.group(1)),
        "sampling_seconds": float(completed_match.group(2)),
        "training_seconds": float(completed_match.group(3)),
        "peak_memory_bytes": int(float(completed_match.group(4)) * 2**30),
        "memory_limit_bytes": None, "model_size_bytes": model_path.stat().st_size,
        "params": metadata.get("params", {}), "evaluations": {},
        "reload_consistent": True, "recovered": True,
    }
    save_json(summary_path, result)
    return result


def ensure_formal_outputs_resumable() -> None:
    completed_outputs = [
        str(FINAL_OUTPUTS[key]) for key in ("report", "importance", "predictions")
        if FINAL_OUTPUTS[key].exists()
    ]
    if completed_outputs:
        raise FileExistsError(f"Refusing to overwrite completed two-stage outputs: {completed_outputs}")
    for horizon in HORIZONS:
        for component in STAGES:
            path = FINAL_OUTPUTS[f"{component}_{horizon}"]
            if path.exists() and not is_resumable_metadata(read_model_metadata(path), component, horizon):
                raise FileExistsError(f"Existing two-stage model is not a matching resumable output: {path}")


def train_component(
    pipeline,
    dataset_path: Path,
    config: dict,
    bucket_counts: dict,
    category_maps: dict,
    run_stage: str,
    component: str,
    horizon: str,
    checkpoint: Path,
    logger: logging.Logger,
) -> dict:
    guard = pipeline.MemoryGuard(f"two-stage-{run_stage}-{component}-{horizon}", logger)
    settings = config["lightgbm_training"]
    features = horizon_features(config, horizon)
    if features != lightgbm_horizon_features(config, horizon):
        raise RuntimeError("Two-stage feature scope differs from LightGBM V2")
    if component == "classifier":
        base_train_rates = {str(k): float(v) for k, v in settings["sampling_rates"].items()}
        base_valid_rates = {str(k): float(v) for k, v in settings["valid_sampling_rates"].items()}
        params = classifier_params(config)
    else:
        base_train_rates = regressor_sampling_rates()
        base_valid_rates = regressor_sampling_rates()
        params = regressor_params(config)
    if run_stage == "smoke":
        train_rates = pipeline.scaled_rates(base_train_rates, bucket_counts["train"][horizon], 100_000)
        valid_rates = pipeline.scaled_rates(base_valid_rates, bucket_counts["valid"][horizon], 30_000)
        max_rows = {"train": 100_000, "valid": 30_000}
        rounds, early_stopping = 300, 40
    else:
        train_rates, valid_rates = base_train_rates, base_valid_rates
        max_rows = {"train": None, "valid": None}
        rounds, early_stopping = settings["num_boost_round"], settings["early_stopping_rounds"]
    sampled_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        dataset_path, horizon, {"train": train_rates, "valid": valid_rates}, category_maps,
        features, config["time_feature_settings"]["base_month"], guard, logger, max_rows,
        None, component == "classifier" and run_stage == "formal",
    )
    sampling_seconds = time.perf_counter() - sampled_started
    raw_train = np.expm1(samples["train"]["y"].astype("float64"))
    raw_valid = np.expm1(samples["valid"]["y"].astype("float64"))
    if component == "classifier":
        train_y, valid_y = binary_target(raw_train).astype("float32"), binary_target(raw_valid).astype("float32")
    else:
        if (raw_train <= 0).any() or (raw_valid <= 0).any():
            raise RuntimeError("Tweedie component received non-positive target")
        train_y, valid_y = raw_train.astype("float32"), raw_valid.astype("float32")
    logger.info(
        "Training %s %s %s train=%s valid=%s features=%d",
        run_stage, component, horizon, f"{len(train_y):,}", f"{len(valid_y):,}", len(features),
    )
    estimated = int((len(train_y) + len(valid_y)) * (len(features) * 3 + 64))
    guard.ensure_capacity(estimated, "before_dataset_construction")
    started = time.perf_counter()
    booster, evaluations = train_booster(
        samples["train"]["x"], train_y, samples["train"]["weight"],
        samples["valid"]["x"], valid_y, samples["valid"]["weight"],
        categorical_features=pipeline.CATEGORY_FEATURES, params=params,
        num_boost_round=rounds, early_stopping_rounds=early_stopping,
        callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)],
        construction_check=lambda label: guard.check(label),
    )
    training_seconds = time.perf_counter() - started
    check_x = samples["valid"]["x"].iloc[:1000]
    prediction = booster.predict(check_x, num_iteration=booster.best_iteration)
    if component == "classifier":
        prediction = np.clip(prediction, 0.0, 1.0)
    else:
        prediction = np.clip(prediction, 0.0, None)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid {component} {horizon} predictions")
    metadata = {
        "experiment": "two_stage_lightgbm", "run_stage": run_stage, "component": component,
        "horizon": horizon, "feature_names": features, "categorical_features": pipeline.CATEGORY_FEATURES,
        "category_maps": category_maps, "time_base": config["time_feature_settings"]["base_month"],
        "params": params, "best_iteration": int(booster.best_iteration),
        "sampling_rates": train_rates, "valid_sampling_rates": valid_rates,
        "target": "has_sales" if component == "classifier" else "raw_positive_qty",
        "lightgbm_version": lgb.__version__, "python_version": platform.python_version(),
    }
    save_model_bundle(booster, checkpoint, metadata)
    loaded, loaded_metadata = load_model_bundle(checkpoint)
    reloaded = loaded.predict(check_x, num_iteration=loaded_metadata["best_iteration"])
    if component == "classifier":
        reloaded = np.clip(reloaded, 0.0, 1.0)
    else:
        reloaded = np.clip(reloaded, 0.0, None)
    np.testing.assert_allclose(prediction, reloaded, rtol=1e-7, atol=1e-8)
    result = {
        "run_stage": run_stage, "component": component, "horizon": horizon,
        "train_rows": len(train_y), "valid_rows": len(valid_y), "feature_count": len(features),
        "best_iteration": int(booster.best_iteration), "sampling_seconds": sampling_seconds,
        "training_seconds": training_seconds, "peak_memory_bytes": guard.peak,
        "memory_limit_bytes": guard.limit, "model_size_bytes": checkpoint.stat().st_size,
        "params": params, "evaluations": evaluations,
        "train_distribution": samples["train"]["distribution"],
        "valid_distribution": samples["valid"]["distribution"],
        "reload_consistent": True,
    }
    del samples, raw_train, raw_valid, train_y, valid_y, booster, loaded, check_x, prediction, reloaded
    gc.collect()
    guard.check("component_released")
    logger.info(
        "Completed %s %s %s best_iteration=%d sampling=%.1fs training=%.1fs peak=%.2fGiB",
        run_stage, component, horizon, result["best_iteration"], sampling_seconds, training_seconds,
        result["peak_memory_bytes"] / 2**30,
    )
    return result


def evaluation_columns(config: dict) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    columns = ["month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m"]
    for horizon in HORIZONS:
        columns.extend(column for column in horizon_features(config, horizon) if column not in runtime)
    return list(dict.fromkeys(columns))


def evaluate_full(
    pipeline,
    dataset_path: Path,
    bundles: dict[str, dict[str, tuple]],
    config: dict,
    output_path: Path,
    logger: logging.Logger,
) -> tuple[dict, float, int]:
    guard = pipeline.MemoryGuard("two-stage-full-evaluation", logger)
    binary_acc = {s: {h: StreamingBinaryMetrics() for h in HORIZONS} for s in ("valid", "test")}
    reg_acc = {s: {h: LongTailStreamingMetrics() for h in HORIZONS} for s in ("valid", "test")}
    combined_acc = {s: {h: LongTailStreamingMetrics() for h in HORIZONS} for s in ("valid", "test")}
    columns = evaluation_columns(config)
    temporary = output_path.with_suffix(".partial.parquet")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    parquet = pq.ParquetFile(dataset_path)
    started = time.perf_counter()
    try:
        for row_group in range(parquet.num_row_groups):
            split_table = parquet.read_row_group(row_group, columns=["split"])
            split_all = split_table.column("split").to_pandas().astype(str).to_numpy()
            selected = np.flatnonzero(np.isin(split_all, ["valid", "test"]))
            del split_table, split_all
            if not selected.size:
                continue
            frame = parquet.read_row_group(row_group, columns=columns).take(pa.array(selected)).to_pandas()
            split_values = frame["split"].astype(str).to_numpy()
            predictions = {}
            targets = {}
            for horizon in HORIZONS:
                classifier, classifier_meta = bundles[horizon]["classifier"]
                regressor, regressor_meta = bundles[horizon]["regressor"]
                if classifier_meta["feature_names"] != regressor_meta["feature_names"]:
                    raise RuntimeError(f"Classifier/regressor feature mismatch for {horizon}")
                x = pipeline.prepare_evaluation_frame(frame, classifier_meta)
                p_sale = np.clip(classifier.predict(x, num_iteration=classifier_meta["best_iteration"]), 0.0, 1.0)
                conditional = np.clip(regressor.predict(x, num_iteration=regressor_meta["best_iteration"]), 0.0, None)
                combined = combine_predictions(p_sale, conditional)
                if not all(np.isfinite(value).all() for value in (p_sale, conditional, combined)):
                    raise RuntimeError(f"Invalid two-stage {horizon} prediction at row group {row_group}")
                target = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                predictions[horizon] = {"p_sale": p_sale, "conditional": conditional, "combined": combined}
                targets[horizon] = target
                for split in ("valid", "test"):
                    mask = split_values == split
                    if not mask.any():
                        continue
                    binary_acc[split][horizon].update(target[mask], p_sale[mask])
                    positive = mask & (target > 0)
                    if positive.any():
                        reg_acc[split][horizon].update(target[positive], conditional[positive])
                    combined_acc[split][horizon].update(target[mask], combined[mask])
                del x
            test_mask = split_values == "test"
            if test_mask.any():
                output = pd.DataFrame({
                    "month": frame.loc[test_mask, "month"].astype(str),
                    "site_no": frame.loc[test_mask, "site_no"].astype(str),
                    "item_id": frame.loc[test_mask, "item_id"].astype(str),
                    "target_qty_1m": targets["1m"][test_mask].astype("float32"),
                    "target_qty_2m": targets["2m"][test_mask].astype("float32"),
                    "p_sale_1m": predictions["1m"]["p_sale"][test_mask].astype("float32"),
                    "conditional_qty_1m": predictions["1m"]["conditional"][test_mask].astype("float32"),
                    "two_stage_pred_1m": predictions["1m"]["combined"][test_mask].astype("float32"),
                    "p_sale_2m": predictions["2m"]["p_sale"][test_mask].astype("float32"),
                    "conditional_qty_2m": predictions["2m"]["conditional"][test_mask].astype("float32"),
                    "two_stage_pred_2m": predictions["2m"]["combined"][test_mask].astype("float32"),
                })
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=["month", "site_no"])
                writer.write_table(table, row_group_size=250_000)
                del output, table
            del frame, predictions, targets
            gc.collect()
            guard.check("evaluation", row_group)
            if (row_group + 1) % 10 == 0 or row_group + 1 == parquet.num_row_groups:
                logger.info("Two-stage full evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        parquet.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No test prediction rows were written")
    temporary.replace(output_path)
    classification = {s: {} for s in ("valid", "test")}
    for horizon in HORIZONS:
        valid_initial = binary_acc["valid"][horizon].compute([0.5])
        threshold = valid_initial["best_f1_threshold"]
        classification["valid"][horizon] = binary_acc["valid"][horizon].compute([0.5, threshold])
        classification["test"][horizon] = binary_acc["test"][horizon].compute([0.5, threshold])
        classification["valid"][horizon]["diagnostic_threshold_source"] = "valid_best_f1"
        classification["test"][horizon]["diagnostic_threshold_source"] = "valid_best_f1"
        classification["test"][horizon]["valid_best_f1_threshold"] = threshold
    return {
        "classification": classification,
        "conditional_regression": {
            s: {h: reg_acc[s][h].compute() for h in HORIZONS} for s in ("valid", "test")
        },
        "combined": {
            s: {h: combined_acc[s][h].compute() for h in HORIZONS} for s in ("valid", "test")
        },
    }, time.perf_counter() - started, guard.peak


def load_lightgbm_v2_metrics() -> dict:
    candidates = sorted((ROOT / "logs/lightgbm/v2_logl2").glob("formal_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("LightGBM V2 formal metrics are missing")
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return {
        split: {h: payload[f"{split}_metrics"][h]["v2_log_l2"] for h in HORIZONS}
        for split in ("valid", "test")
    }


def write_importance(bundles: dict[str, dict[str, tuple]], path: Path) -> None:
    parts = []
    for horizon in HORIZONS:
        for component in STAGES:
            booster, metadata = bundles[horizon][component]
            frame = pd.DataFrame({
                "horizon": horizon, "component": component, "feature": metadata["feature_names"],
                "gain_importance": booster.feature_importance(importance_type="gain"),
                "split_importance": booster.feature_importance(importance_type="split"),
            }).sort_values("gain_importance", ascending=False, ignore_index=True)
            frame["rank"] = np.arange(1, len(frame) + 1)
            parts.append(frame)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def render_report(summary: dict, path: Path) -> None:
    lines = [
        "# 两阶段模型报告", "", "## 模型定义", "",
        "分类器使用 LightGBM binary 与逆采样概率权重；条件回归器仅使用正销量样本，采用 Tweedie power=1.4 直接拟合原始销量。",
        "主预测固定为 `p_sale * conditional_qty`，valid 最佳 F1 阈值只用于分类诊断，不参与主预测。PR-AUC/ROC-AUC 使用 65,536 桶流式近似。", "",
        "## 训练资源", "",
        "| 阶段 | 目标 | 组件 | train | valid | 最佳轮次 | 采样秒数 | 训练秒数 | 峰值 RAM GiB | 模型 MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_stage in ("smoke", "formal"):
        for horizon in HORIZONS:
            for component in STAGES:
                value = summary["stages"][run_stage][horizon][component]
                lines.append(
                    f"| {run_stage} | {horizon} | {component} | {value['train_rows']:,} | {value['valid_rows']:,} | "
                    f"{value['best_iteration']} | {value['sampling_seconds']:.1f} | {value['training_seconds']:.1f} | "
                    f"{value['peak_memory_bytes']/2**30:.2f} | {value['model_size_bytes']/2**20:.2f} |"
                )
    lines.extend(["", "## 分类器完整评价", ""])
    for split in ("valid", "test"):
        lines.extend([
            f"### {split}", "",
            "| 目标 | PR-AUC | ROC-AUC | Logloss | 阈值 | Precision | Recall | F1 | 真实正例率 | 预测正例率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for horizon in HORIZONS:
            value = summary["evaluation"]["classification"][split][horizon]
            diagnostic = value.get("valid_best_f1_threshold", value["best_f1_threshold"])
            for label, threshold in (("0.5", 0.5), ("valid-best-F1", diagnostic)):
                key = f"{threshold:g}"
                metric = value["threshold_metrics"][key]
                lines.append(
                    f"| {horizon} | {fmt(value['pr_auc'])} | {fmt(value['roc_auc'])} | {fmt(value['logloss'])} | "
                    f"{label}={threshold:.5f} | {fmt(metric['precision'])} | {fmt(metric['recall'])} | {fmt(metric['f1'])} | "
                    f"{fmt(value['positive_rate'])} | {fmt(metric['predicted_positive_rate'])} |"
                )
    lines.extend(["", "## 条件回归器：真实正销量样本", ""])
    for split in ("valid", "test"):
        lines.extend([f"### {split}", "", "| 目标 | 分层 | 样本量 | MAE | RMSE | WAPE | 真实总量 | 预测总量 | 总量偏差 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for horizon in HORIZONS:
            values = summary["evaluation"]["conditional_regression"][split][horizon]
            for segment in REGRESSION_SEGMENTS:
                value = values[segment]
                lines.append(
                    f"| {horizon} | {segment} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
                    f"{fmt(value['wape'])}% | {fmt(value['target_sum'],0)} | {fmt(value['prediction_sum'],0)} | "
                    f"{fmt(value['total_bias_rate'],2)}% |"
                )
    lines.extend(["", "## 最终组合预测与 LightGBM V2 局部比较", ""])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            lines.extend([
                f"### {split} / {horizon}", "",
                "| 模型 | 分层 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for model_name, values in (
                ("LightGBM V2", summary["lightgbm_v2_metrics"][split][horizon]),
                ("two_stage", summary["evaluation"]["combined"][split][horizon]),
            ):
                for segment in SEGMENTS:
                    value = values[segment]
                    lines.append(
                        f"| {model_name} | {segment} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
                        f"{fmt(value['smape'])}% | {fmt(value['wape'])}% | {fmt(value['target_sum'],0)} | "
                        f"{fmt(value['prediction_sum'],0)} | {fmt(value['total_bias_rate'],2)}% |"
                    )
                zero = values["0"]
                lines.append(
                    f"\n{model_name} 零销量诊断：平均预测 {fmt(zero.get('mean_prediction'))}，>0.5 "
                    f"{fmt(100*zero.get('prediction_gt_0_5_rate', math.nan),2)}%，>1 "
                    f"{fmt(100*zero.get('prediction_gt_1_rate', math.nan),2)}%。\n"
                )
    lines.extend(["", "## 局部结论", ""])
    for horizon in HORIZONS:
        two = summary["evaluation"]["combined"]["test"][horizon]
        lgbm = summary["lightgbm_v2_metrics"]["test"][horizon]
        mae_gain = 100 * (lgbm["overall"]["mae"] - two["overall"]["mae"]) / lgbm["overall"]["mae"]
        lines.append(
            f"- {horizon}：相对 LightGBM V2 的 overall MAE 变化为 {mae_gain:+.2f}%；two-stage 总量偏差 "
            f"{two['overall']['total_bias_rate']:.2f}%，LightGBM V2 为 {lgbm['overall']['total_bias_rate']:.2f}%。"
        )
        lines.append(
            f"  nonzero/5-20/20+ WAPE：two-stage {two['nonzero']['wape']:.2f}%/{two['5-20']['wape']:.2f}%/"
            f"{two['20+']['wape']:.2f}%，LightGBM V2 {lgbm['nonzero']['wape']:.2f}%/{lgbm['5-20']['wape']:.2f}%/"
            f"{lgbm['20+']['wape']:.2f}%。"
        )
        zero = two["0"]
        lines.append(
            f"  零销量平均预测 {zero['mean_prediction']:.4f}，>0.5 / >1 比例 "
            f"{100*zero['prediction_gt_0_5_rate']:.2f}% / {100*zero['prediction_gt_1_rate']:.2f}%。"
        )
    lines.extend([
        "", "## 工程审计", "",
        "- 两阶段与 LightGBM V2 使用相同特征信息范围；所有类别映射只在 train 拟合，未知类别为 -1。",
        "- 分类器仅使用归一化逆采样概率权重，未叠加 `scale_pos_weight` 或 `is_unbalance`。",
        "- test 未参与早停、参数选择或阈值选择；诊断阈值只来自完整 valid。",
        f"- 完整评价耗时 {summary['evaluation_seconds']:.1f} 秒，峰值 RAM {summary['evaluation_peak_bytes']/2**30:.2f} GiB；测试预测文件 {summary['prediction_size_bytes']/2**20:.2f} MiB。",
        "- 本阶段未训练或调整其他模型，也未生成六模型综合比较报告。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def update_storage_manifest(paths: list[Path]) -> None:
    manifest = ROOT / "reports/storage_manifest.md"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else "# Storage Manifest\n"
    marker = "## Two-Stage 正式产物"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    lines = [marker, "", "| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |", "|---|---|---:|---|---|---|"]
    for artifact in paths:
        if not artifact.exists():
            continue
        relative = artifact.relative_to(ROOT).as_posix()
        if artifact.suffix == ".txt" and "models/final" in relative:
            purpose, safe = "正式模型", "否"
        elif artifact.suffix == ".parquet":
            purpose, safe = "正式测试预测", "是（可由正式模型重建）"
        else:
            purpose, safe = "代码、报告或特征重要性", "否"
        lines.append(f"| `{relative}` | {purpose} | {artifact.stat().st_size/2**20:.2f} MiB | 是 | {safe} | `scripts/15_train_two_stage.py` |")
    manifest.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_smoke_summary() -> dict:
    path = ROOT / "logs/two_stage/smoke_summary.json"
    if not path.exists():
        raise RuntimeError("Formal training requires a successful two-stage smoke test")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wenxuan two-stage LightGBM models.")
    parser.add_argument("--stage", choices=("all", "smoke", "formal", "evaluate"), default="all")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logging(run_id)
    pipeline = pipeline_module()
    config = load_feature_config()
    dataset_path = project_path(args.dataset or config["dataset"])
    checkpoint_dir = ROOT / "models/checkpoints/two_stage"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/cache/two_stage").mkdir(parents=True, exist_ok=True)
    logger.info(
        "Python=%s LightGBM=%s dataset=%s RAM_available=%.2f GiB",
        platform.python_version(), lgb.__version__, dataset_path, pipeline.physical_memory()[1] / 2**30,
    )
    if args.stage in ("all", "formal"):
        ensure_formal_outputs_resumable()
    bucket_counts, category_maps = pipeline.load_or_build_scan_metadata(dataset_path, logger)
    summary = {
        "stages": {}, "python_version": platform.python_version(), "lightgbm_version": lgb.__version__,
        "log_path": str(log_path),
    }

    if args.stage == "evaluate":
        bundles = {
            h: {component: load_model_bundle(FINAL_OUTPUTS[f"{component}_{h}"]) for component in STAGES}
            for h in HORIZONS
        }
        summary["stages"] = {"smoke": load_smoke_summary(), "formal": json.loads((ROOT / "logs/two_stage/formal_training_summary.json").read_text(encoding="utf-8"))}
        evaluation, elapsed, peak = evaluate_full(pipeline, dataset_path, bundles, config, FINAL_OUTPUTS["predictions"], logger)
        summary.update({
            "evaluation": evaluation, "lightgbm_v2_metrics": load_lightgbm_v2_metrics(),
            "evaluation_seconds": elapsed, "evaluation_peak_bytes": peak,
            "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size,
        })
        write_importance(bundles, FINAL_OUTPUTS["importance"])
        render_report(summary, FINAL_OUTPUTS["report"])
        save_json(ROOT / "logs/two_stage/formal_run_summary.json", summary)
        update_storage_manifest([ROOT / "src/models/two_stage_model.py", Path(__file__), *FINAL_OUTPUTS.values()])
        return

    requested = ("smoke", "formal") if args.stage == "all" else (args.stage,)
    for run_stage in requested:
        if run_stage == "formal" and "smoke" not in summary["stages"]:
            summary["stages"]["smoke"] = load_smoke_summary()
        summary["stages"][run_stage] = {}
        for horizon in HORIZONS:
            summary["stages"][run_stage][horizon] = {}
            for component in STAGES:
                checkpoint = checkpoint_dir / f"{run_stage}_{component}_{horizon}.txt"
                destination = FINAL_OUTPUTS[f"{component}_{horizon}"] if run_stage == "formal" else None
                if destination is not None and destination.exists():
                    result = recover_component_result(destination, component, horizon)
                    logger.info("Resuming after completed formal %s %s; model left unchanged", component, horizon)
                else:
                    result = train_component(
                        pipeline, dataset_path, config, bucket_counts, category_maps,
                        run_stage, component, horizon, checkpoint, logger,
                    )
                    if destination is not None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(checkpoint, destination)
                        result["model_size_bytes"] = destination.stat().st_size
                        save_json(component_summary_path(component, horizon), result)
                summary["stages"][run_stage][horizon][component] = result
                gc.collect()
        if run_stage == "smoke":
            save_json(ROOT / "logs/two_stage/smoke_summary.json", summary["stages"][run_stage])
            logger.info("Two-stage smoke passed for all four models")
        else:
            save_json(ROOT / "logs/two_stage/formal_training_summary.json", summary["stages"][run_stage])
    if "formal" not in requested:
        return
    # Keep no completed boosters resident while building the next training matrix.
    # Load all four only after formal training has released its sample frames.
    formal_bundles = {
        h: {component: load_model_bundle(FINAL_OUTPUTS[f"{component}_{h}"]) for component in STAGES}
        for h in HORIZONS
    }
    evaluation, elapsed, peak = evaluate_full(
        pipeline, dataset_path, formal_bundles, config, FINAL_OUTPUTS["predictions"], logger,
    )
    summary.update({
        "evaluation": evaluation, "lightgbm_v2_metrics": load_lightgbm_v2_metrics(),
        "evaluation_seconds": elapsed, "evaluation_peak_bytes": peak,
        "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size,
    })
    write_importance(formal_bundles, FINAL_OUTPUTS["importance"])
    render_report(summary, FINAL_OUTPUTS["report"])
    save_json(ROOT / "logs/two_stage/formal_run_summary.json", summary)
    update_storage_manifest([ROOT / "src/models/two_stage_model.py", Path(__file__), *FINAL_OUTPUTS.values()])
    for transient in checkpoint_dir.glob("*.txt"):
        transient.unlink()
    for transient in (ROOT / "data/cache/two_stage").glob("*"):
        if transient.is_file():
            transient.unlink()
    logger.info("Two-stage stage completed successfully")


if __name__ == "__main__":
    main()
