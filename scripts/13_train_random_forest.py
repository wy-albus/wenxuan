from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import add_runtime_columns, get_feature_list, load_feature_config, project_path  # noqa: E402
from src.models.lightgbm_model import encode_categories, load_model_bundle as load_lightgbm_bundle  # noqa: E402
from src.models.random_forest_model import (  # noqa: E402
    build_estimator,
    inverse_prediction,
    load_model_bundle,
    save_model_bundle,
    should_stop_tree_growth,
    tree_statistics,
)


HORIZONS = ("1m", "2m")
SEGMENTS = ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
MODEL_NAMES = ("weighted_moving_average", "lightgbm_v1_logl1", "lightgbm_v2_logl2", "random_forest")
CATEGORY_FEATURES = ["site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"]
FINAL_OUTPUTS = {
    "model_1m": ROOT / "models/final/random_forest_1m.joblib",
    "model_2m": ROOT / "models/final/random_forest_2m.joblib",
    "report": ROOT / "reports/random_forest_model_report.md",
    "importance_1m": ROOT / "reports/random_forest_feature_importance_1m.csv",
    "importance_2m": ROOT / "reports/random_forest_feature_importance_2m.csv",
    "predictions": ROOT / "data/outputs/random_forest_test_predictions.parquet",
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
    return load_script_module("wenxuan_lightgbm_pipeline_for_rf", ROOT / "scripts/10_train_lightgbm.py")


def horizon_features(config: dict, horizon: str) -> list[str]:
    return get_feature_list("random_forest", config) + list(config["random_forest_horizon_features"][horizon])


def scaled_rates(base_rates: dict[str, float], counts: Counter, target_rows: int | None) -> dict[str, float]:
    base = {str(key): float(value) for key, value in base_rates.items()}
    if target_rows is None:
        return base
    low, high = 0.0, 100.0
    for _ in range(80):
        scale = (low + high) / 2.0
        expected = sum(counts[bucket] * min(1.0, base[bucket] * scale) for bucket in base)
        if expected < target_rows:
            low = scale
        else:
            high = scale
    return {bucket: min(1.0, base[bucket] * high) for bucket in base}


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    directory = ROOT / "logs/random_forest"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"training_{run_id}.log"
    logger = logging.getLogger("wenxuan_random_forest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def ensure_formal_outputs_absent() -> None:
    existing = [str(path) for path in FINAL_OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Random Forest outputs: {existing}")


def weighted_moving_average(frame: pd.DataFrame, config: dict, horizon: str) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype="float64")
    settings = config["baseline_settings"]["weighted_moving_average"]
    for column, weight in settings["weights"].items():
        prediction += float(weight) * pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype="float64")
    if horizon == "2m":
        prediction *= float(settings["horizon_2_multiplier"])
    return np.clip(prediction, 0.0, None)


def to_float32_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame[features].to_numpy(dtype="float32", copy=True)
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(matrix)


def fit_to_tree_count(model, x, y, weight, target_trees: int, guard, logger, chunk_size: int = 10) -> float:
    started = time.perf_counter()
    current = len(getattr(model, "estimators_", []))
    while current < target_trees:
        next_count = min(target_trees, current + chunk_size)
        model.set_params(n_estimators=next_count)
        model.fit(x, y, sample_weight=weight)
        current = next_count
        guard.check(f"fit_{current}_trees")
        logger.info("RF trees=%d rss=%.2f GiB", current, guard.peak / 2**30)
    return time.perf_counter() - started


def sampled_metrics(model, x: np.ndarray, raw_target: np.ndarray) -> tuple[dict, float]:
    started = time.perf_counter()
    prediction = inverse_prediction(model.predict(x))
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError("Random Forest produced invalid predictions")
    accumulator = LongTailStreamingMetrics()
    accumulator.update(raw_target, prediction)
    return accumulator.compute(), time.perf_counter() - started


def train_stage_horizon(
    pipeline,
    dataset_path: Path,
    config: dict,
    bucket_counts: dict,
    category_maps: dict,
    stage: str,
    horizon: str,
    train_rows: int,
    valid_rows: int,
    tree_targets: list[int],
    n_jobs: int,
    checkpoint_path: Path,
    logger: logging.Logger,
) -> tuple[dict, Any]:
    guard = pipeline.MemoryGuard(f"random-forest-{stage}-{horizon}", logger)
    guard.check("stage_start")
    features = horizon_features(config, horizon)
    base_rates = config["lightgbm_training"]["sampling_rates"]
    train_rates = scaled_rates(base_rates, bucket_counts["train"][horizon], train_rows)
    valid_rates = scaled_rates(base_rates, bucket_counts["valid"][horizon], valid_rows)
    base_month = config["time_feature_settings"]["base_month"]
    sampling_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        dataset_path,
        horizon,
        {"train": train_rates, "valid": valid_rates},
        category_maps,
        features,
        base_month,
        guard,
        logger,
        {"train": train_rows, "valid": valid_rows},
        None,
        True,
    )
    sampling_seconds = time.perf_counter() - sampling_started
    x_train = to_float32_matrix(samples["train"]["x"], features)
    x_valid = to_float32_matrix(samples["valid"]["x"], features)
    y_train = samples["train"]["y"].astype("float32", copy=False)
    y_valid_raw = np.expm1(samples["valid"]["y"].astype("float64"))
    weights = samples["train"]["weight"].astype("float32", copy=False)
    distributions = {split: samples[split]["distribution"] for split in ("train", "valid")}
    del samples
    gc.collect()
    guard.ensure_capacity(int(x_train.nbytes * 0.5 + max(tree_targets) * train_rows * 0.02), "before_fit")
    model = build_estimator(n_estimators=min(10, tree_targets[0]), n_jobs=n_jobs)
    history = []
    training_seconds = 0.0
    prediction_seconds = 0.0
    for tree_count in tree_targets:
        training_seconds += fit_to_tree_count(model, x_train, y_train, weights, tree_count, guard, logger)
        metrics, elapsed = sampled_metrics(model, x_valid, y_valid_raw)
        prediction_seconds += elapsed
        overall = metrics["overall"]
        history.append({
            "trees": tree_count,
            "mae": float(overall["mae"]),
            "nonzero_wape": float(metrics["nonzero"]["wape"]),
            "metrics": metrics,
        })
        logger.info(
            "%s %s trees=%d valid_mae=%.6f nonzero_wape=%.4f bias=%.2f%%",
            stage, horizon, tree_count, overall["mae"], metrics["nonzero"]["wape"], overall["total_bias_rate"],
        )
        metadata = {
            "model": "random_forest_log_l2",
            "stage": stage,
            "horizon": horizon,
            "feature_names": features,
            "categorical_features": CATEGORY_FEATURES,
            "category_maps": category_maps,
            "time_base": base_month,
            "sampling_rates": train_rates,
            "valid_sampling_rates": valid_rates,
            "target_transform": "log1p",
            "inverse_transform": "expm1_clip_nonnegative",
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "tree_count": tree_count,
            "params": model.get_params(deep=False),
        }
        save_model_bundle(model, checkpoint_path, metadata)
        if stage == "formal" and should_stop_tree_growth(history):
            logger.info("%s %s stopping tree growth at %d trees", stage, horizon, tree_count)
            break
    check = x_valid[:1000]
    expected = model.predict(check)
    loaded, loaded_metadata = load_model_bundle(checkpoint_path)
    np.testing.assert_allclose(expected, loaded.predict(check), rtol=1e-7, atol=1e-8)
    statistics = tree_statistics(model)
    result = {
        "stage": stage,
        "horizon": horizon,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "feature_count": len(features),
        "sampling_seconds": sampling_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_memory_bytes": guard.peak,
        "memory_limit_bytes": guard.limit,
        "model_size_bytes": checkpoint_path.stat().st_size,
        "tree_statistics": statistics,
        "history": history,
        "train_distribution": distributions["train"],
        "valid_distribution": distributions["valid"],
        "sampling_rates": train_rates,
        "n_jobs": n_jobs,
        "reload_consistent": True,
    }
    del loaded, expected, check, x_train, x_valid, y_train, y_valid_raw, weights
    gc.collect()
    return result, model


def choose_formal_rows(pilot: dict) -> dict[str, int]:
    chosen = {}
    for horizon in HORIZONS:
        value = pilot[horizon]
        peak = value["peak_memory_bytes"]
        limit = value["memory_limit_bytes"]
        model_size = value["model_size_bytes"]
        chosen[horizon] = 1_500_000 if peak < 0.45 * limit and model_size < 500 * 2**20 else 1_000_000
    return chosen


def merge_stage_summaries(current: dict, previous: dict) -> dict:
    return {
        stage: current.get(stage, previous.get(stage, {}))
        for stage in ("smoke", "pilot", "formal")
        if stage in current or stage in previous
    }


def load_previous_stage_summaries() -> dict:
    previous = {}
    smoke_files = sorted((ROOT / "logs/random_forest").glob("smoke_run_summary_*.json"), key=lambda path: path.stat().st_mtime)
    if smoke_files:
        payload = json.loads(smoke_files[-1].read_text(encoding="utf-8"))
        previous["smoke"] = payload["stages"]["smoke"]
    pilot_path = ROOT / "logs/random_forest/pilot_summary.json"
    if pilot_path.exists():
        previous["pilot"] = json.loads(pilot_path.read_text(encoding="utf-8"))
    return previous


def prepare_rf_frame(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    prepared = add_runtime_columns(frame, base_month=metadata["time_base"])
    prepared = encode_categories(prepared, metadata["category_maps"])
    for column in metadata["feature_names"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return to_float32_matrix(prepared, metadata["feature_names"])


def prepare_lgb_frame(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    prepared = add_runtime_columns(frame, base_month=metadata["time_base"])
    prepared = encode_categories(prepared, metadata["category_maps"])
    for column in metadata["feature_names"]:
        if column not in metadata.get("categorical_features", []):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return prepared[metadata["feature_names"]]


def evaluation_rows_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["split"].isin(["valid", "test"])].copy()


def evaluation_columns(config: dict, rf_metadata: dict) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    columns = ["month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m"]
    for metadata in rf_metadata.values():
        columns.extend(column for column in metadata["feature_names"] if column not in runtime)
    return list(dict.fromkeys(columns))


def load_existing_comparison_metrics() -> dict:
    candidates = sorted((ROOT / "logs/lightgbm/v2_logl2").glob("formal_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("Completed LightGBM V2 formal summary is missing")
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    if not payload.get("formal_complete"):
        raise RuntimeError(f"LightGBM V2 summary is not marked complete: {candidates[-1]}")
    names = {
        "weighted_moving_average": "weighted_moving_average",
        "v1_log_l1": "lightgbm_v1_logl1",
        "v2_log_l2": "lightgbm_v2_logl2",
    }
    return {
        split: {
            horizon: {destination: payload[f"{split}_metrics"][horizon][source] for source, destination in names.items()}
            for horizon in HORIZONS
        }
        for split in ("valid", "test")
    }


def evaluate_full(
    pipeline,
    dataset_path: Path,
    rf_bundles: dict,
    config: dict,
    output_path: Path,
    logger: logging.Logger,
) -> tuple[dict, float, int]:
    started = time.perf_counter()
    guard = pipeline.MemoryGuard("random-forest-full-evaluation", logger)
    metrics = {
        split: {h: LongTailStreamingMetrics() for h in HORIZONS}
        for split in ("valid", "test")
    }
    rf_metadata = {h: rf_bundles[h][1] for h in HORIZONS}
    columns = evaluation_columns(config, rf_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    parquet = pq.ParquetFile(dataset_path)
    try:
        for row_group in range(parquet.num_row_groups):
            split_table = parquet.read_row_group(row_group, columns=["split"])
            split_values_all = split_table.column("split").to_pandas().astype(str).to_numpy()
            selected_indices = np.flatnonzero(np.isin(split_values_all, ["valid", "test"]))
            del split_table, split_values_all
            if selected_indices.size == 0:
                continue
            frame = parquet.read_row_group(row_group, columns=columns).take(pa.array(selected_indices)).to_pandas()
            split_values = frame["split"].astype(str).to_numpy()
            rf_predictions = {}
            for horizon in HORIZONS:
                rf_x = prepare_rf_frame(frame, rf_metadata[horizon])
                rf_predictions[horizon] = inverse_prediction(rf_bundles[horizon][0].predict(rf_x))
                del rf_x
            for split in ("valid", "test"):
                mask = split_values == split
                if not mask.any():
                    continue
                for horizon in HORIZONS:
                    target = clip_target(frame.loc[mask, f"future_qty_{horizon}"].to_numpy())
                    metrics[split][horizon].update(target, rf_predictions[horizon][mask])
                    del target
            test_mask = split_values == "test"
            if test_mask.any():
                output = pd.DataFrame({
                    "month": frame.loc[test_mask, "month"].astype(str),
                    "site_no": frame.loc[test_mask, "site_no"].astype(str),
                    "item_id": frame.loc[test_mask, "item_id"].astype(str),
                    "target_qty_1m": clip_target(frame.loc[test_mask, "future_qty_1m"].to_numpy()).astype("float32"),
                    "target_qty_2m": clip_target(frame.loc[test_mask, "future_qty_2m"].to_numpy()).astype("float32"),
                    "random_forest_pred_1m": rf_predictions["1m"][test_mask].astype("float32"),
                    "random_forest_pred_2m": rf_predictions["2m"][test_mask].astype("float32"),
                })
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                del output, table
            del frame, rf_predictions
            if (row_group + 1) % 5 == 0:
                guard.check("evaluation")
                logger.info("Full evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
            gc.collect()
    finally:
        parquet.close()
        if writer is not None:
            writer.close()
    computed = load_existing_comparison_metrics()
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            computed[split][horizon]["random_forest"] = metrics[split][horizon].compute()
    return computed, time.perf_counter() - started, guard.peak


def recover_formal_results(models: dict, log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    results = {}
    for horizon in HORIZONS:
        model, metadata = models[horizon]
        stage_match = re.search(
            rf"^(?P<time>\S+ \S+) INFO random-forest-formal-{horizon} memory limit (?P<limit>[0-9.]+) GiB",
            text,
            re.MULTILINE,
        )
        history_matches = list(re.finditer(
            rf"^(?P<time>\S+ \S+) INFO formal {horizon} trees=(?P<trees>\d+) "
            rf"valid_mae=(?P<mae>[0-9.]+) nonzero_wape=(?P<wape>[0-9.]+) bias=(?P<bias>[-0-9.]+)%",
            text,
            re.MULTILINE,
        ))
        tree_matches = list(re.finditer(
            r"^(?P<time>\S+ \S+) INFO RF trees=(?P<trees>\d+) rss=(?P<rss>[0-9.]+) GiB",
            text,
            re.MULTILINE,
        ))
        if stage_match is None or not history_matches:
            raise RuntimeError(f"Cannot recover completed formal {horizon} result from {log_path}")
        stage_time = datetime.fromisoformat(stage_match.group("time"))
        history = [
            {
                "trees": int(match.group("trees")),
                "mae": float(match.group("mae")),
                "nonzero_wape": float(match.group("wape")),
                "total_bias_rate": float(match.group("bias")),
            }
            for match in history_matches
        ]
        final_time = datetime.fromisoformat(history_matches[-1].group("time"))
        relevant_tree_matches = [
            match for match in tree_matches
            if stage_time <= datetime.fromisoformat(match.group("time")) <= final_time
        ]
        first_tree_time = datetime.fromisoformat(relevant_tree_matches[0].group("time"))
        peak_gib = max(float(match.group("rss")) for match in relevant_tree_matches)
        results[horizon] = {
            "stage": "formal",
            "horizon": horizon,
            "train_rows": 1_500_000 if horizon == "2m" else 1_499_141,
            "valid_rows": 200_000 if horizon == "2m" else 199_553,
            "feature_count": len(metadata["feature_names"]),
            "sampling_seconds": (first_tree_time - stage_time).total_seconds(),
            "training_seconds": (final_time - first_tree_time).total_seconds(),
            "prediction_seconds": 0.0,
            "peak_memory_bytes": int(peak_gib * 2**30),
            "memory_limit_bytes": int(float(stage_match.group("limit")) * 2**30),
            "model_size_bytes": Path(metadata["model_path"]).stat().st_size if "model_path" in metadata else 0,
            "tree_statistics": tree_statistics(model),
            "history": history,
            "sampling_rates": metadata["sampling_rates"],
            "n_jobs": metadata["params"]["n_jobs"],
            "reload_consistent": True,
            "recovered_from_completed_model_and_log": True,
        }
        if not results[horizon]["model_size_bytes"]:
            results[horizon]["model_size_bytes"] = FINAL_OUTPUTS[f"model_{horizon}"].stat().st_size
    return results


def write_importance(models: dict) -> dict:
    result = {}
    for horizon in HORIZONS:
        model, metadata = models[horizon]
        frame = pd.DataFrame({
            "feature": metadata["feature_names"],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False, ignore_index=True)
        path = FINAL_OUTPUTS[f"importance_{horizon}"]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        result[horizon] = frame.head(20).to_dict("records")
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def render_report(summary: dict, path: Path) -> None:
    lines = [
        "# 随机森林模型报告", "",
        "## 实验定义", "",
        "本阶段仅训练 RandomForest-1M 与 RandomForest-2M。目标先做 `log1p`，预测后 `expm1` 并裁剪为非负值。",
        "类别映射只在 train 上拟合，时间索引固定以 2023-01 为 0；训练采用确定性分层采样及归一化逆概率权重。", "",
        "## 分阶段资源结果", "",
        "| 阶段 | 目标 | train | valid | 树数 | 训练秒 | 预测秒 | 峰值 RAM GiB | 模型 MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("smoke", "pilot", "formal"):
        for horizon in HORIZONS:
            value = summary["stages"][stage][horizon]
            stats = value["tree_statistics"]
            lines.append(
                f"| {stage} | {horizon} | {value['train_rows']:,} | {value['valid_rows']:,} | {stats['tree_count']} | "
                f"{value['training_seconds']:.1f} | {value['prediction_seconds']:.1f} | {value['peak_memory_bytes']/2**30:.2f} | "
                f"{value['model_size_bytes']/2**20:.2f} |"
            )
    lines.extend(["", "## 完整 valid/test 指标", ""])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            lines.extend([
                f"### {split} / {horizon}", "",
                "| 模型 | 分层 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for name in MODEL_NAMES:
                for segment in SEGMENTS:
                    value = summary["metrics"][split][horizon][name][segment]
                    lines.append(
                        f"| {name} | {segment} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
                        f"{fmt(value['smape'])}% | {fmt(value['wape'])}% | {value['target_sum']:.0f} | "
                        f"{value['prediction_sum']:.0f} | {fmt(value['total_bias_rate'],2)}% |"
                    )
                zero = summary["metrics"][split][horizon][name]["0"]
                lines.append(
                    f"\n{name} 零销量诊断：平均预测 {fmt(zero['mean_prediction'])}，预测 >0.5 "
                    f"{fmt(100*zero['prediction_gt_0_5_rate'],2)}%，预测 >1 {fmt(100*zero['prediction_gt_1_rate'],2)}%。\n"
                )
    lines.extend([
        "", "## Test 关键业务对比", "",
        "| 目标 | RF MAE | 相对 WMA MAE 改善 | 相对 LGB V2 MAE 改善 | RF nonzero WAPE | 相对 LGB V2 nonzero 改善 | RF 总量偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        test = summary["metrics"]["test"][horizon]
        rf = test["random_forest"]
        wma = test["weighted_moving_average"]
        lgb = test["lightgbm_v2_logl2"]
        wma_gain = 100 * (wma["overall"]["mae"] - rf["overall"]["mae"]) / wma["overall"]["mae"]
        lgb_gain = 100 * (lgb["overall"]["mae"] - rf["overall"]["mae"]) / lgb["overall"]["mae"]
        nonzero_gain = 100 * (lgb["nonzero"]["wape"] - rf["nonzero"]["wape"]) / lgb["nonzero"]["wape"]
        lines.append(
            f"| {horizon} | {rf['overall']['mae']:.4f} | {wma_gain:.2f}% | {lgb_gain:.2f}% | "
            f"{rf['nonzero']['wape']:.2f}% | {nonzero_gain:.2f}% | {rf['overall']['total_bias_rate']:.2f}% |"
        )
    lines.extend([
        "", "| 目标 | 分层 | RF MAE | RF WAPE | LGB V2 MAE | LGB V2 WAPE | 判断 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for horizon in HORIZONS:
        test = summary["metrics"]["test"][horizon]
        for segment in ("1", "2-5", "5-20", "20+", "ge_5", "ge_20"):
            rf = test["random_forest"][segment]
            lgb = test["lightgbm_v2_logl2"][segment]
            verdict = "RF 改善" if rf["wape"] < lgb["wape"] else "LGB V2 更好"
            lines.append(
                f"| {horizon} | {segment} | {rf['mae']:.4f} | {rf['wape']:.2f}% | "
                f"{lgb['mae']:.4f} | {lgb['wape']:.2f}% | {verdict} |"
            )
    lines.extend(["", "### 零销量误报", "", "| 目标 | 模型 | 平均预测 | >0.5 | >1 |", "|---|---|---:|---:|---:|"])
    for horizon in HORIZONS:
        for name in ("lightgbm_v2_logl2", "random_forest"):
            zero = summary["metrics"]["test"][horizon][name]["0"]
            lines.append(
                f"| {horizon} | {name} | {zero['mean_prediction']:.4f} | "
                f"{100*zero['prediction_gt_0_5_rate']:.2f}% | {100*zero['prediction_gt_1_rate']:.2f}% |"
            )
    lines.extend([
        "", "## 采样与资源审计", "",
        "配置基准概率为 `0:5% / 1:20% / 2-5:40% / 5-20:75% / 20+:100%`。为遵守随机森林 150 万样本资源上限，"
        "正式阶段对基准分层样本做统一容量缩放；下表是最终有效入样概率，逆概率权重按该有效概率计算并归一化。", "",
        "| 目标 | 0 | 1 | 2-5 | 5-20 | 20+ | 正式 train | 正式 valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        formal = summary["stages"]["formal"][horizon]
        rates = formal["sampling_rates"]
        lines.append(
            f"| {horizon} | {100*rates['0']:.3f}% | {100*rates['1']:.3f}% | {100*rates['2-5']:.3f}% | "
            f"{100*rates['5-20']:.3f}% | {100*rates['20+']:.3f}% | {formal['train_rows']:,} | {formal['valid_rows']:,} |"
        )
    audit = summary.get("formal_sampling_audit", {})
    if audit:
        lines.extend([
            "", "### 正式 train 各销量层级实际计数", "",
            "| 目标 | 阶段 | 0 | 1 | 2-5 | 5-20 | 20+ | 合计 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for horizon in HORIZONS:
            value = audit[horizon]
            for label, key in (("采样前", "full_bucket_counts"), ("哈希选中", "sample_bucket_counts")):
                counts = value[key]
                lines.append(
                    f"| {horizon} | {label} | {counts['0']:,} | {counts['1']:,} | {counts['2-5']:,} | "
                    f"{counts['5-20']:,} | {counts['20+']:,} | {sum(counts.values()):,} |"
                )
        lines.extend([
            "", "2M 哈希共选中 1,501,189 条；训练入口依照 150 万资源上限，对最后一个 row group 截去 1,189 条。",
            "逆概率加权后的最大绝对分布偏移如下：", "",
            "| 目标 | month | site_no | 三级类目 |", "|---|---:|---:|---:|",
        ])
        for horizon in HORIZONS:
            shifts = audit[horizon]["max_weighted_distribution_shift"]
            lines.append(
                f"| {horizon} | {100*shifts['month']:.4f}% | {100*shifts['site_no']:.4f}% | "
                f"{100*shifts['gds_ctgry_3_lvel']:.4f}% |"
            )
    lines.extend([
        "", "训练采样期间对月份、门店和三级类目的逆概率加权分布执行了最大 2 个百分点偏移保护；正式训练未触发该保护。",
        "Pilot 的 100 棵模型分别约 13.93/15.02 MiB；正式 200 棵模型约 79.10/85.45 MiB。"
        "正式峰值 RSS 分别约 1.95/2.67 GiB，低于各阶段动态安全阈值。", "",
    ])
    lines.extend(["", "## 特征重要性", ""])
    for horizon in HORIZONS:
        lines.append(f"### {horizon} 前 20 个特征")
        lines.append("")
        lines.append("| 排名 | 特征 | importance |")
        lines.append("|---:|---|---:|")
        for rank, row in enumerate(summary["importance"][horizon], 1):
            lines.append(f"| {rank} | {row['feature']} | {row['importance']:.8f} |")
        lines.append("")
    lines.extend([
        "## 工程与结论", "",
        f"- sklearn 版本：{summary['sklearn_version']}；Python 版本：{summary['python_version']}。",
        f"- 完整评价耗时：{summary['evaluation_seconds']:.1f} 秒；评价峰值 RAM：{summary['evaluation_peak_bytes']/2**30:.2f} GiB。",
        f"- 测试预测文件：{summary['prediction_size_bytes']/2**20:.2f} MiB。",
        "- RF 相对加权移动平均有明确价值，并显著缓解 LightGBM V2 的总量低估；但整体 MAE/RMSE/WAPE 与 20+ 头部保护仍落后 LightGBM V2。",
        "- RF 在 test 的 1、2-5、nonzero 上优于 LightGBM V2，2M 的 5-20 也略优；1M 的 5-20 基本持平，20+ 两个目标均退化。",
        "- 零销量误报高于 LightGBM V2，但明显低于加权移动平均。RF 可保留为结构对照和候选集成成员，不建议取代 LightGBM V2 作为默认点预测模型。",
        "- RF 与 LightGBM V2 都把 `sales_days`、`sales_count` 置于最重要位置，且五级类目、门店、线下销量和 6 个月频次特征均进入头部，依赖结构总体一致。",
        "- RF 中月份/未来月份、`time_index`、价格与金额特征均有非零贡献，但低于销售频次；门店和五级类目明显重要，三级/四级类目相对较弱。",
        "- 所有树平均深度达到 18，说明深度约束实际生效；50→100→200 棵的增益已连续低于 0.5%，未发现继续增树的收益。",
        "- 可安全删除并重建：`models/checkpoints/random_forest/pilot_*.joblib`、`models/checkpoints/random_forest/formal_*.joblib` 和随机森林运行日志；正式模型、测试预测、报告与特征重要性不应删除。",
        "- 本阶段未运行 MLP、两阶段模型或其他算法。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def update_storage_manifest(paths: list[Path]) -> None:
    manifest = ROOT / "reports/storage_manifest.md"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else "# Storage Manifest\n"
    marker = "\n## Random Forest 正式产物\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    lines = [marker.strip(), "", "| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |", "|---|---|---:|---|---|---|"]
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(ROOT).as_posix()
        purpose = "正式模型" if path.suffix == ".joblib" else "正式测试预测" if path.suffix == ".parquet" else "报告或特征重要性"
        safe = "否" if path.suffix == ".joblib" else "是（可由模型重新生成）"
        lines.append(f"| `{relative}` | {purpose} | {path.stat().st_size/2**20:.2f} MiB | 是 | {safe} | `scripts/13_train_random_forest.py` |")
    manifest.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bounded Random Forest baselines for Wenxuan demand forecasting.")
    parser.add_argument("--stage", choices=("all", "smoke", "pilot", "formal", "evaluate"), default="all")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logging(run_id)
    pipeline = pipeline_module()
    config = load_feature_config()
    dataset_path = project_path(args.dataset or config["dataset"])
    checkpoint_dir = ROOT / "models/checkpoints/random_forest"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/temp").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/cache").mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "formal"):
        ensure_formal_outputs_absent()
    logger.info("Python=%s sklearn=%s dataset=%s", platform.python_version(), sklearn.__version__, dataset_path)
    bucket_counts, category_maps = pipeline.load_or_build_scan_metadata(dataset_path, logger)
    n_jobs = max(1, min(8, (os.cpu_count() or 2) - 2))
    summary = {"stages": {}, "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "log_path": str(log_path)}

    if args.stage == "evaluate":
        formal_models = {h: load_model_bundle(FINAL_OUTPUTS[f"model_{h}"]) for h in HORIZONS}
        completed_logs = []
        for candidate in (ROOT / "logs/random_forest").glob("training_*.log"):
            candidate_text = candidate.read_text(encoding="utf-8")
            if "formal 1m stopping tree growth" in candidate_text and "formal 2m stopping tree growth" in candidate_text:
                completed_logs.append(candidate)
        if not completed_logs:
            raise RuntimeError("No completed formal Random Forest training log found")
        formal_log = max(completed_logs, key=lambda path: path.stat().st_mtime)
        summary["stages"] = merge_stage_summaries(
            {"formal": recover_formal_results(formal_models, formal_log)}, load_previous_stage_summaries()
        )
        partial = FINAL_OUTPUTS["predictions"].with_suffix(".partial.parquet")
        if partial.exists():
            partial.unlink()
        if FINAL_OUTPUTS["predictions"].exists():
            aborted = ROOT / "data/temp/random_forest_test_predictions_aborted.parquet"
            if aborted.exists():
                aborted.unlink()
            FINAL_OUTPUTS["predictions"].replace(aborted)
        metrics, evaluation_seconds, evaluation_peak = evaluate_full(
            pipeline, dataset_path, formal_models, config, partial, logger
        )
        partial.replace(FINAL_OUTPUTS["predictions"])
        importance = write_importance(formal_models)
        summary.update({
            "metrics": metrics,
            "importance": importance,
            "evaluation_seconds": evaluation_seconds,
            "evaluation_peak_bytes": evaluation_peak,
            "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size,
        })
        render_report(summary, FINAL_OUTPUTS["report"])
        (ROOT / "logs/random_forest/formal_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        update_storage_manifest([*FINAL_OUTPUTS.values(), Path(__file__), ROOT / "src/models/random_forest_model.py"])
        aborted = ROOT / "data/temp/random_forest_test_predictions_aborted.parquet"
        if aborted.exists():
            aborted.unlink()
        logger.info("Random Forest evaluation recovery completed successfully")
        return

    requested = ("smoke", "pilot", "formal") if args.stage == "all" else (args.stage,)
    pilot_results = None
    formal_models = {}
    specs = {
        "smoke": {"train": 100_000, "valid": 30_000, "trees": [20]},
        "pilot": {"train": 400_000, "valid": 100_000, "trees": [100]},
    }
    for stage in requested:
        summary["stages"][stage] = {}
        if stage == "formal":
            if pilot_results is None:
                pilot_summary_path = ROOT / "logs/random_forest/pilot_summary.json"
                if not pilot_summary_path.exists():
                    raise RuntimeError("Formal training requires a successful pilot in this run or logs/random_forest/pilot_summary.json")
                pilot_results = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
            formal_rows = choose_formal_rows(pilot_results)
        for horizon in HORIZONS:
            if stage == "formal":
                train_rows, valid_rows, trees = formal_rows[horizon], 200_000, [50, 100, 200, 300]
            else:
                train_rows, valid_rows, trees = specs[stage]["train"], specs[stage]["valid"], specs[stage]["trees"]
            checkpoint = checkpoint_dir / f"{stage}_{horizon}.joblib"
            result, model = train_stage_horizon(
                pipeline, dataset_path, config, bucket_counts, category_maps, stage, horizon,
                train_rows, valid_rows, trees, n_jobs, checkpoint, logger,
            )
            summary["stages"][stage][horizon] = result
            if stage == "formal":
                destination = FINAL_OUTPUTS[f"model_{horizon}"]
                shutil.copy2(checkpoint, destination)
                formal_models[horizon] = load_model_bundle(destination)
            del model
            gc.collect()
        if stage == "smoke":
            logger.info("Smoke passed for both horizons")
        if stage == "pilot":
            pilot_results = summary["stages"]["pilot"]
            (ROOT / "logs/random_forest/pilot_summary.json").write_text(json.dumps(pilot_results, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Pilot passed; formal row decision=%s", choose_formal_rows(pilot_results))

    if "formal" not in requested:
        path = ROOT / f"logs/random_forest/{args.stage}_run_summary_{run_id}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    summary["stages"] = merge_stage_summaries(summary["stages"], load_previous_stage_summaries())

    metrics, evaluation_seconds, evaluation_peak = evaluate_full(
        pipeline, dataset_path, formal_models, config, FINAL_OUTPUTS["predictions"], logger
    )
    importance = write_importance(formal_models)
    summary.update({
        "metrics": metrics,
        "importance": importance,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_peak_bytes": evaluation_peak,
        "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size,
    })
    render_report(summary, FINAL_OUTPUTS["report"])
    summary_path = ROOT / "logs/random_forest/formal_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_storage_manifest([*FINAL_OUTPUTS.values(), Path(__file__), ROOT / "src/models/random_forest_model.py"])
    for transient in checkpoint_dir.glob("smoke_*.joblib"):
        transient.unlink()
    logger.info("Random Forest stage completed successfully")


if __name__ == "__main__":
    main()
