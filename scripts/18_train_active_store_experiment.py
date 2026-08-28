from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.mc_metrics import (  # noqa: E402
    MCClassificationAccumulator,
    MCLevelConfig,
    TrendAccumulator,
    load_mc_level_config,
    quantity_to_mc,
)
from src.evaluation.metrics import LongTailStreamingMetrics, StreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import get_feature_list, load_feature_config  # noqa: E402
from src.models.active_store_common import (  # noqa: E402
    encode_categories_with_unknown,
    fit_category_maps_with_unknown,
)
from src.models.lightgbm_model import load_model_bundle, save_model_bundle  # noqa: E402


HORIZONS = ("1m", "2m")
CATEGORY_FEATURES = ["site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"]
DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
MC_CONFIG_PATH = ROOT / "config/mc_sales_levels.yaml"
CHECKPOINT_DIR = ROOT / "models/checkpoints/active_store_experiment"

FINAL_OUTPUTS = {
    **{f"v2_{h}": ROOT / f"models/final/lightgbm_v2_logl2_active_store_{h}.txt" for h in HORIZONS},
    **{f"two_classifier_{h}": ROOT / f"models/final/two_stage_classifier_active_store_{h}.txt" for h in HORIZONS},
    **{f"two_regressor_{h}": ROOT / f"models/final/two_stage_regressor_active_store_{h}.txt" for h in HORIZONS},
    **{f"mc_{h}": ROOT / f"models/final/lightgbm_mc_active_store_{h}.txt" for h in HORIZONS},
    "v2_predictions": ROOT / "data/outputs/lightgbm_v2_logl2_active_store_test_predictions.parquet",
    "two_predictions": ROOT / "data/outputs/two_stage_active_store_test_predictions.parquet",
    "mc_predictions": ROOT / "data/outputs/lightgbm_mc_active_store_test_predictions.parquet",
    "joint_predictions": ROOT / "data/outputs/active_store_joint_test_predictions.parquet",
    "v2_report": ROOT / "reports/lightgbm_v2_logl2_active_store_model_report.md",
    "two_report": ROOT / "reports/two_stage_active_store_model_report.md",
    "mc_report": ROOT / "reports/lightgbm_mc_active_store_report.md",
    "experiment_report": ROOT / "reports/active_store_experiment_report.md",
    "quantity_metrics": ROOT / "reports/active_store_quantity_metrics.csv",
    "trend_metrics": ROOT / "reports/active_store_trend_metrics.csv",
    "mc_metrics": ROOT / "reports/active_store_mc_metrics.csv",
    "fair_comparison": ROOT / "reports/active_store_fair_comparison.csv",
}

JOINT_OUTPUT_COLUMNS = (
    "month", "site_no", "item_id",
    "target_qty_1m", "lightgbm_v2_logl2_pred_1m", "two_stage_pred_1m",
    "target_qty_2m", "lightgbm_v2_logl2_pred_2m", "two_stage_pred_2m",
    "target_mc_1m", "regression_mapped_mc_1m", "lightgbm_v2_logl2_mapped_mc_1m",
    "two_stage_mapped_mc_1m", "direct_mc_pred_1m",
    "target_mc_2m", "regression_mapped_mc_2m", "lightgbm_v2_logl2_mapped_mc_2m",
    "two_stage_mapped_mc_2m", "direct_mc_pred_2m",
)

QUANTITY_CANDIDATE_METRIC_ORDER = (
    ("wape", "min"), ("total_bias_rate", "abs_min"),
    ("trend_macro_f1", "max"),
)
MC_CANDIDATE_METRIC_ORDER = (
    ("macro_f1", "max"), ("high_level_recall", "max"),
    ("weighted_f1", "max"), ("logloss", "min"),
)


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_module():
    return _load_script_module("wenxuan_active_store_lgb_pipeline", ROOT / "scripts/10_train_lightgbm.py")


def fit_train_category_maps(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    return fit_category_maps_with_unknown(frame, columns)


def encode_with_unknown_rates(
    frame: pd.DataFrame, maps: dict[str, dict[str, int]]
) -> tuple[pd.DataFrame, dict[str, float]]:
    return encode_categories_with_unknown(frame, maps)


def quantity_to_mc_strict(values, config: MCLevelConfig, horizon: str) -> np.ndarray:
    """Map quantity to configured MC; 2M order is qty/2 -> clip -> floor(x + 0.5) -> map."""
    return quantity_to_mc(values, config, horizon)


def resolve_mc_target(frame: pd.DataFrame, horizon: str, config: MCLevelConfig) -> np.ndarray:
    canonical = quantity_to_mc_strict(frame[f"future_qty_{horizon}"], config, horizon)
    column = f"target_mc_{horizon}"
    if column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        integer = np.isfinite(values) & (values == np.floor(values))
        if not integer.all():
            bad = np.flatnonzero(~integer)[:5].tolist()
            raise ValueError(f"{column} must contain finite integer labels; bad row positions={bad}")
        provided = values.astype("int32")
        if not np.isin(provided, config.codes).all():
            bad_values = sorted(set(provided[~np.isin(provided, config.codes)].tolist()))
            raise ValueError(f"{column} contains labels outside configured MC codes: {bad_values}")
        mismatch = provided != canonical
        if mismatch.any():
            bad = np.flatnonzero(mismatch)[:5].tolist()
            raise ValueError(
                f"{column} disagrees with canonical quantity mapping at row positions={bad}"
            )
    return canonical


def combine_two_stage(probability, conditional_quantity) -> np.ndarray:
    probability = np.clip(np.nan_to_num(np.asarray(probability, dtype="float64")), 0.0, 1.0)
    conditional = np.clip(np.nan_to_num(np.asarray(conditional_quantity, dtype="float64")), 0.0, None)
    return probability * conditional


def select_best_candidate(
    candidates: Iterable[dict[str, Any]], metric_order: tuple[tuple[str, str], ...]
) -> dict[str, Any]:
    rows = list(candidates)
    if not rows:
        raise ValueError("No candidate scores")

    def key(row: dict[str, Any]):
        metrics = []
        for name, direction in metric_order:
            value = float(row[name])
            if direction == "abs_min":
                value = abs(value)
            if not math.isfinite(value):
                value = -math.inf if direction == "max" else math.inf
            metrics.append(-value if direction == "max" else value)
        iteration = row["iteration"]
        tie_iteration = tuple(iteration) if isinstance(iteration, (list, tuple)) else (int(iteration),)
        return (*metrics, sum(tie_iteration), *tie_iteration)

    return min(rows, key=key)


def candidate_window(center: int, maximum: int, radius: int = 200, step: int = 50) -> list[int]:
    if maximum < 1 or step < 1:
        raise ValueError("maximum and step must be positive")
    center = min(max(1, int(center)), int(maximum))
    start = max(1, center - radius)
    stop = min(maximum, center + radius)
    values = set(range(start, stop + 1, step))
    values.update((center, stop, maximum if maximum <= stop else stop))
    return sorted(values)


def candidate_window_for_booster(
    booster,
    center: int,
    radius: int = 200,
    step: int = 50,
    global_step: int = 100,
    max_candidates: int = 64,
) -> list[int]:
    actual_iterations = int(booster.current_iteration())
    if actual_iterations < 1:
        raise RuntimeError("Booster has no trained iterations")
    if global_step < 1 or max_candidates < 4:
        raise ValueError("global_step must be positive and max_candidates must be at least four")
    proxy = min(max(1, int(center)), actual_iterations)
    global_grid = {1, proxy, actual_iterations, *range(global_step, actual_iterations + 1, global_step)}
    local_grid = set(candidate_window(proxy, actual_iterations, radius, step))
    required = {1, proxy, actual_iterations}
    candidates = sorted(global_grid | local_grid)
    if len(candidates) <= max_candidates:
        return candidates
    optional = [value for value in candidates if value not in required]
    slots = max_candidates - len(required)
    indices = np.linspace(0, len(optional) - 1, num=slots, dtype="int64")
    return sorted(required | {optional[index] for index in indices})


def bounded_candidate_pairs(
    left: list[int], right: list[int], max_pairs: int = 256
) -> list[tuple[int, int]]:
    if not left or not right or max_pairs < max(len(left), len(right), 4):
        raise ValueError("max_pairs must cover both candidate axes")
    full_size = len(left) * len(right)
    if full_size <= max_pairs:
        return [(a, b) for a in left for b in right]
    pairs = {
        (left[0], right[0]), (left[0], right[-1]),
        (left[-1], right[0]), (left[-1], right[-1]),
    }
    axis_size = max(len(left), len(right))
    for index in range(axis_size):
        left_index = round(index * (len(left) - 1) / max(1, axis_size - 1))
        right_index = round(index * (len(right) - 1) / max(1, axis_size - 1))
        pairs.add((left[left_index], right[right_index]))
    remaining = max_pairs - len(pairs)
    if remaining > 0:
        flat_indices = np.linspace(0, full_size - 1, num=remaining, dtype="int64")
        for flat in flat_indices:
            pairs.add((left[int(flat) // len(right)], right[int(flat) % len(right)]))
    return sorted(pairs, key=lambda pair: (sum(pair), pair))[:max_pairs]


def trend_inputs(current, target, prediction, horizon: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_values = clip_target(current)
    target_values = clip_target(target)
    prediction_values = clip_target(prediction)
    if horizon == "2m":
        target_values = target_values / 2.0
        prediction_values = prediction_values / 2.0
    elif horizon != "1m":
        raise ValueError(f"Unsupported horizon: {horizon}")
    return current_values, target_values, prediction_values


def multiclass_logloss(true_codes, probabilities, codes: tuple[int, ...]) -> float:
    true = np.asarray(true_codes, dtype="int32")
    probability = np.asarray(probabilities, dtype="float64")
    if probability.ndim != 2 or probability.shape[0] != len(true) or probability.shape[1] != len(codes):
        raise ValueError("Multiclass probabilities do not match target/config dimensions")
    code_to_index = {code: index for index, code in enumerate(codes)}
    try:
        indices = np.array([code_to_index[int(value)] for value in true], dtype="int64")
    except KeyError as exc:
        raise ValueError(f"Unknown MC target code: {exc.args[0]}") from exc
    selected = probability[np.arange(len(true)), indices]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean()) if len(true) else math.nan


def mc_disagreement_diagnostics(target, regression_mapped, direct) -> dict[str, float | int]:
    target_values = np.asarray(target, dtype="int32")
    regression_values = np.asarray(regression_mapped, dtype="int32")
    direct_values = np.asarray(direct, dtype="int32")
    if not (target_values.shape == regression_values.shape == direct_values.shape):
        raise ValueError("MC diagnostic arrays must have equal shapes")
    same = regression_values == direct_values
    regression_correct = regression_values == target_values
    direct_correct = direct_values == target_values
    counts = {
        "count": int(len(target_values)),
        "same_count": int(same.sum()),
        "regression_correct_direct_wrong_count": int((regression_correct & ~direct_correct).sum()),
        "direct_correct_regression_wrong_count": int((direct_correct & ~regression_correct).sum()),
        "both_wrong_count": int((~regression_correct & ~direct_correct).sum()),
    }
    total = counts["count"]
    counts.update({
        "same_rate": counts["same_count"] / total if total else math.nan,
        "regression_correct_direct_wrong_rate": counts["regression_correct_direct_wrong_count"] / total if total else math.nan,
        "direct_correct_regression_wrong_rate": counts["direct_correct_regression_wrong_count"] / total if total else math.nan,
        "both_wrong_rate": counts["both_wrong_count"] / total if total else math.nan,
    })
    return counts


class MCDisagreementAccumulator:
    def __init__(self) -> None:
        self.counts = Counter()

    def update(self, target, regression_mapped, direct) -> None:
        result = mc_disagreement_diagnostics(target, regression_mapped, direct)
        for key in (
            "count", "same_count", "regression_correct_direct_wrong_count",
            "direct_correct_regression_wrong_count", "both_wrong_count",
        ):
            self.counts[key] += int(result[key])

    def compute(self) -> dict[str, float | int]:
        total = self.counts["count"]
        result = dict(self.counts)
        for name in (
            "same", "regression_correct_direct_wrong", "direct_correct_regression_wrong", "both_wrong",
        ):
            result[f"{name}_rate"] = result.get(f"{name}_count", 0) / total if total else math.nan
        return result


def refuse_existing_outputs(paths: Iterable[Path]) -> None:
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing active-store outputs: " + ", ".join(existing))


def iter_horizon_row_groups(
    path: str | Path,
    horizon: str,
    columns: Iterable[str],
    splits: set[str] | None = None,
    max_row_groups: int | None = None,
) -> Iterator[pd.DataFrame]:
    availability = f"target_available_{horizon}"
    requested = list(dict.fromkeys([*columns, availability]))
    parquet = pq.ParquetFile(path)
    missing = set(requested).difference(parquet.schema.names)
    if missing:
        parquet.close()
        raise ValueError(f"Missing active-store columns: {sorted(missing)}")
    count = parquet.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.num_row_groups)
    try:
        for index in range(count):
            frame = parquet.read_row_group(index, columns=requested).to_pandas()
            mask = frame[availability].eq(1)
            if splits is not None:
                mask &= frame["split"].isin(splits)
            eligible = frame.loc[mask, list(dict.fromkeys(columns))].reset_index(drop=True)
            if not eligible.empty:
                yield eligible
    finally:
        parquet.close()


class UnknownRateAccumulator:
    def __init__(self, columns: Iterable[str]):
        self.unknown = Counter({column: 0 for column in columns})
        self.total = Counter({column: 0 for column in columns})

    def update(self, frame: pd.DataFrame, maps: dict[str, dict[str, int]]) -> None:
        for column, mapping in maps.items():
            missing = "__MISSING__" if "__MISSING__" in mapping else "unknown"
            values = frame[column].fillna(missing).astype("string")
            self.unknown[column] += int((~values.isin(mapping)).sum())
            self.total[column] += len(frame)

    def compute(self) -> dict[str, float]:
        return {column: self.unknown[column] / self.total[column] if self.total[column] else 0.0 for column in self.total}


def _setup_logging() -> logging.Logger:
    directory = ROOT / "logs/active_store_experiment"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wenxuan_active_store_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(directory / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _horizon_features(config: dict, horizon: str) -> list[str]:
    features = get_feature_list("lightgbm", config)
    two_stage = get_feature_list("two_stage", config)
    if two_stage != features:
        raise RuntimeError("Two-stage and LightGBM feature scopes must match")
    return features + list(config["lightgbm_horizon_features"][horizon])


def _base_params(config: dict) -> dict[str, Any]:
    params = dict(config["lightgbm_training"]["params"])
    for key in ("objective", "metric", "tweedie_variance_power", "num_class"):
        params.pop(key, None)
    params.update({"num_threads": max(1, (os.cpu_count() or 2) - 2), "first_metric_only": True})
    return params


def _train_full_booster(
    train: dict,
    valid: dict,
    train_y: np.ndarray,
    valid_y: np.ndarray,
    params: dict,
    categorical_features: list[str],
    rounds: int,
    guard,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, dict]:
    train_set = lgb.Dataset(
        train["x"], label=train_y, weight=train["weight"],
        feature_name=list(train["x"].columns), categorical_feature=categorical_features,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        valid["x"], label=valid_y, weight=valid["weight"], reference=train_set,
        feature_name=list(valid["x"].columns), categorical_feature=categorical_features,
        free_raw_data=True,
    )
    evaluations: dict = {}
    booster = lgb.train(
        params, train_set, num_boost_round=rounds, valid_sets=[valid_set], valid_names=["valid"],
        callbacks=training_callbacks(evaluations, guard, early_stopping_rounds),
    )
    return booster, evaluations


def training_callbacks(evaluations: dict, guard, early_stopping_rounds: int) -> list:
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive")
    return [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=True),
        lgb.record_evaluation(evaluations), lgb.log_evaluation(100), guard.callback(10, 100),
    ]


def _proxy_best_iteration(evaluations: dict, metric: str, direction: str = "min") -> int:
    values = np.asarray(evaluations["valid"][metric], dtype="float64")
    index = int(np.nanargmax(values) if direction == "max" else np.nanargmin(values))
    return index + 1


def _sample_horizon(pipeline, dataset_path: Path, horizon: str, maps: dict, config: dict, guard, smoke: bool):
    settings = config["lightgbm_training"]
    rates = {
        "train": settings["sampling_rates"],
        "valid": settings["valid_sampling_rates"],
    }
    limits = {"train": 20_000 if smoke else None, "valid": 8_000 if smoke else None}
    return pipeline.collect_train_valid_samples(
        dataset_path, horizon, rates, maps, _horizon_features(config, horizon),
        config["time_feature_settings"]["base_month"], guard, logging.getLogger("wenxuan_active_store_experiment"),
        limits, 2 if smoke else None, not smoke,
    )


def _metadata(kind: str, horizon: str, features: list[str], maps: dict, params: dict, iteration: int, **extra) -> dict:
    return {
        "experiment": "active_store_experiment", "model_kind": kind, "horizon": horizon,
        "feature_names": features, "categorical_features": CATEGORY_FEATURES,
        "category_maps": maps, "time_base": "2023-01", "params": params,
        "best_iteration": int(iteration), "selection_split": "complete_valid_original_scale",
        "lightgbm_version": lgb.__version__, "python_version": platform.python_version(), **extra,
    }


def _pruned(booster: lgb.Booster, iteration: int) -> lgb.Booster:
    return lgb.Booster(model_str=booster.model_to_string(num_iteration=int(iteration)))


def _physical_eval_columns(config: dict, features: Iterable[str], horizon: str, include_mc: bool = False) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    columns = [
        "month", "site_no", "item_id", "split", "total_qty",
        "future_qty_1m", "future_qty_2m",
    ]
    columns.extend(column for column in features if column not in runtime)
    if include_mc:
        columns.append(f"target_mc_{horizon}")
    return list(dict.fromkeys(columns))


def _prepare_eval(pipeline, frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    return pipeline.prepare_feature_frame(
        frame, metadata["feature_names"], metadata["category_maps"], metadata["time_base"]
    )


def evaluate_quantity_candidates(
    pipeline,
    dataset_path: Path,
    horizon: str,
    booster: lgb.Booster,
    metadata: dict,
    candidates: list[int],
    guard,
    transform: str = "log1p",
    max_row_groups: int | None = None,
) -> list[dict]:
    accumulators = {iteration: StreamingMetrics() for iteration in candidates}
    trend_accumulators = {iteration: TrendAccumulator() for iteration in candidates}
    columns = _physical_eval_columns(load_feature_config(), metadata["feature_names"], horizon)
    for number, frame in enumerate(
        iter_horizon_row_groups(dataset_path, horizon, columns, {"valid"}, max_row_groups), start=1
    ):
        target = clip_target(frame[f"future_qty_{horizon}"])
        current = clip_target(frame["total_qty"])
        x = _prepare_eval(pipeline, frame, metadata)
        for iteration, accumulator in accumulators.items():
            prediction = booster.predict(x, num_iteration=iteration)
            if transform == "log1p":
                prediction = np.expm1(prediction)
            prediction = np.clip(prediction, 0.0, None)
            accumulator.update(target, prediction)
            trend_current, trend_target, trend_prediction = trend_inputs(current, target, prediction, horizon)
            trend_accumulators[iteration].update(trend_current, trend_target, trend_prediction)
        if number % 10 == 0:
            guard.check("candidate-valid")
    return [
        {
            "iteration": iteration, **accumulator.compute(),
            "trend_macro_f1": trend_accumulators[iteration].compute()["macro_f1"],
        }
        for iteration, accumulator in accumulators.items()
    ]


def evaluate_two_stage_candidates(
    pipeline,
    dataset_path: Path,
    horizon: str,
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    metadata: dict,
    classifier_candidates: list[int],
    regressor_candidates: list[int],
    guard,
    max_row_groups: int | None = None,
) -> list[dict]:
    pairs = bounded_candidate_pairs(classifier_candidates, regressor_candidates)
    accumulators = {pair: StreamingMetrics() for pair in pairs}
    trend_accumulators = {pair: TrendAccumulator() for pair in pairs}
    columns = _physical_eval_columns(load_feature_config(), metadata["feature_names"], horizon)
    for number, frame in enumerate(
        iter_horizon_row_groups(dataset_path, horizon, columns, {"valid"}, max_row_groups), start=1
    ):
        target = clip_target(frame[f"future_qty_{horizon}"])
        current = clip_target(frame["total_qty"])
        x = _prepare_eval(pipeline, frame, metadata)
        probabilities = {
            iteration: np.clip(classifier.predict(x, num_iteration=iteration), 0.0, 1.0)
            for iteration in classifier_candidates
        }
        conditionals = {
            iteration: np.clip(regressor.predict(x, num_iteration=iteration), 0.0, None)
            for iteration in regressor_candidates
        }
        for pair, accumulator in accumulators.items():
            prediction = combine_two_stage(probabilities[pair[0]], conditionals[pair[1]])
            accumulator.update(target, prediction)
            trend_current, trend_target, trend_prediction = trend_inputs(current, target, prediction, horizon)
            trend_accumulators[pair].update(trend_current, trend_target, trend_prediction)
        if number % 10 == 0:
            guard.check("two-stage-candidate-valid")
    return [
        {
            "iteration": pair, **accumulator.compute(),
            "trend_macro_f1": trend_accumulators[pair].compute()["macro_f1"],
        }
        for pair, accumulator in accumulators.items()
    ]


def evaluate_mc_candidates(
    pipeline,
    dataset_path: Path,
    horizon: str,
    booster: lgb.Booster,
    metadata: dict,
    candidates: list[int],
    mc_config: MCLevelConfig,
    guard,
    max_row_groups: int | None = None,
) -> list[dict]:
    accumulators = {iteration: MCClassificationAccumulator(mc_config) for iteration in candidates}
    logloss_sums = Counter({iteration: 0.0 for iteration in candidates})
    logloss_counts = Counter({iteration: 0 for iteration in candidates})
    parquet = pq.ParquetFile(dataset_path)
    try:
        has_mc_target = f"target_mc_{horizon}" in parquet.schema.names
    finally:
        parquet.close()
    columns = _physical_eval_columns(
        load_feature_config(), metadata["feature_names"], horizon, include_mc=has_mc_target
    )
    for number, frame in enumerate(
        iter_horizon_row_groups(dataset_path, horizon, columns, {"valid"}, max_row_groups), start=1
    ):
        target = resolve_mc_target(frame, horizon, mc_config)
        x = _prepare_eval(pipeline, frame, metadata)
        for iteration, accumulator in accumulators.items():
            probabilities = booster.predict(x, num_iteration=iteration)
            class_index = np.argmax(probabilities, axis=1)
            prediction = np.asarray(mc_config.codes, dtype="int32")[class_index]
            accumulator.update(target, prediction)
            logloss_sums[iteration] += multiclass_logloss(target, probabilities, mc_config.codes) * len(target)
            logloss_counts[iteration] += len(target)
        if number % 10 == 0:
            guard.check("mc-candidate-valid")
    rows = []
    for iteration, accumulator in accumulators.items():
        result = accumulator.compute()
        rows.append({
            "iteration": iteration, "macro_f1": result["macro_f1"],
            "high_level_recall": result["high_level_recall"],
            "weighted_f1": result["weighted_f1"],
            "logloss": logloss_sums[iteration] / logloss_counts[iteration],
        })
    return rows


def _save_selected_bundle(booster: lgb.Booster, path: Path, metadata: dict, iteration: int) -> None:
    selected = _pruned(booster, iteration)
    save_model_bundle(selected, path, metadata)
    loaded, loaded_metadata = load_model_bundle(path)
    if loaded_metadata["best_iteration"] != iteration or loaded.num_trees() != selected.num_trees():
        raise RuntimeError(f"Bundle reload mismatch: {path}")


def train_horizon_models(
    pipeline,
    dataset_path: Path,
    horizon: str,
    category_maps: dict,
    config: dict,
    mc_config: MCLevelConfig,
    rounds: int,
    smoke: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Path], dict]:
    features = _horizon_features(config, horizon)
    guard = pipeline.MemoryGuard(f"active-store-{horizon}", logger)
    samples = _sample_horizon(pipeline, dataset_path, horizon, category_maps, config, guard, smoke)
    train_raw = np.expm1(samples["train"]["y"].astype("float64"))
    valid_raw = np.expm1(samples["valid"]["y"].astype("float64"))
    base = _base_params(config)
    safety_patience = min(
        int(config["lightgbm_training"]["early_stopping_rounds"]), max(1, rounds // 4)
    )
    paths: dict[str, Path] = {}
    summary: dict[str, Any] = {}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    max_row_groups = 2 if smoke else None

    logger.info("Training V2 Log-L2 %s serially", horizon)
    v2_params = {**base, "objective": "regression_l2", "metric": "l2"}
    v2, v2_eval = _train_full_booster(
        samples["train"], samples["valid"], np.log1p(train_raw).astype("float32"),
        np.log1p(valid_raw).astype("float32"), v2_params, CATEGORY_FEATURES, rounds, guard,
        safety_patience,
    )
    v2_proxy = _proxy_best_iteration(v2_eval, "l2")
    v2_candidates = candidate_window_for_booster(
        v2, v2_proxy, radius=min(200, rounds), step=max(1, min(50, rounds // 4 or 1))
    )
    v2_meta = _metadata("lightgbm_v2_logl2", horizon, features, category_maps, v2_params, v2_proxy, target_transform="log1p")
    v2_scores = evaluate_quantity_candidates(
        pipeline, dataset_path, horizon, v2, v2_meta, v2_candidates, guard, "log1p", max_row_groups
    )
    v2_best = select_best_candidate(v2_scores, QUANTITY_CANDIDATE_METRIC_ORDER)
    v2_path = CHECKPOINT_DIR / f"lightgbm_v2_logl2_{horizon}.txt"
    v2_meta.update(best_iteration=int(v2_best["iteration"]), candidate_scores=v2_scores)
    _save_selected_bundle(v2, v2_path, v2_meta, int(v2_best["iteration"]))
    paths["v2"] = v2_path
    summary["v2"] = v2_best
    del v2
    gc.collect()

    logger.info("Training two-stage classifier %s serially", horizon)
    classifier_params = {**base, "objective": "binary", "metric": "binary_logloss"}
    classifier, classifier_eval = _train_full_booster(
        samples["train"], samples["valid"], (train_raw > 0).astype("float32"),
        (valid_raw > 0).astype("float32"), classifier_params, CATEGORY_FEATURES, rounds, guard,
        safety_patience,
    )
    classifier_proxy = _proxy_best_iteration(classifier_eval, "binary_logloss")

    logger.info("Training two-stage conditional Tweedie regressor %s serially", horizon)
    positive_train = train_raw > 0
    positive_valid = valid_raw > 0
    reg_train = {
        "x": samples["train"]["x"].loc[positive_train].reset_index(drop=True),
        "weight": samples["train"]["weight"][positive_train],
    }
    reg_valid = {
        "x": samples["valid"]["x"].loc[positive_valid].reset_index(drop=True),
        "weight": samples["valid"]["weight"][positive_valid],
    }
    reg_train["weight"] = reg_train["weight"] / reg_train["weight"].mean()
    reg_valid["weight"] = reg_valid["weight"] / reg_valid["weight"].mean()
    regressor_params = {**base, "objective": "tweedie", "metric": "tweedie", "tweedie_variance_power": 1.4}
    regressor, regressor_eval = _train_full_booster(
        reg_train, reg_valid, train_raw[positive_train].astype("float32"), valid_raw[positive_valid].astype("float32"),
        regressor_params, CATEGORY_FEATURES, rounds, guard, safety_patience,
    )
    regressor_proxy = _proxy_best_iteration(regressor_eval, "tweedie")
    step = max(1, min(50, rounds // 4 or 1))
    classifier_candidates = candidate_window_for_booster(
        classifier, classifier_proxy, min(150, rounds), step
    )
    regressor_candidates = candidate_window_for_booster(
        regressor, regressor_proxy, min(150, rounds), step
    )
    two_meta = _metadata("two_stage", horizon, features, category_maps, classifier_params, classifier_proxy)
    two_scores = evaluate_two_stage_candidates(
        pipeline, dataset_path, horizon, classifier, regressor, two_meta,
        classifier_candidates, regressor_candidates, guard, max_row_groups,
    )
    two_best = select_best_candidate(two_scores, QUANTITY_CANDIDATE_METRIC_ORDER)
    classifier_iteration, regressor_iteration = map(int, two_best["iteration"])
    classifier_path = CHECKPOINT_DIR / f"two_stage_classifier_{horizon}.txt"
    regressor_path = CHECKPOINT_DIR / f"two_stage_regressor_{horizon}.txt"
    classifier_meta = _metadata(
        "two_stage_classifier", horizon, features, category_maps, classifier_params, classifier_iteration,
        combination="p_sale * conditional_quantity", joint_candidate_scores=two_scores,
    )
    regressor_meta = _metadata(
        "two_stage_regressor", horizon, features, category_maps, regressor_params, regressor_iteration,
        tweedie_variance_power=1.4, conditional_on_positive=True,
        combination="p_sale * conditional_quantity", joint_candidate_scores=two_scores,
    )
    _save_selected_bundle(classifier, classifier_path, classifier_meta, classifier_iteration)
    _save_selected_bundle(regressor, regressor_path, regressor_meta, regressor_iteration)
    paths.update(two_classifier=classifier_path, two_regressor=regressor_path)
    summary["two_stage"] = two_best
    del classifier, regressor, reg_train, reg_valid
    gc.collect()

    logger.info("Training direct MC multiclass %s serially", horizon)
    train_codes = quantity_to_mc_strict(train_raw, mc_config, horizon)
    valid_codes = quantity_to_mc_strict(valid_raw, mc_config, horizon)
    code_to_index = {code: index for index, code in enumerate(mc_config.codes)}
    train_labels = np.array([code_to_index[int(code)] for code in train_codes], dtype="int32")
    valid_labels = np.array([code_to_index[int(code)] for code in valid_codes], dtype="int32")
    mc_params = {**base, "objective": "multiclass", "metric": "multi_logloss", "num_class": len(mc_config.codes)}
    mc, mc_eval = _train_full_booster(
        samples["train"], samples["valid"], train_labels, valid_labels,
        mc_params, CATEGORY_FEATURES, rounds, guard, safety_patience,
    )
    mc_proxy = _proxy_best_iteration(mc_eval, "multi_logloss")
    mc_candidates = candidate_window_for_booster(mc, mc_proxy, min(200, rounds), step)
    mc_meta = _metadata(
        "lightgbm_mc", horizon, features, category_maps, mc_params, mc_proxy,
        mc_codes=list(mc_config.codes),
        label_source="canonical configured quantity mapping; target_mc field is validation-only",
    )
    mc_scores = evaluate_mc_candidates(
        pipeline, dataset_path, horizon, mc, mc_meta, mc_candidates, mc_config, guard, max_row_groups
    )
    mc_best = select_best_candidate(mc_scores, MC_CANDIDATE_METRIC_ORDER)
    mc_iteration = int(mc_best["iteration"])
    mc_path = CHECKPOINT_DIR / f"lightgbm_mc_{horizon}.txt"
    mc_meta.update(best_iteration=mc_iteration, candidate_scores=mc_scores)
    _save_selected_bundle(mc, mc_path, mc_meta, mc_iteration)
    paths["mc"] = mc_path
    summary["mc"] = mc_best
    del mc, samples, train_raw, valid_raw, train_codes, valid_codes
    gc.collect()
    guard.check("horizon-released")
    return paths, summary


class OldPredictionIndex:
    """Disk-backed lookup for only the two requested historical model outputs."""

    V2_PATH = ROOT / "data/outputs/lightgbm_v2_logl2_test_predictions.parquet"
    TWO_PATH = ROOT / "data/outputs/two_stage_test_predictions.parquet"

    def __init__(self, path: Path, v2_path: Path | None = None, two_path: Path | None = None):
        self.path = path
        self.v2_path = Path(v2_path) if v2_path is not None else self.V2_PATH
        self.two_path = Path(two_path) if two_path is not None else self.TWO_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=FILE;"
            "CREATE TABLE old_v2 ("
            "month TEXT, site_no TEXT, item_id TEXT, old_v2_1m REAL, old_v2_2m REAL, "
            "PRIMARY KEY(month, site_no, item_id));"
            "CREATE TABLE old_two ("
            "month TEXT, site_no TEXT, item_id TEXT, old_two_1m REAL, old_two_2m REAL, "
            "PRIMARY KEY(month, site_no, item_id));"
            "CREATE TEMP TABLE lookup_keys (seq INTEGER PRIMARY KEY, month TEXT, site_no TEXT, item_id TEXT);"
        )

    def build(self, logger: logging.Logger) -> None:
        for path in (self.v2_path, self.two_path):
            if not path.exists():
                raise FileNotFoundError(f"Fair comparison requires {path}")
        specifications = (
            (
                self.v2_path, "old_v2",
                ["month", "site_no", "item_id", "lightgbm_v2_logl2_pred_1m", "lightgbm_v2_logl2_pred_2m"],
                "INSERT INTO old_v2 VALUES (?,?,?,?,?)",
            ),
            (
                self.two_path, "old_two",
                ["month", "site_no", "item_id", "two_stage_pred_1m", "two_stage_pred_2m"],
                "INSERT INTO old_two VALUES (?,?,?,?,?)",
            ),
        )
        for path, source, columns, statement in specifications:
            parquet = pq.ParquetFile(path)
            inserted = 0
            try:
                for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
                    frame = batch.to_pandas()
                    rows = zip(
                        frame["month"].astype(str), frame["site_no"].astype(str),
                        frame["item_id"].astype(str), frame[columns[3]], frame[columns[4]],
                    )
                    try:
                        self.connection.executemany(statement, rows)
                    except sqlite3.IntegrityError as exc:
                        self.connection.rollback()
                        raise ValueError(f"Duplicate historical prediction key in {source}") from exc
                    inserted += len(frame)
                    if inserted % 1_000_000 < len(frame):
                        self.connection.commit()
                        logger.info("Indexed %,d %s rows", inserted, source)
                self.connection.commit()
            finally:
                parquet.close()

    def lookup(self, keys: pd.DataFrame) -> pd.DataFrame:
        self.connection.execute("DELETE FROM lookup_keys")
        rows = zip(range(len(keys)), keys["month"].astype(str), keys["site_no"].astype(str), keys["item_id"].astype(str))
        self.connection.executemany("INSERT INTO lookup_keys VALUES (?,?,?,?)", rows)
        query = (
            "SELECT k.seq,v.old_v2_1m,v.old_v2_2m,t.old_two_1m,t.old_two_2m "
            "FROM lookup_keys k "
            "JOIN old_v2 v USING(month,site_no,item_id) "
            "JOIN old_two t USING(month,site_no,item_id) ORDER BY k.seq"
        )
        return pd.read_sql_query(query, self.connection)

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)


class ParquetOutputSet:
    def __init__(self, destinations: dict[str, Path]):
        self.destinations = destinations
        self.temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in destinations.items()}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.closed = False
        for path in self.temporary.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)

    def write(self, name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in self.writers:
            dictionary = [column for column in ("month", "site_no") if column in frame.columns]
            self.writers[name] = pq.ParquetWriter(self.temporary[name], table.schema, compression="zstd", use_dictionary=dictionary)
        self.writers[name].write_table(table, row_group_size=250_000)

    def close(self, require_all: bool = True) -> None:
        if self.closed:
            return
        for writer in self.writers.values():
            writer.close()
        self.closed = True
        missing = set(self.destinations).difference(self.writers)
        if require_all and missing:
            raise RuntimeError(f"No test predictions written for: {sorted(missing)}")

    def abort(self) -> None:
        try:
            self.close(require_all=False)
        except Exception:
            return
        for path in self.temporary.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _prediction_columns(config: dict, bundles: dict, schema_names: set[str]) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    columns = [
        "month", "site_no", "item_id", "split", "total_qty",
        "future_qty_1m", "future_qty_2m", "target_available_1m", "target_available_2m",
    ]
    columns.extend(column for column in ("target_mc_1m", "target_mc_2m") if column in schema_names)
    for horizon in HORIZONS:
        for _, metadata in bundles[horizon].values():
            columns.extend(column for column in metadata["feature_names"] if column not in runtime)
    return list(dict.fromkeys(columns))


def _predict_horizon(pipeline, frame: pd.DataFrame, bundles: dict, horizon: str, mc_config: MCLevelConfig) -> dict[str, np.ndarray]:
    v2, v2_meta = bundles[horizon]["v2"]
    classifier, classifier_meta = bundles[horizon]["two_classifier"]
    regressor, regressor_meta = bundles[horizon]["two_regressor"]
    mc, mc_meta = bundles[horizon]["mc"]
    if not (v2_meta["feature_names"] == classifier_meta["feature_names"] == regressor_meta["feature_names"] == mc_meta["feature_names"]):
        raise RuntimeError(f"Feature order mismatch for {horizon}")
    x = _prepare_eval(pipeline, frame, v2_meta)
    v2_pred = np.clip(np.expm1(v2.predict(x, num_iteration=v2_meta["best_iteration"])), 0.0, None)
    probability = np.clip(classifier.predict(x, num_iteration=classifier_meta["best_iteration"]), 0.0, 1.0)
    conditional = np.clip(regressor.predict(x, num_iteration=regressor_meta["best_iteration"]), 0.0, None)
    two_pred = combine_two_stage(probability, conditional)
    class_index = np.argmax(mc.predict(x, num_iteration=mc_meta["best_iteration"]), axis=1)
    direct_mc = np.asarray(mc_config.codes, dtype="int32")[class_index]
    return {
        "v2": v2_pred, "p_sale": probability, "conditional": conditional, "two": two_pred,
        "direct_mc": direct_mc, "v2_mc": quantity_to_mc_strict(v2_pred, mc_config, horizon),
        "two_mc": quantity_to_mc_strict(two_pred, mc_config, horizon),
    }


def evaluate_and_write(
    pipeline,
    dataset_path: Path,
    checkpoint_paths: dict[str, dict[str, Path]],
    config: dict,
    mc_config: MCLevelConfig,
    logger: logging.Logger,
    include_old_comparison: bool,
    max_row_groups: int | None = None,
) -> tuple[dict, ParquetOutputSet]:
    bundles = {
        horizon: {name: load_model_bundle(path) for name, path in paths.items()}
        for horizon, paths in checkpoint_paths.items()
    }
    quantity = {
        split: {h: {model: LongTailStreamingMetrics() for model in ("lightgbm_v2_logl2", "two_stage")} for h in HORIZONS}
        for split in ("valid", "test")
    }
    trend = {
        split: {h: {model: TrendAccumulator() for model in ("lightgbm_v2_logl2", "two_stage")} for h in HORIZONS}
        for split in ("valid", "test")
    }
    mc_metrics = {
        split: {h: {model: MCClassificationAccumulator(mc_config) for model in ("direct_mc", "v2_mapped_mc", "two_stage_mapped_mc")} for h in HORIZONS}
        for split in ("valid", "test")
    }
    mc_disagreement = {
        split: {
            h: {model: MCDisagreementAccumulator() for model in ("v2_mapped_mc", "two_stage_mapped_mc")}
            for h in HORIZONS
        }
        for split in ("valid", "test")
    }
    unknown = {split: UnknownRateAccumulator(CATEGORY_FEATURES) for split in ("valid", "test")}
    fair = {
        "quantity": {
            h: {family: {version: LongTailStreamingMetrics() for version in ("old", "active_store")} for family in ("lightgbm_v2_logl2", "two_stage")}
            for h in HORIZONS
        },
        "trend": {
            h: {family: {version: TrendAccumulator() for version in ("old", "active_store")} for family in ("lightgbm_v2_logl2", "two_stage")}
            for h in HORIZONS
        },
        "mc": {
            h: {family: {version: MCClassificationAccumulator(mc_config) for version in ("old", "active_store")} for family in ("lightgbm_v2_logl2", "two_stage")}
            for h in HORIZONS
        },
    }
    outputs = ParquetOutputSet({
        "v2": FINAL_OUTPUTS["v2_predictions"], "two": FINAL_OUTPUTS["two_predictions"],
        "mc": FINAL_OUTPUTS["mc_predictions"], "joint": FINAL_OUTPUTS["joint_predictions"],
    })
    old_index = None
    if include_old_comparison:
        old_index = OldPredictionIndex(CHECKPOINT_DIR / "old_prediction_index.sqlite")
        old_index.build(logger)
    parquet = pq.ParquetFile(dataset_path)
    columns = _prediction_columns(config, bundles, set(parquet.schema.names))
    row_groups = parquet.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.num_row_groups)
    guard = pipeline.MemoryGuard("active-store-full-evaluation", logger)
    evaluation_completed = False
    try:
        for row_group in range(row_groups):
            frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
            relevant = frame[frame["split"].isin(["valid", "test"])].copy()
            if relevant.empty:
                continue
            for split in ("valid", "test"):
                split_frame = relevant[relevant["split"].eq(split)]
                if not split_frame.empty:
                    unknown[split].update(split_frame, bundles["1m"]["v2"][1]["category_maps"])

            per_horizon: dict[str, tuple[np.ndarray, pd.DataFrame, dict]] = {}
            for horizon in HORIZONS:
                eligible_mask = relevant[f"target_available_{horizon}"].eq(1).to_numpy()
                eligible = relevant.loc[eligible_mask]
                if eligible.empty:
                    continue
                target = clip_target(eligible[f"future_qty_{horizon}"])
                target_mc = resolve_mc_target(eligible, horizon, mc_config)
                predictions = _predict_horizon(pipeline, eligible, bundles, horizon, mc_config)
                per_horizon[horizon] = (eligible_mask, eligible, {"target": target, "target_mc": target_mc, **predictions})
                current = clip_target(eligible["total_qty"])
                for split in ("valid", "test"):
                    mask = eligible["split"].eq(split).to_numpy()
                    if not mask.any():
                        continue
                    for model, key in (("lightgbm_v2_logl2", "v2"), ("two_stage", "two")):
                        quantity[split][horizon][model].update(target[mask], predictions[key][mask])
                        trend_current, trend_target, trend_prediction = trend_inputs(
                            current[mask], target[mask], predictions[key][mask], horizon
                        )
                        trend[split][horizon][model].update(trend_current, trend_target, trend_prediction)
                    for model, key in (("direct_mc", "direct_mc"), ("v2_mapped_mc", "v2_mc"), ("two_stage_mapped_mc", "two_mc")):
                        mc_metrics[split][horizon][model].update(target_mc[mask], predictions[key][mask])
                    for model, key in (("v2_mapped_mc", "v2_mc"), ("two_stage_mapped_mc", "two_mc")):
                        mc_disagreement[split][horizon][model].update(
                            target_mc[mask], predictions[key][mask], predictions["direct_mc"][mask]
                        )

            test = relevant[relevant["split"].eq("test") & relevant["target_available_1m"].eq(1)].copy()
            if not test.empty:
                positions = pd.Series(np.arange(len(relevant)), index=relevant.index)
                output = test[["month", "site_no", "item_id"]].copy()
                test_positions = positions.loc[test.index].to_numpy()
                prediction_rows: dict[str, dict[str, np.ndarray]] = {}
                for horizon in HORIZONS:
                    available = test[f"target_available_{horizon}"].eq(1).to_numpy()
                    size = len(test)
                    values = {
                        "target": np.full(size, np.nan, dtype="float64"),
                        "target_mc": np.full(size, -1, dtype="int32"),
                        "v2": np.full(size, np.nan), "p_sale": np.full(size, np.nan),
                        "conditional": np.full(size, np.nan), "two": np.full(size, np.nan),
                        "v2_mc": np.full(size, -1, dtype="int32"), "two_mc": np.full(size, -1, dtype="int32"),
                        "direct_mc": np.full(size, -1, dtype="int32"),
                    }
                    if available.any() and horizon in per_horizon:
                        _, eligible, result = per_horizon[horizon]
                        eligible_positions = positions.loc[eligible.index].to_numpy()
                        by_position = {position: idx for idx, position in enumerate(eligible_positions)}
                        source = np.array([by_position[position] for position in test_positions[available]], dtype="int64")
                        for key in values:
                            values[key][available] = result[key][source]
                    prediction_rows[horizon] = values

                for horizon, values in prediction_rows.items():
                    output[f"target_qty_{horizon}"] = values["target"].astype("float32")
                    output[f"lightgbm_v2_logl2_pred_{horizon}"] = values["v2"].astype("float32")
                    output[f"p_sale_{horizon}"] = values["p_sale"].astype("float32")
                    output[f"conditional_qty_{horizon}"] = values["conditional"].astype("float32")
                    output[f"two_stage_pred_{horizon}"] = values["two"].astype("float32")
                    output[f"target_mc_{horizon}"] = values["target_mc"]
                    output[f"lightgbm_v2_logl2_mapped_mc_{horizon}"] = values["v2_mc"]
                    output[f"two_stage_mapped_mc_{horizon}"] = values["two_mc"]
                    output[f"regression_mapped_mc_{horizon}"] = values["v2_mc"]
                    output[f"direct_mc_pred_{horizon}"] = values["direct_mc"]
                outputs.write("v2", output[["month", "site_no", "item_id", "target_qty_1m", "lightgbm_v2_logl2_pred_1m", "target_qty_2m", "lightgbm_v2_logl2_pred_2m"]])
                outputs.write("two", output[["month", "site_no", "item_id", "target_qty_1m", "p_sale_1m", "conditional_qty_1m", "two_stage_pred_1m", "target_qty_2m", "p_sale_2m", "conditional_qty_2m", "two_stage_pred_2m"]])
                outputs.write("mc", output[["month", "site_no", "item_id", "target_mc_1m", "direct_mc_pred_1m", "target_mc_2m", "direct_mc_pred_2m"]])
                outputs.write("joint", output[list(JOINT_OUTPUT_COLUMNS)])

                if old_index is not None:
                    matched = old_index.lookup(output[["month", "site_no", "item_id"]])
                    if not matched.empty:
                        seq = matched["seq"].to_numpy(dtype="int64")
                        current_values = clip_target(test["total_qty"].to_numpy(dtype="float64")[seq])
                        for horizon in HORIZONS:
                            target_values = output[f"target_qty_{horizon}"].to_numpy(dtype="float64")[seq]
                            target_mc_values = output[f"target_mc_{horizon}"].to_numpy(dtype="int32")[seq]
                            available = np.isfinite(target_values)
                            for family, old_column, new_column in (
                                ("lightgbm_v2_logl2", f"old_v2_{horizon}", f"lightgbm_v2_logl2_pred_{horizon}"),
                                ("two_stage", f"old_two_{horizon}", f"two_stage_pred_{horizon}"),
                            ):
                                old_values = matched[old_column].to_numpy(dtype="float64")
                                new_values = output[new_column].to_numpy(dtype="float64")[seq]
                                common = available & np.isfinite(old_values) & np.isfinite(new_values)
                                fair["quantity"][horizon][family]["old"].update(target_values[common], old_values[common])
                                fair["quantity"][horizon][family]["active_store"].update(target_values[common], new_values[common])
                                old_current, old_target, old_prediction = trend_inputs(
                                    current_values[common], target_values[common], old_values[common], horizon
                                )
                                new_current, new_target, new_prediction = trend_inputs(
                                    current_values[common], target_values[common], new_values[common], horizon
                                )
                                fair["trend"][horizon][family]["old"].update(old_current, old_target, old_prediction)
                                fair["trend"][horizon][family]["active_store"].update(new_current, new_target, new_prediction)
                                old_mc = quantity_to_mc_strict(old_values[common], mc_config, horizon)
                                new_mc = quantity_to_mc_strict(new_values[common], mc_config, horizon)
                                fair["mc"][horizon][family]["old"].update(target_mc_values[common], old_mc)
                                fair["mc"][horizon][family]["active_store"].update(target_mc_values[common], new_mc)
            if (row_group + 1) % 10 == 0:
                guard.check("full-evaluation", row_group + 1)
                logger.info("Evaluated row groups %d/%d", row_group + 1, row_groups)
        evaluation_completed = True
    finally:
        parquet.close()
        if evaluation_completed:
            outputs.close(require_all=True)
        else:
            outputs.abort()
        if old_index is not None:
            old_index.close()

    return {
        "quantity": {s: {h: {m: a.compute() for m, a in models.items()} for h, models in hs.items()} for s, hs in quantity.items()},
        "trend": {s: {h: {m: a.compute() for m, a in models.items()} for h, models in hs.items()} for s, hs in trend.items()},
        "mc": {s: {h: {m: a.compute() for m, a in models.items()} for h, models in hs.items()} for s, hs in mc_metrics.items()},
        "mc_disagreement": {
            s: {h: {m: a.compute() for m, a in models.items()} for h, models in hs.items()}
            for s, hs in mc_disagreement.items()
        },
        "unknown_rates": {split: accumulator.compute() for split, accumulator in unknown.items()},
        "fair": {
            metric_family: {
                h: {f: {v: a.compute() for v, a in versions.items()} for f, versions in families.items()}
                for h, families in horizons.items()
            }
            for metric_family, horizons in fair.items()
        },
        "evaluation_peak_bytes": guard.peak,
    }, outputs


def _quantity_metric_rows(metrics: dict) -> list[dict]:
    rows = []
    for split, horizons in metrics.items():
        for horizon, models in horizons.items():
            for model, segments in models.items():
                for segment, values in segments.items():
                    rows.append({"split": split, "horizon": horizon, "model": model, "segment": segment, **values})
    return rows


def _trend_metric_rows(metrics: dict) -> list[dict]:
    rows = []
    for split, horizons in metrics.items():
        for horizon, models in horizons.items():
            for model, values in models.items():
                matrix = values["confusion_matrix"]
                base = {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
                rows.append({"section": "aggregate", "split": split, "horizon": horizon, "model": model, **base})
                for true_index, true_label in enumerate(TrendAccumulator.LABELS):
                    for pred_index, pred_label in enumerate(TrendAccumulator.LABELS):
                        rows.append({
                            "section": "confusion", "split": split, "horizon": horizon, "model": model,
                            "true_label": true_label, "pred_label": pred_label, "count": int(matrix[true_index, pred_index]),
                        })
    return rows


def _mc_metric_rows(metrics: dict, config: MCLevelConfig, disagreement: dict | None = None) -> list[dict]:
    rows = []
    for split, horizons in metrics.items():
        for horizon, models in horizons.items():
            for model, values in models.items():
                matrix = values["confusion_matrix"]
                base = {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
                if disagreement is not None and model in disagreement[split][horizon]:
                    base.update(disagreement[split][horizon][model])
                rows.append({"section": "aggregate", "split": split, "horizon": horizon, "model": model, **base})
                for index, code in enumerate(config.codes):
                    rows.append({
                        "section": "class", "split": split, "horizon": horizon, "model": model,
                        "true_mc": code, "precision": values["precision"][index],
                        "recall": values["recall"][index], "f1": values["f1"][index],
                    })
                    for pred_index, pred_code in enumerate(config.codes):
                        rows.append({
                            "section": "confusion", "split": split, "horizon": horizon, "model": model,
                            "true_mc": code, "pred_mc": pred_code, "count": int(matrix[index, pred_index]),
                        })
    return rows


def _fair_rows(metrics: dict) -> list[dict]:
    rows = []
    for metric_family, horizons in metrics.items():
        output_family = "mapped_mc" if metric_family == "mc" else metric_family
        for horizon, families in horizons.items():
            for family, versions in families.items():
                for version, values in versions.items():
                    if metric_family == "quantity":
                        for segment, segment_values in values.items():
                            rows.append({
                                "metric_family": output_family, "section": "segment", "split": "test",
                                "horizon": horizon, "model_family": family, "dataset_version": version,
                                "common_sample_only": True, "segment": segment, **segment_values,
                            })
                    else:
                        base = {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
                        rows.append({
                            "metric_family": output_family, "section": "aggregate", "split": "test",
                            "horizon": horizon, "model_family": family, "dataset_version": version,
                            "common_sample_only": True, **base,
                        })
    return rows


def _temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _write_csv_temp(path: Path, rows: list[dict]) -> None:
    temporary = _temporary(path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "N/A" if not math.isfinite(value) else f"{value:.{digits}f}"


def _report_table(training: dict, model: str) -> list[str]:
    lines = ["| Horizon | Best iteration | Selection metric |", "|---|---:|---:|"]
    for horizon in HORIZONS:
        values = training[horizon][model]
        metric = values.get("wape", values.get("macro_f1"))
        lines.append(f"| {horizon} | {values['iteration']} | {_fmt(metric)} |")
    return lines


def read_structural_zero_summary() -> dict[str, str]:
    summary: dict[str, str] = {}
    sources = (
        ROOT / "reports/store_activity_audit.md",
        ROOT / "reports/model_dataset_active_store_report.md",
    )
    wanted = (
        "Old model rows after store last active month", "Final unified rows",
        "Old rows after store closure", "Active-store zero-current-sales ratio",
        "Difference between old and active-store unified row count",
    )
    for path in sources:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", maxsplit=1)
            if any(token.lower() in key.lower() for token in wanted):
                summary[key.strip()] = value.strip()
    if not summary:
        summary["Structural-zero source reports"] = "not available"
    return summary


def _core_summary(evaluation: dict) -> list[dict]:
    rows = []
    mapped_names = {
        "lightgbm_v2_logl2": "v2_mapped_mc",
        "two_stage": "two_stage_mapped_mc",
    }
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            direct = evaluation["mc"][split][horizon]["direct_mc"]
            for model, mapped_model in mapped_names.items():
                quantity = evaluation["quantity"][split][horizon][model]["overall"]
                trend = evaluation["trend"][split][horizon][model]
                mapped = evaluation["mc"][split][horizon][mapped_model]
                rows.append({
                    "split": split, "horizon": horizon, "model": model,
                    "wape": quantity["wape"], "total_bias_rate": quantity["total_bias_rate"],
                    "trend_macro_f1": trend["macro_f1"], "mapped_mc_macro_f1": mapped["macro_f1"],
                    "direct_mc_macro_f1": direct["macro_f1"],
                })
    return rows


def _fair_summary(evaluation: dict) -> list[dict]:
    rows = []
    specifications = (
        ("quantity", "wape", lambda value: value["overall"]["wape"]),
        ("quantity", "total_bias_rate", lambda value: value["overall"]["total_bias_rate"]),
        ("trend", "macro_f1", lambda value: value["macro_f1"]),
        ("mc", "mapped_mc_macro_f1", lambda value: value["macro_f1"]),
    )
    for metric_family, metric, extractor in specifications:
        for horizon, families in evaluation["fair"][metric_family].items():
            for family, versions in families.items():
                old = float(extractor(versions["old"]))
                new = float(extractor(versions["active_store"]))
                rows.append({
                    "horizon": horizon, "model_family": family, "metric": metric,
                    "old": old, "active_store": new, "change": new - old,
                })
    return rows


def _disagreement_summary(evaluation: dict) -> list[dict]:
    rows = []
    for split, horizons in evaluation["mc_disagreement"].items():
        for horizon, models in horizons.items():
            direct_accuracy = evaluation["mc"][split][horizon]["direct_mc"]["accuracy"]
            for model, values in models.items():
                rows.append({
                    "split": split, "horizon": horizon, "model": model,
                    "mapped_accuracy": evaluation["mc"][split][horizon][model]["accuracy"],
                    "direct_accuracy": direct_accuracy, **values,
                })
    return rows


def build_experiment_report_lines(
    structural: dict[str, Any],
    core_summary: list[dict],
    fair_summary: list[dict],
    disagreement_summary: list[dict],
    unknown_rates: dict[str, dict[str, float]],
) -> list[str]:
    lines = [
        "# Active-Store Experiment Report", "", "## Scope", "",
        "- Trained serially: V2 Log-L2, binary + conditional Tweedie two-stage, and direct multiclass LightGBM for 1M and 2M.",
        "- Baseline, Random Forest, MLP and federated training were not run.",
        "- Complete valid/test evaluation is row-group streaming. Historical comparison uses only old V2 and old two-stage predictions on their common test keys; no six-model merged table is produced.",
        "- Dataset validation enforces that 2M eligibility implies 1M eligibility; test outputs may therefore use the eligible 1M rows as their base.",
        "- Candidate selection uses a bounded coarse-to-fine full-valid search: a global iteration grid plus a fine grid around the sampled-valid proxy optimum.",
        "- 2M MC uses qty/2 -> clip -> floor(x + 0.5) -> configured mapping; 2M trend also uses monthly-average target and prediction.", "",
        "## Structural-Zero Correction", "",
        "| Measure | Value |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in structural.items())
    lines.extend([
        "", "## Core Valid/Test Metrics", "",
        "| Split | Horizon | Model | WAPE | Bias % | Trend Macro-F1 | Mapped MC Macro-F1 | Direct MC Macro-F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in core_summary:
        lines.append(
            f"| {row['split']} | {row['horizon']} | {row['model']} | {_fmt(row.get('wape'))} | "
            f"{_fmt(row.get('total_bias_rate'))} | {_fmt(row.get('trend_macro_f1'))} | "
            f"{_fmt(row.get('mapped_mc_macro_f1'))} | {_fmt(row.get('direct_mc_macro_f1'))} |"
        )
    lines.extend([
        "", "## Common-Sample Old/New Changes", "",
        "| Horizon | Model | Metric | Old | Active-store | Change |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in fair_summary:
        lines.append(
            f"| {row['horizon']} | {row['model_family']} | {row['metric']} | {_fmt(row.get('old'))} | "
            f"{_fmt(row.get('active_store'))} | {_fmt(row.get('change'))} |"
        )
    lines.extend([
        "", "## Mapped MC vs Direct MC", "",
        "| Split | Horizon | Regression model | Same | Mapped accuracy | Direct accuracy | Mapped right/direct wrong | Direct right/mapped wrong | Both wrong |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in disagreement_summary:
        lines.append(
            f"| {row['split']} | {row['horizon']} | {row['model']} | {_fmt(row.get('same_rate'))} | "
            f"{_fmt(row.get('mapped_accuracy'))} | {_fmt(row.get('direct_accuracy'))} | "
            f"{_fmt(row.get('regression_correct_direct_wrong_rate'))} | "
            f"{_fmt(row.get('direct_correct_regression_wrong_rate'))} | {_fmt(row.get('both_wrong_rate'))} |"
        )
    lines.extend(["", "## Category Unknown Rates", "", "| Split | Feature | Unknown rate |", "|---|---|---:|"])
    for split, rates in unknown_rates.items():
        for feature, rate in rates.items():
            lines.append(f"| {split} | `{feature}` | {rate:.4%} |")
    lines.extend([
        "", "## Detailed Artifacts", "",
        "Detailed segment, confusion-matrix and comparison records are in `reports/active_store_quantity_metrics.csv`, `reports/active_store_trend_metrics.csv`, `reports/active_store_mc_metrics.csv`, and `reports/active_store_fair_comparison.csv`.", "",
    ])
    return lines


def write_reports(training: dict, evaluation: dict) -> None:
    report_payloads = {
        "v2_report": [
            "# Active-Store LightGBM V2 Log-L2 Report", "",
            "Best iteration is selected on complete eligible validation data in original quantity scale by WAPE, absolute total bias, trend Macro-F1, then the smaller iteration.", "",
            "The full-valid candidate set uses a bounded coarse-to-fine search: a global grid across all trained rounds plus a fine grid around the sampled-valid proxy optimum.", "",
            *_report_table(training, "v2"), "",
        ],
        "two_report": [
            "# Active-Store Two-Stage Report", "",
            "The classifier is binary and the positive-only conditional regressor uses Tweedie power 1.4. Candidate pairs are selected by complete-validation WAPE, absolute bias, trend Macro-F1, then fewer total component rounds for p(sale) * conditional quantity.", "",
            "Each component candidate axis spans all trained rounds with a global coarse grid and proxy-local fine grid; bounded pair sampling avoids an unbounded Cartesian product.", "",
            *_report_table(training, "two_stage"), "",
        ],
        "mc_report": [
            "# Active-Store Direct MC Report", "",
            "MC iterations are selected on complete-validation Macro-F1. Ties use high-level recall, Weighted-F1, true multiclass logloss, then the smaller iteration.", "",
            "The full-valid candidate set combines a global coarse grid over all trained rounds with a fine grid around the sampled-valid proxy optimum.", "",
            *_report_table(training, "mc"), "",
        ],
        "experiment_report": build_experiment_report_lines(
            read_structural_zero_summary(), _core_summary(evaluation), _fair_summary(evaluation),
            _disagreement_summary(evaluation), evaluation["unknown_rates"],
        ),
    }
    for key, lines in report_payloads.items():
        path = _temporary(FINAL_OUTPUTS[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


def prepare_formal_artifacts(checkpoint_paths: dict[str, dict[str, Path]], training: dict, evaluation: dict, mc_config: MCLevelConfig) -> None:
    for horizon in HORIZONS:
        mapping = {
            "v2": f"v2_{horizon}", "two_classifier": f"two_classifier_{horizon}",
            "two_regressor": f"two_regressor_{horizon}", "mc": f"mc_{horizon}",
        }
        for source_key, destination_key in mapping.items():
            destination = FINAL_OUTPUTS[destination_key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_paths[horizon][source_key], _temporary(destination))
    _write_csv_temp(FINAL_OUTPUTS["quantity_metrics"], _quantity_metric_rows(evaluation["quantity"]))
    _write_csv_temp(FINAL_OUTPUTS["trend_metrics"], _trend_metric_rows(evaluation["trend"]))
    _write_csv_temp(
        FINAL_OUTPUTS["mc_metrics"],
        _mc_metric_rows(evaluation["mc"], mc_config, evaluation["mc_disagreement"]),
    )
    _write_csv_temp(FINAL_OUTPUTS["fair_comparison"], _fair_rows(evaluation["fair"]))
    write_reports(training, evaluation)


def promote_formal_artifacts(outputs: dict[str, Path] | None = None) -> None:
    destinations = FINAL_OUTPUTS if outputs is None else outputs
    refuse_existing_outputs(destinations.values())
    missing = [str(path) for path in destinations.values() if not _temporary(path).exists()]
    if missing:
        raise RuntimeError("Formal artifact staging incomplete: " + ", ".join(missing))
    for destination in destinations.values():
        _temporary(destination).rename(destination)


def update_storage_manifest() -> None:
    manifest = ROOT / "reports/storage_manifest.md"
    marker = "<!-- ACTIVE_STORE_EXPERIMENT_ARTIFACTS -->"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else "# Storage Manifest\n"
    existing = existing.split(marker, maxsplit=1)[0].rstrip()
    lines = [
        marker, "", "## Active-Store Experiment Artifacts", "",
        "| File | Purpose | Size MiB | Generated by |", "|---|---|---:|---|",
    ]
    for path in [Path(__file__), ROOT / "tests/test_active_store_training.py", *FINAL_OUTPUTS.values()]:
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size / 2**20 if path.exists() else 0.0
        lines.append(f"| `{relative}` | Active-store training/evaluation artifact | {size:.2f} | `scripts/18_train_active_store_experiment.py` |")
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(existing + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(manifest)


def validate_dataset(path: Path, config: dict, mc_config: MCLevelConfig | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    try:
        required = {
            "month", "site_no", "item_id", "split", "total_qty",
            "future_qty_1m", "future_qty_2m", "target_available_1m", "target_available_2m",
            *CATEGORY_FEATURES,
        }
        runtime = set(config.get("runtime_time_features", []))
        for horizon in HORIZONS:
            required.update(column for column in _horizon_features(config, horizon) if column not in runtime)
        missing = required.difference(parquet.schema.names)
        if missing:
            raise ValueError(f"Active-store dataset is missing columns: {sorted(missing)}")
        mc_target_columns = [column for column in ("target_mc_1m", "target_mc_2m") if column in parquet.schema.names]
        if mc_target_columns and mc_config is None:
            mc_config = load_mc_level_config(MC_CONFIG_PATH)
        validation_columns = ["target_available_1m", "target_available_2m"]
        for horizon in HORIZONS:
            if f"target_mc_{horizon}" in mc_target_columns:
                validation_columns.extend((f"future_qty_{horizon}", f"target_mc_{horizon}"))
        for row_group in range(parquet.num_row_groups):
            availability = parquet.read_row_group(
                row_group, columns=list(dict.fromkeys(validation_columns))
            ).to_pandas()
            invalid = availability["target_available_2m"].eq(1) & ~availability["target_available_1m"].eq(1)
            if invalid.any():
                positions = np.flatnonzero(invalid.to_numpy())[:5].tolist()
                raise ValueError(
                    "2M eligibility requires 1M eligibility; "
                    f"row_group={row_group}, row_positions={positions}"
                )
            for horizon in HORIZONS:
                target_column = f"target_mc_{horizon}"
                if target_column not in mc_target_columns:
                    continue
                eligible = availability[f"target_available_{horizon}"].eq(1)
                if eligible.any():
                    try:
                        resolve_mc_target(availability.loc[eligible], horizon, mc_config)
                    except ValueError as exc:
                        raise ValueError(f"row_group={row_group}: {exc}") from exc
    finally:
        parquet.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the isolated active-store LightGBM experiment.")
    parser.add_argument("--stage", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()
    logger = _setup_logging()
    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    config = load_feature_config()
    mc_config = load_mc_level_config(MC_CONFIG_PATH)
    validate_dataset(dataset_path, config, mc_config)
    pipeline = pipeline_module()
    if args.stage == "formal":
        refuse_existing_outputs(FINAL_OUTPUTS.values())
    rounds = args.max_rounds or (20 if args.stage == "smoke" else int(config["lightgbm_training"]["num_boost_round"]))
    if rounds < 1:
        raise ValueError("--max-rounds must be positive")

    scan_guard = pipeline.MemoryGuard("active-store-category-scan", logger)
    category_maps = pipeline.fit_category_maps_streaming(dataset_path, logger, scan_guard)
    checkpoint_paths: dict[str, dict[str, Path]] = {}
    training: dict[str, dict] = {}
    for horizon in HORIZONS:
        paths, summary = train_horizon_models(
            pipeline, dataset_path, horizon, category_maps, config, mc_config,
            rounds, args.stage == "smoke", logger,
        )
        checkpoint_paths[horizon] = paths
        training[horizon] = summary
        gc.collect()

    if args.stage == "smoke":
        summary_path = ROOT / "logs/active_store_experiment/smoke_summary.json"
        summary_path.write_text(json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Active-store smoke training completed; no formal artifacts were touched")
        return

    evaluation, _ = evaluate_and_write(
        pipeline, dataset_path, checkpoint_paths, config, mc_config, logger,
        include_old_comparison=True,
    )
    prepare_formal_artifacts(checkpoint_paths, training, evaluation, mc_config)
    promote_formal_artifacts()
    update_storage_manifest()
    logger.info("Active-store formal experiment completed and published atomically per artifact")


if __name__ == "__main__":
    main()
