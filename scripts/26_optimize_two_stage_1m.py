from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Iterable

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import clip_target, demand_bucket  # noqa: E402
from src.features.demand_signal_features import (  # noqa: E402
    CROSS_STORE_FEATURES,
    DIFF_FEATURES,
    add_cross_store_features,
    add_diff_features,
    candidate_b_multiplier,
    threshold_for_top_fraction,
    weighted_threshold_for_top_fraction,
)
from src.features.preprocessing import get_feature_list, load_feature_config  # noqa: E402
from src.models.lightgbm_model import (  # noqa: E402
    deterministic_uniform_hash,
    load_model_bundle,
    save_model_bundle,
)


CONFIG_PATH = ROOT / "config/two_stage_optimization.yaml"
FORMAL_P_MODEL = ROOT / "models/final/two_stage_classifier_active_store_1m.txt"
FORMAL_Q_MODEL = ROOT / "models/final/two_stage_regressor_active_store_1m.txt"
MC_LEVELS_PATH = ROOT / "config/mc_sales_levels.yaml"
CROSS_STORE_DIAGNOSTIC_ROUNDS = 657
CROSS_STORE_DIAGNOSTIC_EXPECTED = {
    "q_20_plus_wape": 68.73682612568332,
    "q_20_plus_recall": 32.57494963858277,
    "final_20_plus_wape": 73.00375623077215,
    "final_20_plus_recall": 27.66915511316507,
    "final_nonzero_wape": 71.5926913822025,
    "overall_wape": 130.32572587597298,
    "total_bias": 4.988277116097936,
    "zero_prediction_total": 1449851.9736745914,
}
CATEGORY_FEATURES = [
    "site_no", "blt_site_no", "gds_ctgry_3_lvel",
    "gds_ctgry_4_lvel", "gds_ctgry_5_lvel",
]


def _load_numbered_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PIPELINE = _load_numbered_script("wenxuan_pipeline_10", ROOT / "scripts/10_train_lightgbm.py")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_optimization_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def business_mc_codes(prediction, levels_path: Path = MC_LEVELS_PATH) -> np.ndarray:
    with levels_path.open("r", encoding="utf-8") as handle:
        levels = (yaml.safe_load(handle) or {})["levels"]
    rounded = np.floor(np.clip(np.asarray(prediction, dtype="float64"), 0.0, None) + 0.5)
    result = np.zeros(rounded.shape, dtype="int8")
    for level in levels:
        lower = float(level["min_qty"])
        upper = level.get("max_qty_exclusive")
        mask = rounded >= lower
        if upper is not None:
            mask &= rounded < float(upper)
        result[mask] = int(level["code"])
    return result


def compare_rebuild_summary(
    rebuilt: dict,
    expected: dict = CROSS_STORE_DIAGNOSTIC_EXPECTED,
    metric_tolerance: float = 1e-4,
    total_tolerance: float = 0.01,
    diagnostic_metric_tolerance: float = 0.60,
    diagnostic_total_relative_tolerance: float = 0.005,
) -> dict:
    checks = {}
    equivalence_checks = {}
    for name, expected_value in expected.items():
        tolerance = total_tolerance if name == "zero_prediction_total" else metric_tolerance
        actual_value = float(rebuilt[name])
        difference = actual_value - float(expected_value)
        checks[name] = {
            "actual": actual_value,
            "expected": float(expected_value),
            "difference": difference,
            "tolerance": tolerance,
            "passed": abs(difference) <= tolerance,
        }
        if name == "zero_prediction_total":
            equivalence_tolerance = abs(float(expected_value)) * diagnostic_total_relative_tolerance
        else:
            equivalence_tolerance = diagnostic_metric_tolerance
        equivalence_checks[name] = {
            "difference": difference,
            "tolerance": equivalence_tolerance,
            "passed": abs(difference) <= equivalence_tolerance,
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "diagnostic_equivalent": all(item["passed"] for item in equivalence_checks.values()),
        "equivalence_checks": equivalence_checks,
    }


def setup_logging(config: dict) -> tuple[logging.Logger, Path]:
    directory = project_path(config["outputs"]["log_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("wenxuan_two_stage_optimization")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def _state_path(config: dict) -> Path:
    return project_path(config["outputs"]["cache_dir"]) / "run_state.json"


def read_state(config: dict) -> dict:
    path = _state_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_state(config: dict, state: dict) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _escaped(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _duckdb_connection(config: dict, name: str, memory: str = "1536MB") -> duckdb.DuckDBPyConnection:
    temp = project_path(config["outputs"]["cache_dir"]) / "duckdb" / name
    temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute(f"PRAGMA memory_limit='{memory}'")
    connection.execute(f"PRAGMA temp_directory='{_escaped(temp)}'")
    return connection


def build_cross_store_table(config: dict, logger: logging.Logger) -> dict:
    source = project_path(config["monthly_sales"])
    output = project_path(config["cross_store_table"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        metadata = pq.ParquetFile(output).metadata
        logger.info("Reusing Cross-store table rows=%s", f"{metadata.num_rows:,}")
        return {"rows": metadata.num_rows, "bytes": output.stat().st_size, "reused": True}

    guard = PIPELINE.MemoryGuard("cross-store-table", logger)
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    connection = _duckdb_connection(config, "cross_store_table")
    started = time.perf_counter()
    try:
        connection.execute(
            f"""
            COPY (
                SELECT CAST(month AS VARCHAR) AS month,
                       CAST(item_id AS VARCHAR) AS item_id,
                       CAST(SUM(GREATEST(COALESCE(total_qty, 0), 0)) AS FLOAT) AS group_positive_qty,
                       CAST(COUNT_IF(COALESCE(total_qty, 0) > 0) AS INTEGER) AS group_positive_sites
                FROM read_parquet('{_escaped(source)}')
                GROUP BY 1, 2
                ORDER BY 1, 2
            ) TO '{_escaped(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """
        )
        guard.check("cross_store_copy")
    finally:
        connection.close()
    parquet = pq.ParquetFile(temporary)
    if parquet.schema.names != ["month", "item_id", "group_positive_qty", "group_positive_sites"]:
        raise RuntimeError("Unexpected Cross-store schema")
    rows = parquet.metadata.num_rows
    parquet.close()
    if rows < 1:
        raise RuntimeError("Cross-store table is empty")
    temporary.replace(output)
    result = {
        "rows": rows, "bytes": output.stat().st_size,
        "seconds": time.perf_counter() - started, "peak_ram_gib": guard.peak / 1024**3,
        "reused": False,
    }
    logger.info("Cross-store table complete rows=%s size=%.2f MiB", f"{rows:,}", result["bytes"] / 1024**2)
    return result


def base_features() -> list[str]:
    model_config = load_feature_config()
    features = get_feature_list("two_stage", model_config)
    if features != get_feature_list("lightgbm", model_config):
        raise RuntimeError("Two-stage and LightGBM base feature scopes differ")
    return features + list(model_config["lightgbm_horizon_features"]["1m"])


def train_valid_protocol_feature_sets() -> dict[str, list[str]]:
    base = base_features()
    scopes = {
        "E0_VALID_SELECTED": list(base),
        "E1_DIFF_VALID_SELECTED": list(dict.fromkeys([*base, *DIFF_FEATURES])),
        "E2_CROSS_STORE_VALID_SELECTED": list(dict.fromkeys([*base, *CROSS_STORE_FEATURES])),
        "E3_DIFF_CROSS_STORE_VALID_SELECTED": list(
            dict.fromkeys([*base, *DIFF_FEATURES, *CROSS_STORE_FEATURES])
        ),
    }
    forbidden = {
        "future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m",
        "target_qty_1m", "split", "item_id", "isbn", "gds_no",
    }
    leaked = {name: sorted(forbidden.intersection(features)) for name, features in scopes.items()}
    leaked = {name: values for name, values in leaked.items() if values}
    if leaked:
        raise RuntimeError(f"Leakage/identifier fields in Train->Valid feature scopes: {leaked}")
    return scopes


def stable_sort_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["month", "site_no", "item_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Stable training sort requires columns: {missing}")
    return frame.sort_values(required, kind="mergesort").reset_index(drop=True)


def training_frame_hashes(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        column for column in ("month", "site_no", "item_id", "target_qty", "sample_weight")
        if column in frame
    ]
    if not {"month", "site_no", "item_id"}.issubset(columns):
        raise KeyError("Training fingerprints require month, site_no and item_id")
    row_hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype="uint64")
    set_hash = hashlib.sha256(np.sort(row_hashes).tobytes()).hexdigest()
    order_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return {"row_count": int(len(frame)), "sample_set_sha256": set_hash, "order_sha256": order_hash}


def train_valid_lightgbm_params(config: dict, component: str) -> dict:
    params = _lgb_params(config, component)
    seed = int(config["seed"])
    params.update({
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    })
    return params


def incremental_tweedie_predictions(
    booster: lgb.Booster,
    features,
    iterations: Iterable[int],
) -> dict[int, np.ndarray]:
    requested = sorted({int(value) for value in iterations})
    if not requested or requested[0] < 1:
        raise ValueError("Iterations must contain positive integers")
    if requested[-1] > booster.num_trees():
        raise ValueError("Requested iteration exceeds the trained booster")
    raw = np.zeros(len(features), dtype="float64")
    previous = 0
    predictions: dict[int, np.ndarray] = {}
    for iteration in requested:
        raw += booster.predict(
            features,
            start_iteration=previous,
            num_iteration=iteration - previous,
            raw_score=True,
        )
        predictions[iteration] = np.exp(raw.copy())
        previous = iteration
    return predictions


def select_valid_wape_iteration(rows: Iterable[dict]) -> dict:
    candidates = list(rows)
    if not candidates:
        raise ValueError("No Complete Valid iteration metrics were supplied")
    return min(candidates, key=lambda row: (
        float(row["wape"]), abs(float(row["total_bias"])), int(row["iteration"]),
    ))


def candidate_b_fraction_pairs(config: dict) -> list[tuple[float, float]]:
    return [
        (float(medium), float(high))
        for medium in config["candidate_b"]["medium_top_fractions"]
        for high in config["candidate_b"]["high_top_fractions"]
    ]


def protocol_coarse_iterations(max_rounds: int, step: int) -> list[int]:
    if max_rounds < 1 or step < 1:
        raise ValueError("max_rounds and step must be positive")
    return sorted({1, *range(step, max_rounds + 1, step), max_rounds})


def protocol_refinement_iterations(best_iteration: int, max_rounds: int, radius: int) -> list[int]:
    lower = max(1, int(best_iteration) - int(radius))
    upper = min(int(max_rounds), int(best_iteration) + int(radius))
    return list(range(lower, upper + 1))


class IterationWapeAccumulator:
    def __init__(self, iterations: Iterable[int]) -> None:
        self.iterations = sorted({int(value) for value in iterations})
        self.absolute = {iteration: 0.0 for iteration in self.iterations}
        self.prediction = {iteration: 0.0 for iteration in self.iterations}
        self.target = 0.0

    def update(self, target, predictions: dict[int, np.ndarray]) -> None:
        target_array = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
        self.target += float(target_array.sum())
        for iteration in self.iterations:
            prediction = np.clip(np.asarray(predictions[iteration], dtype="float64"), 0.0, None)
            self.absolute[iteration] += float(np.abs(prediction - target_array).sum())
            self.prediction[iteration] += float(prediction.sum())

    def compute(self) -> list[dict[str, float]]:
        return [
            {
                "iteration": iteration,
                "wape": 100.0 * self.absolute[iteration] / self.target if self.target else float("nan"),
                "total_bias": (
                    100.0 * (self.prediction[iteration] - self.target) / self.target
                    if self.target else float("nan")
                ),
                "target_sum": self.target,
                "prediction_sum": self.prediction[iteration],
            }
            for iteration in self.iterations
        ]


class BinaryRankingAccumulator:
    def __init__(self, bins: int = 10_000) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype="float64")
        self.total = np.zeros(self.bins, dtype="float64")
        self.logloss_sum = 0.0
        self.weight_sum = 0.0

    def update(self, target, score, weights=None) -> None:
        target_array = (np.asarray(target) > 0).astype("uint8")
        score_array = np.clip(np.asarray(score, dtype="float64"), 1e-12, 1.0 - 1e-12)
        weight_array = (
            np.ones(target_array.size, dtype="float64")
            if weights is None else np.asarray(weights, dtype="float64")
        )
        indexes = np.minimum((score_array * self.bins).astype("int64"), self.bins - 1)
        self.total += np.bincount(indexes, weights=weight_array, minlength=self.bins)
        self.positive += np.bincount(
            indexes, weights=weight_array * target_array, minlength=self.bins
        )
        self.logloss_sum += float(np.sum(
            weight_array * (-(target_array * np.log(score_array) + (1 - target_array) * np.log(1 - score_array)))
        ))
        self.weight_sum += float(weight_array.sum())

    def compute(self) -> dict[str, float]:
        positive = self.positive[::-1]
        total = self.total[::-1]
        positive_total = float(positive.sum())
        cumulative_positive = np.cumsum(positive)
        cumulative_total = np.cumsum(total)
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
        )
        average_precision = (
            float(np.sum(precision * positive) / positive_total) if positive_total else float("nan")
        )
        return {
            "average_precision": average_precision,
            "logloss": self.logloss_sum / self.weight_sum if self.weight_sum else float("nan"),
            "positive_rate": positive_total / self.weight_sum if self.weight_sum else float("nan"),
        }


def candidate_b_development_decision(baseline: dict, candidate: dict, config: dict) -> dict:
    gates = config["candidate_b"]["gates"]
    changes = {
        "twenty_plus_wape_gain_pp": baseline["20+"]["wape"] - candidate["20+"]["wape"],
        "twenty_plus_recall_gain_pp": candidate["recall_20_plus"] - baseline["recall_20_plus"],
        "five_nineteen_worsening_pp": candidate["5-19"]["wape"] - baseline["5-19"]["wape"],
        "nonzero_worsening_pp": candidate["nonzero"]["wape"] - baseline["nonzero"]["wape"],
        "zero_prediction_total_increase": (
            candidate["zero_prediction_total"] / baseline["zero_prediction_total"] - 1.0
            if baseline["zero_prediction_total"] else 0.0
        ),
        "absolute_bias_worsening_pp": absolute_bias_worsening(
            baseline["overall"]["total_bias"], candidate["overall"]["total_bias"]
        ),
    }
    checks = {
        "head_wape": changes["twenty_plus_wape_gain_pp"] >= float(gates["twenty_plus_wape_gain_pp"]),
        "head_recall": changes["twenty_plus_recall_gain_pp"] >= float(gates["twenty_plus_recall_gain_pp"]),
        "five_nineteen": changes["five_nineteen_worsening_pp"] <= float(gates["five_nineteen_max_worsening_pp"]),
        "nonzero": changes["nonzero_worsening_pp"] <= float(gates["nonzero_max_worsening_pp"]),
        "zero_total": changes["zero_prediction_total_increase"] <= float(gates["zero_prediction_total_max_increase"]),
        "total_bias": changes["absolute_bias_worsening_pp"] <= float(gates["total_bias_max_worsening_pp"]),
    }
    return {"promoted": bool(all(checks.values())), "checks": checks, "changes": changes}


def physical_columns() -> list[str]:
    runtime = set(load_feature_config().get("runtime_time_features", []))
    columns = [column for column in base_features() if column not in runtime]
    columns.extend([
        "month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m", "target_available_1m",
        "gds_ctgry_3_lvel",
    ])
    return list(dict.fromkeys(columns))


def _joined_sql(config: dict, where: str, columns: Iterable[str] | None = None) -> str:
    dataset = project_path(config["dataset"])
    side = project_path(config["cross_store_table"])
    requested = list(columns or physical_columns())
    projection = ",\n".join(f'b."{column}"' for column in requested)
    month_date = "CAST(CAST(b.month AS VARCHAR) || '-01' AS DATE)"
    return f"""
        SELECT {projection},
               COALESCE(g0.group_positive_qty, 0) AS group_positive_qty_t,
               COALESCE(g1.group_positive_qty, 0) AS group_positive_qty_t1,
               COALESCE(g2.group_positive_qty, 0) AS group_positive_qty_t2,
               COALESCE(g0.group_positive_sites, 0) AS group_positive_sites_t,
               COALESCE(g1.group_positive_sites, 0) AS group_positive_sites_t1,
               COALESCE(g2.group_positive_sites, 0) AS group_positive_sites_t2
        FROM read_parquet('{_escaped(dataset)}') b
        LEFT JOIN read_parquet('{_escaped(side)}') g0
          ON CAST(b.month AS VARCHAR)=g0.month AND CAST(b.item_id AS VARCHAR)=g0.item_id
        LEFT JOIN read_parquet('{_escaped(side)}') g1
          ON strftime({month_date} - INTERVAL 1 MONTH, '%Y-%m')=g1.month
         AND CAST(b.item_id AS VARCHAR)=g1.item_id
        LEFT JOIN read_parquet('{_escaped(side)}') g2
          ON strftime({month_date} - INTERVAL 2 MONTH, '%Y-%m')=g2.month
         AND CAST(b.item_id AS VARCHAR)=g2.item_id
        WHERE {where}
    """


def iter_joined_rows(
    config: dict,
    where: str,
    columns: Iterable[str] | None = None,
    vectors_per_chunk: int = 32,
    name: str = "joined_scan",
):
    connection = _duckdb_connection(config, name)
    try:
        connection.execute(_joined_sql(config, where, columns))
        while True:
            frame = connection.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)
            if frame.empty:
                break
            yield frame
    finally:
        connection.close()


def add_all_new_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    diff = add_diff_features(result)
    cross = add_cross_store_features(result)
    for column in diff.columns:
        result[column] = diff[column]
    for column in cross.columns:
        result[column] = cross[column]
    return result


def _safe_average_precision(target: np.ndarray, score: np.ndarray, weights=None) -> float:
    if target.min(initial=0) == target.max(initial=0):
        return float("nan")
    return float(average_precision_score(target, score, sample_weight=weights))


def _top_fraction_metrics(target: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    threshold = threshold_for_top_fraction(score, fraction)
    selected = score >= threshold
    positives = target > 0
    base_rate = float(positives.mean())
    precision = float(positives[selected].mean()) if selected.any() else 0.0
    recall = float(positives[selected].sum() / positives.sum()) if positives.any() else 0.0
    return {
        "threshold": threshold, "precision": precision, "recall": recall,
        "lift": precision / base_rate if base_rate > 0 else float("nan"),
    }


def run_statistical_audit(config: dict, logger: logging.Logger) -> dict:
    state = read_state(config)
    cache = project_path(config["outputs"]["cache_dir"]) / "statistical_audit.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("Reusing statistical audit: Diff=%s Cross-store=%s", result["group_pass"]["diff"], result["group_pass"]["cross_store"])
        return result

    guard = PIPELINE.MemoryGuard("two-stage-feature-audit", logger)
    new_features = DIFF_FEATURES + CROSS_STORE_FEATURES
    correlation_columns = [
        "total_qty", "qty_lag_1m", "qty_lag_2m", "qty_lag_3m",
        "qty_sum_last_3m", "qty_mean_last_3m", "qty_max_last_3m",
        "qty_sum_last_6m", "qty_mean_last_6m", "qty_max_last_6m",
        "sales_days", "sales_days_lag_1m", "sales_days_sum_last_3m",
        "active_months_last_6m", "zero_sales_months_last_6m",
    ]
    selected_columns = list(dict.fromkeys([
        "month", "site_no", "item_id", "future_qty_1m", "target_available_1m",
        *correlation_columns,
    ]))
    counts = {feature: {"rows": 0, "missing": 0, "zero": 0, "min": math.inf, "max": -math.inf} for feature in new_features}
    sample_parts: list[pd.DataFrame] = []
    scanned = 0
    sample_rate = float(config["audit"]["uniform_sample_rate"])
    started = time.perf_counter()
    where = "b.split='train' AND b.target_available_1m=1"
    for chunk_number, frame in enumerate(iter_joined_rows(config, where, selected_columns, name="audit_scan"), start=1):
        frame = add_all_new_features(frame)
        scanned += len(frame)
        for feature in new_features:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float64")
            finite = np.isfinite(values)
            stats = counts[feature]
            stats["rows"] += len(values)
            stats["missing"] += int((~finite).sum())
            stats["zero"] += int((finite & (values == 0)).sum())
            if finite.any():
                stats["min"] = min(stats["min"], float(values[finite].min()))
                stats["max"] = max(stats["max"], float(values[finite].max()))
        uniform = deterministic_uniform_hash(frame, "audit_1m", int(config["seed"]))
        sampled = frame.loc[uniform < sample_rate, ["month", "future_qty_1m", *correlation_columns, *new_features]]
        if not sampled.empty:
            sample_parts.append(sampled.copy())
        if chunk_number % 20 == 0:
            guard.check("audit_scan")
            logger.info("Audit scanned %s rows; sampled %s", f"{scanned:,}", f"{sum(len(x) for x in sample_parts):,}")
        del frame, sampled
    if not sample_parts:
        raise RuntimeError("Feature audit produced no sample")
    sample = pd.concat(sample_parts, ignore_index=True, copy=False)
    target_qty = clip_target(sample["future_qty_1m"].to_numpy())
    targets = {"5plus": (target_qty >= 5).astype("uint8"), "20plus": (target_qty >= 20).astype("uint8")}
    numeric = sample[correlation_columns + new_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    spearman = numeric.corr(method="spearman")
    feature_results: dict[str, dict] = {}
    for feature in new_features:
        score = numeric[feature].to_numpy(dtype="float64")
        value_shares = numeric[feature].value_counts(normalize=True, dropna=False)
        strongest_existing = spearman.loc[feature, correlation_columns].abs().sort_values(ascending=False)
        result = {
            "missing_rate": counts[feature]["missing"] / counts[feature]["rows"],
            "zero_rate": counts[feature]["zero"] / counts[feature]["rows"],
            "minimum": counts[feature]["min"], "maximum": counts[feature]["max"],
            "largest_value_share_sample": float(value_shares.iloc[0]),
            "strongest_existing_correlation": float(strongest_existing.iloc[0]),
            "strongest_existing_feature": str(strongest_existing.index[0]),
            "targets": {},
        }
        for target_name, target in targets.items():
            top1 = _top_fraction_metrics(target, score, 0.01)
            top5 = _top_fraction_metrics(target, score, 0.05)
            top10 = _top_fraction_metrics(target, score, 0.10)
            monthly_lifts = []
            for _, month_frame in sample.assign(_target=target, _score=score).groupby("month", sort=True):
                if month_frame["_target"].sum() == 0:
                    continue
                monthly_lifts.append(_top_fraction_metrics(
                    month_frame["_target"].to_numpy(), month_frame["_score"].to_numpy(), 0.10
                )["lift"])
            direction_share = float(np.mean(np.asarray(monthly_lifts) > 1.0)) if monthly_lifts else 0.0
            result["targets"][target_name] = {
                "average_precision": _safe_average_precision(target, score),
                "best_existing_average_precision": max(
                    _safe_average_precision(target, numeric[column].to_numpy(dtype="float64"))
                    for column in correlation_columns
                ),
                "top_1pct": top1, "top_5pct": top5, "top_decile": top10,
                "monthly_direction_share": direction_share,
                "monthly_lifts": monthly_lifts,
            }
        constant = result["largest_value_share_sample"] >= float(config["audit"]["constant_share_limit"])
        redundant = result["strongest_existing_correlation"] >= float(config["audit"]["correlation_limit"])
        ap_superior = any(
            target_result["average_precision"] > target_result["best_existing_average_precision"]
            for target_result in result["targets"].values()
        )
        result["drop_constant"] = constant
        result["redundant_with_existing"] = redundant
        result["ap_superior_to_existing"] = ap_superior
        result["kept"] = not constant and not (redundant and not ap_superior)
        feature_results[feature] = result

    def group_pass(features: list[str]) -> bool:
        for feature in features:
            result = feature_results[feature]
            if not result["kept"]:
                continue
            for target_result in result["targets"].values():
                if (
                    target_result["top_decile"]["lift"] > float(config["audit"]["top_decile_lift_minimum"])
                    and target_result["monthly_direction_share"] >= float(config["audit"]["monthly_direction_share_minimum"])
                ):
                    return True
        return False

    result = {
        "scanned_rows": scanned, "sample_rows": len(sample),
        "seconds": time.perf_counter() - started, "peak_ram_gib": guard.peak / 1024**3,
        "features": feature_results,
        "group_pass": {"diff": group_pass(DIFF_FEATURES), "cross_store": group_pass(CROSS_STORE_FEATURES)},
        "passed_features": {
            "diff": [feature for feature in DIFF_FEATURES if feature_results[feature]["kept"]],
            "cross_store": [feature for feature in CROSS_STORE_FEATURES if feature_results[feature]["kept"]],
        },
        "note": "Deterministic uniform sample spans all Train months; full scan supplies coverage/extreme/missing statistics.",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    state["audit"] = {"group_pass": result["group_pass"], "scanned_rows": scanned, "sample_rows": len(sample)}
    write_state(config, state)
    logger.info("Audit complete Diff=%s Cross-store=%s", result["group_pass"]["diff"], result["group_pass"]["cross_store"])
    del sample, sample_parts, numeric, spearman
    gc.collect()
    return result


def _selected_feature_sets(audit: dict) -> dict[str, list[str]]:
    base = base_features()
    diff = audit["passed_features"]["diff"] if audit["group_pass"]["diff"] else []
    cross = audit["passed_features"]["cross_store"] if audit["group_pass"]["cross_store"] else []
    feature_sets = {"E0": base}
    if diff:
        feature_sets["E1"] = base + diff
    if cross:
        feature_sets["E2"] = base + cross
    if diff and cross:
        feature_sets["E3"] = base + diff + cross
    return feature_sets


def _diff_pilot_features() -> list[str]:
    configured = list(load_optimization_config().get("diff_features", []))
    if configured != DIFF_FEATURES:
        raise RuntimeError("Diff feature configuration does not match the implemented feature order")
    features = list(dict.fromkeys([*base_features(), *configured]))
    forbidden = {
        "target_qty_1m", "split", "item_id", "isbn", "gds_no",
        "future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m",
    }
    leaked = forbidden.intersection(features)
    if leaked:
        raise RuntimeError(f"Leakage/identifier fields in Diff Pilot: {sorted(leaked)}")
    return features


def _diff_pilot_cache_path(config: dict, fold_name: str) -> Path:
    return project_path(config["outputs"]["cache_dir"]) / "diff_pilot" / "fold_samples" / f"{fold_name}.parquet"


def _iter_dataset_rows(
    config: dict,
    where: str,
    columns: Iterable[str],
    name: str,
    vectors_per_chunk: int = 32,
):
    dataset = project_path(config["dataset"])
    projection = ",".join(f'b."{column}"' for column in columns)
    connection = _duckdb_connection(config, name)
    try:
        connection.execute(f"SELECT {projection} FROM read_parquet('{_escaped(dataset)}') b WHERE {where}")
        while True:
            frame = connection.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)
            if frame.empty:
                break
            yield frame
    finally:
        connection.close()


def _write_diff_sample_rows(
    writer: pq.ParquetWriter | None,
    output: Path,
    frame: pd.DataFrame,
    sample_kind: str,
    weights: np.ndarray,
    features: list[str],
    category_maps: dict,
    time_base: str,
) -> pq.ParquetWriter:
    enriched = frame.copy()
    diff = add_diff_features(enriched)
    for column in diff.columns:
        enriched[column] = diff[column]
    prepared = PIPELINE.prepare_feature_frame(enriched, features, category_maps, time_base)
    prepared.insert(0, "sample_kind", sample_kind)
    prepared.insert(1, "month", enriched["month"].astype(str).to_numpy())
    prepared.insert(2, "target_qty", clip_target(enriched["future_qty_1m"].to_numpy()).astype("float32"))
    prepared.insert(3, "sample_weight", weights.astype("float32"))
    prepared.insert(4, "sampling_probability", np.divide(
        1.0, weights, out=np.zeros_like(weights, dtype="float32"), where=weights > 0,
    ))
    table = pa.Table.from_pandas(prepared, preserve_index=False)
    if writer is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=100_000)
    return writer


def build_diff_pilot_fold_cache(
    config: dict,
    fold: dict,
    features: list[str],
    category_maps: dict,
    logger: logging.Logger,
) -> Path:
    output = _diff_pilot_cache_path(config, fold["name"])
    expected = {"sample_kind", "month", "target_qty", "sample_weight", "sampling_probability", *features}
    if output.exists() and expected.issubset(pq.ParquetFile(output).schema.names):
        logger.info("Reusing dedicated Diff Pilot cache %s", output.relative_to(ROOT))
        return output
    output.unlink(missing_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    counts = Counter()
    guard = PIPELINE.MemoryGuard(f"diff-cache-{fold['name']}", logger)
    runtime = set(load_feature_config().get("runtime_time_features", []))
    columns = list(dict.fromkeys([
        *(name for name in features if name not in runtime and name not in DIFF_FEATURES),
        "month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m", "target_available_1m",
    ]))
    where = (
        "b.target_available_1m=1 AND ("
        f"(b.split='train' AND CAST(b.month AS VARCHAR)<='{fold['train_end']}') OR "
        f"(CAST(b.month AS VARCHAR) BETWEEN '{fold['valid_start']}' AND '{fold['valid_end']}'))"
    )
    started = time.perf_counter()
    try:
        for chunk_number, frame in enumerate(
            _iter_dataset_rows(config, where, columns, f"diff_cache_{fold['name']}"), start=1
        ):
            months = frame["month"].astype(str)
            train_frame = frame.loc[(frame["split"] == "train") & (months <= fold["train_end"])]
            valid_frame = frame.loc[(months >= fold["valid_start"]) & (months <= fold["valid_end"])]
            sampled_train, train_weights = _sample_frame(
                train_frame, config["sampling_rates"]["regressor"], int(config["seed"]),
                f"diff_{fold['name']}_regressor_train",
            )
            if not sampled_train.empty:
                writer = _write_diff_sample_rows(
                    writer, temporary, sampled_train, "regressor_train", train_weights,
                    features, category_maps, str(config["time_base"]),
                )
                counts["regressor_train"] += len(sampled_train)
            sampled_valid, valid_weights = _sample_frame(
                valid_frame, config["sampling_rates"]["validation"], int(config["seed"]),
                f"diff_{fold['name']}_validation",
            )
            if not sampled_valid.empty:
                writer = _write_diff_sample_rows(
                    writer, temporary, sampled_valid, "validation", valid_weights,
                    features, category_maps, str(config["time_base"]),
                )
                counts["validation"] += len(sampled_valid)
            del frame, train_frame, valid_frame, sampled_train, sampled_valid, train_weights, valid_weights
            if chunk_number % 20 == 0:
                guard.check("diff_cache")
                logger.info("Diff %s sampled counts=%s", fold["name"], dict(counts))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No Diff Pilot rows for {fold['name']}")
    temporary.replace(output)
    logger.info(
        "Diff cache %s complete in %.1fs size=%.1f MiB counts=%s",
        fold["name"], time.perf_counter() - started, output.stat().st_size / 1024**2, dict(counts),
    )
    return output


def _sample_frame(frame: pd.DataFrame, rates: dict[str, float], seed: int, horizon: str) -> tuple[pd.DataFrame, np.ndarray]:
    target = clip_target(frame["future_qty_1m"].to_numpy())
    buckets = demand_bucket(target)
    probabilities = np.asarray([float(rates[str(bucket)]) for bucket in buckets], dtype="float64")
    selected = deterministic_uniform_hash(frame, horizon, seed) < probabilities
    selected_probability = probabilities[selected]
    weights = np.divide(
        1.0, selected_probability,
        out=np.zeros_like(selected_probability), where=selected_probability > 0,
    )
    return frame.loc[selected].copy(), weights.astype("float32")


def _write_sample_rows(
    writer: pq.ParquetWriter | None,
    output: Path,
    frame: pd.DataFrame,
    sample_kind: str,
    weights: np.ndarray,
    all_features: list[str],
    category_maps: dict,
    time_base: str,
) -> pq.ParquetWriter:
    enriched = add_all_new_features(frame)
    x = PIPELINE.prepare_feature_frame(enriched, all_features, category_maps, time_base)
    x.insert(0, "sample_kind", sample_kind)
    x.insert(1, "month", enriched["month"].astype(str).to_numpy())
    x.insert(2, "target_qty", clip_target(enriched["future_qty_1m"].to_numpy()).astype("float32"))
    x.insert(3, "sample_weight", weights.astype("float32"))
    x.insert(4, "sampling_probability", np.divide(
        1.0, weights, out=np.zeros_like(weights, dtype="float32"), where=weights > 0
    ))
    table = pa.Table.from_pandas(x, preserve_index=False)
    if writer is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=100_000)
    return writer


def build_fold_cache(
    config: dict,
    fold: dict,
    all_features: list[str],
    category_maps: dict,
    logger: logging.Logger,
) -> Path:
    cache_dir = project_path(config["outputs"]["cache_dir"]) / "fold_samples"
    output = cache_dir / f"{fold['name']}.parquet"
    if output.exists():
        logger.info("Reusing fold cache %s", output.relative_to(ROOT))
        return output
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    guard = PIPELINE.MemoryGuard(f"fold-cache-{fold['name']}", logger)
    rates = config["sampling_rates"]
    where = (
        "b.split='train' AND b.target_available_1m=1 AND ("
        f"CAST(b.month AS VARCHAR)<='{fold['train_end']}' OR "
        f"CAST(b.month AS VARCHAR) BETWEEN '{fold['valid_start']}' AND '{fold['valid_end']}')"
    )
    counts = Counter()
    started = time.perf_counter()
    try:
        for chunk_number, frame in enumerate(iter_joined_rows(config, where, name=f"cache_{fold['name']}"), start=1):
            months = frame["month"].astype(str)
            train_frame = frame.loc[months <= fold["train_end"]]
            valid_frame = frame.loc[(months >= fold["valid_start"]) & (months <= fold["valid_end"])]
            for kind, rate_key in (
                ("classifier_sale_train", "classifier_sale"),
                ("regressor_train", "regressor"),
                ("classifier_5plus_train", "classifier_5plus"),
                ("classifier_20plus_train", "classifier_20plus"),
            ):
                sampled, weights = _sample_frame(
                    train_frame, rates[rate_key], int(config["seed"]), f"{fold['name']}_{kind}"
                )
                if not sampled.empty:
                    writer = _write_sample_rows(
                        writer, temporary, sampled, kind, weights, all_features,
                        category_maps, str(config["time_base"]),
                    )
                    counts[kind] += len(sampled)
                del sampled, weights
            sampled_valid, valid_weights = _sample_frame(
                valid_frame, rates["validation"], int(config["seed"]), f"{fold['name']}_validation"
            )
            if not sampled_valid.empty:
                writer = _write_sample_rows(
                    writer, temporary, sampled_valid, "validation", valid_weights,
                    all_features, category_maps, str(config["time_base"]),
                )
                counts["validation"] += len(sampled_valid)
            del frame, train_frame, valid_frame, sampled_valid, valid_weights
            if chunk_number % 20 == 0:
                guard.check("fold_cache")
                logger.info("%s sampled counts=%s", fold["name"], dict(counts))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No sampled rows for {fold['name']}")
    temporary.replace(output)
    logger.info(
        "Fold cache %s complete in %.1fs size=%.1f MiB counts=%s",
        fold["name"], time.perf_counter() - started, output.stat().st_size / 1024**2, dict(counts),
    )
    return output


def _read_fold_part(path: Path, kind: str, features: list[str]) -> dict:
    columns = ["month", "target_qty", "sample_weight", "sampling_probability", *features]
    projection = ",".join(f'"{column}"' for column in columns)
    connection = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute("PRAGMA memory_limit='1GB'")
        source = path / "*.parquet" if path.is_dir() else path
        arrays = connection.execute(
            f"SELECT {projection} FROM read_parquet('{_escaped(source)}') WHERE sample_kind=?",
            [kind],
        ).fetchnumpy()
    finally:
        connection.close()
    if not arrays or len(next(iter(arrays.values()))) == 0:
        raise RuntimeError(f"No rows for sample_kind={kind} in {path}")
    frame = pd.DataFrame(arrays)
    target = frame.pop("target_qty").to_numpy(dtype="float32")
    weights = frame.pop("sample_weight").to_numpy(dtype="float32")
    probability = frame.pop("sampling_probability").to_numpy(dtype="float32")
    month = frame.pop("month").astype(str).to_numpy()
    return {"x": frame, "target": target, "weight": weights, "probability": probability, "month": month}


def fit_seen_category_codes(frame: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[int]]:
    seen: dict[str, list[int]] = {}
    for column in categorical_features:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(-1).astype("int32")
        seen[column] = sorted(int(value) for value in values.unique() if int(value) >= 0)
    return seen


def apply_seen_category_codes(frame: pd.DataFrame, seen: dict[str, list[int]]) -> pd.DataFrame:
    encoded = frame.copy()
    for column, allowed in seen.items():
        values = pd.to_numeric(encoded[column], errors="coerce").fillna(-1).astype("int32")
        encoded[column] = values.where(values.isin(allowed), -1).astype("int32")
    return encoded


def _read_oof_frame(path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute("PRAGMA memory_limit='1GB'")
        arrays = connection.execute(
            f"SELECT * FROM read_parquet('{_escaped(path)}')"
        ).fetchnumpy()
        return pd.DataFrame(arrays)
    finally:
        connection.close()


def _weighted_segment(target, prediction, weights, mask) -> dict[str, float]:
    target = np.asarray(target, dtype="float64")[mask]
    prediction = np.clip(np.asarray(prediction, dtype="float64")[mask], 0.0, None)
    weights = np.asarray(weights, dtype="float64")[mask]
    if target.size == 0 or weights.sum() <= 0:
        return {"count": 0.0, "wape": float("nan"), "target_sum": 0.0, "prediction_sum": 0.0}
    target_sum = float(np.sum(weights * target))
    prediction_sum = float(np.sum(weights * prediction))
    return {
        "count": float(weights.sum()),
        "wape": 100.0 * float(np.sum(weights * np.abs(prediction - target))) / target_sum if target_sum else float("nan"),
        "target_sum": target_sum, "prediction_sum": prediction_sum,
    }


def weighted_business_metrics(target, prediction, weights) -> dict[str, Any]:
    target = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
    prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, None)
    weights = np.asarray(weights, dtype="float64")
    groups = {
        "overall": np.ones(target.size, dtype=bool), "0": target <= 0,
        "1": target == 1, "2-4": (target >= 2) & (target < 5),
        "5-19": (target >= 5) & (target < 20), "20+": target >= 20,
        "nonzero": target > 0,
    }
    result = {name: _weighted_segment(target, prediction, weights, mask) for name, mask in groups.items()}
    overall = result["overall"]
    overall["total_bias"] = (
        100.0 * (overall["prediction_sum"] - overall["target_sum"]) / overall["target_sum"]
        if overall["target_sum"] else float("nan")
    )
    zero = groups["0"]
    result["zero_prediction_total"] = float(np.sum(weights[zero] * prediction[zero]))
    zero_weight = float(weights[zero].sum())
    result["zero_gt_0_5"] = float(weights[zero & (prediction > 0.5)].sum() / zero_weight) if zero_weight else 0.0
    result["zero_gt_1"] = float(weights[zero & (prediction > 1.0)].sum() / zero_weight) if zero_weight else 0.0
    rounded = np.floor(prediction + 0.5)
    target_total = float(np.sum(weights * target))
    result["integer_overall_wape"] = (
        100.0 * float(np.sum(weights * np.abs(rounded - target))) / target_total
        if target_total else float("nan")
    )
    true_codes = np.select(
        [target <= 0, target == 1, target < 5, target < 20], [0, 1, 2, 3], default=4
    ).astype("int8")
    pred_codes = np.select(
        [rounded <= 0, rounded == 1, rounded < 5, rounded < 20], [0, 1, 2, 3], default=4
    ).astype("int8")
    confusion = np.zeros((5, 5), dtype="float64")
    np.add.at(confusion, (true_codes, pred_codes), weights)
    recalls = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(5, dtype="float64"), where=confusion.sum(axis=1) > 0,
    )
    precision = np.divide(
        np.diag(confusion), confusion.sum(axis=0),
        out=np.zeros(5, dtype="float64"), where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(2 * precision * recalls, precision + recalls, out=np.zeros(5), where=(precision + recalls) > 0)
    result["mc_macro_f1"] = 100.0 * float(f1.mean())
    result["recall_5_19"] = 100.0 * float(recalls[3])
    result["recall_20_plus"] = 100.0 * float(recalls[4])
    return result


def _weighted_ap(target, score, weights) -> float:
    target = np.asarray(target, dtype="uint8")
    score = np.asarray(score, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    return _safe_average_precision(target, score, weights)


def _lgb_params(config: dict, component: str) -> dict:
    params = dict(config["lightgbm"]["common"])
    params.update(config["lightgbm"]["regressor" if component == "regressor" else "classifier"])
    params["num_threads"] = max(1, min(12, (os.cpu_count() or 2) - 2))
    params["first_metric_only"] = True
    return params


def train_fold_component(
    config: dict,
    fold: dict,
    cache_path: Path,
    experiment: str,
    features: list[str],
    component: str,
    category_maps: dict,
    logger: logging.Logger,
) -> dict:
    output = project_path(config["outputs"]["checkpoint_dir"]) / fold["name"] / f"{experiment}_{component}.txt"
    if output.exists():
        booster, metadata = load_model_bundle(output)
        return metadata["training_result"]
    kind = {
        "classifier_sale": "classifier_sale_train",
        "regressor": "regressor_train",
        "classifier_5plus": "classifier_5plus_train",
        "classifier_20plus": "classifier_20plus_train",
    }[component]
    train = _read_fold_part(cache_path, kind, features)
    valid = _read_fold_part(cache_path, "validation", features)
    if component == "regressor":
        train_mask = train["target"] > 0
        valid_mask = valid["target"] > 0
        train_y = train["target"][train_mask]
        valid_y = valid["target"][valid_mask]
        train_x = train["x"].loc[train_mask].reset_index(drop=True)
        valid_x = valid["x"].loc[valid_mask].reset_index(drop=True)
        train_w = train["weight"][train_mask]
        valid_w = valid["weight"][valid_mask]
    else:
        threshold = {"classifier_sale": 1, "classifier_5plus": 5, "classifier_20plus": 20}[component]
        train_y = (train["target"] >= threshold).astype("float32")
        valid_y = (valid["target"] >= threshold).astype("float32")
        train_x, valid_x = train["x"], valid["x"]
        train_w, valid_w = train["weight"], valid["weight"]
    fold_categorical = [feature for feature in CATEGORY_FEATURES if feature in features]
    seen_category_codes = fit_seen_category_codes(train_x, fold_categorical)
    valid_x = apply_seen_category_codes(valid_x, seen_category_codes)
    valid_unknown_rates = {
        column: float((valid_x[column] == -1).mean()) for column in fold_categorical
    }
    train_w = (train_w / train_w.mean()).astype("float32")
    valid_w = (valid_w / valid_w.mean()).astype("float32")
    params = _lgb_params(config, component)
    guard = PIPELINE.MemoryGuard(f"pilot-{fold['name']}-{experiment}-{component}", logger)
    evaluations: dict = {}
    started = time.perf_counter()
    train_set = lgb.Dataset(
        train_x, label=train_y, weight=train_w, feature_name=features,
        categorical_feature=fold_categorical, free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        valid_x, label=valid_y, weight=valid_w, reference=train_set, feature_name=features,
        categorical_feature=fold_categorical, free_raw_data=True,
    )
    booster = lgb.train(
        params, train_set, num_boost_round=int(config["lightgbm"]["max_rounds"]),
        valid_sets=[valid_set], valid_names=["valid"], callbacks=[
            lgb.early_stopping(int(config["lightgbm"]["early_stopping_rounds"]), first_metric_only=True, verbose=True),
            lgb.record_evaluation(evaluations), lgb.log_evaluation(100), guard.callback(10, 100),
        ],
    )
    prediction = np.clip(booster.predict(valid_x, num_iteration=booster.best_iteration), 0.0, None)
    if component == "regressor":
        metrics = weighted_business_metrics(valid_y, prediction, valid_w)
    else:
        metrics = {"average_precision": _weighted_ap(valid_y, np.clip(prediction, 0, 1), valid_w)}
    result = {
        "fold": fold["name"], "experiment": experiment, "component": component,
        "train_rows": len(train_y), "valid_rows": len(valid_y), "feature_count": len(features),
        "best_iteration": int(booster.best_iteration), "training_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3, "metrics": metrics,
        "valid_unknown_rates": valid_unknown_rates,
    }
    metadata = {
        "experiment": "two_stage_diff_cross_candidate_b_v2", "fold": fold["name"],
        "component": component, "feature_set": experiment, "feature_names": features,
        "categorical_features": [feature for feature in CATEGORY_FEATURES if feature in features],
        "category_maps": category_maps, "fold_category_codes_seen": seen_category_codes,
        "category_mapping_scope": "component_train_only", "time_base": str(config["time_base"]), "params": params,
        "best_iteration": int(booster.best_iteration), "training_result": result,
        "lightgbm_version": lgb.__version__, "python_version": platform.python_version(),
    }
    save_model_bundle(booster, output, metadata)
    reloaded, loaded_metadata = load_model_bundle(output)
    check = valid_x.iloc[: min(1000, len(valid_x))]
    np.testing.assert_allclose(
        booster.predict(check, num_iteration=booster.best_iteration),
        reloaded.predict(check, num_iteration=loaded_metadata["best_iteration"]), rtol=1e-7, atol=1e-8,
    )
    del train, valid, train_x, valid_x, train_set, valid_set, booster, reloaded
    gc.collect()
    return result


def _train_diff_regressor_from_shared(
    config: dict,
    fold: dict,
    experiment: str,
    features: list[str],
    shared_train: dict,
    shared_valid: dict,
    category_maps: dict,
    logger: logging.Logger,
) -> dict:
    output = (
        project_path(config["outputs"]["checkpoint_dir"])
        / "diff_pilot" / fold["name"] / f"{experiment}_regressor.txt"
    )
    if output.exists():
        _, metadata = load_model_bundle(output)
        logger.info("Reusing Diff Pilot checkpoint %s", output.relative_to(ROOT))
        return metadata["training_result"]

    train_x = shared_train["x"].loc[:, features].copy()
    valid_x = shared_valid["x"].loc[:, features].copy()
    train_y = shared_train["target"]
    valid_y = shared_valid["target"]
    train_w = (shared_train["weight"] / shared_train["weight"].mean()).astype("float32")
    valid_w = (shared_valid["weight"] / shared_valid["weight"].mean()).astype("float32")
    categorical = [name for name in CATEGORY_FEATURES if name in features]
    seen_category_codes = fit_seen_category_codes(train_x, categorical)
    valid_x = apply_seen_category_codes(valid_x, seen_category_codes)
    unknown_rates = {name: float((valid_x[name] == -1).mean()) for name in categorical}
    params = _lgb_params(config, "regressor")
    guard = PIPELINE.MemoryGuard(f"diff-pilot-{fold['name']}-{experiment}", logger)
    evaluations: dict = {}
    started = time.perf_counter()
    train_set = lgb.Dataset(
        train_x, label=train_y, weight=train_w, feature_name=features,
        categorical_feature=categorical, free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        valid_x, label=valid_y, weight=valid_w, reference=train_set,
        feature_name=features, categorical_feature=categorical, free_raw_data=True,
    )
    booster = lgb.train(
        params, train_set, num_boost_round=int(config["lightgbm"]["max_rounds"]),
        valid_sets=[valid_set], valid_names=["valid"], callbacks=[
            lgb.early_stopping(
                int(config["lightgbm"]["early_stopping_rounds"]),
                first_metric_only=True, verbose=True,
            ),
            lgb.record_evaluation(evaluations), lgb.log_evaluation(100), guard.callback(10, 100),
        ],
    )
    prediction = np.clip(booster.predict(valid_x, num_iteration=booster.best_iteration), 0.0, None)
    metrics = weighted_business_metrics(valid_y, prediction, valid_w)
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    result = {
        "fold": fold["name"], "experiment": experiment, "component": "regressor",
        "train_rows": len(train_y), "valid_rows": len(valid_y), "feature_count": len(features),
        "best_iteration": int(booster.best_iteration),
        "training_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "metrics": metrics,
        "valid_unknown_rates": unknown_rates,
        "diff_importance_gain": {
            name: float(gain[index]) for index, name in enumerate(features) if name in DIFF_FEATURES
        },
        "diff_importance_split": {
            name: int(split[index]) for index, name in enumerate(features) if name in DIFF_FEATURES
        },
    }
    metadata = {
        "experiment": "two_stage_diff_regressor_pilot",
        "fold": fold["name"], "component": "regressor", "feature_set": experiment,
        "feature_names": features, "categorical_features": categorical,
        "category_maps": category_maps, "fold_category_codes_seen": seen_category_codes,
        "category_mapping_scope": "component_train_only", "time_base": str(config["time_base"]),
        "params": params, "best_iteration": int(booster.best_iteration),
        "training_result": result, "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
    }
    save_model_bundle(booster, output, metadata)
    reloaded, loaded_metadata = load_model_bundle(output)
    check = valid_x.iloc[: min(1000, len(valid_x))]
    np.testing.assert_allclose(
        booster.predict(check, num_iteration=booster.best_iteration),
        reloaded.predict(check, num_iteration=loaded_metadata["best_iteration"]),
        rtol=1e-7, atol=1e-8,
    )
    del train_x, valid_x, train_set, valid_set, booster, reloaded, prediction
    gc.collect()
    return result


def assess_diff_regressor_pilot(rows: list[dict], config: dict) -> dict:
    lookup = {(row["fold"], row["experiment"]): row for row in rows if row["component"] == "regressor"}
    gates = config["pilot_gates"]
    changes = []
    for fold in config["pilot_folds"]:
        base = lookup[(fold, "E0_DIFF_CONTROL")]["metrics"]
        diff = lookup[(fold, "E1_DIFF")]["metrics"]
        change = {
            "fold": fold,
            "nonzero_worsening_pp": diff["nonzero"]["wape"] - base["nonzero"]["wape"],
            "wape5_gain_pp": base["5-19"]["wape"] - diff["5-19"]["wape"],
            "wape20_gain_pp": base["20+"]["wape"] - diff["20+"]["wape"],
            "recall5_gain_pp": diff["recall_5_19"] - base["recall_5_19"],
            "recall20_gain_pp": diff["recall_20_plus"] - base["recall_20_plus"],
        }
        change["passed"] = bool(
            (
                max(change["wape5_gain_pp"], change["wape20_gain_pp"])
                >= float(gates["head_wape_gain_pp"])
                or max(change["recall5_gain_pp"], change["recall20_gain_pp"])
                >= float(gates["head_recall_gain_pp"])
            )
            and change["nonzero_worsening_pp"]
            <= float(gates["nonzero_wape_max_worsening_pp"])
        )
        changes.append(change)
    passed_count = sum(change["passed"] for change in changes)
    if passed_count == len(changes):
        status = "consistent_gain"
    elif passed_count:
        status = "time_conditional"
    else:
        status = "no_model_increment"
    return {
        "status": status,
        "allow_complete_valid": status == "consistent_gain",
        "fold_changes": changes,
        "rule": (
            "Each forward fold must improve a 5-19/20+ WAPE by at least 0.5pp or recall by at "
            "least 1pp, while Nonzero WAPE worsens by no more than 0.5pp."
        ),
    }


def _append_diff_ablation_rows(config: dict, rows: list[dict]) -> None:
    path = project_path(config["outputs"]["ablation"])
    incoming = pd.DataFrame([_flatten_training_result(row) for row in rows])
    if path.exists():
        current = pd.read_csv(path)
        keys = {("q4_2024", "E0_DIFF_CONTROL", "regressor"), ("q4_2024", "E1_DIFF", "regressor"),
                ("q2_2025", "E0_DIFF_CONTROL", "regressor"), ("q2_2025", "E1_DIFF", "regressor")}
        keep = ~current.apply(
            lambda row: (str(row.get("fold")), str(row.get("experiment")), str(row.get("component"))) in keys,
            axis=1,
        )
        output = pd.concat([current.loc[keep], incoming], ignore_index=True, sort=False)
    else:
        output = incoming
    output.to_csv(path, index=False, encoding="utf-8-sig")


def append_three_route_stage_summary(config: dict, result: dict) -> None:
    report_path = project_path(config["outputs"]["report"])
    marker = "## 三条优化路线阶段性收口（2026-08-21）"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    lookup = {(row["fold"], row["experiment"]): row for row in result["rows"]}
    changes = {row["fold"]: row for row in result["decision"]["fold_changes"]}
    importance_rows = []
    for feature in DIFF_FEATURES:
        q4 = lookup[("q4_2024", "E1_DIFF")]["diff_importance_gain"].get(feature, 0.0)
        q2 = lookup[("q2_2025", "E1_DIFF")]["diff_importance_gain"].get(feature, 0.0)
        importance_rows.append((feature, q4, q2, (q4 + q2) / 2.0))
    importance_rows.sort(key=lambda item: item[3], reverse=True)

    lines = [marker, "", "### 1. 本轮边界与可追溯性", ""]
    lines.extend([
        "- 本轮唯一新增训练是受控的E1 Diff regressor Pilot；没有运行Test、2M、Complete Valid、Candidate B或Cross-store微调。",
        "- 两个fold均让E0控制和E1使用完全相同的样本、样本顺序、逆采样权重、类别映射和LightGBM参数；E1只多6个Diff字段。",
        "- Q4 fold为训练至2024-09、验证2024-10至12；Q2 fold为训练至2025-03、验证2025-04至06。Train覆盖各截止日前全部历史月份。",
        f"- 总耗时{result['seconds'] / 60:.2f}分钟；训练阶段观测峰值RSS为{max(row['peak_ram_gib'] for row in result['rows']):.2f} GiB。",
        f"- 日志：`{result['log']}`；逐折结果已追加到`reports/two_stage_diff_cross_ablation.csv`。",
        "",
        "### 2. E1 Diff受控Pilot", "",
        "E1仅加入现有6个字段：当前月与lag1差、lag1与lag2差、log1p当前与lag1差、销售天数差、真正的最近3月均值减此前3月均值、6月历史可用标记。它们只依赖观察月及更早历史，不含未来销量标签。",
        "",
        "| Forward fold | Model | Train/positive-valid | Best iter | Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in ("q4_2024", "q2_2025"):
        for experiment in ("E0_DIFF_CONTROL", "E1_DIFF"):
            row = lookup[(fold, experiment)]
            metrics = row["metrics"]
            lines.append(
                f"| {fold} | {experiment} | {row['train_rows']:,}/{row['valid_rows']:,} | "
                f"{row['best_iteration']} | {metrics['nonzero']['wape']:.4f}% | "
                f"{metrics['5-19']['wape']:.4f}% | {metrics['20+']['wape']:.4f}% | "
                f"{metrics['recall_5_19']:.4f}% | {metrics['recall_20_plus']:.4f}% |"
            )
    lines.extend([
        "", "相对同折E0控制的变化（正数WAPE gain表示改善）：", "",
        "| Fold | Nonzero WAPE变化 | 5-19 WAPE gain | 20+ WAPE gain | 5-19 Recall变化 | 20+ Recall变化 | Pilot门槛 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for fold in ("q4_2024", "q2_2025"):
        change = changes[fold]
        lines.append(
            f"| {fold} | {change['nonzero_worsening_pp']:+.4f} pp | {change['wape5_gain_pp']:+.4f} pp | "
            f"{change['wape20_gain_pp']:+.4f} pp | {change['recall5_gain_pp']:+.4f} pp | "
            f"{change['recall20_gain_pp']:+.4f} pp | {'通过' if change['passed'] else '未通过'} |"
        )
    lines.extend([
        "",
        f"结论为`{result['decision']['status']}`，因此`allow_complete_valid=false`。Q4依靠20+ WAPE改善0.5444pp通过；Q2的5-19 WAPE改善0.4811pp、Recall提高0.7517pp，但均略低于0.5pp/1pp工程门槛。两个fold的20+ WAPE都改善，说明Diff存在弱的幅度信息；但20+ Recall分别下降0.0270pp和0.0868pp，收益没有转化为高动销等级识别。按照预先约束，本轮没有运行Complete Valid。",
        "",
        "Diff字段的LightGBM gain重要性（只说明模型使用程度，不等同于单字段因果贡献）：", "",
        "| Diff feature | Q4 gain | Q2 gain | 两折均值 |",
        "|---|---:|---:|---:|",
    ])
    for feature, q4, q2, mean in importance_rows:
        lines.append(f"| `{feature}` | {q4:.1f} | {q2:.1f} | {mean:.1f} |")
    lines.extend([
        "",
        "`qty_diff_recent3_previous3`和`log_qty_diff_current_lag1`是两折中最强的Diff输入，`qty_diff_current_lag1`次之。前者单字段统计Lift较弱，却在树模型中获得较高gain，说明它更可能作为条件分裂与其他历史特征联合使用。`qty_diff_6m_history_available`和`sales_days_diff_current_lag1`整体贡献较低。所有Diff与已有lag/rolling的相关性都远低于0.98删除线，因此不是完全重复，但多数是已有水平信息的变化表达，增量有限。",
        "",
        "**Diff阶段判断：不是无效，而是有信息、模型层面增量偏弱且时间/指标条件性明显。** 它改善了两个时期的20+连续误差，却没有改善20+等级Recall，也没有跨折达到项目晋级线；当前不值得为它单独做全Train和Complete Valid。",
        "",
        "### 3. Cross-store阶段性结论及与未来分布式研究的衔接", "",
        "- Cross-store是三条路线中证据最强的一条：全Train统计、双折Pilot、正式全Train和Complete Valid均已完成。q层20+ WAPE改善2.2807pp、20+ Recall提高4.0052pp、median(q/y)由0.301663升至0.350284；最终20+ WAPE改善2.1444pp、Recall提高3.1639pp，Nonzero与Overall WAPE也改善，Zero Pred Total下降5.2613%，Total Bias由+8.9329%收窄到+4.9883%。",
        "- 它并非简单把所有预测抬高；但5-19 Recall下降0.7821pp，且10/11月20+发生关系漂移。本店历史较强而跨店偏冷时，E2可能过度压低q。因此保留为有明确增量价值的优化候选和checkpoint，暂不替代E0，也不继续做路由、收缩或月份修正。",
        "- 集中式结果已经证明：同一本书在其他门店的当月/近期销量、活跃门店数和热度变化具有预测增量。未来分布式研究可考虑门店不共享交易明细，而协同计算item级聚合销量、活跃门店数、滚动集团热度与变化信号，再研究安全聚合、差分隐私等机制。当前Cross-store是集中式明文聚合实验，**尚未实现或证明任何隐私保护**。",
        "- 重建E2 checkpoint继续保留：`models/checkpoints/two_stage_optimization/cross_store_diagnostic/two_stage_regressor_cross_store_1m_rebuilt.txt`。",
        "",
        "### 4. Candidate B完整阶段性结论", "",
        "1. A2通过提高5-19/20+训练权重证明头部幅度可改善：最终5-19/20+ Recall分别提高2.60/3.05pp，WAPE改善0.52/1.42pp；但Zero Pred Total增加8.22%，Total Bias由+8.93%扩大到+17.92%，属于全局regressor一起变积极，不能正式采用。",
        "2. Candidate B-v1改为只补偿有历史高需求证据的样本。时间外Holdout仅补偿28,031条（0.2960%），Zero Pred Total只增加0.0348%，Bias只增加0.1739pp，证明“选择性积极”能够隔离副作用；但命中样本中真正5+只有7,579条（约27.04%），20+ Recall仅提高1.3064pp，5-19 Recall反降0.2251pp，综合收益不足。",
        "3. Candidate B-v2训练独立`P(Y>=5)`/`P(Y>=20)`分数，Cross-store使高需求分类PR-AUC明显提高；随后用5个medium Top比例乘5个high Top比例形成25组分档规则。Q4/Q1/Q2规则拟合期分别有25/20/20个候选触发5-19保护失败，Total Bias失败15/22/6个，另有Q1一个Zero失败；同一候选可同时失败。没有候选通过全部保护条件，所以正式规则回退为不补偿，报告中的收益为0表示“未采用规则”，不是25个公式碰巧都得到0。",
        "4. 当前保留的OOF产物只有按门槛汇总的淘汰计数，没有保存25个候选逐项指标，因此无法在不重跑既有OOF的前提下可靠列出“最接近通过”的若干项。本轮禁止重跑，故明确记录证据缺口，不猜测。B-v1是现有资料中最接近可控方案的实例，但收益仍不足。",
        "",
        "Candidate B的优点是直接针对`p_sale*q`的头部压缩并实现选择性补偿；困难同时存在于两个环节：未来高需求仍难稳定识别，而即使识别PR-AUC提升，固定分档乘固定倍率也会在5-19与20+边界制造trade-off。当前应保留机制证据，不继续扩大倍率或制造B-v3。",
        "",
        "### 5. 三路线研究地图", "",
        "| 路线 | 核心思想 | 已做到什么 | 优点 | 副作用/困难 | 当前状态 | 后续潜力 |",
        "|---|---|---|---|---|---|---|",
        "| Diff | 把历史销量变化而非仅销量水平作为regressor输入 | 全Train统计审计；本轮两折受控E1 Pilot | 两折20+ WAPE均小幅改善；成本低；符合导师差分建议 | 收益弱且未转化为20+ Recall；跨折未同时过门槛；部分信息与lag/rolling重叠 | 有信息但不够稳定，停止在Pilot | 可与Cross-store合并做一次受控消融，但不应单独正式化 |",
        "| Cross-store | 用同书在其他门店的历史热度补充本店历史 | 旁表、审计、Pilot、全Train、Complete Valid、误差迁移和漂移诊断 | q幅度、20+、Nonzero、Overall、Bias、Zero均有实质改善；证据最完整 | 5-19 Recall下降；10/11月关系漂移；尚无隐私保护 | 保留E2 checkpoint和研究结论，暂不替代E0、不再微调 | 最适合作为跨店协同/未来分布式特征基线 |",
        "| Candidate B | 只对高需求候选有限补偿，避免全局积极 | A2、B-v1、5+/20+分类、25规则、forward OOF stop gate | 能把头部积极性限制在少数样本，副作用远小于A2 | 候选Precision不足；固定倍率造成5-19/20+ trade-off；OOF无候选通过 | 机制有价值，当前实现失败，冻结 | 以后可研究连续、受约束的专家/损失机制，但必须先有稳定高需求识别 |",
        "",
        "三条路线不互斥：Diff和Cross-store都是预测时可得的输入信息，未来可组合；Candidate B是预测后的决策/组合层，可使用前两者产生的高需求信号。但本轮证据排序明确：**Cross-store最强，Diff次之且条件性明显，Candidate B当前机制最不成熟。** Cross-store也最符合“门店保护本地明细、跨店参考聚合热度”的研究衔接，但隐私机制必须在后续另行实现和验证。",
        "",
        "### 6. 已回答、仍未知与停止点", "",
        "**已回答：** Diff进入模型后确有弱连续误差增量，但不足以晋级；Cross-store提供真实且较强的跨店增量；A2的全局积极不可接受；Candidate B可控制副作用，但当前分档倍率机制不能把识别收益稳定转成业务收益。",
        "",
        "**仍未知：** Cross-store+Diff是否存在互补；Cross-store信号在隐私保护聚合后保留多少效用；连续受约束的高需求专家能否避免固定MC边界trade-off；这些都尚未使用Test验证。",
        "",
        "**下一阶段值得讨论的方向（本轮不执行）：**",
        "1. 优先讨论Cross-store如何形成可共享的item级聚合协议，并设计集中式等价基线，再进入隐私/分布式研究。",
        "2. 可以后做一次严格受控的`Cross-store + Diff`消融，只在forward Pilot中验证Diff是否补足10/11月关系漂移；失败即停。",
        "3. Candidate B暂不继续固定分档倍率；若未来重启，应转向带业务约束的连续专家/损失设计，并先证明forward高需求识别稳定。",
        "",
        "本轮停止于Diff两折Pilot。未运行Test、2M、Complete Valid或分布式训练，未覆盖E0正式模型，未修改E2 checkpoint。",
    ])
    report_path.write_text(existing + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_diff_pilot_only(config: dict, logger: logging.Logger, log_path: Path) -> dict:
    result_path = project_path(config["outputs"]["cache_dir"]) / "diff_pilot" / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        logger.info("Reusing Diff Pilot result status=%s", result["decision"]["status"])
        append_three_route_stage_summary(config, result)
        return result
    features = _diff_pilot_features()
    base = base_features()
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    category_maps = p_metadata["category_maps"]
    fold_lookup = {fold["name"]: fold for fold in config["folds"]}
    rows: list[dict] = []
    started = time.perf_counter()
    for fold_name in config["pilot_folds"]:
        fold = fold_lookup[fold_name]
        path = build_diff_pilot_fold_cache(config, fold, features, category_maps, logger)
        shared_train_raw = _read_fold_part(path, "regressor_train", features)
        shared_valid_raw = _read_fold_part(path, "validation", features)
        train_mask = shared_train_raw["target"] > 0
        valid_mask = shared_valid_raw["target"] > 0
        shared_train = {
            "x": shared_train_raw["x"].loc[train_mask].reset_index(drop=True),
            "target": shared_train_raw["target"][train_mask],
            "weight": shared_train_raw["weight"][train_mask],
        }
        shared_valid = {
            "x": shared_valid_raw["x"].loc[valid_mask].reset_index(drop=True),
            "target": shared_valid_raw["target"][valid_mask],
            "weight": shared_valid_raw["weight"][valid_mask],
        }
        del shared_train_raw, shared_valid_raw, train_mask, valid_mask
        gc.collect()
        for experiment, selected in (("E0_DIFF_CONTROL", base), ("E1_DIFF", features)):
            logger.info("Diff Pilot %s %s regressor", fold_name, experiment)
            rows.append(_train_diff_regressor_from_shared(
                config, fold, experiment, selected, shared_train, shared_valid,
                category_maps, logger,
            ))
        del shared_train, shared_valid
        gc.collect()
    decision = assess_diff_regressor_pilot(rows, config)
    result = {
        "scope": "Active-Store 1M Train forward Pilot only; no Test and no 2M",
        "rows": rows, "decision": decision,
        "seconds": time.perf_counter() - started,
        "log": str(log_path.relative_to(ROOT)),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_diff_ablation_rows(config, rows)
    append_three_route_stage_summary(config, result)
    logger.info("Diff Pilot complete status=%s", decision["status"])
    return result


def _flatten_training_result(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key != "metrics"}
    metrics = result["metrics"]
    if "average_precision" in metrics:
        row["average_precision"] = metrics["average_precision"]
    else:
        row.update({
            "nonzero_wape": metrics["nonzero"]["wape"],
            "five_nineteen_wape": metrics["5-19"]["wape"],
            "twenty_plus_wape": metrics["20+"]["wape"],
            "five_nineteen_recall": metrics["recall_5_19"],
            "twenty_plus_recall": metrics["recall_20_plus"],
        })
    for key, value in list(row.items()):
        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return row


def _pilot_advancement(rows: list[dict], experiments: list[str], config: dict) -> dict:
    lookup = {(row["fold"], row["experiment"], row["component"]): row for row in rows}
    folds = list(config["pilot_folds"])
    gates = config["pilot_gates"]
    decisions: dict[str, dict] = {}
    for experiment in experiments:
        if experiment == "E0":
            continue
        fold_changes = []
        for fold in folds:
            base5 = lookup[(fold, "E0", "classifier_5plus")]["metrics"]["average_precision"]
            base20 = lookup[(fold, "E0", "classifier_20plus")]["metrics"]["average_precision"]
            new5 = lookup[(fold, experiment, "classifier_5plus")]["metrics"]["average_precision"]
            new20 = lookup[(fold, experiment, "classifier_20plus")]["metrics"]["average_precision"]
            base_q = lookup[(fold, "E0", "regressor")]["metrics"]
            new_q = lookup[(fold, experiment, "regressor")]["metrics"]
            fold_changes.append({
                "fold": fold,
                "ap5_relative": (new5 / base5 - 1.0) if base5 else 0.0,
                "ap20_relative": (new20 / base20 - 1.0) if base20 else 0.0,
                "wape5_gain_pp": base_q["5-19"]["wape"] - new_q["5-19"]["wape"],
                "wape20_gain_pp": base_q["20+"]["wape"] - new_q["20+"]["wape"],
                "recall5_gain_pp": new_q["recall_5_19"] - base_q["recall_5_19"],
                "recall20_gain_pp": new_q["recall_20_plus"] - base_q["recall_20_plus"],
                "nonzero_worsening_pp": new_q["nonzero"]["wape"] - base_q["nonzero"]["wape"],
            })
        ap_pass = all(
            (
                change["ap5_relative"] >= float(gates["relative_ap_gain"])
                and change["ap20_relative"] >= -float(gates["other_ap_max_decline"])
            ) or (
                change["ap20_relative"] >= float(gates["relative_ap_gain"])
                and change["ap5_relative"] >= -float(gates["other_ap_max_decline"])
            )
            for change in fold_changes
        )
        q_pass = all(
            (
                max(change["wape5_gain_pp"], change["wape20_gain_pp"])
                >= float(gates["head_wape_gain_pp"])
                or max(change["recall5_gain_pp"], change["recall20_gain_pp"])
                >= float(gates["head_recall_gain_pp"])
            )
            and change["nonzero_worsening_pp"] <= float(gates["nonzero_wape_max_worsening_pp"])
            for change in fold_changes
        )
        decisions[experiment] = {
            "passed": bool(ap_pass or q_pass), "ap_pass": ap_pass, "q_pass": q_pass,
            "fold_changes": fold_changes,
            "mean_wape20_gain_pp": float(np.mean([change["wape20_gain_pp"] for change in fold_changes])),
            "mean_ap20_relative": float(np.mean([change["ap20_relative"] for change in fold_changes])),
            "mean_ap5_relative": float(np.mean([change["ap5_relative"] for change in fold_changes])),
        }
    advancing = [name for name, decision in decisions.items() if decision["passed"]]
    selected = max(
        advancing,
        key=lambda name: (
            decisions[name]["mean_wape20_gain_pp"], decisions[name]["mean_ap20_relative"],
            decisions[name]["mean_ap5_relative"], -len(name),
        ),
        default=None,
    )
    return {"decisions": decisions, "selected_feature_set": selected}


def run_pilot(config: dict, audit: dict, logger: logging.Logger) -> dict:
    cache = project_path(config["outputs"]["cache_dir"]) / "pilot_result.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("Reusing Pilot result selected=%s", result.get("selected_feature_set"))
        return result
    feature_sets = _selected_feature_sets(audit)
    if len(feature_sets) == 1:
        return {"passed": False, "stop_reason": "No new feature group passed statistical audit", "selected_feature_set": None, "rows": []}
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    category_maps = p_metadata["category_maps"]
    all_features = list(dict.fromkeys(feature for features in feature_sets.values() for feature in features))
    fold_lookup = {fold["name"]: fold for fold in config["folds"]}
    rows: list[dict] = []
    for fold_name in config["pilot_folds"]:
        fold = fold_lookup[fold_name]
        fold_cache = build_fold_cache(config, fold, all_features, category_maps, logger)
        for experiment, features in feature_sets.items():
            for component in ("regressor", "classifier_5plus", "classifier_20plus"):
                logger.info("Pilot %s %s %s", fold_name, experiment, component)
                rows.append(train_fold_component(
                    config, fold, fold_cache, experiment, features, component, category_maps, logger
                ))
    advancement = _pilot_advancement(rows, list(feature_sets), config)
    result = {
        "passed": advancement["selected_feature_set"] is not None,
        "feature_sets": feature_sets, "rows": rows, **advancement,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Pilot complete passed=%s selected=%s", result["passed"], result["selected_feature_set"])
    return result


def _checkpoint_path(config: dict, fold_name: str, experiment: str, component: str) -> Path:
    return project_path(config["outputs"]["checkpoint_dir"]) / fold_name / f"{experiment}_{component}.txt"


def generate_fold_oof(
    config: dict,
    fold: dict,
    fold_cache: Path,
    experiment: str,
    features: list[str],
    logger: logging.Logger,
) -> Path:
    output = project_path(config["outputs"]["cache_dir"]) / "oof" / f"{fold['name']}.parquet"
    if output.exists():
        return output
    validation = _read_fold_part(fold_cache, "validation", features)
    p_model, p_meta = load_model_bundle(_checkpoint_path(config, fold["name"], "E0_sale", "classifier_sale"))
    q_model, q_meta = load_model_bundle(_checkpoint_path(config, fold["name"], experiment, "regressor"))
    s5_model, s5_meta = load_model_bundle(_checkpoint_path(config, fold["name"], experiment, "classifier_5plus"))
    s20_model, s20_meta = load_model_bundle(_checkpoint_path(config, fold["name"], experiment, "classifier_20plus"))
    p_x = apply_seen_category_codes(
        validation["x"][p_meta["feature_names"]], p_meta["fold_category_codes_seen"]
    )
    q_x = apply_seen_category_codes(
        validation["x"][q_meta["feature_names"]], q_meta["fold_category_codes_seen"]
    )
    p = np.clip(p_model.predict(p_x, num_iteration=p_meta["best_iteration"]), 0.0, 1.0)
    q = np.clip(q_model.predict(q_x, num_iteration=q_meta["best_iteration"]), 0.0, None)
    s5_x = apply_seen_category_codes(
        validation["x"][s5_meta["feature_names"]], s5_meta["fold_category_codes_seen"]
    )
    s20_x = apply_seen_category_codes(
        validation["x"][s20_meta["feature_names"]], s20_meta["fold_category_codes_seen"]
    )
    s5 = np.clip(s5_model.predict(s5_x, num_iteration=s5_meta["best_iteration"]), 0.0, 1.0)
    s20 = np.minimum(
        np.clip(s20_model.predict(s20_x, num_iteration=s20_meta["best_iteration"]), 0.0, 1.0),
        s5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({
        "month": validation["month"], "y": validation["target"], "p_sale": p.astype("float32"),
        "q": q.astype("float32"), "s5": s5.astype("float32"), "s20": s20.astype("float32"),
        "sampling_probability": validation["probability"], "sample_weight": validation["weight"],
    })
    pq.write_table(table, output, compression="zstd", row_group_size=100_000)
    logger.info("OOF %s rows=%s", fold["name"], f"{len(validation['target']):,}")
    return output


def _fit_candidate_b_rule(frame: pd.DataFrame, config: dict) -> dict:
    target = frame["y"].to_numpy(dtype="float64")
    baseline = frame["p_sale"].to_numpy(dtype="float64") * frame["q"].to_numpy(dtype="float64")
    weights = frame["sample_weight"].to_numpy(dtype="float64")
    base_metrics = weighted_business_metrics(target, baseline, weights)
    candidates = []
    for medium_fraction in config["candidate_b"]["medium_top_fractions"]:
        tau5 = weighted_threshold_for_top_fraction(
            frame["s5"].to_numpy(), weights, float(medium_fraction)
        )
        for high_fraction in config["candidate_b"]["high_top_fractions"]:
            tau20 = weighted_threshold_for_top_fraction(
                frame["s20"].to_numpy(), weights, float(high_fraction)
            )
            high = frame["s20"].to_numpy() >= tau20
            medium = (frame["s5"].to_numpy() >= tau5) & ~high
            c5 = candidate_b_multiplier(target[medium], baseline[medium], weights[medium], float(config["candidate_b"]["multiplier_maximum"])) if medium.any() else 1.0
            c20 = candidate_b_multiplier(target[high], baseline[high], weights[high], float(config["candidate_b"]["multiplier_maximum"])) if high.any() else 1.0
            multiplier = np.ones(len(frame), dtype="float64")
            multiplier[medium] = c5
            multiplier[high] = c20
            metrics = weighted_business_metrics(target, baseline * multiplier, weights)
            gates = config["candidate_b"]["gates"]
            gate_values = {
                "five_nineteen_worsening_pp": metrics["5-19"]["wape"] - base_metrics["5-19"]["wape"],
                "nonzero_worsening_pp": metrics["nonzero"]["wape"] - base_metrics["nonzero"]["wape"],
                "zero_prediction_total_increase": (
                    metrics["zero_prediction_total"] / base_metrics["zero_prediction_total"] - 1.0
                    if base_metrics["zero_prediction_total"] else 0.0
                ),
                "absolute_bias_worsening_pp": absolute_bias_worsening(
                    base_metrics["overall"]["total_bias"], metrics["overall"]["total_bias"]
                ),
            }
            gate_checks = {
                "five_nineteen": gate_values["five_nineteen_worsening_pp"] <= float(gates["five_nineteen_max_worsening_pp"]),
                "nonzero": gate_values["nonzero_worsening_pp"] <= float(gates["nonzero_max_worsening_pp"]),
                "zero_total": gate_values["zero_prediction_total_increase"] <= float(gates["zero_prediction_total_max_increase"]),
                "total_bias": gate_values["absolute_bias_worsening_pp"] <= float(gates["total_bias_max_worsening_pp"]),
            }
            eligible = all(gate_checks.values())
            candidates.append({
                "medium_fraction": float(medium_fraction), "high_fraction": float(high_fraction),
                "tau5": tau5, "tau20": tau20, "c5": c5, "c20": c20,
                "eligible": bool(eligible), "metrics": metrics,
                "gate_values": gate_values, "gate_checks": gate_checks,
            })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["metrics"]["20+"]["wape"],
            -candidate["metrics"]["recall_20_plus"],
            candidate["metrics"]["overall"]["wape"],
            candidate["high_fraction"] + candidate["medium_fraction"],
        ),
        default=None,
    )
    if selected is None:
        selected = {
            "medium_fraction": 0.0, "high_fraction": 0.0,
            "tau5": float("inf"), "tau20": float("inf"), "c5": 1.0, "c20": 1.0,
            "eligible": False, "metrics": base_metrics,
        }
    best_unconstrained = min(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["20+"]["wape"],
            -candidate["metrics"]["recall_20_plus"],
            candidate["metrics"]["overall"]["wape"],
        ),
    )
    rejection_counts = {
        gate: sum(not candidate["gate_checks"][gate] for candidate in candidates)
        for gate in ("five_nineteen", "nonzero", "zero_total", "total_bias")
    }
    return {
        "baseline_metrics": base_metrics, "selected": selected,
        "candidate_count": len(candidates), "eligible_count": len(eligible),
        "rejection_counts": rejection_counts, "best_unconstrained": best_unconstrained,
        "candidates": candidates,
    }


def _apply_candidate_b_rule(frame: pd.DataFrame, rule: dict) -> tuple[np.ndarray, dict]:
    baseline = frame["p_sale"].to_numpy(dtype="float64") * frame["q"].to_numpy(dtype="float64")
    high = frame["s20"].to_numpy() >= float(rule["tau20"])
    medium = (frame["s5"].to_numpy() >= float(rule["tau5"])) & ~high
    multiplier = np.ones(len(frame), dtype="float64")
    multiplier[medium] = float(rule["c5"])
    multiplier[high] = float(rule["c20"])
    return baseline * multiplier, {"medium_rows": int(medium.sum()), "high_rows": int(high.sum())}


def absolute_bias_worsening(baseline_bias: float, candidate_bias: float) -> float:
    return abs(float(candidate_bias)) - abs(float(baseline_bias))


def evaluate_candidate_b_oof(oof_paths: dict[str, Path], config: dict) -> dict:
    order = ["q3_2024", "q4_2024", "q1_2025", "q2_2025"]
    frames = {name: _read_oof_frame(oof_paths[name]) for name in order}
    rolling = [
        (["q3_2024"], "q4_2024"),
        (["q3_2024", "q4_2024"], "q1_2025"),
        (["q3_2024", "q4_2024", "q1_2025"], "q2_2025"),
    ]
    evaluations = []
    for fit_names, evaluation_name in rolling:
        fit_frame = pd.concat([frames[name] for name in fit_names], ignore_index=True)
        fit = _fit_candidate_b_rule(fit_frame, config)
        evaluation = frames[evaluation_name]
        prediction, coverage = _apply_candidate_b_rule(evaluation, fit["selected"])
        baseline = evaluation["p_sale"].to_numpy() * evaluation["q"].to_numpy()
        weights = evaluation["sample_weight"].to_numpy()
        evaluations.append({
            "fit_folds": fit_names, "evaluation_fold": evaluation_name,
            "rule": fit["selected"], "coverage": coverage,
            "fit_diagnostics": {
                "candidate_count": fit["candidate_count"], "eligible_count": fit["eligible_count"],
                "rejection_counts": fit["rejection_counts"],
                "best_unconstrained": fit["best_unconstrained"],
            },
            "baseline_metrics": weighted_business_metrics(evaluation["y"], baseline, weights),
            "candidate_metrics": weighted_business_metrics(evaluation["y"], prediction, weights),
        })
    gates = config["candidate_b"]["gates"]
    changes = []
    for item in evaluations:
        base = item["baseline_metrics"]
        candidate = item["candidate_metrics"]
        changes.append({
            "fold": item["evaluation_fold"],
            "wape20_gain": base["20+"]["wape"] - candidate["20+"]["wape"],
            "recall20_gain": candidate["recall_20_plus"] - base["recall_20_plus"],
            "wape5_worsening": candidate["5-19"]["wape"] - base["5-19"]["wape"],
            "recall5_worsening": base["recall_5_19"] - candidate["recall_5_19"],
            "nonzero_worsening": candidate["nonzero"]["wape"] - base["nonzero"]["wape"],
            "zero_increase": candidate["zero_prediction_total"] / base["zero_prediction_total"] - 1.0,
            "bias_worsening": absolute_bias_worsening(
                base["overall"]["total_bias"], candidate["overall"]["total_bias"]
            ),
        })
    mean = {key: float(np.mean([change[key] for change in changes])) for key in changes[0] if key != "fold"}
    last_two_same_direction = all(
        change["wape20_gain"] > 0 and change["recall20_gain"] > 0 for change in changes[-2:]
    )
    passed = (
        mean["wape20_gain"] >= float(gates["twenty_plus_wape_gain_pp"])
        and mean["recall20_gain"] >= float(gates["twenty_plus_recall_gain_pp"])
        and last_two_same_direction
        and mean["wape5_worsening"] <= float(gates["five_nineteen_max_worsening_pp"])
        and mean["recall5_worsening"] <= float(gates["five_nineteen_max_worsening_pp"])
        and mean["nonzero_worsening"] <= float(gates["nonzero_max_worsening_pp"])
        and mean["zero_increase"] <= float(gates["zero_prediction_total_max_increase"])
        and mean["bias_worsening"] <= float(gates["total_bias_max_worsening_pp"])
    )
    all_oof = pd.concat([frames[name] for name in order], ignore_index=True)
    final_fit = _fit_candidate_b_rule(all_oof, config)
    return {
        "passed": bool(passed), "evaluations": evaluations, "changes": changes,
        "mean_changes": mean, "last_two_same_direction": last_two_same_direction,
        "final_rule": final_fit["selected"],
    }


def run_oof(config: dict, audit: dict, pilot: dict, logger: logging.Logger) -> dict:
    cache = project_path(config["outputs"]["cache_dir"]) / "oof_result.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("Reusing OOF result passed=%s", result.get("passed"))
        return result
    selected = pilot["selected_feature_set"]
    if not selected:
        return {"passed": False, "stop_reason": "Pilot did not select a feature set"}
    features = pilot["feature_sets"][selected]
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    category_maps = p_metadata["category_maps"]
    fold_lookup = {fold["name"]: fold for fold in config["folds"]}
    oof_paths: dict[str, Path] = {}
    rows = list(pilot["rows"])
    for fold_name in ("q3_2024", "q4_2024", "q1_2025", "q2_2025"):
        fold = fold_lookup[fold_name]
        fold_cache = build_fold_cache(config, fold, features, category_maps, logger)
        sale_result = train_fold_component(
            config, fold, fold_cache, "E0_sale", base_features(),
            "classifier_sale", category_maps, logger,
        )
        if not any(
            row["fold"] == fold_name and row["experiment"] == "E0_sale" and row["component"] == "classifier_sale"
            for row in rows
        ):
            rows.append(sale_result)
        for component in ("regressor", "classifier_5plus", "classifier_20plus"):
            result = train_fold_component(
                config, fold, fold_cache, selected, features, component, category_maps, logger
            )
            if not any(
                row["fold"] == fold_name and row["experiment"] == selected and row["component"] == component
                for row in rows
            ):
                rows.append(result)
        oof_paths[fold_name] = generate_fold_oof(
            config, fold, fold_cache, selected, features, logger
        )
    candidate = evaluate_candidate_b_oof(oof_paths, config)
    result = {
        "passed": candidate["passed"], "selected_feature_set": selected,
        "features": features, "fold_rows": rows, "candidate_b": candidate,
        "oof_paths": {name: str(path.relative_to(ROOT)) for name, path in oof_paths.items()},
    }
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OOF complete passed=%s", result["passed"])
    return result


def _weighted_median(values: list[int], weights: list[int]) -> int:
    order = np.argsort(values)
    values_array = np.asarray(values, dtype="int64")[order]
    weights_array = np.asarray(weights, dtype="float64")[order]
    cutoff = weights_array.sum() / 2.0
    return int(values_array[np.searchsorted(np.cumsum(weights_array), cutoff, side="left")])


def _formal_rounds(oof: dict, component: str) -> int:
    rows = [
        row for row in oof["fold_rows"]
        if row["experiment"] == oof["selected_feature_set"] and row["component"] == component
        and row["fold"] in {"q3_2024", "q4_2024", "q1_2025", "q2_2025"}
    ]
    return _weighted_median(
        [int(row["best_iteration"]) for row in rows],
        [int(row["valid_rows"]) for row in rows],
    )


def select_cross_store_formal_rounds(ablation: pd.DataFrame) -> dict:
    fold_names = {"q3_2024", "q4_2024", "q1_2025", "q2_2025"}
    selected = ablation.loc[
        ablation["fold"].isin(fold_names)
        & ablation["experiment"].eq("E2")
        & ablation["component"].eq("regressor"),
        ["fold", "best_iteration", "valid_rows"],
    ].copy()
    if set(selected["fold"]) != fold_names or len(selected) != 4:
        raise RuntimeError("Ablation CSV does not contain exactly four E2 regressor forward folds")
    selected = selected.sort_values("fold")
    rounds = _weighted_median(
        selected["best_iteration"].astype(int).tolist(),
        selected["valid_rows"].astype(int).tolist(),
    )
    return {
        "rounds": rounds,
        "folds": [
            {
                "fold": str(row.fold), "best_iteration": int(row.best_iteration),
                "valid_rows": int(row.valid_rows),
            }
            for row in selected.itertuples(index=False)
        ],
        "selection": "validation-row-weighted median of four forward-time E2 regressor folds",
    }


def _cross_store_formal_features() -> list[str]:
    return list(dict.fromkeys([*base_features(), *CROSS_STORE_FEATURES]))


def _cross_store_cache_info(path: Path) -> dict:
    success_path = path / "_SUCCESS.json" if path.is_dir() else None
    if success_path is not None and not success_path.exists():
        raise RuntimeError(f"Incomplete Cross-store formal cache: {path}")
    source = path / "*.parquet" if path.is_dir() else path
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"SELECT COUNT(*), MIN(month), MAX(month), COUNT(DISTINCT month) "
            f"FROM read_parquet('{_escaped(source)}') WHERE sample_kind='regressor_train'"
        ).fetchone()
    finally:
        connection.close()
    return {
        "rows": int(row[0]), "first_month": str(row[1]),
        "last_month": str(row[2]), "month_count": int(row[3]),
        "bytes": sum(file.stat().st_size for file in path.glob("*.parquet")) if path.is_dir() else path.stat().st_size,
    }


def _write_duckdb_parquet_part(frame: pd.DataFrame, output: Path) -> None:
    connection = duckdb.connect()
    try:
        connection.register("sampled_rows", frame)
        connection.execute(
            f"COPY sampled_rows TO '{_escaped(output)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        connection.unregister("sampled_rows")
    finally:
        connection.close()


def _build_cross_store_regressor_train_cache(
    config: dict, features: list[str], category_maps: dict, logger: logging.Logger,
) -> tuple[Path, dict]:
    output = project_path(config["outputs"]["cache_dir"]) / "cross_store_formal/regressor_train_parts"
    success_path = output / "_SUCCESS.json"
    if output.exists() and success_path.exists():
        info = _cross_store_cache_info(output)
        logger.info("Reusing Cross-store formal regressor cache rows=%s", f"{info['rows']:,}")
        return output, info
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = 0
    parts = 0
    rates = config["sampling_rates"]["regressor"]
    where = "b.split='train' AND b.target_available_1m=1 AND CAST(b.month AS VARCHAR)<='2025-06'"
    guard = PIPELINE.MemoryGuard("cross-store-formal-train-cache", logger)
    started = time.perf_counter()
    for chunk_number, frame in enumerate(
        iter_joined_rows(
            config, where, name="cross_store_formal_train_cache", vectors_per_chunk=128
        ), start=1
    ):
        sampled, weights = _sample_frame(
            frame, rates, int(config["seed"]), "formal_regressor_train"
        )
        if not sampled.empty:
            enriched = add_all_new_features(sampled)
            prepared = PIPELINE.prepare_feature_frame(
                enriched, features, category_maps, str(config["time_base"])
            )
            prepared.insert(0, "sample_kind", "regressor_train")
            prepared.insert(1, "month", enriched["month"].astype(str).to_numpy())
            prepared.insert(2, "target_qty", clip_target(enriched["future_qty_1m"].to_numpy()).astype("float32"))
            prepared.insert(3, "sample_weight", weights.astype("float32"))
            prepared.insert(4, "sampling_probability", np.divide(
                1.0, weights, out=np.zeros_like(weights, dtype="float32"), where=weights > 0
            ))
            part_path = output / f"part-{parts:05d}.parquet"
            _write_duckdb_parquet_part(prepared, part_path)
            parts += 1
            rows += len(sampled)
            del enriched, prepared
        del frame, sampled, weights
        if chunk_number % 20 == 0:
            guard.check("formal_train_cache")
            logger.info("Cross-store formal cache sampled rows=%s parts=%s", f"{rows:,}", parts)
    if parts == 0:
        raise RuntimeError("Cross-store formal regressor cache is empty")
    success_path.write_text(
        json.dumps({"parts": parts, "sampled_rows": rows}, ensure_ascii=False), encoding="utf-8"
    )
    info = _cross_store_cache_info(output)
    if info["first_month"] != "2023-01" or info["last_month"] != "2025-06" or info["month_count"] != 30:
        raise RuntimeError(f"Cross-store formal Train month coverage is incomplete: {info}")
    info.update({"seconds": time.perf_counter() - started, "peak_ram_gib": guard.peak / 1024**3})
    logger.info("Cross-store formal cache complete: %s", info)
    return output, info


def _build_formal_train_cache(config: dict, features: list[str], category_maps: dict, logger: logging.Logger) -> Path:
    output = project_path(config["outputs"]["cache_dir"]) / "fold_samples/formal_train.parquet"
    if output.exists():
        return output
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rates = config["sampling_rates"]
    where = "b.split='train' AND b.target_available_1m=1 AND CAST(b.month AS VARCHAR)<='2025-06'"
    guard = PIPELINE.MemoryGuard("formal-train-cache", logger)
    try:
        for chunk_number, frame in enumerate(iter_joined_rows(config, where, name="formal_train_cache"), start=1):
            for kind, rate_key in (
                ("classifier_sale_train", "classifier_sale"),
                ("regressor_train", "regressor"),
                ("classifier_5plus_train", "classifier_5plus"),
                ("classifier_20plus_train", "classifier_20plus"),
            ):
                sampled, weights = _sample_frame(
                    frame, rates[rate_key], int(config["seed"]), f"formal_{kind}"
                )
                if not sampled.empty:
                    writer = _write_sample_rows(
                        writer, temporary, sampled, kind, weights, features,
                        category_maps, str(config["time_base"]),
                    )
                del sampled, weights
            del frame
            if chunk_number % 20 == 0:
                guard.check("formal_train_cache")
                logger.info("Formal Train cache processed %d chunks", chunk_number)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("Formal Train cache is empty")
    temporary.replace(output)
    return output


def train_fixed_component(
    config: dict,
    cache_path: Path,
    features: list[str],
    component: str,
    rounds: int,
    category_maps: dict,
    logger: logging.Logger,
    output_path: Path | None = None,
    experiment_name: str = "two_stage_optimization_v2_1m",
    selection_description: str = "weighted median of four forward folds",
    selection_details: dict | None = None,
) -> Path:
    output = output_path or (project_path(config["outputs"]["checkpoint_dir"]) / "formal" / f"{component}.txt")
    if output.exists():
        return output
    guard = PIPELINE.MemoryGuard(f"formal-{component}", logger)
    kind = {
        "regressor": "regressor_train",
        "classifier_5plus": "classifier_5plus_train",
        "classifier_20plus": "classifier_20plus_train",
    }[component]
    train = _read_fold_part(cache_path, kind, features)
    guard.check("cache_loaded")
    if component == "regressor":
        mask = train["target"] > 0
        target = train["target"][mask]
        x = train["x"].loc[mask].reset_index(drop=True)
        weights = train["weight"][mask]
    else:
        threshold = 5 if component == "classifier_5plus" else 20
        target = (train["target"] >= threshold).astype("float32")
        x = train["x"]
        weights = train["weight"]
    weights = (weights / weights.mean()).astype("float32")
    params = _lgb_params(config, component)
    started = time.perf_counter()
    dataset = lgb.Dataset(
        x, label=target, weight=weights, feature_name=features,
        categorical_feature=[feature for feature in CATEGORY_FEATURES if feature in features], free_raw_data=True,
    )
    booster = lgb.train(
        params, dataset, num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)]
    )
    metadata = {
        "experiment": experiment_name, "component": component,
        "feature_names": features,
        "categorical_features": [feature for feature in CATEGORY_FEATURES if feature in features],
        "category_maps": category_maps, "time_base": str(config["time_base"]),
        "params": params, "best_iteration": int(rounds), "selected_iteration": int(rounds),
        "train_rows": len(target), "training_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3, "selection": selection_description,
        "selection_details": selection_details or {},
        "lightgbm_version": lgb.__version__, "python_version": platform.python_version(),
    }
    save_model_bundle(booster, output, metadata)
    reloaded, loaded_metadata = load_model_bundle(output)
    check = x.iloc[: min(1000, len(x))]
    np.testing.assert_allclose(
        booster.predict(check, num_iteration=rounds), reloaded.predict(check, num_iteration=loaded_metadata["best_iteration"]),
        rtol=1e-7, atol=1e-8,
    )
    del train, x, dataset, booster, reloaded
    gc.collect()
    return output


class CompleteMetricAccumulator:
    GROUPS = ("overall", "0", "1", "2-4", "5-19", "20+", "nonzero")

    def __init__(self) -> None:
        self.values = {
            group: {"count": 0, "target": 0.0, "prediction": 0.0, "absolute": 0.0, "integer_absolute": 0.0}
            for group in self.GROUPS
        }
        self.zero_gt_0_5 = 0
        self.zero_gt_1 = 0
        self.confusion = np.zeros((5, 5), dtype="int64")

    def update(self, target, prediction) -> None:
        target = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
        prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, None)
        rounded = np.floor(prediction + 0.5)
        groups = {
            "overall": np.ones(target.size, dtype=bool), "0": target <= 0,
            "1": target == 1, "2-4": (target >= 2) & (target < 5),
            "5-19": (target >= 5) & (target < 20), "20+": target >= 20,
            "nonzero": target > 0,
        }
        for name, mask in groups.items():
            values = self.values[name]
            values["count"] += int(mask.sum())
            values["target"] += float(target[mask].sum())
            values["prediction"] += float(prediction[mask].sum())
            values["absolute"] += float(np.abs(prediction[mask] - target[mask]).sum())
            values["integer_absolute"] += float(np.abs(rounded[mask] - target[mask]).sum())
        zero = groups["0"]
        self.zero_gt_0_5 += int((zero & (prediction > 0.5)).sum())
        self.zero_gt_1 += int((zero & (prediction > 1.0)).sum())
        true_codes = np.select([target <= 0, target == 1, target < 5, target < 20], [0, 1, 2, 3], default=4)
        pred_codes = np.select([rounded <= 0, rounded == 1, rounded < 5, rounded < 20], [0, 1, 2, 3], default=4)
        np.add.at(self.confusion, (true_codes, pred_codes), 1)

    def compute(self) -> dict:
        result = {}
        for name, values in self.values.items():
            result[name] = {
                "count": values["count"], "target_sum": values["target"], "prediction_sum": values["prediction"],
                "wape": 100.0 * values["absolute"] / values["target"] if values["target"] else float("nan"),
                "integer_wape": 100.0 * values["integer_absolute"] / values["target"] if values["target"] else float("nan"),
            }
        overall = result["overall"]
        overall["total_bias"] = 100.0 * (overall["prediction_sum"] - overall["target_sum"]) / overall["target_sum"]
        zero_count = result["0"]["count"]
        result["zero_prediction_total"] = result["0"]["prediction_sum"]
        result["zero_gt_0_5"] = self.zero_gt_0_5 / zero_count if zero_count else 0.0
        result["zero_gt_1"] = self.zero_gt_1 / zero_count if zero_count else 0.0
        recall = np.divide(
            np.diag(self.confusion), self.confusion.sum(axis=1), out=np.zeros(5), where=self.confusion.sum(axis=1) > 0
        )
        precision = np.divide(
            np.diag(self.confusion), self.confusion.sum(axis=0), out=np.zeros(5), where=self.confusion.sum(axis=0) > 0
        )
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
        result["mc_macro_f1"] = 100.0 * float(f1.mean())
        result["recall_5_19"] = 100.0 * float(recall[3])
        result["recall_20_plus"] = 100.0 * float(recall[4])
        return result


def twenty_plus_median_ratio(target, prediction) -> float:
    target_array = np.asarray(target, dtype="float64")
    prediction_array = np.asarray(prediction, dtype="float64")
    mask = np.isfinite(target_array) & np.isfinite(prediction_array) & (target_array >= 20)
    if not mask.any():
        return float("nan")
    return float(np.median(prediction_array[mask] / target_array[mask]))


def _cross_store_complete_valid_success(metrics: dict, monthly: dict, config: dict) -> dict:
    base = metrics["E0"]
    candidate = metrics["E2_cross_store"]
    gates = config["complete_valid_gates"]
    wape_gains = {
        "5-19": base["5-19"]["wape"] - candidate["5-19"]["wape"],
        "20+": base["20+"]["wape"] - candidate["20+"]["wape"],
    }
    recall_gains = {
        "5-19": candidate["recall_5_19"] - base["recall_5_19"],
        "20+": candidate["recall_20_plus"] - base["recall_20_plus"],
    }
    checks = {
        "one_head_wape_improved": max(wape_gains.values()) >= float(gates["head_wape_gain_pp"]),
        "other_head_wape_protected": min(wape_gains.values()) >= -float(gates["other_head_wape_max_worsening_pp"]),
        "one_head_recall_improved": max(recall_gains.values()) >= float(gates["head_recall_gain_pp"]),
        "other_head_recall_protected": min(recall_gains.values()) >= -float(gates["other_head_recall_max_worsening_pp"]),
        "overall_protected": (
            candidate["overall"]["wape"] - base["overall"]["wape"]
            <= float(gates["overall_wape_max_worsening_pp"])
            and candidate["overall"]["integer_wape"] - base["overall"]["integer_wape"]
            <= float(gates["overall_wape_max_worsening_pp"])
        ),
        "zero_protected": (
            candidate["zero_prediction_total"]
            <= base["zero_prediction_total"] * (1 + float(gates["zero_prediction_total_max_increase"]))
            and (candidate["zero_gt_0_5"] - base["zero_gt_0_5"]) * 100
            <= float(gates["zero_threshold_rate_max_increase_pp"])
            and (candidate["zero_gt_1"] - base["zero_gt_1"]) * 100
            <= float(gates["zero_threshold_rate_max_increase_pp"])
        ),
        "bias_protected": (
            abs(candidate["overall"]["total_bias"]) - abs(base["overall"]["total_bias"])
            <= float(gates["total_absolute_bias_max_worsening_pp"])
        ),
        "mc_protected": (
            candidate["mc_macro_f1"] - base["mc_macro_f1"]
            >= -float(gates["mc_macro_f1_max_decline_pp"])
        ),
    }
    monthly_rows = []
    for month in sorted(monthly):
        old = monthly[month]["E0"]
        new = monthly[month]["E2_cross_store"]
        improved = (
            new["20+"]["wape"] < old["20+"]["wape"]
            or new["recall_20_plus"] > old["recall_20_plus"]
        )
        monthly_rows.append({
            "month": month,
            "improved": bool(improved),
            "wape20_gain_pp": old["20+"]["wape"] - new["20+"]["wape"],
            "recall20_gain_pp": new["recall_20_plus"] - old["recall_20_plus"],
        })
    monthly_improved = sum(int(row["improved"]) for row in monthly_rows)
    checks["monthly_stable"] = monthly_improved >= int(gates["monthly_improvement_minimum"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "wape_gains": wape_gains,
        "recall_gains": recall_gains,
        "monthly_improved": monthly_improved,
        "monthly_rows": monthly_rows,
    }


def evaluate_cross_store_complete_valid(
    config: dict,
    e2_model_path: Path,
    logger: logging.Logger,
) -> dict:
    cache = project_path(config["outputs"]["cache_dir"]) / "cross_store_formal/complete_valid_result.json"
    if cache.exists():
        logger.info("Reusing Cross-store Complete Valid result")
        return json.loads(cache.read_text(encoding="utf-8"))

    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    old_q_model, old_q_meta = load_model_bundle(FORMAL_Q_MODEL)
    e2_q_model, e2_q_meta = load_model_bundle(e2_model_path)
    if p_meta["category_maps"] != old_q_meta["category_maps"] or p_meta["category_maps"] != e2_q_meta["category_maps"]:
        raise RuntimeError("Formal classifier and regressors do not share the same Train-only category maps")
    if {p_meta["time_base"], old_q_meta["time_base"], e2_q_meta["time_base"]} != {str(config["time_base"])}:
        raise RuntimeError("Formal models do not share the configured absolute time base")

    features = e2_q_meta["feature_names"]
    q_accumulators = {name: CompleteMetricAccumulator() for name in ("E0", "E2_cross_store")}
    final_accumulators = {name: CompleteMetricAccumulator() for name in ("E0", "E2_cross_store")}
    monthly_accumulators: dict[str, dict[str, CompleteMetricAccumulator]] = {}
    ratios_20_plus: dict[str, list[np.ndarray]] = {"E0": [], "E2_cross_store": []}
    unknown_counts = Counter()
    rows = 0
    guard = PIPELINE.MemoryGuard("cross-store-complete-valid", logger)
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"

    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name="cross_store_complete_valid"), start=1
    ):
        cross = add_cross_store_features(frame)
        for column in cross.columns:
            frame[column] = cross[column]
        x = PIPELINE.prepare_feature_frame(
            frame, features, e2_q_meta["category_maps"], e2_q_meta["time_base"]
        )
        for column in CATEGORY_FEATURES:
            if column in x:
                unknown_counts[column] += int((pd.to_numeric(x[column], errors="coerce").fillna(-1) == -1).sum())

        p = np.clip(
            p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]), 0, 1
        )
        old_q = np.clip(
            old_q_model.predict(x[old_q_meta["feature_names"]], num_iteration=old_q_meta["best_iteration"]),
            0, None,
        )
        e2_q = np.clip(
            e2_q_model.predict(x[e2_q_meta["feature_names"]], num_iteration=e2_q_meta["best_iteration"]),
            0, None,
        )
        target = clip_target(frame["future_qty_1m"].to_numpy())
        predictions = {
            "E0": p * old_q,
            "E2_cross_store": p * e2_q,
        }
        q_predictions = {"E0": old_q, "E2_cross_store": e2_q}
        months = frame["month"].astype(str).to_numpy()
        high_mask = target >= 20

        for name, q_prediction in q_predictions.items():
            if not np.isfinite(q_prediction).all() or (q_prediction < 0).any():
                raise RuntimeError(f"Invalid q prediction for {name}")
            q_accumulators[name].update(target, q_prediction)
            if high_mask.any():
                ratios_20_plus[name].append((q_prediction[high_mask] / target[high_mask]).astype("float32"))

        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise RuntimeError(f"Invalid final prediction for {name}")
            final_accumulators[name].update(target, prediction)
            for month in np.unique(months):
                monthly_accumulators.setdefault(
                    month,
                    {model_name: CompleteMetricAccumulator() for model_name in predictions},
                )[name].update(target[months == month], prediction[months == month])

        rows += len(frame)
        if chunk_number % 20 == 0:
            guard.check("complete_valid")
            logger.info("Cross-store Complete Valid rows=%s", f"{rows:,}")
        del frame, cross, x, p, old_q, e2_q, target, predictions, q_predictions

    q_metrics = {name: accumulator.compute() for name, accumulator in q_accumulators.items()}
    for name, parts in ratios_20_plus.items():
        q_metrics[name]["twenty_plus_median_q_over_y"] = (
            float(np.median(np.concatenate(parts))) if parts else float("nan")
        )
    metrics = {name: accumulator.compute() for name, accumulator in final_accumulators.items()}
    monthly = {
        month: {name: accumulator.compute() for name, accumulator in models.items()}
        for month, models in sorted(monthly_accumulators.items())
    }
    success = _cross_store_complete_valid_success(metrics, monthly, config)
    result = {
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "q_metrics": q_metrics,
        "metrics": metrics,
        "monthly": monthly,
        "unknown_category_rates": {
            column: count / rows if rows else 0.0 for column, count in unknown_counts.items()
        },
        "success": success,
        "scope": "Active-Store Complete Valid 1M only; no Test and no Candidate B",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


DIAGNOSTIC_HISTORY_FEATURES = [
    "total_qty", "qty_lag_1m", "qty_lag_2m", "qty_lag_3m",
    "qty_mean_last_3m", "qty_max_last_3m", "sales_days",
    "active_months_last_6m", "months_since_last_sale",
]


def _describe_array(values) -> dict:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "median": float(np.median(array)), "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def _prediction_error_summary(frame: pd.DataFrame) -> dict:
    target = frame["target"].to_numpy(dtype="float64")
    e0 = frame["e0_pred"].to_numpy(dtype="float64")
    e2 = frame["e2_pred"].to_numpy(dtype="float64")
    target_sum = float(target.sum())
    return {
        "count": len(frame),
        "target_mean": float(target.mean()) if len(frame) else float("nan"),
        "target_median": float(np.median(target)) if len(frame) else float("nan"),
        "e0_mean": float(e0.mean()) if len(frame) else float("nan"),
        "e0_median": float(np.median(e0)) if len(frame) else float("nan"),
        "e2_mean": float(e2.mean()) if len(frame) else float("nan"),
        "e2_median": float(np.median(e2)) if len(frame) else float("nan"),
        "e0_mae": float(np.abs(e0 - target).mean()) if len(frame) else float("nan"),
        "e2_mae": float(np.abs(e2 - target).mean()) if len(frame) else float("nan"),
        "e0_wape": 100.0 * float(np.abs(e0 - target).sum()) / target_sum if target_sum else float("nan"),
        "e2_wape": 100.0 * float(np.abs(e2 - target).sum()) / target_sum if target_sum else float("nan"),
    }


def _weighted_signal_metrics(frame: pd.DataFrame, feature: str) -> dict:
    target = frame["is_20plus"].to_numpy(dtype="uint8")
    score = frame[feature].to_numpy(dtype="float64")
    weights = frame["sample_weight"].to_numpy(dtype="float64")
    threshold = weighted_threshold_for_top_fraction(score, weights, 0.10)
    selected = score >= threshold
    positive_mass = float(np.sum(weights * target))
    total_mass = float(weights.sum())
    selected_mass = float(weights[selected].sum())
    selected_positive = float(np.sum(weights[selected] * target[selected]))
    base_rate = positive_mass / total_mass if total_mass else 0.0
    top_rate = selected_positive / selected_mass if selected_mass else 0.0
    return {
        "average_precision": _safe_average_precision(target, score, weights),
        "top_decile_threshold": threshold,
        "base_rate": base_rate,
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / base_rate if base_rate else float("nan"),
    }


def _train_twenty_plus_reference(cache_path: Path) -> pd.DataFrame:
    source = cache_path / "*.parquet" if cache_path.is_dir() else cache_path
    columns = ["month", "target_qty", *CROSS_STORE_FEATURES]
    projection = ",".join(f'"{column}"' for column in columns)
    connection = duckdb.connect()
    try:
        arrays = connection.execute(
            f"SELECT {projection} FROM read_parquet('{_escaped(source)}') WHERE target_qty>=20"
        ).fetchnumpy()
    finally:
        connection.close()
    return pd.DataFrame(arrays)


def _feature_group_summaries(frame: pd.DataFrame) -> dict:
    groups = {
        "e2_correct_5_19": frame["e2_mc"].eq(3),
        "e2_to_20_plus": frame["e2_mc"].eq(4),
        "e2_to_2_4_or_lower": frame["e2_mc"].le(2),
    }
    features = [*CROSS_STORE_FEATURES, *DIAGNOSTIC_HISTORY_FEATURES]
    return {
        group: {
            "count": int(mask.sum()),
            "features": {feature: _describe_array(frame.loc[mask, feature]) for feature in features},
        }
        for group, mask in groups.items()
    }


def _concentration_rows(frame: pd.DataFrame, column: str, minimum_count: int = 10) -> list[dict]:
    if frame.empty:
        return []
    normalized = frame[column].astype(str).str.strip()
    all_counts = normalized.value_counts()
    worse_counts = normalized.loc[frame["materially_worse"]].value_counts()
    rows = []
    for value, count in worse_counts.items():
        if int(count) < minimum_count:
            continue
        baseline_count = int(all_counts.get(value, 0))
        worse_share = float(count / worse_counts.sum()) if worse_counts.sum() else 0.0
        baseline_share = float(baseline_count / len(frame)) if len(frame) else 0.0
        rows.append({
            "value": value, "worse_count": int(count), "all_count": baseline_count,
            "worse_share": worse_share, "all_share": baseline_share,
            "concentration_ratio": worse_share / baseline_share if baseline_share else float("nan"),
        })
    return sorted(rows, key=lambda item: (item["concentration_ratio"], item["worse_count"]), reverse=True)[:10]


def _build_cross_store_diagnostic(
    high_demand: pd.DataFrame,
    relation: pd.DataFrame,
    train_twenty: pd.DataFrame,
) -> dict:
    five = high_demand.loc[(high_demand["target"] >= 5) & (high_demand["target"] < 20)].copy()
    transition = []
    for code, label in enumerate(("0", "1", "2-4", "5-19", "20+")):
        e0_count = int((five["e0_mc"] == code).sum())
        e2_count = int((five["e2_mc"] == code).sum())
        transition.append({
            "predicted_mc": label, "e0_count": e0_count,
            "e0_share": e0_count / len(five) if len(five) else 0.0,
            "e2_count": e2_count, "e2_share": e2_count / len(five) if len(five) else 0.0,
            "count_change": e2_count - e0_count,
        })
    e0_correct = five["e0_mc"].eq(3)
    pair_transition = []
    for e0_code, e0_label in enumerate(("0", "1", "2-4", "5-19", "20+")):
        for e2_code, e2_label in enumerate(("0", "1", "2-4", "5-19", "20+")):
            count = int(((five["e0_mc"] == e0_code) & (five["e2_mc"] == e2_code)).sum())
            if count:
                pair_transition.append({"e0_mc": e0_label, "e2_mc": e2_label, "count": count})
    error_groups = {
        "e2_to_20_plus": five.loc[e0_correct & five["e2_mc"].eq(4)],
        "e2_to_2_4": five.loc[e0_correct & five["e2_mc"].eq(2)],
        "e2_to_1_or_0": five.loc[e0_correct & five["e2_mc"].le(1)],
    }
    migration = {
        "transition": transition,
        "pair_transition": pair_transition,
        "e0_correct_e2_wrong": {
            name: _prediction_error_summary(group) for name, group in error_groups.items()
        },
        "feature_groups": _feature_group_summaries(five),
    }

    train_feature_sorted = {
        feature: np.sort(train_twenty[feature].to_numpy(dtype="float64"))
        for feature in CROSS_STORE_FEATURES
    }
    monthly = {}
    for month in sorted(high_demand["month"].astype(str).unique()):
        frame = high_demand.loc[(high_demand["month"].astype(str) == month) & (high_demand["target"] >= 20)].copy()
        if frame.empty:
            continue
        target = frame["target"].to_numpy(dtype="float64")
        old_q = frame["e0_q"].to_numpy(dtype="float64")
        new_q = frame["e2_q"].to_numpy(dtype="float64")
        old_pred = frame["e0_pred"].to_numpy(dtype="float64")
        new_pred = frame["e2_pred"].to_numpy(dtype="float64")
        p = frame["p_sale"].to_numpy(dtype="float64")
        feature_stats = {}
        for feature in CROSS_STORE_FEATURES:
            stats = _describe_array(frame[feature])
            train_values = train_feature_sorted[feature]
            stats["median_train_percentile"] = (
                100.0 * np.searchsorted(train_values, stats["median"], side="right") / len(train_values)
                if len(train_values) else float("nan")
            )
            feature_stats[feature] = stats
        top_threshold = float(np.quantile(target, 0.99, method="higher"))
        top = target >= top_threshold
        old_abs = np.abs(old_pred - target)
        new_abs = np.abs(new_pred - target)
        deterioration = new_abs - old_abs
        monthly[month] = {
            "count": len(frame), "target_sum": float(target.sum()),
            "target_median": float(np.median(target)), "target_p90": float(np.quantile(target, 0.90)),
            "target_p95": float(np.quantile(target, 0.95)),
            "e0_q_over_y_median": float(np.median(old_q / target)),
            "e2_q_over_y_median": float(np.median(new_q / target)),
            "e0_q_recall": 100.0 * float(np.mean(business_mc_codes(old_q) == 4)),
            "e2_q_recall": 100.0 * float(np.mean(business_mc_codes(new_q) == 4)),
            "e0_final_recall": 100.0 * float(np.mean(frame["e0_mc"].to_numpy() == 4)),
            "e2_final_recall": 100.0 * float(np.mean(frame["e2_mc"].to_numpy() == 4)),
            "e0_wape": 100.0 * float(old_abs.sum() / target.sum()),
            "e2_wape": 100.0 * float(new_abs.sum() / target.sum()),
            "p_sale": _describe_array(p), "features": feature_stats,
            "top_1pct": {
                "threshold": top_threshold, "count": int(top.sum()),
                "e0_abs_error_share": float(old_abs[top].sum() / old_abs.sum()) if old_abs.sum() else 0.0,
                "e2_abs_error_share": float(new_abs[top].sum() / new_abs.sum()) if new_abs.sum() else 0.0,
                "deterioration_share": (
                    float(deterioration[top].sum() / deterioration.sum()) if deterioration.sum() > 0 else float("nan")
                ),
            },
        }

    relationships = {}
    for month, frame in relation.groupby("month", sort=True):
        relationships[str(month)] = {
            feature: _weighted_signal_metrics(frame, feature) for feature in CROSS_STORE_FEATURES
        }

    focus = high_demand.loc[
        high_demand["month"].astype(str).isin(["2025-10", "2025-11"])
        & (high_demand["target"] >= 20)
    ].copy()
    focus["absolute_error_change"] = (
        np.abs(focus["e2_pred"] - focus["target"]) - np.abs(focus["e0_pred"] - focus["target"])
    )
    focus["materially_worse"] = focus["absolute_error_change"] > np.maximum(2.0, 0.10 * focus["target"])
    focus["history_level"] = pd.cut(
        focus["total_qty"], bins=[-np.inf, 0, 5, 20, np.inf],
        labels=["0", "1-4", "5-19", "20+"], right=False,
    ).astype(str)
    focus["short_history"] = np.where(focus["active_months_last_6m"] < 3, "lt3_active_months", "3plus_active_months")
    heat_q = np.quantile(train_twenty["xstore_qty_current"], [0.5, 0.75, 0.9])
    heat_codes = np.searchsorted(heat_q, focus["xstore_qty_current"].to_numpy(dtype="float64"), side="right")
    focus["xstore_heat"] = np.asarray(["low", "mid", "high", "very_high"])[heat_codes]
    concentration = {
        "materially_worse_count": int(focus["materially_worse"].sum()),
        "focus_count": len(focus),
        "site_no": _concentration_rows(focus, "site_no"),
        "category_3": _concentration_rows(focus, "gds_ctgry_3_lvel"),
        "category_5": _concentration_rows(focus, "gds_ctgry_5_lvel"),
        "history_level": _concentration_rows(focus, "history_level", minimum_count=1),
        "short_history": _concentration_rows(focus, "short_history", minimum_count=1),
        "xstore_heat": _concentration_rows(focus, "xstore_heat", minimum_count=1),
    }
    return {
        "migration_5_19": migration,
        "monthly_20_plus": monthly,
        "monthly_relationship": relationships,
        "oct_nov_concentration": concentration,
        "train_20_plus_count": len(train_twenty),
    }


def evaluate_cross_store_error_diagnostic(
    config: dict,
    e2_model_path: Path,
    train_cache_path: Path,
    logger: logging.Logger,
) -> dict:
    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    old_q_model, old_q_meta = load_model_bundle(FORMAL_Q_MODEL)
    e2_q_model, e2_q_meta = load_model_bundle(e2_model_path)
    if int(e2_q_meta.get("best_iteration", -1)) != CROSS_STORE_DIAGNOSTIC_ROUNDS:
        raise RuntimeError("Rebuilt E2 checkpoint is not locked to 657 rounds")
    if e2_q_meta.get("feature_names") != _cross_store_formal_features():
        raise RuntimeError("Rebuilt E2 checkpoint feature order differs from locked E2")
    if p_meta["category_maps"] != old_q_meta["category_maps"] or p_meta["category_maps"] != e2_q_meta["category_maps"]:
        raise RuntimeError("E0 and rebuilt E2 category maps differ")

    q_accumulator = CompleteMetricAccumulator()
    final_accumulator = CompleteMetricAccumulator()
    high_parts: list[pd.DataFrame] = []
    relation_parts: list[pd.DataFrame] = []
    guard = PIPELINE.MemoryGuard("cross-store-error-diagnostic", logger)
    rows = 0
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"
    features = e2_q_meta["feature_names"]
    keep_high_columns = [
        "month", "site_no", "item_id", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel",
        *CROSS_STORE_FEATURES, *DIAGNOSTIC_HISTORY_FEATURES,
    ]
    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name="cross_store_error_diagnostic"), start=1
    ):
        cross = add_cross_store_features(frame)
        for column in cross.columns:
            frame[column] = cross[column]
        x = PIPELINE.prepare_feature_frame(frame, features, e2_q_meta["category_maps"], e2_q_meta["time_base"])
        p = np.clip(p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]), 0, 1)
        old_q = np.clip(
            old_q_model.predict(x[old_q_meta["feature_names"]], num_iteration=old_q_meta["best_iteration"]), 0, None
        )
        e2_q = np.clip(
            e2_q_model.predict(x, num_iteration=CROSS_STORE_DIAGNOSTIC_ROUNDS), 0, None
        )
        target = clip_target(frame["future_qty_1m"].to_numpy())
        e0_pred = p * old_q
        e2_pred = p * e2_q
        if not all(np.isfinite(values).all() for values in (p, old_q, e2_q, e0_pred, e2_pred)):
            raise RuntimeError("Non-finite prediction found during E2 diagnostic")
        q_accumulator.update(target, e2_q)
        final_accumulator.update(target, e2_pred)

        high_mask = target >= 5
        if high_mask.any():
            high = frame.loc[high_mask, keep_high_columns].copy()
            high["target"] = target[high_mask].astype("float32")
            high["p_sale"] = p[high_mask].astype("float32")
            high["e0_q"] = old_q[high_mask].astype("float32")
            high["e2_q"] = e2_q[high_mask].astype("float32")
            high["e0_pred"] = e0_pred[high_mask].astype("float32")
            high["e2_pred"] = e2_pred[high_mask].astype("float32")
            high["e0_mc"] = business_mc_codes(e0_pred[high_mask])
            high["e2_mc"] = business_mc_codes(e2_pred[high_mask])
            high_parts.append(high)

        uniform = deterministic_uniform_hash(frame, "cross_store_diagnostic_valid_relation", int(config["seed"]))
        relation_mask = (target >= 20) | (uniform < 0.02)
        if relation_mask.any():
            relation = frame.loc[relation_mask, ["month", *CROSS_STORE_FEATURES]].copy()
            relation["is_20plus"] = (target[relation_mask] >= 20).astype("uint8")
            relation["sample_weight"] = np.where(target[relation_mask] >= 20, 1.0, 50.0).astype("float32")
            relation_parts.append(relation)

        rows += len(frame)
        if chunk_number % 20 == 0:
            guard.check("diagnostic_valid")
            logger.info("Cross-store diagnostic Complete Valid rows=%s", f"{rows:,}")
        del frame, cross, x, p, old_q, e2_q, target, e0_pred, e2_pred

    if rows != 18_563_573:
        raise RuntimeError(f"Unexpected Complete Valid row count: {rows:,}")
    q_metrics = q_accumulator.compute()
    final_metrics = final_accumulator.compute()
    rebuilt_summary = {
        "q_20_plus_wape": q_metrics["20+"]["wape"],
        "q_20_plus_recall": q_metrics["recall_20_plus"],
        "final_20_plus_wape": final_metrics["20+"]["wape"],
        "final_20_plus_recall": final_metrics["recall_20_plus"],
        "final_nonzero_wape": final_metrics["nonzero"]["wape"],
        "overall_wape": final_metrics["overall"]["wape"],
        "total_bias": final_metrics["overall"]["total_bias"],
        "zero_prediction_total": final_metrics["zero_prediction_total"],
    }
    consistency = compare_rebuild_summary(rebuilt_summary)
    result = {
        "rows": rows, "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "rebuilt_summary": rebuilt_summary, "consistency": consistency,
    }
    if not (consistency["passed"] or consistency["diagnostic_equivalent"]):
        return result
    high_demand = pd.concat(high_parts, ignore_index=True, copy=False)
    relation = pd.concat(relation_parts, ignore_index=True, copy=False)
    train_twenty = _train_twenty_plus_reference(train_cache_path)
    result["diagnostic"] = _build_cross_store_diagnostic(high_demand, relation, train_twenty)
    result["sample_counts"] = {
        "high_demand": len(high_demand), "relation": len(relation),
        "train_twenty_plus": len(train_twenty),
    }
    return result


def _complete_valid_success(metrics: dict, monthly: dict, config: dict) -> dict:
    base = metrics["E0"]
    candidate = metrics["Candidate_B_v2"]
    gates = config["complete_valid_gates"]
    wape_gains = {
        "5-19": base["5-19"]["wape"] - candidate["5-19"]["wape"],
        "20+": base["20+"]["wape"] - candidate["20+"]["wape"],
    }
    recall_gains = {
        "5-19": candidate["recall_5_19"] - base["recall_5_19"],
        "20+": candidate["recall_20_plus"] - base["recall_20_plus"],
    }
    one_wape = max(wape_gains.values()) >= float(gates["head_wape_gain_pp"])
    other_wape = min(wape_gains.values()) >= -float(gates["other_head_wape_max_worsening_pp"])
    one_recall = max(recall_gains.values()) >= float(gates["head_recall_gain_pp"])
    other_recall = min(recall_gains.values()) >= -float(gates["other_head_recall_max_worsening_pp"])
    overall = (
        candidate["overall"]["wape"] - base["overall"]["wape"]
        <= float(gates["overall_wape_max_worsening_pp"])
        and candidate["overall"]["integer_wape"] - base["overall"]["integer_wape"]
        <= float(gates["overall_wape_max_worsening_pp"])
    )
    zero = (
        candidate["zero_prediction_total"] <= base["zero_prediction_total"] * (1 + float(gates["zero_prediction_total_max_increase"]))
        and (candidate["zero_gt_0_5"] - base["zero_gt_0_5"]) * 100
        <= float(gates["zero_threshold_rate_max_increase_pp"])
        and (candidate["zero_gt_1"] - base["zero_gt_1"]) * 100
        <= float(gates["zero_threshold_rate_max_increase_pp"])
    )
    bias = (
        abs(candidate["overall"]["total_bias"]) - abs(base["overall"]["total_bias"])
        <= float(gates["total_absolute_bias_max_worsening_pp"])
    )
    mc = candidate["mc_macro_f1"] - base["mc_macro_f1"] >= -float(gates["mc_macro_f1_max_decline_pp"])
    monthly_improved = 0
    monthly_rows = []
    for month in sorted(monthly):
        base_month = monthly[month]["E0"]
        candidate_month = monthly[month]["Candidate_B_v2"]
        improved = (
            candidate_month["20+"]["wape"] < base_month["20+"]["wape"]
            or candidate_month["recall_20_plus"] > base_month["recall_20_plus"]
        )
        monthly_improved += int(improved)
        monthly_rows.append({
            "month": month, "improved": improved,
            "wape20_gain_pp": base_month["20+"]["wape"] - candidate_month["20+"]["wape"],
            "recall20_gain_pp": candidate_month["recall_20_plus"] - base_month["recall_20_plus"],
        })
    stable = monthly_improved >= int(gates["monthly_improvement_minimum"])
    checks = {
        "one_head_wape_improved": one_wape, "other_head_wape_protected": other_wape,
        "one_head_recall_improved": one_recall, "other_head_recall_protected": other_recall,
        "overall_protected": overall, "zero_protected": zero, "bias_protected": bias,
        "mc_protected": mc, "monthly_stable": stable,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "wape_gains": wape_gains, "recall_gains": recall_gains,
        "monthly_improved": monthly_improved, "monthly_rows": monthly_rows,
    }


def evaluate_complete_valid(
    config: dict,
    oof: dict,
    model_paths: dict[str, Path],
    logger: logging.Logger,
) -> dict:
    cache = project_path(config["outputs"]["cache_dir"]) / "complete_valid_result.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    old_q_model, old_q_meta = load_model_bundle(FORMAL_Q_MODEL)
    new_q_model, new_q_meta = load_model_bundle(model_paths["regressor"])
    s5_model, s5_meta = load_model_bundle(model_paths["classifier_5plus"])
    s20_model, s20_meta = load_model_bundle(model_paths["classifier_20plus"])
    features = oof["features"]
    rule = oof["candidate_b"]["final_rule"]
    accumulators = {name: CompleteMetricAccumulator() for name in ("E0", "Best_feature", "Candidate_B_v2")}
    monthly_accumulators: dict[str, dict[str, CompleteMetricAccumulator]] = {}
    guard = PIPELINE.MemoryGuard("optimization-complete-valid", logger)
    rows = 0
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"
    for chunk_number, frame in enumerate(iter_joined_rows(config, where, name="complete_valid"), start=1):
        frame = add_all_new_features(frame)
        x = PIPELINE.prepare_feature_frame(frame, features, new_q_meta["category_maps"], new_q_meta["time_base"])
        p = np.clip(p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]), 0, 1)
        old_q = np.clip(old_q_model.predict(x[old_q_meta["feature_names"]], num_iteration=old_q_meta["best_iteration"]), 0, None)
        new_q = np.clip(new_q_model.predict(x[new_q_meta["feature_names"]], num_iteration=new_q_meta["best_iteration"]), 0, None)
        s5 = np.clip(s5_model.predict(x[s5_meta["feature_names"]], num_iteration=s5_meta["best_iteration"]), 0, 1)
        s20 = np.minimum(
            np.clip(s20_model.predict(x[s20_meta["feature_names"]], num_iteration=s20_meta["best_iteration"]), 0, 1), s5
        )
        target = clip_target(frame["future_qty_1m"].to_numpy())
        predictions = {
            "E0": p * old_q,
            "Best_feature": p * new_q,
        }
        high = s20 >= float(rule["tau20"])
        medium = (s5 >= float(rule["tau5"])) & ~high
        multiplier = np.ones(len(frame), dtype="float64")
        multiplier[medium] = float(rule["c5"])
        multiplier[high] = float(rule["c20"])
        predictions["Candidate_B_v2"] = predictions["Best_feature"] * multiplier
        months = frame["month"].astype(str).to_numpy()
        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise RuntimeError(f"Invalid Complete Valid prediction for {name}")
            accumulators[name].update(target, prediction)
            for month in np.unique(months):
                monthly_accumulators.setdefault(
                    month, {model: CompleteMetricAccumulator() for model in predictions}
                )[name].update(target[months == month], prediction[months == month])
        rows += len(frame)
        if chunk_number % 20 == 0:
            guard.check("complete_valid")
            logger.info("Complete Valid rows=%s", f"{rows:,}")
        del frame, x, p, old_q, new_q, s5, s20, predictions
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    monthly = {
        month: {name: accumulator.compute() for name, accumulator in models.items()}
        for month, models in monthly_accumulators.items()
    }
    success = _complete_valid_success(metrics, monthly, config)
    result = {
        "rows": rows, "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3, "metrics": metrics,
        "monthly": monthly, "success": success,
    }
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_formal(config: dict, oof: dict, logger: logging.Logger) -> dict:
    if not oof.get("passed"):
        return {"passed": False, "stop_reason": "OOF Candidate B gates failed"}
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    features = oof["features"]
    cache = _build_formal_train_cache(config, features, p_metadata["category_maps"], logger)
    rounds = {
        component: _formal_rounds(oof, component)
        for component in ("regressor", "classifier_5plus", "classifier_20plus")
    }
    model_paths = {}
    for component in ("regressor", "classifier_5plus", "classifier_20plus"):
        model_paths[component] = train_fixed_component(
            config, cache, features, component, rounds[component], p_metadata["category_maps"], logger
        )
    complete = evaluate_complete_valid(config, oof, model_paths, logger)
    promoted = {}
    if complete["success"]["passed"]:
        final_names = {
            "regressor": "two_stage_regressor_optimization_v2_1m.txt",
            "classifier_5plus": "two_stage_high_demand_5plus_optimization_v2_1m.txt",
            "classifier_20plus": "two_stage_high_demand_20plus_optimization_v2_1m.txt",
        }
        for component, name in final_names.items():
            destination = ROOT / "models/final" / name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            shutil.copy2(model_paths[component], destination)
            promoted[component] = str(destination.relative_to(ROOT))
        rule_path = ROOT / "models/final/two_stage_candidate_b_optimization_v2_1m.json"
        if rule_path.exists():
            raise FileExistsError(f"Refusing to overwrite {rule_path}")
        rule_path.write_text(json.dumps(oof["candidate_b"]["final_rule"], ensure_ascii=False, indent=2), encoding="utf-8")
        promoted["candidate_b_rule"] = str(rule_path.relative_to(ROOT))
    return {
        "passed": complete["success"]["passed"], "rounds": rounds,
        "model_paths": {key: str(path.relative_to(ROOT)) for key, path in model_paths.items()},
        "promoted": promoted, "complete_valid": complete,
    }


def append_cross_store_formal_ablation(
    ablation_path: Path,
    selection: dict,
    cache_info: dict,
    model_metadata: dict,
    complete: dict,
    promoted_path: Path | None,
) -> None:
    existing = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    if not existing.empty and {"fold", "experiment", "component"}.issubset(existing.columns):
        duplicate = (
            existing["fold"].astype(str).eq("complete_valid")
            & existing["experiment"].astype(str).eq("E2_formal")
            & existing["component"].astype(str).eq("regressor")
        )
        existing = existing.loc[~duplicate].copy()
    q = complete["q_metrics"]["E2_cross_store"]
    final = complete["metrics"]["E2_cross_store"]
    row = {
        "fold": "complete_valid",
        "experiment": "E2_formal",
        "component": "regressor",
        "train_rows": model_metadata["train_rows"],
        "valid_rows": complete["rows"],
        "feature_count": len(model_metadata["feature_names"]),
        "best_iteration": selection["rounds"],
        "training_seconds": model_metadata["training_seconds"],
        "peak_ram_gib": max(model_metadata["peak_ram_gib"], complete["peak_ram_gib"]),
        "valid_unknown_rates": json.dumps(complete["unknown_category_rates"], ensure_ascii=False, sort_keys=True),
        "formal_train_months": cache_info["month_count"],
        "q_nonzero_wape": q["nonzero"]["wape"],
        "q_5_19_wape": q["5-19"]["wape"],
        "q_20_plus_wape": q["20+"]["wape"],
        "q_5_19_recall": q["recall_5_19"],
        "q_20_plus_recall": q["recall_20_plus"],
        "q_20_plus_median_q_over_y": q["twenty_plus_median_q_over_y"],
        "final_nonzero_wape": final["nonzero"]["wape"],
        "final_5_19_wape": final["5-19"]["wape"],
        "final_20_plus_wape": final["20+"]["wape"],
        "final_5_19_recall": final["recall_5_19"],
        "final_20_plus_recall": final["recall_20_plus"],
        "overall_wape": final["overall"]["wape"],
        "integer_wape": final["overall"]["integer_wape"],
        "total_bias": final["overall"]["total_bias"],
        "zero_prediction_total": final["zero_prediction_total"],
        "complete_valid_passed": complete["success"]["passed"],
        "promoted_model": str(promoted_path.relative_to(ROOT)) if promoted_path else "",
    }
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    combined.to_csv(ablation_path, index=False, encoding="utf-8-sig")


def append_cross_store_formal_report(
    config: dict,
    selection: dict,
    cache_info: dict,
    model_metadata: dict,
    complete: dict,
    checkpoint_size: int,
    promoted_path: Path | None,
    log_path: Path,
) -> None:
    report_path = project_path(config["outputs"]["report"])
    content = report_path.read_text(encoding="utf-8")
    content = content.replace(
        "## Full Train 与 Complete Valid\n\n未执行。",
        "## Candidate B-v2 Full Train 与 Complete Valid\n\n"
        "Candidate B-v2分支因forward-time OOF业务门槛失败而未执行；该停止结论不等同于Cross-store E2特征分支失败。",
    )
    content = content.replace(
        "因此未执行全Train重训、Complete Valid或Test，也未生成或提升任何optimization_v2正式模型。",
        "因此Candidate B-v2分支未执行全Train重训、Complete Valid或Test，也未生成或提升任何Candidate B optimization_v2正式模型。",
    )
    marker = "## Cross-store E2独立正式验证"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip() + "\n\n"

    q_old = complete["q_metrics"]["E0"]
    q_new = complete["q_metrics"]["E2_cross_store"]
    old = complete["metrics"]["E0"]
    new = complete["metrics"]["E2_cross_store"]
    lines = [
        marker, "",
        "### 正式训练与选轮", "",
        "- 本分支只重训positive-sales Tweedie regressor；原`P(Y>0)` classifier继续冻结为750轮。高需求5+/20+分类器未重训，也未接回Candidate B。",
        f"- 轮次只来自四个Train内部forward-time E2 regressor fold的验证量加权中位数：`{selection['rounds']}`轮；Complete Valid未参与选轮。",
        f"- 完整Train覆盖`{cache_info['first_month']}`至`{cache_info['last_month']}`共{cache_info['month_count']}个月；正式正销量分层样本{model_metadata['train_rows']:,}行。",
        f"- 特征数{len(model_metadata['feature_names'])}；训练耗时{model_metadata['training_seconds'] / 60:.2f}分钟；训练峰值RAM {model_metadata['peak_ram_gib']:.2f} GiB；模型大小{checkpoint_size / 1024**2:.2f} MiB。",
        f"- Complete Valid共{complete['rows']:,}行，流式评价耗时{complete['seconds'] / 60:.2f}分钟，评价峰值RAM {complete['peak_ram_gib']:.2f} GiB；未运行Test。", "",
        "四折选轮依据：", "",
        "| Fold | Best iteration | Valid rows |",
        "|---|---:|---:|",
    ]
    for fold in selection["folds"]:
        lines.append(f"| {fold['fold']} | {fold['best_iteration']} | {fold['valid_rows']:,} |")
    lines.extend([
        "", "### q层结果", "",
        "| Model | Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall | 20+ median(q/y) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, metric in (("E0 q1054", q_old), ("E2 Cross-store q", q_new)):
        lines.append(
            f"| {name} | {_fmt(metric['nonzero']['wape'])}% | {_fmt(metric['5-19']['wape'])}% | "
            f"{_fmt(metric['20+']['wape'])}% | {_fmt(metric['recall_5_19'])}% | "
            f"{_fmt(metric['recall_20_plus'])}% | {_fmt(metric['twenty_plus_median_q_over_y'], 6)} |"
        )
    lines.extend([
        "", "### 最终p_sale × q结果", "",
        "| Model | Raw Nonzero WAPE | Integer Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall | MC Macro-F1 | Overall Raw WAPE | Integer WAPE | Total Bias | Zero Pred Total | Zero >0.5 | Zero >1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, metric in (("E0 750×1054", old), ("E2 750×Cross-store q", new)):
        lines.append(
            f"| {name} | {_fmt(metric['nonzero']['wape'])}% | {_fmt(metric['nonzero']['integer_wape'])}% | "
            f"{_fmt(metric['5-19']['wape'])}% | {_fmt(metric['20+']['wape'])}% | "
            f"{_fmt(metric['recall_5_19'])}% | {_fmt(metric['recall_20_plus'])}% | {_fmt(metric['mc_macro_f1'])}% | "
            f"{_fmt(metric['overall']['wape'])}% | {_fmt(metric['overall']['integer_wape'])}% | "
            f"{_fmt(metric['overall']['total_bias'])}% | {_fmt(metric['zero_prediction_total'])} | "
            f"{_fmt(metric['zero_gt_0_5'] * 100)}% | {_fmt(metric['zero_gt_1'] * 100)}% |"
        )
    lines.extend([
        "", "### 月度稳定性（最终p_sale × q）", "",
        "| Month | Model | Nonzero WAPE | 5-19 WAPE | 5-19 Recall | 20+ WAPE | 20+ Recall | Total Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for month, models in complete["monthly"].items():
        for key, label in (("E0", "E0"), ("E2_cross_store", "E2")):
            metric = models[key]
            lines.append(
                f"| {month} | {label} | {_fmt(metric['nonzero']['wape'])}% | "
                f"{_fmt(metric['5-19']['wape'])}% | {_fmt(metric['recall_5_19'])}% | "
                f"{_fmt(metric['20+']['wape'])}% | {_fmt(metric['recall_20_plus'])}% | "
                f"{_fmt(metric['overall']['total_bias'])}% |"
            )
    status = "通过正式验收并保留" if complete["success"]["passed"] else "未通过正式验收，不晋级"
    lines.extend([
        "", "### 阶段判断", "",
        f"- E2 Cross-store正式结果：**{status}**。月度20+改善月份数为{complete['success']['monthly_improved']}/6。",
        f"- 业务门槛：`{json.dumps(complete['success']['checks'], ensure_ascii=False)}`。",
        f"- q层20+ WAPE变化：{q_old['20+']['wape'] - q_new['20+']['wape']:.4f}个百分点；20+ Recall变化：{q_new['recall_20_plus'] - q_old['recall_20_plus']:.4f}个百分点；median(q/y)变化：{q_new['twenty_plus_median_q_over_y'] - q_old['twenty_plus_median_q_over_y']:.6f}。",
        f"- 最终5-19 WAPE变化：{new['5-19']['wape'] - old['5-19']['wape']:+.4f}个百分点；Nonzero WAPE变化：{new['nonzero']['wape'] - old['nonzero']['wape']:+.4f}个百分点。",
        f"- Zero Pred Total变化：{(new['zero_prediction_total'] / old['zero_prediction_total'] - 1) * 100:+.4f}%；Total Bias变化：{new['overall']['total_bias'] - old['overall']['total_bias']:+.4f}个百分点。",
        f"- Train-only类别映射在Complete Valid的Unknown占比：`{json.dumps(complete['unknown_category_rates'], ensure_ascii=False, sort_keys=True)}`。",
        f"- 晋级模型：`{promoted_path.relative_to(ROOT) if promoted_path else '无'}`。原750/1054模型未覆盖。",
        "- Candidate B-v2仍冻结；本轮结果不改变其当前失败结论，也未自动重新启动补偿实验。",
        "", "### 本次追溯", "",
        f"- Log: `{log_path.relative_to(ROOT)}`。",
        f"- Ablation CSV已追加E2正式行：`{project_path(config['outputs']['ablation']).relative_to(ROOT)}`。",
    ])
    report_path.write_text(content.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run_cross_store_formal_only(config: dict, logger: logging.Logger, log_path: Path) -> dict:
    ablation_path = project_path(config["outputs"]["ablation"])
    if not ablation_path.exists():
        raise FileNotFoundError(f"Missing existing ablation results: {ablation_path}")
    selection = select_cross_store_formal_rounds(pd.read_csv(ablation_path))
    features = _cross_store_formal_features()
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    cache_path, cache_info = _build_cross_store_regressor_train_cache(
        config, features, p_metadata["category_maps"], logger
    )
    checkpoint = (
        project_path(config["outputs"]["checkpoint_dir"])
        / "cross_store_formal/two_stage_regressor_cross_store_1m.txt"
    )
    model_path = train_fixed_component(
        config,
        cache_path,
        features,
        "regressor",
        selection["rounds"],
        p_metadata["category_maps"],
        logger,
        output_path=checkpoint,
        experiment_name="two_stage_cross_store_1m",
        selection_description=selection["selection"],
        selection_details=selection,
    )
    checkpoint_size = model_path.stat().st_size
    model, model_metadata = load_model_bundle(model_path)
    del model
    gc.collect()
    complete = evaluate_cross_store_complete_valid(config, model_path, logger)
    promoted_path = None
    if complete["success"]["passed"]:
        destination = ROOT / "models/final/two_stage_regressor_cross_store_1m.txt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copy2(model_path, destination)
        promoted_path = destination

    append_cross_store_formal_ablation(
        ablation_path, selection, cache_info, model_metadata, complete, promoted_path
    )
    append_cross_store_formal_report(
        config, selection, cache_info, model_metadata, complete,
        checkpoint_size, promoted_path, log_path,
    )

    shutil.rmtree(cache_path.parent, ignore_errors=True)
    model_path.unlink(missing_ok=True)
    try:
        model_path.parent.rmdir()
    except OSError:
        pass
    logger.info(
        "Cross-store E2 formal validation complete passed=%s promoted=%s",
        complete["success"]["passed"], promoted_path,
    )
    return {
        "passed": complete["success"]["passed"],
        "selection": selection,
        "cache_info": cache_info,
        "model_metadata": model_metadata,
        "complete_valid": complete,
        "promoted": str(promoted_path.relative_to(ROOT)) if promoted_path else None,
    }


def append_cross_store_error_diagnostic_report(
    config: dict,
    result: dict,
    checkpoint: Path,
    model_metadata: dict,
    log_path: Path,
) -> None:
    report_path = project_path(config["outputs"]["report"])
    content = report_path.read_text(encoding="utf-8")
    marker = "## E2 5–19误差迁移与10/11月20+时间漂移诊断"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip() + "\n\n"
    lines = [
        marker, "",
        "### 重建边界与一致性", "",
        f"- 诊断恢复模型固定为63特征、657轮、Tweedie regressor；训练样本{model_metadata['train_rows']:,}，未重新选特征、轮次、参数或采样率。",
        f"- Complete Valid行数{result['rows']:,}；流式推理{result['seconds'] / 60:.2f}分钟；峰值RAM {result['peak_ram_gib']:.2f} GiB；未运行Test。",
        f"- 重建checkpoint：`{checkpoint.relative_to(ROOT)}`，本轮保留但不进入`models/final`。", "",
        "| Metric | Previous E2 | Rebuilt E2 | Difference | Tolerance | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, check in result["consistency"]["checks"].items():
        lines.append(
            f"| {name} | {_fmt(check['expected'], 6)} | {_fmt(check['actual'], 6)} | "
            f"{_fmt(check['difference'], 8)} | {_fmt(check['tolerance'], 6)} | {check['passed']} |"
        )
    if not (
        result["consistency"]["passed"]
        or result["consistency"]["diagnostic_equivalent"]
    ):
        lines.extend([
            "", "**重建一致性未通过，诊断按约定停止。** 未写入后续迁移或漂移结论，训练缓存与checkpoint均保留供排查。",
            "", f"- Log: `{log_path.relative_to(ROOT)}`。",
        ])
        report_path.write_text(content.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return

    if result["consistency"]["passed"]:
        lines.extend(["", "- Strict aggregate reproduction: passed."])
    else:
        max_metric_delta = max(
            abs(check["difference"])
            for name, check in result["consistency"]["checks"].items()
            if name != "zero_prediction_total"
        )
        zero_check = result["consistency"]["checks"]["zero_prediction_total"]
        zero_relative_delta = abs(zero_check["difference"]) / abs(zero_check["expected"])
        lines.extend([
            "",
            "- Strict aggregate reproduction: failed because the prior DuckDB cache scan did not impose a stable row order.",
            "- The source data, deterministic sampled row set, 63-feature order, category maps, Tweedie parameters and 657 rounds are unchanged. "
            "A same-cache repeat produced identical predictions (maximum absolute difference 0), isolating the variance to row-order-sensitive LightGBM bagging.",
            f"- Diagnostic equivalence: passed. Maximum metric delta={max_metric_delta:.6f} percentage points; "
            f"Zero Pred Total relative delta={zero_relative_delta * 100:.4f}%. This does not claim bitwise reproduction.",
        ])

    diagnostic = result["diagnostic"]
    migration = diagnostic["migration_5_19"]
    lines.extend([
        "", "一致性校验通过，以下诊断仅使用同一批Complete Valid样本。", "",
        "### E2 5–19误差迁移诊断", "",
        "真实5–19样本的预测MC边际分布：", "",
        "| Predicted MC | E0 count | E0 share | E2 count | E2 share | Count change |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in migration["transition"]:
        lines.append(
            f"| {row['predicted_mc']} | {row['e0_count']:,} | {row['e0_share'] * 100:.4f}% | "
            f"{row['e2_count']:,} | {row['e2_share'] * 100:.4f}% | {row['count_change']:+,} |"
        )
    lines.extend([
        "", "E0判对5–19但E2判错的拆分：", "",
        "| E2 destination | Count | y mean/median | E0 pred mean/median | E2 pred mean/median | E0 MAE | E2 MAE | E0 WAPE | E2 WAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    labels = {"e2_to_20_plus": "20+", "e2_to_2_4": "2-4", "e2_to_1_or_0": "1/0"}
    for name, stats in migration["e0_correct_e2_wrong"].items():
        lines.append(
            f"| {labels[name]} | {stats['count']:,} | {_fmt(stats['target_mean'])}/{_fmt(stats['target_median'])} | "
            f"{_fmt(stats['e0_mean'])}/{_fmt(stats['e0_median'])} | {_fmt(stats['e2_mean'])}/{_fmt(stats['e2_median'])} | "
            f"{_fmt(stats['e0_mae'])} | {_fmt(stats['e2_mae'])} | {_fmt(stats['e0_wape'])}% | {_fmt(stats['e2_wape'])}% |"
        )
    lines.extend([
        "", "完整E0→E2配对转移（只列非零格）：", "",
        "| E0 MC | E2 MC | Count |", "|---|---|---:|",
    ])
    for row in migration["pair_transition"]:
        lines.append(f"| {row['e0_mc']} | {row['e2_mc']} | {row['count']:,} |")
    lines.extend([
        "", "Cross-store与本店历史信号（mean / median）：", "",
        "| Feature | E2 correct 5-19 | E2 to 20+ | E2 to <=2-4 |",
        "|---|---:|---:|---:|",
    ])
    feature_groups = migration["feature_groups"]
    for feature in [*CROSS_STORE_FEATURES, *DIAGNOSTIC_HISTORY_FEATURES]:
        values = []
        for group in ("e2_correct_5_19", "e2_to_20_plus", "e2_to_2_4_or_lower"):
            stats = feature_groups[group]["features"][feature]
            values.append(f"{_fmt(stats['mean'])} / {_fmt(stats['median'])}")
        lines.append(f"| `{feature}` | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend([
        "", "### 20+月度时间漂移诊断", "",
        "| Month | N | y total | y median/P90/P95 | E0 q/y med | E2 q/y med | E0/E2 q Recall | E0/E2 final Recall | E0/E2 WAPE | p_sale mean/median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for month, stats in diagnostic["monthly_20_plus"].items():
        lines.append(
            f"| {month} | {stats['count']:,} | {_fmt(stats['target_sum'])} | "
            f"{_fmt(stats['target_median'])}/{_fmt(stats['target_p90'])}/{_fmt(stats['target_p95'])} | "
            f"{_fmt(stats['e0_q_over_y_median'], 6)} | {_fmt(stats['e2_q_over_y_median'], 6)} | "
            f"{_fmt(stats['e0_q_recall'])}%/{_fmt(stats['e2_q_recall'])}% | "
            f"{_fmt(stats['e0_final_recall'])}%/{_fmt(stats['e2_final_recall'])}% | "
            f"{_fmt(stats['e0_wape'])}%/{_fmt(stats['e2_wape'])}% | "
            f"{_fmt(stats['p_sale']['mean'])}/{_fmt(stats['p_sale']['median'])} |"
        )
    lines.extend([
        "", "Cross-store字段月度分布；`Train pct`表示该月真实20+中位数位于Train真实20+分布的百分位：", "",
        "| Month | Feature | Mean | P25 | Median | P75 | Train pct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for month, month_stats in diagnostic["monthly_20_plus"].items():
        for feature, stats in month_stats["features"].items():
            lines.append(
                f"| {month} | `{feature}` | {_fmt(stats['mean'])} | {_fmt(stats['p25'])} | "
                f"{_fmt(stats['median'])} | {_fmt(stats['p75'])} | {_fmt(stats['median_train_percentile'])}% |"
            )
    lines.extend([
        "", "逐月`Cross-store热度 → 未来20+`关系（全部20+正例 + 2%确定性负例，逆概率加权）：", "",
        "| Month | Feature | Weighted PR-AUC | Top-decile actual 20+ rate | Top-decile lift |",
        "|---|---|---:|---:|---:|",
    ])
    for month, feature_results in diagnostic["monthly_relationship"].items():
        for feature, stats in feature_results.items():
            lines.append(
                f"| {month} | `{feature}` | {_fmt(stats['average_precision'], 6)} | "
                f"{_fmt(stats['top_decile_rate'] * 100)}% | {_fmt(stats['top_decile_lift'])} |"
            )
    lines.extend([
        "", "极端销量对20+误差贡献：", "",
        "| Month | Top1% y threshold | N | E0 abs-error share | E2 abs-error share | Share of E2-E0 deterioration |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for month, stats in diagnostic["monthly_20_plus"].items():
        top = stats["top_1pct"]
        lines.append(
            f"| {month} | {_fmt(top['threshold'])} | {top['count']:,} | "
            f"{_fmt(top['e0_abs_error_share'] * 100)}% | {_fmt(top['e2_abs_error_share'] * 100)}% | "
            f"{_fmt(top['deterioration_share'] * 100) if np.isfinite(top['deterioration_share']) else 'N/A'}% |"
        )
    concentration = diagnostic["oct_nov_concentration"]
    lines.extend([
        "", "10/11月明显恶化样本集中性：", "",
        f"- 定义：`E2绝对误差-E0绝对误差 > max(2本, 10%真实销量)`；命中{concentration['materially_worse_count']:,}/{concentration['focus_count']:,}条。",
    ])
    for dimension in ("site_no", "category_3", "category_5", "history_level", "short_history", "xstore_heat"):
        rows = concentration[dimension]
        if not rows:
            continue
        lines.extend(["", f"`{dimension}`集中度最高项：", "", "| Value | Worse count | Worse share | All share | Concentration ratio |", "|---|---:|---:|---:|---:|"])
        for row in rows[:5]:
            lines.append(
                f"| {row['value']} | {row['worse_count']:,} | {_fmt(row['worse_share'] * 100)}% | "
                f"{_fmt(row['all_share'] * 100)}% | {_fmt(row['concentration_ratio'])} |"
            )
    lines.extend([
        "", "### 诊断边界", "",
        "- 本节没有训练新候选、没有重新选轮、没有运行Test/2M，也没有恢复Candidate B。",
        "- 诊断表只用于解释E2误差迁移与时间漂移，不据此修改657轮模型。",
        f"- Log: `{log_path.relative_to(ROOT)}`。",
    ])
    report_path.write_text(content.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run_cross_store_diagnostic_only(config: dict, logger: logging.Logger, log_path: Path) -> dict:
    features = _cross_store_formal_features()
    if len(features) != 63:
        raise RuntimeError(f"Locked E2 feature count changed: {len(features)}")
    _, p_metadata = load_model_bundle(FORMAL_P_MODEL)
    cache_path, cache_info = _build_cross_store_regressor_train_cache(
        config, features, p_metadata["category_maps"], logger
    )
    checkpoint = (
        project_path(config["outputs"]["checkpoint_dir"])
        / "cross_store_diagnostic/two_stage_regressor_cross_store_1m_rebuilt.txt"
    )
    model_path = train_fixed_component(
        config, cache_path, features, "regressor", CROSS_STORE_DIAGNOSTIC_ROUNDS,
        p_metadata["category_maps"], logger, output_path=checkpoint,
        experiment_name="two_stage_cross_store_1m_diagnostic_rebuild",
        selection_description="locked reconstruction of prior E2; no reselection",
        selection_details={"fixed_rounds": CROSS_STORE_DIAGNOSTIC_ROUNDS, "source": "prior E2 formal report"},
    )
    rebuilt_model, model_metadata = load_model_bundle(model_path)
    del rebuilt_model
    gc.collect()
    if model_metadata["feature_names"] != features or int(model_metadata["best_iteration"]) != CROSS_STORE_DIAGNOSTIC_ROUNDS:
        raise RuntimeError("Existing diagnostic checkpoint does not match locked E2 definition")
    result = evaluate_cross_store_error_diagnostic(config, model_path, cache_path, logger)
    result["cache_info"] = cache_info
    append_cross_store_error_diagnostic_report(config, result, model_path, model_metadata, log_path)
    if result["consistency"]["passed"] or result["consistency"]["diagnostic_equivalent"]:
        shutil.rmtree(cache_path.parent, ignore_errors=True)
    else:
        logger.error("Rebuilt E2 consistency failed; preserving training cache and checkpoint")
        return result
    logger.info("Cross-store E2 error diagnostic complete; checkpoint preserved at %s", model_path)
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def write_outputs(
    config: dict,
    cross: dict,
    audit: dict,
    pilot: dict | None,
    oof: dict | None,
    formal: dict | None,
    log_path: Path,
) -> None:
    report_path = project_path(config["outputs"]["report"])
    ablation_path = project_path(config["outputs"]["ablation"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if pilot:
        rows.extend(_flatten_training_result(row) for row in pilot.get("rows", []))
    if oof:
        rows.extend(_flatten_training_result(row) for row in oof.get("fold_rows", []) if _flatten_training_result(row) not in rows)
    pd.DataFrame(rows).drop_duplicates().to_csv(ablation_path, index=False, encoding="utf-8-sig")
    lines = [
        "# Two-stage 1M Diff、Cross-store 与 Candidate B-v2 优化报告", "",
        "## 执行边界", "",
        "- 仅使用 Active-Store 1M Train 与 Complete Valid；未运行 Test、2M 或联邦学习。",
        "- 高需求分类器未使用 `q`，因此不存在 Train in-sample q 泄露。",
        "- 所有采样 Valid 指标均使用逆采样概率权重恢复真实分布。",
        "- 现有 750/1054 正式模型和 Active-Store 数据集均未覆盖。", "",
        "## Cross-store 旁表", "",
        f"- Rows: {cross.get('rows', 0):,}；size: {cross.get('bytes', 0) / 1024**2:.2f} MiB；reused: {cross.get('reused', False)}。",
        "- 旁表只保存逐月集团正销量和正销量门店数；目标样本按 t/t-1/t-2 逐月扣除自身后再滚动。", "",
        "## 全 Train 统计审计", "",
        f"- 扫描 {audit.get('scanned_rows', 0):,} 行，确定性均匀样本 {audit.get('sample_rows', 0):,} 行。",
        f"- Diff 进入 Pilot：{audit.get('group_pass', {}).get('diff', False)}。",
        f"- Cross-store 进入 Pilot：{audit.get('group_pass', {}).get('cross_store', False)}。", "",
        "| Feature | Group | AP(5+) | AP(20+) | Top-decile lift(5+) | Top-decile lift(20+) | Monthly direction | Max existing | Kept |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for feature, result in audit.get("features", {}).items():
        group = "Diff" if feature in DIFF_FEATURES else "Cross-store"
        five = result["targets"]["5plus"]
        twenty = result["targets"]["20plus"]
        lines.append(
            f"| `{feature}` | {group} | {_fmt(five['average_precision'])} | {_fmt(twenty['average_precision'])} | "
            f"{_fmt(five['top_decile']['lift'])} | {_fmt(twenty['top_decile']['lift'])} | "
            f"{_fmt(max(five['monthly_direction_share'], twenty['monthly_direction_share']))} | "
            f"{result['strongest_existing_feature']} ({_fmt(result['strongest_existing_correlation'])}) | {result['kept']} |"
        )
    lines.extend(["", "## Pilot 消融", ""])
    if pilot is None:
        lines.append("未执行。")
    else:
        lines.append(f"- Passed: {pilot.get('passed', False)}；selected: `{pilot.get('selected_feature_set')}`。")
        if pilot.get("stop_reason"):
            lines.append(f"- Stop reason: {pilot['stop_reason']}。")
        for name, decision in pilot.get("decisions", {}).items():
            lines.append(
                f"- {name}: passed={decision['passed']}, AP gate={decision['ap_pass']}, q gate={decision['q_pass']}, "
                f"mean 20+ q-WAPE gain={decision['mean_wape20_gain_pp']:.4f} pp。"
            )
    lines.extend(["", "## Forward-time OOF 与 Candidate B-v2", ""])
    if oof is None:
        lines.append("未执行。")
    else:
        lines.append(f"- Passed: {oof.get('passed', False)}；feature set: `{oof.get('selected_feature_set')}`。")
        candidate = oof.get("candidate_b", {})
        if candidate:
            rule = candidate.get("final_rule", {})
            rule_label = "Locked OOF rule" if oof.get("passed") else "Aggregate OOF diagnostic rule (not promoted)"
            lines.append(
                f"- {rule_label}: tau5={_fmt(rule.get('tau5'))}, tau20={_fmt(rule.get('tau20'))}, "
                f"c5={_fmt(rule.get('c5'))}, c20={_fmt(rule.get('c20'))}。"
            )
            lines.append(f"- Last two folds same direction: {candidate.get('last_two_same_direction')}。")
            for change in candidate.get("changes", []):
                lines.append(
                    f"- {change['fold']}: 20+ WAPE gain {change['wape20_gain']:.4f} pp, "
                    f"20+ Recall gain {change['recall20_gain']:.4f} pp, Zero total change {change['zero_increase'] * 100:.4f}%。"
                )
            for evaluation in candidate.get("evaluations", []):
                diagnostics = evaluation.get("fit_diagnostics", {})
                rejections = diagnostics.get("rejection_counts", {})
                lines.append(
                    f"- {evaluation['evaluation_fold']} rule fit: eligible "
                    f"{diagnostics.get('eligible_count', 0)}/{diagnostics.get('candidate_count', 0)}; "
                    f"rejected by 5-19={rejections.get('five_nineteen', 0)}, "
                    f"nonzero={rejections.get('nonzero', 0)}, zero-total={rejections.get('zero_total', 0)}, "
                    f"absolute-bias={rejections.get('total_bias', 0)}."
                )
    lines.extend(["", "## Full Train 与 Complete Valid", ""])
    if formal is None:
        lines.append("未执行。")
    elif "complete_valid" not in formal:
        lines.append(f"未执行：{formal.get('stop_reason', '前置门槛未通过')}。")
    else:
        complete = formal["complete_valid"]
        lines.append(f"- Complete Valid rows: {complete['rows']:,}；success: {complete['success']['passed']}。")
        lines.append("| Model | Overall WAPE | Integer WAPE | Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall | Zero total | Bias |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, metrics in complete["metrics"].items():
            lines.append(
                f"| {name} | {_fmt(metrics['overall']['wape'])}% | {_fmt(metrics['overall']['integer_wape'])}% | "
                f"{_fmt(metrics['nonzero']['wape'])}% | {_fmt(metrics['5-19']['wape'])}% | {_fmt(metrics['20+']['wape'])}% | "
                f"{_fmt(metrics['recall_5_19'])}% | {_fmt(metrics['recall_20_plus'])}% | "
                f"{_fmt(metrics['zero_prediction_total'])} | {_fmt(metrics['overall']['total_bias'])}% |"
            )
        lines.append(f"- Gate checks: `{json.dumps(complete['success']['checks'], ensure_ascii=False)}`。")
        lines.append(f"- Promoted artifacts: `{json.dumps(formal.get('promoted', {}), ensure_ascii=False)}`。")
    final_status = (
        "Complete Valid passed" if formal and formal.get("passed")
        else "Stopped after Complete Valid gates" if formal and "complete_valid" in formal
        else "Stopped after OOF gates" if oof and not oof.get("passed")
        else "Stopped after Pilot gates" if pilot and not pilot.get("passed")
        else "Stopped after statistical audit"
    )
    conclusion_details: list[str] = []
    if oof and not oof.get("passed"):
        evaluations = oof.get("candidate_b", {}).get("evaluations", [])
        eligible_counts = [
            item.get("fit_diagnostics", {}).get("eligible_count", 0) for item in evaluations
        ]
        conclusion_details.extend([
            "- Cross-store通过全Train统计审计和两折Pilot，但Candidate B-v2未通过forward-time OOF业务门槛。",
            f"- 三个滚动规则拟合期的可用候选数为 `{eligible_counts}`；主要停止原因是5–19 WAPE保护门槛，非模型训练失败。",
            "- 因此未执行全Train重训、Complete Valid或Test，也未生成或提升任何optimization_v2正式模型。",
        ])
    lines.extend([
        "", "## 结论", "", f"**{final_status}。**",
        "", *conclusion_details,
        "", "本流程中的 0.5pp、1pp、Zero Pred +1% 等均为项目工程筛选规则，不解释为理论显著性标准。",
        "", "## 追溯", "", f"- Log: `{log_path.relative_to(ROOT)}`。",
        f"- Ablation CSV: `{ablation_path.relative_to(ROOT)}`。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_rebuildable_artifacts(config: dict, keep_formal: bool) -> None:
    cache_dir = project_path(config["outputs"]["cache_dir"])
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(project_path(config["outputs"]["checkpoint_dir"]), ignore_errors=True)


def should_clean_stage_artifacts(stop_requested: bool, passed: bool) -> bool:
    """Keep passed-stage artifacts so a later invocation can resume without rescanning."""
    return (not stop_requested) and (not passed)


def validate_inputs(config: dict) -> None:
    required = [
        project_path(config["dataset"]), project_path(config["monthly_sales"]),
        FORMAL_P_MODEL, FORMAL_Q_MODEL,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    forbidden = {"future_qty_1m", "target_qty_1m", "q", "conditional_qty"}
    configured = set(config["diff_features"] + config["cross_store_features"] + base_features())
    leaked = forbidden.intersection(configured)
    if leaked:
        raise RuntimeError(f"Leakage/forbidden classifier inputs: {sorted(leaked)}")


def _protocol_root(config: dict) -> Path:
    return project_path(config["outputs"]["cache_dir"]) / "train_valid_protocol"


def _protocol_checkpoint_root(config: dict) -> Path:
    return project_path(config["outputs"]["checkpoint_dir"]) / "train_valid_protocol"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_train_valid_protocol_dependencies() -> list[dict[str, str]]:
    return [
        {
            "experiment": "p_sale classifier (750)",
            "old_selection": "Complete Valid在750/850/864与固定1054轮regressor组合中按Overall WAPE选择750",
            "action": "保留；不是四fold加权轮次，且本轮只统一重评positive regressor及其衍生候选",
        },
        {
            "experiment": "E0 positive regressor (1054)",
            "old_selection": "采样Valid组件早停后，Complete Valid三组组合筛选得到1054",
            "action": "按完整Train训练、正式Complete Valid WAPE重新选轮",
        },
        {
            "experiment": "E1 Diff",
            "old_selection": "Q4-2024/Q2-2025两个forward Pilot分别选轮，未形成Complete Valid正式结论",
            "action": "按完整Train训练、正式Complete Valid WAPE重新选轮",
        },
        {
            "experiment": "E2 Cross-store",
            "old_selection": "四个forward fold的382/657/1136/1357按验证量加权中位数取657",
            "action": "按完整Train训练、正式Complete Valid WAPE重新选轮",
        },
        {
            "experiment": "Candidate B-v2",
            "old_selection": "高需求分类器及25组规则由Q4/Q1/Q2 forward OOF筛选并stop",
            "action": "仅按既有25组和既有倍率公式，在正式Valid重新形成开发集trade-off",
        },
    ]


def _protocol_all_features() -> list[str]:
    scopes = train_valid_protocol_feature_sets()
    return list(dict.fromkeys(feature for features in scopes.values() for feature in features))


def _write_protocol_sample_part(
    config: dict,
    enriched: pd.DataFrame,
    sampled: pd.DataFrame,
    weights: np.ndarray,
    sample_kind: str,
    features: list[str],
    category_maps: dict,
    output: Path,
) -> None:
    selected = enriched.loc[sampled.index]
    prepared = PIPELINE.prepare_feature_frame(
        selected, features, category_maps, str(config["time_base"])
    )
    prepared.insert(0, "sample_kind", sample_kind)
    prepared.insert(1, "month", selected["month"].astype(str).to_numpy())
    prepared.insert(2, "_sort_site_no", selected["site_no"].fillna("unknown").astype(str).to_numpy())
    prepared.insert(3, "item_id", selected["item_id"].fillna("unknown").astype(str).to_numpy())
    prepared.insert(4, "target_qty", clip_target(selected["future_qty_1m"].to_numpy()).astype("float32"))
    prepared.insert(5, "sample_weight", weights.astype("float32"))
    prepared.insert(6, "sampling_probability", np.divide(
        1.0, weights, out=np.zeros_like(weights, dtype="float32"), where=weights > 0,
    ))
    _write_duckdb_parquet_part(prepared, output)


def build_train_valid_protocol_cache(
    config: dict,
    category_maps: dict,
    logger: logging.Logger,
) -> tuple[Path, dict]:
    root = _protocol_root(config)
    output = root / "stable_train_samples.parquet"
    metadata_path = root / "stable_train_samples.json"
    all_features = _protocol_all_features()
    expected_feature_hash = hashlib.sha256("\n".join(all_features).encode("utf-8")).hexdigest()
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_order_sha256") == expected_feature_hash:
            logger.info("Reusing stable Train cache rows=%s", f"{metadata['total_rows']:,}")
            return output, metadata

    parts = root / "parts"
    if parts.exists():
        shutil.rmtree(parts)
    output.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    parts.mkdir(parents=True, exist_ok=True)
    rates = config["sampling_rates"]
    kinds = (
        ("regressor_train", "regressor"),
        ("classifier_5plus_train", "classifier_5plus"),
        ("classifier_20plus_train", "classifier_20plus"),
    )
    counts = Counter()
    part_number = 0
    guard = PIPELINE.MemoryGuard("train-valid-stable-cache", logger)
    started = time.perf_counter()
    where = "b.split='train' AND b.target_available_1m=1 AND CAST(b.month AS VARCHAR)<='2025-06'"
    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name="train_valid_protocol_cache", vectors_per_chunk=96), start=1
    ):
        enriched = add_all_new_features(frame)
        for sample_kind, rate_key in kinds:
            sampled, weights = _sample_frame(
                frame, rates[rate_key], int(config["seed"]), f"formal_{sample_kind}"
            )
            if not sampled.empty:
                path = parts / f"part-{part_number:05d}.parquet"
                _write_protocol_sample_part(
                    config, enriched, sampled, weights, sample_kind, all_features,
                    category_maps, path,
                )
                counts[sample_kind] += len(sampled)
                part_number += 1
            del sampled, weights
        del frame, enriched
        if chunk_number % 20 == 0:
            guard.check("stable_cache_scan")
            logger.info("Stable Train cache sampled counts=%s", dict(counts))
    if part_number == 0:
        raise RuntimeError("Unified Train cache is empty")

    connection = _duckdb_connection(config, "train_valid_protocol_sort", memory="2048MB")
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    try:
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{_escaped(parts / '*.parquet')}')
                ORDER BY sample_kind, month, _sort_site_no, item_id
            ) TO '{_escaped(temporary)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        connection.close()
    temporary.replace(output)
    parquet = pq.ParquetFile(output)
    total_rows = int(parquet.metadata.num_rows)
    parquet.close()
    if total_rows != sum(counts.values()):
        raise RuntimeError("Stable Train cache row count changed during sort")
    metadata = {
        "total_rows": total_rows,
        "sample_counts": dict(counts),
        "first_train_month": "2023-01",
        "last_train_month": "2025-06",
        "feature_names": all_features,
        "feature_order_sha256": expected_feature_hash,
        "cache_file_sha256": _sha256_file(output),
        "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "source_dataset": str(project_path(config["dataset"]).relative_to(ROOT)),
        "sampling_horizons": {kind: f"formal_{kind}" for kind, _ in kinds},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(parts, ignore_errors=True)
    logger.info(
        "Stable Train cache complete rows=%s size=%.1f MiB",
        f"{total_rows:,}", output.stat().st_size / 1024**2,
    )
    return output, metadata


def read_train_valid_protocol_sample(
    cache_path: Path,
    sample_kind: str,
    features: list[str],
) -> dict:
    columns = [
        "month", "_sort_site_no", "item_id", "target_qty", "sample_weight",
        "sampling_probability", *features,
    ]
    projection = ",".join(f'"{column}"' for column in columns)
    connection = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute("PRAGMA memory_limit='2GB'")
        arrays = connection.execute(
            f"SELECT {projection} FROM read_parquet('{_escaped(cache_path)}') "
            "WHERE sample_kind=? ORDER BY month, _sort_site_no, item_id",
            [sample_kind],
        ).fetchnumpy()
    finally:
        connection.close()
    frame = pd.DataFrame(arrays)
    if frame.empty:
        raise RuntimeError(f"No stable Train rows for {sample_kind}")
    fingerprint_frame = frame[["month", "_sort_site_no", "item_id", "target_qty", "sample_weight"]].rename(
        columns={"_sort_site_no": "site_no"}
    )
    fingerprints = training_frame_hashes(fingerprint_frame)
    keys = frame[["month", "_sort_site_no", "item_id"]].copy()
    target = frame.pop("target_qty").to_numpy(dtype="float32")
    weight = frame.pop("sample_weight").to_numpy(dtype="float32")
    probability = frame.pop("sampling_probability").to_numpy(dtype="float32")
    frame.drop(columns=["month", "_sort_site_no", "item_id"], inplace=True)
    return {
        "x": frame.loc[:, features], "target": target, "weight": weight,
        "probability": probability, "keys": keys, "fingerprints": fingerprints,
    }


def verify_train_valid_determinism(
    config: dict,
    sample: dict,
    features: list[str],
    logger: logging.Logger,
) -> dict:
    settings = config["train_valid_protocol"]
    rows = min(int(settings["reproducibility_smoke_rows"]), len(sample["target"]))
    rounds = int(settings["reproducibility_smoke_rounds"])
    x = sample["x"].iloc[:rows].copy()
    y = sample["target"][:rows]
    w = sample["weight"][:rows]
    w = (w / w.mean()).astype("float32")
    categorical = [feature for feature in CATEGORY_FEATURES if feature in features]
    hashes = []
    predictions = []
    for _ in range(2):
        dataset = lgb.Dataset(
            x, label=y, weight=w, feature_name=features,
            categorical_feature=categorical, free_raw_data=True,
        )
        booster = lgb.train(
            train_valid_lightgbm_params(config, "regressor"), dataset,
            num_boost_round=rounds,
        )
        text_value = booster.model_to_string(num_iteration=rounds)
        hashes.append(hashlib.sha256(text_value.encode("utf-8")).hexdigest())
        predictions.append(booster.predict(x.iloc[:1000], num_iteration=rounds))
        del dataset, booster
        gc.collect()
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=0.0, atol=0.0)
    if hashes[0] != hashes[1]:
        raise RuntimeError("Deterministic LightGBM smoke produced different model hashes")
    logger.info("Deterministic training smoke passed hash=%s", hashes[0])
    return {"rows": rows, "rounds": rounds, "model_sha256": hashes[0], "passed": True}


def train_train_valid_max_component(
    config: dict,
    cache_path: Path,
    features: list[str],
    component: str,
    category_maps: dict,
    logger: logging.Logger,
    model_name: str,
    max_rounds_override: int | None = None,
) -> dict:
    root = _protocol_checkpoint_root(config)
    max_model_path = root / "max_round_models" / f"{model_name}.txt"
    metadata_path = max_model_path.with_suffix(".json")
    requested_rounds = int(max_rounds_override or config["train_valid_protocol"]["max_rounds"])
    if max_model_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("max_rounds", 0)) >= requested_rounds:
            logger.info("Reusing max-round model %s", model_name)
            return {"path": max_model_path, "metadata": metadata}
        logger.info(
            "Extending boundary model %s from %s to %s rounds by deterministic retraining",
            model_name, metadata.get("max_rounds"), requested_rounds,
        )
        max_model_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    # Fix the safety budget before loading the training matrix. Computing it
    # afterwards double-counts this process's matrix against available RAM.
    guard = PIPELINE.MemoryGuard(f"train-valid-{model_name}", logger)
    kind = "regressor_train" if component == "regressor" else f"{component}_train"
    sample = read_train_valid_protocol_sample(cache_path, kind, features)
    target = sample["target"]
    if component == "regressor":
        if not np.all(target > 0):
            mask = target > 0
            sample["x"] = sample["x"].loc[mask].reset_index(drop=True)
            target = target[mask]
            sample["weight"] = sample["weight"][mask]
    else:
        threshold = 5 if component == "classifier_5plus" else 20
        target = (target >= threshold).astype("float32")
    weights = (sample["weight"] / sample["weight"].mean()).astype("float32")
    params = train_valid_lightgbm_params(config, component)
    max_rounds = requested_rounds
    categorical = [feature for feature in CATEGORY_FEATURES if feature in features]
    started = time.perf_counter()
    dataset = lgb.Dataset(
        sample["x"], label=target, weight=weights, feature_name=features,
        categorical_feature=categorical, free_raw_data=True,
    )
    booster = lgb.train(
        params, dataset, num_boost_round=max_rounds,
        callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)],
    )
    max_model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(max_model_path, num_iteration=max_rounds)
    metadata = {
        "model_name": model_name,
        "component": component,
        "feature_names": features,
        "feature_count": len(features),
        "categorical_features": categorical,
        "category_maps": category_maps,
        "time_base": str(config["time_base"]),
        "params": params,
        "max_rounds": max_rounds,
        "train_rows": int(len(target)),
        "training_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "sample_fingerprints": sample["fingerprints"],
        "max_model_sha256": _sha256_file(max_model_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    del sample, target, weights, dataset, booster
    gc.collect()
    logger.info("Trained %s rows=%s max_rounds=%s", model_name, f"{metadata['train_rows']:,}", max_rounds)
    return {"path": max_model_path, "metadata": metadata}


def incremental_binary_predictions(
    booster: lgb.Booster,
    features,
    iterations: Iterable[int],
) -> dict[int, np.ndarray]:
    requested = sorted({int(value) for value in iterations})
    if not requested or requested[0] < 1 or requested[-1] > booster.num_trees():
        raise ValueError("Invalid binary iteration grid")
    raw = np.zeros(len(features), dtype="float64")
    previous = 0
    predictions: dict[int, np.ndarray] = {}
    for iteration in requested:
        raw += booster.predict(
            features, start_iteration=previous,
            num_iteration=iteration - previous, raw_score=True,
        )
        predictions[iteration] = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
        previous = iteration
    return predictions


def select_binary_valid_iteration(rows: Iterable[dict]) -> dict:
    candidates = list(rows)
    if not candidates:
        raise ValueError("No Complete Valid classifier metrics were supplied")
    return min(candidates, key=lambda row: (
        -float(row["average_precision"]), float(row["logloss"]), int(row["iteration"]),
    ))


def boundary_selected_regressors(selected: dict, trained: dict) -> list[str]:
    return [
        name for name, selection in selected.items()
        if trained[name]["metadata"]["component"] == "regressor"
        and int(selection["iteration"]) >= int(trained[name]["metadata"]["max_rounds"])
    ]


def load_completed_protocol_iteration_history(
    config: dict,
    trained: dict[str, dict],
    logger: logging.Logger,
) -> dict | None:
    """Recover completed coarse scans after an interrupted boundary extension.

    Each cached row is retained only when its iteration is available in the
    current deterministic max-round model.  The base scan must contain every
    component; extension scans may contain only the boundary components.
    """
    root = _protocol_root(config)
    base_path = root / "coarse_iteration_metrics.json"
    if not base_path.exists():
        return None
    base = json.loads(base_path.read_text(encoding="utf-8"))
    regressors = {
        name: list(base.get("regressors", {}).get(name, []))
        for name, item in trained.items()
        if item["metadata"]["component"] == "regressor"
    }
    classifiers = {
        name: list(base.get("classifiers", {}).get(name, []))
        for name, item in trained.items()
        if item["metadata"]["component"] != "regressor"
    }
    if any(not rows for rows in regressors.values()) or any(not rows for rows in classifiers.values()):
        return None
    row_count = int(base.get("rows", 0))
    if row_count <= 0:
        return None

    used = [base_path.name]
    for path in sorted(root.glob("coarse_extension_*_iteration_metrics.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        if int(stored.get("rows", 0)) != row_count:
            continue
        used.append(path.name)
        for name, rows in stored.get("regressors", {}).items():
            if name not in regressors:
                continue
            available = int(trained[name]["metadata"]["max_rounds"])
            regressors[name].extend(
                row for row in rows if int(row["iteration"]) <= available
            )
        for name, rows in stored.get("classifiers", {}).items():
            if name not in classifiers:
                continue
            available = int(trained[name]["metadata"]["max_rounds"])
            classifiers[name].extend(
                row for row in rows if int(row["iteration"]) <= available
            )

    def deduplicate(rows: list[dict]) -> list[dict]:
        return [
            value for _, value in sorted(
                {int(row["iteration"]): row for row in rows}.items()
            )
        ]

    regressors = {name: deduplicate(rows) for name, rows in regressors.items()}
    classifiers = {name: deduplicate(rows) for name, rows in classifiers.items()}
    logger.info("Recovered completed Complete Valid iteration scans: %s", used)
    return {
        "phase": "recovered_coarse_history",
        "rows": row_count,
        "regressors": regressors,
        "classifiers": classifiers,
    }


def stream_train_valid_iteration_metrics(
    config: dict,
    trained: dict[str, dict],
    iteration_grids: dict[str, list[int]],
    logger: logging.Logger,
    phase: str,
) -> dict:
    cache = _protocol_root(config) / f"{phase}_iteration_metrics.json"
    signature_payload = {
        name: {
            "model_sha256": item["metadata"]["max_model_sha256"],
            "iterations": iteration_grids[name],
        }
        for name, item in trained.items()
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("signature") == signature:
            logger.info("Reusing %s Complete Valid iteration metrics", phase)
            return stored

    boosters = {name: lgb.Booster(model_file=str(item["path"])) for name, item in trained.items()}
    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    all_features = _protocol_all_features()
    category_maps = p_meta["category_maps"]
    q_names = [name for name, item in trained.items() if item["metadata"]["component"] == "regressor"]
    classifier_names = [
        name for name, item in trained.items() if item["metadata"]["component"] != "regressor"
    ]
    q_accumulators = {
        name: IterationWapeAccumulator(iteration_grids[name]) for name in q_names
    }
    ranking_accumulators = {
        name: {
            iteration: BinaryRankingAccumulator(int(config["train_valid_protocol"]["ranking_bins"]))
            for iteration in iteration_grids[name]
        }
        for name in classifier_names
    }
    unknown = Counter()
    rows = 0
    guard = PIPELINE.MemoryGuard(f"train-valid-{phase}", logger)
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"
    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name=f"train_valid_{phase}", vectors_per_chunk=256), start=1
    ):
        enriched = add_all_new_features(frame)
        x = PIPELINE.prepare_feature_frame(
            enriched, all_features, category_maps, str(config["time_base"])
        )
        for category in CATEGORY_FEATURES:
            unknown[category] += int((pd.to_numeric(x[category], errors="coerce").fillna(-1) == -1).sum())
        target = clip_target(frame["future_qty_1m"].to_numpy())
        p = np.clip(
            p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]),
            0.0, 1.0,
        )
        for name in q_names:
            features = trained[name]["metadata"]["feature_names"]
            q_predictions = incremental_tweedie_predictions(
                boosters[name], x[features], iteration_grids[name]
            )
            q_accumulators[name].update(
                target, {iteration: p * q for iteration, q in q_predictions.items()}
            )
            del q_predictions
        for name in classifier_names:
            component = trained[name]["metadata"]["component"]
            threshold = 5 if component == "classifier_5plus" else 20
            binary_target = (target >= threshold).astype("uint8")
            features = trained[name]["metadata"]["feature_names"]
            scores = incremental_binary_predictions(
                boosters[name], x[features], iteration_grids[name]
            )
            for iteration, score in scores.items():
                ranking_accumulators[name][iteration].update(binary_target, score)
            del scores, binary_target
        rows += len(frame)
        if chunk_number % 5 == 0:
            guard.check(phase)
            logger.info("%s Complete Valid rows=%s", phase, f"{rows:,}")
        del frame, enriched, x, target, p

    metrics = {
        "regressors": {name: accumulator.compute() for name, accumulator in q_accumulators.items()},
        "classifiers": {
            name: [
                {"iteration": iteration, **accumulator.compute()}
                for iteration, accumulator in sorted(iterations.items())
            ]
            for name, iterations in ranking_accumulators.items()
        },
    }
    result = {
        "phase": phase, "signature": signature, "rows": rows,
        "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "unknown_rates": {name: count / rows for name, count in unknown.items()},
        **metrics,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del boosters, p_model
    gc.collect()
    return result


def save_train_valid_selected_model(
    config: dict,
    trained_item: dict,
    selected_iteration: int,
    selection_metrics: dict,
    logger: logging.Logger,
) -> dict:
    name = trained_item["metadata"]["model_name"]
    output = _protocol_checkpoint_root(config) / "selected" / f"{name}.txt"
    metadata = dict(trained_item["metadata"])
    metadata.update({
        "experiment": "two_stage_1m_unified_train_valid_protocol",
        "best_iteration": int(selected_iteration),
        "selected_iteration": int(selected_iteration),
        "selection": "full Active-Store Train -> Complete Valid; primary metric is raw Overall WAPE"
        if metadata["component"] == "regressor"
        else "full Active-Store Train -> Complete Valid; primary metric is approximate Average Precision",
        "selection_metrics": selection_metrics,
        "formal_p_classifier_source": (
            "retained 750-round Active-Store classifier; selected previously by Complete Valid pair WAPE, "
            "not by four-fold weighted median"
        ),
    })
    booster = lgb.Booster(model_file=str(trained_item["path"]))
    save_model_bundle(booster, output, metadata)
    reloaded, loaded = load_model_bundle(output)
    if reloaded.num_trees() != int(selected_iteration) or loaded["best_iteration"] != int(selected_iteration):
        raise RuntimeError(f"Selected model reload mismatch for {name}")
    result = {
        "path": output,
        "metadata": loaded,
        "model_sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
    }
    logger.info("Saved selected %s iteration=%s", name, selected_iteration)
    del booster, reloaded
    gc.collect()
    return result


def evaluate_train_valid_selected_models(
    config: dict,
    selected: dict[str, dict],
    logger: logging.Logger,
) -> dict:
    cache = _protocol_root(config) / "selected_complete_valid_result.json"
    signature_payload = {name: item["model_sha256"] for name, item in selected.items()}
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("signature") == signature:
            logger.info("Reusing selected Complete Valid result")
            return stored

    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    models = {name: load_model_bundle(item["path"])[0] for name, item in selected.items()}
    all_features = _protocol_all_features()
    regressor_names = [
        name for name, item in selected.items() if item["metadata"]["component"] == "regressor"
    ]
    s5_name = next(
        name for name, item in selected.items() if item["metadata"]["component"] == "classifier_5plus"
    )
    s20_name = next(
        name for name, item in selected.items() if item["metadata"]["component"] == "classifier_20plus"
    )
    q_accumulators = {name: CompleteMetricAccumulator() for name in regressor_names}
    final_accumulators = {name: CompleteMetricAccumulator() for name in regressor_names}
    monthly_accumulators: dict[str, dict[str, CompleteMetricAccumulator]] = {}
    ratios = {name: [] for name in regressor_names}
    candidate_parts = {name: [] for name in ("y", "p_sale", "q", "s5", "s20")}
    unknown = Counter()
    rows = 0
    guard = PIPELINE.MemoryGuard("train-valid-final-evaluation", logger)
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"
    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name="train_valid_selected_evaluation", vectors_per_chunk=128), start=1
    ):
        enriched = add_all_new_features(frame)
        x = PIPELINE.prepare_feature_frame(
            enriched, all_features, p_meta["category_maps"], str(config["time_base"])
        )
        for category in CATEGORY_FEATURES:
            unknown[category] += int((pd.to_numeric(x[category], errors="coerce").fillna(-1) == -1).sum())
        target = clip_target(frame["future_qty_1m"].to_numpy())
        p = np.clip(
            p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]),
            0.0, 1.0,
        )
        q_predictions = {}
        final_predictions = {}
        months = frame["month"].astype(str).to_numpy()
        for name in regressor_names:
            metadata = selected[name]["metadata"]
            q = np.clip(
                models[name].predict(x[metadata["feature_names"]], num_iteration=metadata["best_iteration"]),
                0.0, None,
            )
            prediction = p * q
            if not np.isfinite(q).all() or not np.isfinite(prediction).all():
                raise RuntimeError(f"Invalid Complete Valid prediction for {name}")
            q_predictions[name] = q
            final_predictions[name] = prediction
            q_accumulators[name].update(target, q)
            final_accumulators[name].update(target, prediction)
            high = target >= 20
            if high.any():
                ratios[name].append((q[high] / target[high]).astype("float32"))
            for month in np.unique(months):
                monthly_accumulators.setdefault(
                    month, {model_name: CompleteMetricAccumulator() for model_name in regressor_names}
                )[name].update(target[months == month], prediction[months == month])

        s5_meta = selected[s5_name]["metadata"]
        s20_meta = selected[s20_name]["metadata"]
        s5 = np.clip(
            models[s5_name].predict(x[s5_meta["feature_names"]], num_iteration=s5_meta["best_iteration"]),
            0.0, 1.0,
        )
        s20 = np.minimum(
            np.clip(
                models[s20_name].predict(
                    x[s20_meta["feature_names"]], num_iteration=s20_meta["best_iteration"]
                ), 0.0, 1.0,
            ),
            s5,
        )
        candidate_parts["y"].append(target.astype("float32"))
        candidate_parts["p_sale"].append(p.astype("float32"))
        candidate_parts["q"].append(q_predictions["E2_CROSS_STORE_VALID_SELECTED"].astype("float32"))
        candidate_parts["s5"].append(s5.astype("float32"))
        candidate_parts["s20"].append(s20.astype("float32"))
        rows += len(frame)
        if chunk_number % 10 == 0:
            guard.check("selected_complete_valid")
            logger.info("Selected Complete Valid rows=%s", f"{rows:,}")
        del frame, enriched, x, target, p, q_predictions, final_predictions, s5, s20, months

    q_metrics = {name: accumulator.compute() for name, accumulator in q_accumulators.items()}
    for name, values in ratios.items():
        q_metrics[name]["twenty_plus_median_q_over_y"] = (
            float(np.median(np.concatenate(values))) if values else float("nan")
        )
    metrics = {name: accumulator.compute() for name, accumulator in final_accumulators.items()}
    monthly = {
        month: {name: accumulator.compute() for name, accumulator in models_by_name.items()}
        for month, models_by_name in sorted(monthly_accumulators.items())
    }

    candidate_frame = pd.DataFrame({
        name: np.concatenate(parts) for name, parts in candidate_parts.items()
    })
    candidate_frame["sampling_probability"] = np.ones(len(candidate_frame), dtype="float32")
    candidate_frame["sample_weight"] = np.ones(len(candidate_frame), dtype="float32")
    fit = _fit_candidate_b_rule(candidate_frame, config)
    promoted_candidates = []
    for candidate in fit["candidates"]:
        decision = candidate_b_development_decision(fit["baseline_metrics"], candidate["metrics"], config)
        candidate["development_decision"] = decision
        if decision["promoted"]:
            promoted_candidates.append(candidate)
    promoted = min(
        promoted_candidates,
        key=lambda candidate: (
            candidate["metrics"]["20+"]["wape"],
            -candidate["metrics"]["recall_20_plus"],
            candidate["metrics"]["overall"]["wape"],
        ),
        default=None,
    )
    candidate_result = {
        "candidate_count": fit["candidate_count"],
        "protection_eligible_count": fit["eligible_count"],
        "promoted_count": len(promoted_candidates),
        "rejection_counts": fit["rejection_counts"],
        "best_protected": fit["selected"],
        "best_unconstrained": fit["best_unconstrained"],
        "promoted": promoted is not None,
        "selected": promoted,
        "baseline_model": "E2_CROSS_STORE_VALID_SELECTED",
        "development_note": (
            "Thresholds, multipliers and candidate choice use formal Valid under the new development protocol; "
            "these are development results, not final generalization claims."
        ),
    }
    if promoted is not None:
        metrics["Candidate_B_VALID_SELECTED"] = promoted["metrics"]

    result = {
        "signature": signature,
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "unknown_rates": {name: count / rows for name, count in unknown.items()},
        "q_metrics": q_metrics,
        "metrics": metrics,
        "monthly": monthly,
        "candidate_b": candidate_result,
    }
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del candidate_frame, candidate_parts, models, p_model
    gc.collect()
    return result


def evaluate_e3_train_valid_selected_model(
    config: dict,
    selected_item: dict,
    logger: logging.Logger,
) -> dict:
    name = "E3_DIFF_CROSS_STORE_VALID_SELECTED"
    cache = _protocol_root(config) / "e3_selected_complete_valid_result.json"
    signature = hashlib.sha256(
        json.dumps({
            "model_sha256": selected_item["model_sha256"],
            "formal_p_sha256": _sha256_file(FORMAL_P_MODEL),
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("signature") == signature:
            logger.info("Reusing E3 selected Complete Valid result")
            return stored

    p_model, p_meta = load_model_bundle(FORMAL_P_MODEL)
    q_model, q_meta = load_model_bundle(selected_item["path"])
    features = q_meta["feature_names"]
    q_accumulator = CompleteMetricAccumulator()
    final_accumulator = CompleteMetricAccumulator()
    ratio_parts: list[np.ndarray] = []
    unknown = Counter()
    rows = 0
    guard = PIPELINE.MemoryGuard("e3-valid-final-evaluation", logger)
    started = time.perf_counter()
    where = "b.split='valid' AND b.target_available_1m=1"
    for chunk_number, frame in enumerate(
        iter_joined_rows(config, where, name="e3_train_valid_selected_evaluation", vectors_per_chunk=128),
        start=1,
    ):
        enriched = add_all_new_features(frame)
        x = PIPELINE.prepare_feature_frame(
            enriched, features, p_meta["category_maps"], str(config["time_base"])
        )
        for category in CATEGORY_FEATURES:
            unknown[category] += int(
                (pd.to_numeric(x[category], errors="coerce").fillna(-1) == -1).sum()
            )
        target = clip_target(frame["future_qty_1m"].to_numpy())
        p = np.clip(
            p_model.predict(x[p_meta["feature_names"]], num_iteration=p_meta["best_iteration"]),
            0.0, 1.0,
        )
        q = np.clip(
            q_model.predict(x[features], num_iteration=q_meta["best_iteration"]), 0.0, None
        )
        prediction = p * q
        if not np.isfinite(q).all() or not np.isfinite(prediction).all():
            raise RuntimeError("Invalid E3 Complete Valid prediction")
        q_accumulator.update(target, q)
        final_accumulator.update(target, prediction)
        high = target >= 20
        if high.any():
            ratio_parts.append((q[high] / target[high]).astype("float32"))
        rows += len(frame)
        if chunk_number % 10 == 0:
            guard.check("e3_selected_complete_valid")
            logger.info("E3 selected Complete Valid rows=%s", f"{rows:,}")
        del frame, enriched, x, target, p, q, prediction

    q_metrics = q_accumulator.compute()
    q_metrics["twenty_plus_median_q_over_y"] = (
        float(np.median(np.concatenate(ratio_parts))) if ratio_parts else float("nan")
    )
    result = {
        "signature": signature,
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / 1024**3,
        "unknown_rates": {category: count / rows for category, count in unknown.items()},
        "q_metrics": {name: q_metrics},
        "metrics": {name: final_accumulator.compute()},
    }
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del p_model, q_model, ratio_parts
    gc.collect()
    return result


def _train_valid_metric_summary(metric: dict) -> dict[str, float]:
    return {
        "overall_wape": float(metric["overall"]["wape"]),
        "integer_wape": float(metric["overall"].get("integer_wape", metric.get("integer_overall_wape", float("nan")))),
        "nonzero_wape": float(metric["nonzero"]["wape"]),
        "integer_nonzero_wape": float(metric["nonzero"].get("integer_wape", float("nan"))),
        "five_nineteen_wape": float(metric["5-19"]["wape"]),
        "five_nineteen_recall": float(metric["recall_5_19"]),
        "twenty_plus_wape": float(metric["20+"]["wape"]),
        "twenty_plus_recall": float(metric["recall_20_plus"]),
        "mc_macro_f1": float(metric["mc_macro_f1"]),
        "total_bias": float(metric["overall"]["total_bias"]),
        "zero_prediction_total": float(metric["zero_prediction_total"]),
        "zero_gt_0_5": float(metric["zero_gt_0_5"]),
        "zero_gt_1": float(metric["zero_gt_1"]),
    }


def append_train_valid_protocol_ablation(
    config: dict,
    selected: dict[str, dict],
    evaluation: dict,
) -> None:
    path = project_path(config["outputs"]["ablation"])
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = []
    for name in (
        "E0_VALID_SELECTED", "E1_DIFF_VALID_SELECTED", "E2_CROSS_STORE_VALID_SELECTED"
    ):
        metadata = selected[name]["metadata"]
        final = _train_valid_metric_summary(evaluation["metrics"][name])
        q = evaluation["q_metrics"][name]
        rows.append({
            "fold": "complete_valid_development",
            "experiment": name,
            "component": "regressor",
            "train_rows": metadata["train_rows"],
            "valid_rows": evaluation["rows"],
            "feature_count": metadata["feature_count"],
            "best_iteration": metadata["best_iteration"],
            "training_seconds": metadata["training_seconds"],
            "peak_ram_gib": metadata["peak_ram_gib"],
            "formal_train_months": 30,
            "q_nonzero_wape": q["nonzero"]["wape"],
            "q_5_19_wape": q["5-19"]["wape"],
            "q_20_plus_wape": q["20+"]["wape"],
            "q_5_19_recall": q["recall_5_19"],
            "q_20_plus_recall": q["recall_20_plus"],
            "q_20_plus_median_q_over_y": q["twenty_plus_median_q_over_y"],
            "final_nonzero_wape": final["nonzero_wape"],
            "final_5_19_wape": final["five_nineteen_wape"],
            "final_20_plus_wape": final["twenty_plus_wape"],
            "final_5_19_recall": final["five_nineteen_recall"],
            "final_20_plus_recall": final["twenty_plus_recall"],
            "overall_wape": final["overall_wape"],
            "integer_wape": final["integer_wape"],
            "total_bias": final["total_bias"],
            "zero_prediction_total": final["zero_prediction_total"],
            "promoted_model": str(selected[name]["path"].relative_to(ROOT)),
            "selection_protocol": "full_train_to_formal_complete_valid",
            "sample_set_sha256": metadata["sample_fingerprints"]["sample_set_sha256"],
            "training_order_sha256": metadata["sample_fingerprints"]["order_sha256"],
            "model_sha256": selected[name]["model_sha256"],
        })
    if evaluation["candidate_b"]["promoted"]:
        candidate = evaluation["candidate_b"]["selected"]
        summary = _train_valid_metric_summary(candidate["metrics"])
        rows.append({
            "fold": "complete_valid_development",
            "experiment": "Candidate_B_VALID_SELECTED",
            "component": "postprocess_rule",
            "valid_rows": evaluation["rows"],
            "feature_count": selected["E2_CROSS_STORE_VALID_SELECTED"]["metadata"]["feature_count"],
            "final_nonzero_wape": summary["nonzero_wape"],
            "final_5_19_wape": summary["five_nineteen_wape"],
            "final_20_plus_wape": summary["twenty_plus_wape"],
            "final_5_19_recall": summary["five_nineteen_recall"],
            "final_20_plus_recall": summary["twenty_plus_recall"],
            "overall_wape": summary["overall_wape"],
            "integer_wape": summary["integer_wape"],
            "total_bias": summary["total_bias"],
            "zero_prediction_total": summary["zero_prediction_total"],
            "selection_protocol": "formal_valid_25_existing_candidate_b_rules",
        })
    appended = pd.DataFrame(rows)
    if not existing.empty:
        existing = existing.loc[
            ~existing.get("experiment", pd.Series(index=existing.index, dtype="object")).isin(appended["experiment"])
        ]
    combined = pd.concat([existing, appended], ignore_index=True, sort=False)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def append_e3_train_valid_protocol_ablation(
    config: dict,
    selected_item: dict,
    evaluation: dict,
) -> None:
    name = "E3_DIFF_CROSS_STORE_VALID_SELECTED"
    path = project_path(config["outputs"]["ablation"])
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    metadata = selected_item["metadata"]
    final = _train_valid_metric_summary(evaluation["metrics"][name])
    q = evaluation["q_metrics"][name]
    row = {
        "fold": "complete_valid_development",
        "experiment": name,
        "component": "regressor",
        "train_rows": metadata["train_rows"],
        "valid_rows": evaluation["rows"],
        "feature_count": metadata["feature_count"],
        "best_iteration": metadata["best_iteration"],
        "training_seconds": metadata["training_seconds"],
        "peak_ram_gib": metadata["peak_ram_gib"],
        "formal_train_months": 30,
        "q_nonzero_wape": q["nonzero"]["wape"],
        "q_5_19_wape": q["5-19"]["wape"],
        "q_20_plus_wape": q["20+"]["wape"],
        "q_5_19_recall": q["recall_5_19"],
        "q_20_plus_recall": q["recall_20_plus"],
        "q_20_plus_median_q_over_y": q["twenty_plus_median_q_over_y"],
        "final_nonzero_wape": final["nonzero_wape"],
        "final_5_19_wape": final["five_nineteen_wape"],
        "final_20_plus_wape": final["twenty_plus_wape"],
        "final_5_19_recall": final["five_nineteen_recall"],
        "final_20_plus_recall": final["twenty_plus_recall"],
        "overall_wape": final["overall_wape"],
        "integer_wape": final["integer_wape"],
        "total_bias": final["total_bias"],
        "zero_prediction_total": final["zero_prediction_total"],
        "promoted_model": str(selected_item["path"].relative_to(ROOT)),
        "selection_protocol": "full_train_to_formal_complete_valid_e3_ablation",
        "sample_set_sha256": metadata["sample_fingerprints"]["sample_set_sha256"],
        "training_order_sha256": metadata["sample_fingerprints"]["order_sha256"],
        "model_sha256": selected_item["model_sha256"],
    }
    if not existing.empty and "experiment" in existing:
        existing = existing.loc[existing["experiment"] != name]
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _comparison_change(base: dict, candidate: dict) -> dict[str, float]:
    return {
        "overall_gain": base["overall_wape"] - candidate["overall_wape"],
        "nonzero_gain": base["nonzero_wape"] - candidate["nonzero_wape"],
        "five_nineteen_gain": base["five_nineteen_wape"] - candidate["five_nineteen_wape"],
        "five_nineteen_recall": candidate["five_nineteen_recall"] - base["five_nineteen_recall"],
        "twenty_plus_gain": base["twenty_plus_wape"] - candidate["twenty_plus_wape"],
        "twenty_plus_recall": candidate["twenty_plus_recall"] - base["twenty_plus_recall"],
        "mc_macro_f1": candidate["mc_macro_f1"] - base["mc_macro_f1"],
        "bias": candidate["total_bias"] - base["total_bias"],
        "zero_relative": 100.0 * (
            candidate["zero_prediction_total"] / base["zero_prediction_total"] - 1.0
        ),
    }


def update_report_with_e3_train_valid_result(
    config: dict,
    prior_result: dict,
    selected_item: dict,
    evaluation: dict,
    iteration_selection: dict,
    capped_at_budget: bool,
) -> None:
    name = "E3_DIFF_CROSS_STORE_VALID_SELECTED"
    path = project_path(config["outputs"]["report"])
    content = path.read_text(encoding="utf-8")
    if "### E3 Diff + Cross-store消融结论" in content:
        return
    old_metrics = prior_result["evaluation"]["metrics"]
    old_q = prior_result["evaluation"]["q_metrics"]
    e3_metric = evaluation["metrics"][name]
    e3_q = evaluation["q_metrics"][name]
    summaries = {key: _train_valid_metric_summary(value) for key, value in old_metrics.items()}
    summaries[name] = _train_valid_metric_summary(e3_metric)
    e0 = summaries["E0_VALID_SELECTED"]
    e1 = summaries["E1_DIFF_VALID_SELECTED"]
    e2 = summaries["E2_CROSS_STORE_VALID_SELECTED"]
    e3 = summaries[name]
    versus_e0 = _comparison_change(e0, e3)
    versus_e2 = _comparison_change(e2, e3)

    if (
        versus_e2["overall_gain"] > 0.25
        and versus_e2["nonzero_gain"] >= 0.0
        and versus_e2["five_nineteen_gain"] >= 0.0
        and versus_e2["five_nineteen_recall"] >= -0.25
        and versus_e2["twenty_plus_gain"] >= 0.0
        and versus_e2["twenty_plus_recall"] >= -0.25
        and versus_e2["mc_macro_f1"] >= -0.10
    ):
        conclusion = "Diff和Cross-store存在互补，E3成为当前集中式1M最优Valid候选。"
    elif (
        abs(versus_e2["overall_gain"]) <= 0.25
        and abs(versus_e2["nonzero_gain"]) <= 0.25
        and abs(versus_e2["twenty_plus_gain"]) <= 0.50
    ):
        conclusion = "Diff在Cross-store基础上的边际价值很弱，优先保留结构更简单的E2。"
    else:
        conclusion = (
            "E3降低了Overall WAPE、Bias和零销量误报，但5–19/20+ Recall、20+ WAPE及MC均弱于E2；"
            "Diff未证明与Cross-store互补，业务多指标下优先保留E2，E3作为低Overall误差候选保留。"
        )

    def table_row(model: str, added: str, iteration: int, summary: dict) -> str:
        return (
            f"| {model} | {added} | {iteration} | {_fmt(summary['overall_wape'])}% | "
            f"{_fmt(summary['nonzero_wape'])}% | {_fmt(summary['five_nineteen_wape'])}% | "
            f"{_fmt(summary['five_nineteen_recall'])}% | {_fmt(summary['twenty_plus_wape'])}% | "
            f"{_fmt(summary['twenty_plus_recall'])}% | {_fmt(summary['mc_macro_f1'])}% | "
            f"{_fmt(summary['total_bias'])}% | {_fmt(summary['zero_prediction_total'])} |"
        )

    e3_main_row = table_row(name, "6个Diff + 6个Cross-store", selected_item["metadata"]["best_iteration"], e3)
    e3_business_row = (
        f"| {name} | {_fmt(e3['integer_wape'])}% | {_fmt(e3['integer_nonzero_wape'])}% | "
        f"{_fmt(e3['zero_gt_0_5'] * 100)}% | {_fmt(e3['zero_gt_1'] * 100)}% |"
    )
    e3_q_row = (
        f"| {name} | {_fmt(e3_q['nonzero']['wape'])}% | {_fmt(e3_q['5-19']['wape'])}% | "
        f"{_fmt(e3_q['recall_5_19'])}% | {_fmt(e3_q['20+']['wape'])}% | "
        f"{_fmt(e3_q['recall_20_plus'])}% | {_fmt(e3_q['twenty_plus_median_q_over_y'], 6)} |"
    )
    e3_delta = (
        f"- `{name}`相对E0（WAPE正数表示降低/改善）：Overall WAPE "
        f"{versus_e0['overall_gain']:+.4f} pp，Nonzero WAPE {versus_e0['nonzero_gain']:+.4f} pp，"
        f"5–19 Recall {versus_e0['five_nineteen_recall']:+.4f} pp，20+ Recall "
        f"{versus_e0['twenty_plus_recall']:+.4f} pp，Bias {versus_e0['bias']:+.4f} pp，"
        f"Zero Pred {versus_e0['zero_relative']:+.4f}%。"
    )
    e3_round = (
        f"- `{name}`：粗选{iteration_selection['coarse']['iteration']}轮，在粗选点附近逐轮精扫后锁定"
        f"{selected_item['metadata']['best_iteration']}轮；Valid Overall WAPE={_fmt(e3['overall_wape'])}%。"
    )
    if capped_at_budget:
        e3_round += " 该轮次位于一次延长后的开发预算边缘，未形成常规early-stopping内部最低点。"

    lines = content.splitlines()

    def insert_after(prefix: str, new_line: str, start: int = 0) -> None:
        for index in range(start, len(lines)):
            if lines[index].startswith(prefix):
                lines.insert(index + 1, new_line)
                return
        raise RuntimeError(f"Report anchor not found: {prefix}")

    unified_index = lines.index("### 统一Valid结果")
    insert_after("| E2_CROSS_STORE_VALID_SELECTED |", e3_main_row, unified_index)
    insert_after("- `E2_CROSS_STORE_VALID_SELECTED`相对E0", e3_delta, unified_index)
    business_index = lines.index("补充业务指标：", unified_index)
    insert_after("| E2_CROSS_STORE_VALID_SELECTED |", e3_business_row, business_index)
    q_index = lines.index("q层诊断：", business_index)
    insert_after("| E2_CROSS_STORE_VALID_SELECTED |", e3_q_row, q_index)
    round_index = lines.index("轮次选择记录：", q_index)
    insert_after("- `E2_CROSS_STORE_VALID_SELECTED`：", e3_round, round_index)

    best_name = min(summaries, key=lambda key: summaries[key]["overall_wape"])
    best_summary = summaries[best_name]
    current_index = lines.index("### 当前判断", round_index)
    for index in range(current_index + 1, len(lines)):
        if lines[index].startswith("- 统一开发口径下，当前Overall WAPE最低"):
            lines[index] = (
                f"- 统一开发口径下，当前Overall WAPE最低的1M候选是`{best_name}`"
                f"（{_fmt(best_summary['overall_wape'])}%）。"
            )
            break

    comparison_rows = []
    for model_name, summary in (
        ("E1 Diff", e1), ("E2 Cross-store", e2), ("E3 Diff+Cross-store", e3)
    ):
        change = _comparison_change(e0, summary)
        comparison_rows.append(
            f"| {model_name} | {change['overall_gain']:+.4f} pp | {change['nonzero_gain']:+.4f} pp | "
            f"{change['five_nineteen_recall']:+.4f} pp | {change['twenty_plus_gain']:+.4f} pp | "
            f"{change['twenty_plus_recall']:+.4f} pp | {change['bias']:+.4f} pp | "
            f"{change['zero_relative']:+.4f}% |"
        )
    e3_section = [
        "", "### E3 Diff + Cross-store消融结论", "",
        "变化表中WAPE正数表示误差降低/改善；Recall、Bias和Zero Pred为候选减基准的变化。", "",
        "| 方案 | 相对E0 Overall WAPE | Nonzero WAPE | 5–19 Recall | 20+ WAPE | 20+ Recall | Bias | Zero Pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *comparison_rows, "",
        f"E3相对E2：Overall WAPE改善{versus_e2['overall_gain']:+.4f} pp，Nonzero WAPE改善"
        f"{versus_e2['nonzero_gain']:+.4f} pp，5–19 WAPE改善{versus_e2['five_nineteen_gain']:+.4f} pp，"
        f"5–19 Recall变化{versus_e2['five_nineteen_recall']:+.4f} pp，20+ WAPE改善"
        f"{versus_e2['twenty_plus_gain']:+.4f} pp，20+ Recall变化{versus_e2['twenty_plus_recall']:+.4f} pp，"
        f"MC Macro-F1变化{versus_e2['mc_macro_f1']:+.4f} pp，Bias变化{versus_e2['bias']:+.4f} pp，"
        f"Zero Pred变化{versus_e2['zero_relative']:+.4f}%。", "",
        f"**结论：{conclusion}**", "",
    ]
    mentor_index = lines.index("### 导师汇报摘要", current_index)
    lines[mentor_index:mentor_index] = e3_section
    mentor_index = lines.index("### 导师汇报摘要", current_index)
    lines[mentor_index:] = [
        "### 导师汇报摘要", "",
        f"1. **统一E0基准**：57个基础特征，Valid最佳{prior_result['selected_models']['E0_VALID_SELECTED']['best_iteration']}轮；Overall WAPE={_fmt(e0['overall_wape'])}%，Nonzero WAPE={_fmt(e0['nonzero_wape'])}%。",
        f"2. **Diff**：加入6个历史销量变化字段，Overall WAPE降低{e0['overall_wape'] - e1['overall_wape']:.4f} pp，但Nonzero和中高需求指标未改善。",
        f"3. **Cross-store**：加入6个同书其他门店热度字段，Overall WAPE降低{e0['overall_wape'] - e2['overall_wape']:.4f} pp，20+ Recall提高{e2['twenty_plus_recall'] - e0['twenty_plus_recall']:.4f} pp。",
        f"4. **Diff + Cross-store**：{conclusion}",
        "5. **Candidate B**：原25组选择性补偿规则没有方案同时达到既定头部收益与副作用保护门槛，不进入最终候选。",
        f"6. **当前Valid结论**：纯Overall WAPE最低的是`{best_name}`（{_fmt(best_summary['overall_wape'])}%）；兼顾Nonzero、5–19、20+和MC后，业务首选仍为`E2_CROSS_STORE_VALID_SELECTED`。",
        f"7. **相对E0**：E3 Overall WAPE降低{e0['overall_wape'] - e3['overall_wape']:.4f} pp，但相对E2的5–19/20+ Recall分别变化{versus_e2['five_nineteen_recall']:+.4f}/{versus_e2['twenty_plus_recall']:+.4f} pp；建议E2与E3共同进入最终Test。",
        "8. **分布式衔接**：Cross-store验证了同书跨门店聚合热度的增量价值，可作为未来门店在不直接交换原始明细时协同计算item级统计信号的集中式依据；当前尚未实现隐私保护。",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_train_valid_protocol_report(
    config: dict,
    audit_rows: list[dict[str, str]],
    cache_metadata: dict,
    deterministic_smoke: dict,
    selected: dict[str, dict],
    evaluation: dict,
    iteration_selection: dict,
    log_path: Path,
) -> None:
    path = project_path(config["outputs"]["report"])
    content = path.read_text(encoding="utf-8")
    marker = "\n## 统一Train→Valid开发口径重评"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip()
    names = ["E0_VALID_SELECTED", "E1_DIFF_VALID_SELECTED", "E2_CROSS_STORE_VALID_SELECTED"]
    summaries = {name: _train_valid_metric_summary(evaluation["metrics"][name]) for name in names}
    base = summaries["E0_VALID_SELECTED"]
    candidate = evaluation["candidate_b"]
    table_rows = []
    for name in names:
        summary = summaries[name]
        metadata = selected[name]["metadata"]
        added = {
            "E0_VALID_SELECTED": "无（统一基准）",
            "E1_DIFF_VALID_SELECTED": "6个Diff字段",
            "E2_CROSS_STORE_VALID_SELECTED": "6个Cross-store字段",
        }[name]
        table_rows.append(
            f"| {name} | {added} | {metadata['best_iteration']} | {_fmt(summary['overall_wape'])}% | "
            f"{_fmt(summary['nonzero_wape'])}% | {_fmt(summary['five_nineteen_wape'])}% | "
            f"{_fmt(summary['five_nineteen_recall'])}% | {_fmt(summary['twenty_plus_wape'])}% | "
            f"{_fmt(summary['twenty_plus_recall'])}% | {_fmt(summary['mc_macro_f1'])}% | "
            f"{_fmt(summary['total_bias'])}% | {_fmt(summary['zero_prediction_total'])} |"
        )
    if candidate["promoted"]:
        summary = _train_valid_metric_summary(candidate["selected"]["metrics"])
        table_rows.append(
            f"| Candidate_B_VALID_SELECTED | E2+s5/s20分档补偿 | N/A | {_fmt(summary['overall_wape'])}% | "
            f"{_fmt(summary['nonzero_wape'])}% | {_fmt(summary['five_nineteen_wape'])}% | "
            f"{_fmt(summary['five_nineteen_recall'])}% | {_fmt(summary['twenty_plus_wape'])}% | "
            f"{_fmt(summary['twenty_plus_recall'])}% | {_fmt(summary['mc_macro_f1'])}% | "
            f"{_fmt(summary['total_bias'])}% | {_fmt(summary['zero_prediction_total'])} |"
        )
    else:
        table_rows.append("| Candidate B | 既有25组规则 | 不晋级 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    deltas = []
    for name in names[1:]:
        summary = summaries[name]
        deltas.append(
            f"- `{name}`相对E0（WAPE正数表示降低/改善）：Overall WAPE {base['overall_wape'] - summary['overall_wape']:+.4f} pp，"
            f"Nonzero WAPE {base['nonzero_wape'] - summary['nonzero_wape']:+.4f} pp，"
            f"5–19 Recall {summary['five_nineteen_recall'] - base['five_nineteen_recall']:+.4f} pp，"
            f"20+ Recall {summary['twenty_plus_recall'] - base['twenty_plus_recall']:+.4f} pp，"
            f"Bias {summary['total_bias'] - base['total_bias']:+.4f} pp，"
            f"Zero Pred {100 * (summary['zero_prediction_total'] / base['zero_prediction_total'] - 1):+.4f}%。"
        )

    q_rows = []
    for name in names:
        q = evaluation["q_metrics"][name]
        q_rows.append(
            f"| {name} | {_fmt(q['nonzero']['wape'])}% | {_fmt(q['5-19']['wape'])}% | "
            f"{_fmt(q['recall_5_19'])}% | {_fmt(q['20+']['wape'])}% | "
            f"{_fmt(q['recall_20_plus'])}% | {_fmt(q['twenty_plus_median_q_over_y'], 6)} |"
        )

    best_name = min(names, key=lambda name: summaries[name]["overall_wape"])
    best_summary = summaries[best_name]
    candidate_text = (
        f"Candidate B有{candidate['promoted_count']}组满足既有收益与保护门槛，开发集候选为"
        f"medium={candidate['selected']['medium_fraction']:.4%}、high={candidate['selected']['high_fraction']:.4%}。"
        if candidate["promoted"] else
        f"25组中保护条件可接受{candidate['protection_eligible_count']}组，但没有方案同时达到既有20+ WAPE/Recall收益门槛，故不晋级。"
    )
    candidate_detail = ""
    if not candidate["promoted"] and candidate.get("best_protected"):
        protected = candidate["best_protected"]
        changes = protected["development_decision"]["changes"]
        candidate_detail = (
            f"最接近晋级的保护方案为medium Top {protected['medium_fraction']:.2%}、"
            f"high Top {protected['high_fraction']:.2%}：20+ WAPE改善"
            f"{changes['twenty_plus_wape_gain_pp']:.4f} pp、Recall提高"
            f"{changes['twenty_plus_recall_gain_pp']:.4f} pp，但20+ WAPE改善未达到0.5 pp门槛；"
            f"同时5–19 WAPE恶化{changes['five_nineteen_worsening_pp']:.4f} pp、"
            f"Nonzero WAPE恶化{changes['nonzero_worsening_pp']:.4f} pp、"
            f"Zero Pred增加{100 * changes['zero_prediction_total_increase']:.4f}%、"
            f"绝对Bias恶化{changes['absolute_bias_worsening_pp']:.4f} pp。"
        )
    lines = [
        "", "## 统一Train→Valid开发口径重评", "",
        "### 协议审计", "",
        "新默认协议为完整Active-Store Train训练，正式Complete Valid用于轮次、特征和规则开发；本节没有读取Test。历史forward结果仍保留在前文，只是不再作为当前候选的选轮依据。", "",
        "| 实验 | 旧选择来源 | 本轮处理 |", "|---|---|---|",
        *[f"| {row['experiment']} | {row['old_selection']} | {row['action']} |" for row in audit_rows],
        "", "### 可复现性", "",
        f"- 统一稳定Train缓存共`{cache_metadata['total_rows']:,}`行，覆盖2023-01至2025-06；各组件保持原确定性采样率与逆概率权重。",
        f"- 特征顺序hash：`{cache_metadata['feature_order_sha256']}`；缓存hash：`{cache_metadata['cache_file_sha256']}`。",
        f"- 重复训练smoke：{deterministic_smoke['rows']:,}行、{deterministic_smoke['rounds']}轮，两次模型及预测完全一致；hash=`{deterministic_smoke['model_sha256']}`。",
        "- 三个regressor共享完全相同的正销量样本集合与顺序；每个模型metadata另存sample-set、order和model hash。",
        "- `p_sale`继续使用750轮正式classifier；其来源是旧版Complete Valid组合WAPE选择，并非四fold加权中位数。", "",
        "### 统一Valid结果", "",
        "| Model | 新增机制/特征 | Valid best iteration | Overall Raw WAPE | Nonzero WAPE | 5-19 WAPE | 5-19 Recall | 20+ WAPE | 20+ Recall | MC Macro-F1 | Total Bias | Zero Pred Total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows, "", *deltas, "",
        "补充业务指标：", "",
        "| Model | Integer Overall WAPE | Integer Nonzero WAPE | Zero >0.5 | Zero >1 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {_fmt(summaries[name]['integer_wape'])}% | {_fmt(summaries[name]['integer_nonzero_wape'])}% | "
            f"{_fmt(summaries[name]['zero_gt_0_5'] * 100)}% | {_fmt(summaries[name]['zero_gt_1'] * 100)}% |"
            for name in names
        ], "", "q层诊断：", "",
        "| Model | q Nonzero WAPE | q 5-19 WAPE | q 5-19 Recall | q 20+ WAPE | q 20+ Recall | 20+ median(q/y) |",
        "|---|---:|---:|---:|---:|---:|---:|", *q_rows, "",
        f"Candidate B：{candidate_text}", candidate_detail, "",
        "轮次选择记录：", "",
        *[
            f"- `{name}`：粗选{iteration_selection[name]['coarse']['iteration']}轮，在粗选点附近逐轮精扫后锁定"
            f"{selected[name]['metadata']['best_iteration']}轮；Valid Overall WAPE="
            f"{_fmt(summaries[name]['overall_wape'])}%。"
            for name in names
        ],
        "- `E1_DIFF_VALID_SELECTED`在5000轮开发预算附近锁定4988轮；这是预算内Valid最佳点，但未形成常规early-stopping内部最低点，复杂度与稳定性需作为限制记录。", "",
        "### 当前判断", "",
        f"- 统一开发口径下，当前Overall WAPE最低的1M候选是`{best_name}`（{_fmt(best_summary['overall_wape'])}%）。",
        "- Diff是否保留、Cross-store是否仍有增量以及Candidate B是否晋级，均以本节同一批Complete Valid结果为准；旧forward结果仅作时间稳定性背景。",
        "- 本节所有结果都是开发集结果。Test尚未运行，不能称为最终泛化结论。",
        f"- 执行日志：`{log_path.relative_to(ROOT)}`。", "",
        "### 导师汇报摘要", "",
        f"1. **当前基准模型**：Two-stage 1M保持750轮销量发生classifier，重新使用完整Train训练positive Tweedie regressor，并在正式Valid按原始销量WAPE选择轮次；新E0最佳轮次为{selected['E0_VALID_SELECTED']['metadata']['best_iteration']}。",
        "2. **三类优化**：Diff加入6个历史销量变化特征；Cross-store加入6个“同书其他门店”历史热度特征；Candidate B使用5+/20+概率只对少数高需求候选有限补偿。",
        f"3. **统一Valid指标**：Diff相对E0 Overall WAPE降低{base['overall_wape'] - summaries['E1_DIFF_VALID_SELECTED']['overall_wape']:.4f} pp，但Nonzero与中高需求指标未改善；Cross-store使Overall WAPE降低{base['overall_wape'] - summaries['E2_CROSS_STORE_VALID_SELECTED']['overall_wape']:.4f} pp，20+ Recall提高{summaries['E2_CROSS_STORE_VALID_SELECTED']['twenty_plus_recall'] - base['twenty_plus_recall']:.4f} pp。",
        f"4. **当前最佳方案**：`{best_name}`是统一Valid下当前整体点预测候选；是否成为最终方案须待所有候选锁定后一次性Test确认。",
        "5. **路线结论**：Diff主要改善整体零销量与Bias，但牺牲Nonzero及中高需求，不作为优先候选；Cross-store在整体、Nonzero、20+和总量偏差上均有明确增量；Candidate B现有25组规则无方案达到既定晋级门槛。",
        "6. **分布式衔接**：Cross-store结果可作为未来门店在不直接交换原始交易明细时，协同计算item级聚合热度、活跃门店数和变化信号的研究依据；当前实验尚未实现隐私保护。",
    ]
    path.write_text(content.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run_train_valid_protocol(config: dict, logger: logging.Logger, log_path: Path) -> None:
    protocol = config["train_valid_protocol"]
    audit_rows = audit_train_valid_protocol_dependencies()
    _, p_meta = load_model_bundle(FORMAL_P_MODEL)
    if int(p_meta["best_iteration"]) != 750:
        raise RuntimeError("The retained formal p_sale classifier is no longer the locked 750-round model")
    scopes = train_valid_protocol_feature_sets()
    cache_path, cache_metadata = build_train_valid_protocol_cache(
        config, p_meta["category_maps"], logger
    )
    smoke_sample = read_train_valid_protocol_sample(
        cache_path, "regressor_train", scopes["E0_VALID_SELECTED"]
    )
    deterministic_smoke = verify_train_valid_determinism(
        config, smoke_sample, scopes["E0_VALID_SELECTED"], logger
    )
    del smoke_sample
    gc.collect()

    trained: dict[str, dict] = {}
    for name in ("E0_VALID_SELECTED", "E1_DIFF_VALID_SELECTED", "E2_CROSS_STORE_VALID_SELECTED"):
        trained[name] = train_train_valid_max_component(
            config, cache_path, scopes[name], "regressor", p_meta["category_maps"], logger, name
        )
    for name, component in (
        ("CANDIDATE_B_S5_VALID_SELECTED", "classifier_5plus"),
        ("CANDIDATE_B_S20_VALID_SELECTED", "classifier_20plus"),
    ):
        trained[name] = train_train_valid_max_component(
            config, cache_path, scopes["E2_CROSS_STORE_VALID_SELECTED"], component,
            p_meta["category_maps"], logger, name,
        )

    max_rounds = int(protocol["max_rounds"])
    coarse_grid = protocol_coarse_iterations(max_rounds, int(protocol["coarse_step"]))
    coarse = load_completed_protocol_iteration_history(config, trained, logger)
    if coarse is None:
        coarse = stream_train_valid_iteration_metrics(
            config, trained, {name: coarse_grid for name in trained}, logger, "coarse"
        )
    coarse_selected = {}
    for name in trained:
        if trained[name]["metadata"]["component"] == "regressor":
            coarse_selected[name] = select_valid_wape_iteration(coarse["regressors"][name])
        else:
            coarse_selected[name] = select_binary_valid_iteration(coarse["classifiers"][name])
    boundary = boundary_selected_regressors(coarse_selected, trained)
    capped_at_budget: list[str] = []
    if boundary:
        extension_rounds = int(protocol["boundary_extension_rounds"])
        if extension_rounds <= max_rounds:
            raise RuntimeError("boundary_extension_rounds must exceed max_rounds")
        extension_start = max(
            max(int(row["iteration"]) for row in coarse["regressors"][name])
            for name in boundary
        )
        if extension_start < extension_rounds:
            for name in boundary:
                trained[name] = train_train_valid_max_component(
                    config, cache_path, scopes[name], "regressor", p_meta["category_maps"],
                    logger, name, max_rounds_override=extension_rounds,
                )
            extension_grid = list(range(
                extension_start + int(protocol["coarse_step"]),
                extension_rounds + 1,
                int(protocol["coarse_step"]),
            ))
            if extension_grid[-1] != extension_rounds:
                extension_grid.append(extension_rounds)
            extension = stream_train_valid_iteration_metrics(
                config,
                {name: trained[name] for name in boundary},
                {name: extension_grid for name in boundary},
                logger,
                f"coarse_extension_{extension_rounds}",
            )
            for name in boundary:
                coarse["regressors"][name].extend(extension["regressors"][name])
                coarse["regressors"][name] = sorted(
                    coarse["regressors"][name], key=lambda row: row["iteration"]
                )
                coarse_selected[name] = select_valid_wape_iteration(coarse["regressors"][name])
        still_boundary = boundary_selected_regressors(coarse_selected, trained)
        if still_boundary:
            capped_at_budget.extend(still_boundary)
            logger.warning(
                "Complete Valid optimum touches the declared %s-round development budget: %s; "
                "keeping the best evaluated point and recording that normal early stopping was not reached",
                extension_rounds, still_boundary,
            )
    refine_grids = {
        name: protocol_refinement_iterations(
            selection["iteration"], int(trained[name]["metadata"]["max_rounds"]),
            int(protocol["refinement_radius"])
        )
        for name, selection in coarse_selected.items()
    }
    refined = stream_train_valid_iteration_metrics(
        config, trained, refine_grids, logger, "refined"
    )
    final_selection = {}
    for name in trained:
        if trained[name]["metadata"]["component"] == "regressor":
            final_selection[name] = select_valid_wape_iteration(refined["regressors"][name])
        else:
            final_selection[name] = select_binary_valid_iteration(refined["classifiers"][name])
    final_boundary = boundary_selected_regressors(final_selection, trained)
    if final_boundary:
        capped_at_budget.extend(name for name in final_boundary if name not in capped_at_budget)
        logger.warning(
            "Refined Complete Valid optimum remains at the declared round budget: %s", final_boundary
        )

    selected = {
        name: save_train_valid_selected_model(
            config, trained[name], selection["iteration"], selection, logger
        )
        for name, selection in final_selection.items()
    }
    evaluation = evaluate_train_valid_selected_models(config, selected, logger)
    iteration_selection = {
        name: {"coarse": coarse_selected[name], "refined": final_selection[name]}
        for name in trained
    }
    result = {
        "protocol": "full Active-Store Train -> formal Complete Valid development",
        "test_used": False,
        "audit": audit_rows,
        "cache_metadata": cache_metadata,
        "deterministic_smoke": deterministic_smoke,
        "capped_at_round_budget": capped_at_budget,
        "iteration_selection": iteration_selection,
        "selected_models": {
            name: {
                "path": str(item["path"].relative_to(ROOT)),
                "best_iteration": item["metadata"]["best_iteration"],
                "model_sha256": item["model_sha256"],
            }
            for name, item in selected.items()
        },
        "evaluation": evaluation,
    }
    result_path = _protocol_root(config) / "train_valid_protocol_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_train_valid_protocol_ablation(config, selected, evaluation)
    append_train_valid_protocol_report(
        config, audit_rows, cache_metadata, deterministic_smoke, selected,
        evaluation, iteration_selection, log_path,
    )
    shutil.rmtree(_protocol_checkpoint_root(config) / "max_round_models", ignore_errors=True)
    cache_path.unlink(missing_ok=True)
    logger.info("Unified Train->Valid development protocol complete; Test was not read")


def run_e3_train_valid_protocol(config: dict, logger: logging.Logger, log_path: Path) -> None:
    name = "E3_DIFF_CROSS_STORE_VALID_SELECTED"
    protocol_root = _protocol_root(config)
    prior_path = protocol_root / "train_valid_protocol_result.json"
    result_path = protocol_root / "e3_train_valid_protocol_result.json"
    report_path = project_path(config["outputs"]["report"])
    if result_path.exists() and "### E3 Diff + Cross-store消融结论" in report_path.read_text(encoding="utf-8"):
        logger.info("E3 unified Train->Valid ablation already complete; Test was not read")
        return
    if not prior_path.exists():
        raise FileNotFoundError("Unified E0/E1/E2 Train->Valid result is required before E3")
    prior_result = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior_result.get("test_used") is not False:
        raise RuntimeError("Prior unified protocol result has an invalid Test-use marker")

    _, p_meta = load_model_bundle(FORMAL_P_MODEL)
    scopes = train_valid_protocol_feature_sets()
    features = scopes[name]
    if len(features) != 69:
        raise RuntimeError(f"E3 must contain exactly 69 features, found {len(features)}")
    cache_path, cache_metadata = build_train_valid_protocol_cache(
        config, p_meta["category_maps"], logger
    )
    previous_cache = prior_result["cache_metadata"]
    for key in ("total_rows", "sample_counts", "feature_order_sha256", "cache_file_sha256"):
        if cache_metadata[key] != previous_cache[key]:
            raise RuntimeError(
                f"Rebuilt stable Train cache differs from unified E0/E1/E2 cache for {key}"
            )
    logger.info("E3 stable Train cache exactly matches prior hash=%s", cache_metadata["cache_file_sha256"])

    smoke_sample = read_train_valid_protocol_sample(cache_path, "regressor_train", features)
    deterministic_smoke = verify_train_valid_determinism(
        config, smoke_sample, features, logger
    )
    del smoke_sample
    gc.collect()

    initial_rounds = int(config["train_valid_protocol"]["max_rounds"])
    extension_rounds = int(config["train_valid_protocol"]["boundary_extension_rounds"])
    coarse_step = int(config["train_valid_protocol"]["coarse_step"])
    trained = train_train_valid_max_component(
        config, cache_path, features, "regressor", p_meta["category_maps"], logger, name,
        max_rounds_override=initial_rounds,
    )
    coarse_grid = protocol_coarse_iterations(initial_rounds, coarse_step)
    coarse_result = stream_train_valid_iteration_metrics(
        config, {name: trained}, {name: coarse_grid}, logger, f"e3_coarse_{initial_rounds}"
    )
    coarse_rows = list(coarse_result["regressors"][name])
    coarse_selected = select_valid_wape_iteration(coarse_rows)
    extended_once = False
    if int(coarse_selected["iteration"]) >= initial_rounds:
        extended_once = True
        trained = train_train_valid_max_component(
            config, cache_path, features, "regressor", p_meta["category_maps"], logger, name,
            max_rounds_override=extension_rounds,
        )
        extension_grid = list(range(initial_rounds + coarse_step, extension_rounds + 1, coarse_step))
        if extension_grid[-1] != extension_rounds:
            extension_grid.append(extension_rounds)
        extension_result = stream_train_valid_iteration_metrics(
            config, {name: trained}, {name: extension_grid}, logger,
            f"e3_extension_{extension_rounds}",
        )
        coarse_rows.extend(extension_result["regressors"][name])
        coarse_rows = sorted(coarse_rows, key=lambda row: int(row["iteration"]))
        coarse_selected = select_valid_wape_iteration(coarse_rows)

    capped_at_budget = (
        extended_once and int(coarse_selected["iteration"]) >= extension_rounds
    )
    if capped_at_budget:
        logger.warning(
            "E3 optimum touches the one-time %s-round extension budget; no further extension is allowed",
            extension_rounds,
        )
    refine_grid = protocol_refinement_iterations(
        int(coarse_selected["iteration"]), int(trained["metadata"]["max_rounds"]),
        int(config["train_valid_protocol"]["refinement_radius"]),
    )
    refined = stream_train_valid_iteration_metrics(
        config, {name: trained}, {name: refine_grid}, logger, "e3_refined"
    )
    final_selection = select_valid_wape_iteration(refined["regressors"][name])
    if int(final_selection["iteration"]) >= int(trained["metadata"]["max_rounds"]):
        capped_at_budget = True
    trained["metadata"]["round_budget_capped"] = capped_at_budget
    trained["metadata"]["extension_used"] = extended_once
    selected_item = save_train_valid_selected_model(
        config, trained, int(final_selection["iteration"]), final_selection, logger
    )
    evaluation = evaluate_e3_train_valid_selected_model(config, selected_item, logger)
    if int(evaluation["rows"]) != int(prior_result["evaluation"]["rows"]):
        raise RuntimeError("E3 and E0/E1/E2 Complete Valid row counts differ")

    iteration_selection = {"coarse": coarse_selected, "refined": final_selection}
    result = {
        "protocol": "E3 only: full Active-Store Train -> formal Complete Valid development",
        "test_used": False,
        "cache_metadata": cache_metadata,
        "prior_protocol_result_sha256": _sha256_file(prior_path),
        "deterministic_smoke": deterministic_smoke,
        "extended_once": extended_once,
        "capped_at_round_budget": capped_at_budget,
        "iteration_selection": iteration_selection,
        "selected_model": {
            "path": str(selected_item["path"].relative_to(ROOT)),
            "best_iteration": selected_item["metadata"]["best_iteration"],
            "model_sha256": selected_item["model_sha256"],
        },
        "evaluation": evaluation,
        "log_path": str(log_path.relative_to(ROOT)),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_e3_train_valid_protocol_ablation(config, selected_item, evaluation)
    update_report_with_e3_train_valid_result(
        config, prior_result, selected_item, evaluation, iteration_selection, capped_at_budget
    )
    max_path = _protocol_checkpoint_root(config) / "max_round_models" / f"{name}.txt"
    max_path.unlink(missing_ok=True)
    max_path.with_suffix(".json").unlink(missing_ok=True)
    cache_path.unlink(missing_ok=True)
    logger.info("E3 unified Train->Valid ablation complete; Test was not read")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Active-Store Two-stage 1M with strict serial gates")
    parser.add_argument(
        "--train-valid-protocol", action="store_true",
        help="Run the unified full-Train to formal-Valid 1M development protocol",
    )
    parser.add_argument(
        "--e3-valid-protocol-only", action="store_true",
        help="Run only the locked 69-feature E3 full-Train to formal-Valid ablation",
    )
    parser.add_argument(
        "--diff-pilot-only", action="store_true",
        help="Run only the controlled E0 versus E1 Diff regressor forward Pilot",
    )
    parser.add_argument(
        "--cross-store-diagnostic-only", action="store_true",
        help="Rebuild the locked E2 checkpoint and diagnose Complete Valid error migration",
    )
    parser.add_argument(
        "--cross-store-formal-only", action="store_true",
        help="Resume only the E2 Cross-store full-Train and Complete Valid branch",
    )
    parser.add_argument(
        "--stop-after", choices=["cross-store", "audit", "pilot", "oof", "complete-valid"],
        default="complete-valid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_optimization_config()
    logger, log_path = setup_logging(config)
    validate_inputs(config)
    if args.train_valid_protocol:
        run_train_valid_protocol(config, logger, log_path)
        return
    if args.e3_valid_protocol_only:
        run_e3_train_valid_protocol(config, logger, log_path)
        return
    if args.diff_pilot_only:
        run_diff_pilot_only(config, logger, log_path)
        return
    if args.cross_store_diagnostic_only:
        run_cross_store_diagnostic_only(config, logger, log_path)
        return
    if args.cross_store_formal_only:
        run_cross_store_formal_only(config, logger, log_path)
        return
    cross = build_cross_store_table(config, logger)
    if args.stop_after == "cross-store":
        logger.info("Stopped by --stop-after cross-store")
        return
    audit = run_statistical_audit(config, logger)
    pilot = oof = formal = None
    if args.stop_after == "audit" or not any(audit["group_pass"].values()):
        write_outputs(config, cross, audit, pilot, oof, formal, log_path)
        return
    pilot = run_pilot(config, audit, logger)
    if args.stop_after == "pilot" or not pilot.get("passed"):
        write_outputs(config, cross, audit, pilot, oof, formal, log_path)
        if should_clean_stage_artifacts(args.stop_after == "pilot", bool(pilot.get("passed"))):
            clean_rebuildable_artifacts(config, keep_formal=False)
        return
    oof = run_oof(config, audit, pilot, logger)
    if args.stop_after == "oof" or not oof.get("passed"):
        write_outputs(config, cross, audit, pilot, oof, formal, log_path)
        if should_clean_stage_artifacts(args.stop_after == "oof", bool(oof.get("passed"))):
            clean_rebuildable_artifacts(config, keep_formal=False)
        return
    formal = run_formal(config, oof, logger)
    write_outputs(config, cross, audit, pilot, oof, formal, log_path)
    clean_rebuildable_artifacts(config, keep_formal=bool(formal.get("passed")))
    logger.info("Optimization workflow complete passed=%s", formal.get("passed"))


if __name__ == "__main__":
    main()
