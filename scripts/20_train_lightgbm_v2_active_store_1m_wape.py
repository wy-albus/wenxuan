from __future__ import annotations

import argparse
import gc
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
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import StreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import load_feature_config  # noqa: E402
from src.models.lightgbm_model import load_model_bundle, save_model_bundle  # noqa: E402


DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
BASE_CHECKPOINT_PATH = ROOT / "models/checkpoints/active_store_experiment/lightgbm_v2_logl2_1m.txt"
EXTENDED_CHECKPOINT_PATH = ROOT / "models/checkpoints/active_store_wape/lightgbm_v2_logl2_active_store_wape_1m_2000.txt"
FINAL_MODEL_PATH = ROOT / "models/final/lightgbm_v2_logl2_active_store_wape_1m.txt"
CANDIDATE_PATH = ROOT / "reports/lightgbm_v2_active_store_wape_candidates_1m.csv"
CURVE_PATH = ROOT / "reports/lightgbm_v2_active_store_wape_curve_1m.png"
TEST_PREDICTION_PATH = ROOT / "data/outputs/lightgbm_v2_active_store_wape_1m_test_predictions.parquet"
FAIR_COMPARISON_PATH = ROOT / "reports/lightgbm_v2_active_store_wape_1m_fair_comparison.csv"
REPORT_PATH = ROOT / "reports/lightgbm_v2_active_store_wape_1m_report.md"
OLD_PREDICTION_PATH = ROOT / "data/outputs/lightgbm_v2_logl2_test_predictions.parquet"
LOG_DIR = ROOT / "logs/lightgbm/active_store_wape_1m"
TEMP_DIR = ROOT / "data/temp/active_store_wape_1m"

MAX_ITERATION = 2000
EXPECTED_VALID_ROWS = 18_563_573
EXPECTED_TEST_ROWS = 12_999_597
CATEGORY_FEATURES = ["site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"]


class ExactRangeMetrics:
    """Experiment metrics using 2<=qty<5, 5<=qty<20, and qty>=20 boundaries."""

    def __init__(self) -> None:
        self.segments = {
            name: StreamingMetrics()
            for name in ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
        }
        self.zero_gt_0_5 = 0
        self.zero_gt_1 = 0

    def update(self, target, prediction) -> None:
        target = clip_target(target)
        prediction = clip_target(prediction)
        if target.shape != prediction.shape:
            raise ValueError("target and prediction must have the same shape")
        masks = {
            "overall": np.ones(target.shape, dtype=bool),
            "0": target == 0,
            "1": target == 1,
            "2-5": (target >= 2) & (target < 5),
            "5-20": (target >= 5) & (target < 20),
            "20+": target >= 20,
            "nonzero": target > 0,
            "ge_5": target >= 5,
            "ge_20": target >= 20,
        }
        for name, mask in masks.items():
            if mask.any():
                self.segments[name].update(target[mask], prediction[mask])
        zero_prediction = prediction[masks["0"]]
        self.zero_gt_0_5 += int((zero_prediction > 0.5).sum())
        self.zero_gt_1 += int((zero_prediction > 1.0).sum())

    def compute(self) -> dict[str, dict[str, float | int]]:
        result = {name: accumulator.compute() for name, accumulator in self.segments.items()}
        zero = result["0"]
        count = int(zero["count"])
        zero["wape"] = np.nan
        zero["mean_prediction"] = zero["prediction_sum"] / count if count else np.nan
        zero["prediction_gt_0_5_rate"] = self.zero_gt_0_5 / count if count else np.nan
        zero["prediction_gt_1_rate"] = self.zero_gt_1 / count if count else np.nan
        return result


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_module():
    return _load_script_module("wenxuan_active_store_wape_pipeline", ROOT / "scripts/10_train_lightgbm.py")


def experiment_module():
    return _load_script_module("wenxuan_active_store_wape_experiment", ROOT / "scripts/18_train_active_store_experiment.py")


def extended_candidate_iterations() -> list[int]:
    return [600, 650, 700, 750, 800, 850, 900, 1000, 1200, 1400, 1600, 1800, 2000]


def validate_candidate_rows(rows: list[dict[str, Any]], expected_count: int) -> None:
    if not rows:
        raise ValueError("Candidate table is empty")
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != sorted(set(iterations)):
        raise ValueError("Candidate iterations must be unique and sorted")
    counts = {int(row["count"]) for row in rows}
    if counts != {int(expected_count)}:
        raise ValueError(f"Every candidate must use the same complete Valid count: {sorted(counts)}")
    if set(iterations) != set(extended_candidate_iterations()):
        raise ValueError("Candidate table does not cover the approved 600-2000 search points")
    metric_columns = ("wape", "mae", "rmse", "prediction_sum", "target_sum", "total_bias_rate")
    for row in rows:
        if not all(math.isfinite(float(row[column])) for column in metric_columns):
            raise ValueError(f"Candidate {row['iteration']} contains non-finite metrics")


def wape_search_status(rows: list[dict[str, Any]], best_iteration: int) -> str:
    iterations = sorted(int(row["iteration"]) for row in rows)
    if int(best_iteration) == iterations[-1]:
        return "best_at_search_upper_bound"
    if int(best_iteration) == iterations[0]:
        return "best_at_search_lower_bound"
    return "internal_minimum"


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("wenxuan_active_store_wape_1m")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def _required_inputs() -> None:
    for path in (DATASET_PATH, BASE_CHECKPOINT_PATH, OLD_PREDICTION_PATH):
        if not path.exists():
            raise FileNotFoundError(path)


def _training_params(metadata: dict) -> dict:
    params = dict(metadata["params"])
    params.update(
        objective="regression_l2",
        metric="l2",
        num_threads=max(1, (os.cpu_count() or 2) - 2),
        seed=42,
        verbosity=-1,
    )
    return params


def _extended_checkpoint_metadata(base_metadata: dict, params: dict, training_summary: dict) -> dict:
    metadata = dict(base_metadata)
    metadata.update(
        experiment="active_store_wape_1m_extension",
        model_kind="lightgbm_v2_logl2",
        horizon="1m",
        params=params,
        trained_iterations=MAX_ITERATION,
        search_candidates=extended_candidate_iterations(),
        target_transform="log1p",
        prediction_transform="clip(expm1(prediction), 0, None)",
        selection_metric="complete_valid_original_scale_wape",
        training_summary=training_summary,
        lightgbm_version=lgb.__version__,
        python_version=platform.python_version(),
    )
    metadata.pop("candidate_scores", None)
    metadata.pop("best_iteration", None)
    return metadata


def build_continuation_datasets(
    train: dict,
    valid: dict,
    features: list[str],
    categorical_features: list[str],
    params: dict,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    # LightGBM needs retained raw data when attaching the predictor from init_model.
    train_set = lgb.Dataset(
        train["x"], label=train["y"], weight=train["weight"],
        feature_name=features, categorical_feature=categorical_features,
        params=params, free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid["x"], label=valid["y"], weight=valid["weight"], reference=train_set,
        feature_name=features, categorical_feature=categorical_features,
        params=params, free_raw_data=False,
    )
    return train_set, valid_set


def continue_to_2000(logger: logging.Logger) -> tuple[lgb.Booster, dict]:
    if EXTENDED_CHECKPOINT_PATH.exists():
        booster, metadata = load_model_bundle(EXTENDED_CHECKPOINT_PATH)
        if booster.current_iteration() != MAX_ITERATION:
            raise RuntimeError("Existing extended checkpoint does not contain 2000 iterations")
        logger.info("Reusing extended 2000-iteration checkpoint: %s", EXTENDED_CHECKPOINT_PATH)
        return booster, metadata

    pipeline = pipeline_module()
    config = load_feature_config()
    base, base_metadata = load_model_bundle(BASE_CHECKPOINT_PATH)
    if base.current_iteration() != 700:
        raise RuntimeError(f"Expected the resumable base checkpoint to contain 700 trees, found {base.current_iteration()}")
    features = pipeline.horizon_features(config, "1m")
    if features != base_metadata["feature_names"]:
        raise RuntimeError("Current 1M feature order differs from the 700-tree checkpoint")
    guard = pipeline.MemoryGuard("active-store-wape-1m-train", logger)
    settings = config["lightgbm_training"]
    sample_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        DATASET_PATH,
        "1m",
        {"train": settings["sampling_rates"], "valid": settings["valid_sampling_rates"]},
        base_metadata["category_maps"],
        features,
        config["time_feature_settings"]["base_month"],
        guard,
        logger,
        {"train": None, "valid": None},
        max_row_groups=None,
        enforce_distribution=True,
    )
    sampling_seconds = time.perf_counter() - sample_started
    train, valid = samples["train"], samples["valid"]
    params = _training_params(base_metadata)
    logger.info(
        "Continuing 1M from 700 to 2000 with train=%s valid=%s features=%d",
        f"{len(train['y']):,}", f"{len(valid['y']):,}", len(features),
    )
    estimated = int((len(train["y"]) + len(valid["y"])) * (len(features) * 3 + 64))
    guard.ensure_capacity(estimated, "before_continuation_dataset")
    train_set, valid_set = build_continuation_datasets(
        train, valid, features, CATEGORY_FEATURES, params
    )
    train_set.construct()
    valid_set.construct()
    guard.check("continuation_datasets_constructed")
    del train["x"], valid["x"]
    gc.collect()
    evaluations: dict = {}
    training_started = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ITERATION - base.current_iteration(),
        valid_sets=[valid_set],
        valid_names=["sampled_valid"],
        init_model=base,
        keep_training_booster=True,
        callbacks=[lgb.record_evaluation(evaluations), lgb.log_evaluation(100), guard.callback(10, 100)],
    )
    training_seconds = time.perf_counter() - training_started
    if booster.current_iteration() != MAX_ITERATION:
        raise RuntimeError(f"Continuation stopped at {booster.current_iteration()} instead of {MAX_ITERATION}")
    training_summary = {
        "base_iterations": 700,
        "trained_iterations": booster.current_iteration(),
        "train_rows": len(train["y"]),
        "sampled_valid_rows": len(valid["y"]),
        "sampling_seconds": sampling_seconds,
        "continuation_training_seconds": training_seconds,
        "peak_ram_gib": guard.peak / (1024 ** 3),
        "sampling_distribution": {
            "train": train["distribution"], "valid": valid["distribution"],
        },
        "sampled_valid_l2_history": evaluations.get("sampled_valid", {}).get("l2", []),
    }
    metadata = _extended_checkpoint_metadata(base_metadata, params, training_summary)
    EXTENDED_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(booster, EXTENDED_CHECKPOINT_PATH, metadata)
    reloaded, reloaded_metadata = load_model_bundle(EXTENDED_CHECKPOINT_PATH)
    if reloaded.current_iteration() != MAX_ITERATION or reloaded_metadata["trained_iterations"] != MAX_ITERATION:
        raise RuntimeError("Extended checkpoint reload mismatch")
    logger.info(
        "Saved 2000-iteration checkpoint; sampling %.1fs training %.1fs peak %.2f GiB",
        sampling_seconds, training_seconds, guard.peak / (1024 ** 3),
    )
    del samples, train, valid, train_set, valid_set, base, booster
    gc.collect()
    return reloaded, reloaded_metadata


def _save_curve(frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["iteration"], frame["wape"], marker="o", linewidth=1.6)
    best = frame.loc[frame["wape"].idxmin()]
    axis.scatter([best["iteration"]], [best["wape"]], color="#c62828", zorder=3)
    axis.annotate(
        f"best={int(best['iteration'])}\nWAPE={best['wape']:.4f}%",
        (best["iteration"], best["wape"]), xytext=(8, 10), textcoords="offset points",
    )
    axis.set_xlabel("Boosting iteration")
    axis.set_ylabel("Complete Valid WAPE (%)")
    axis.set_title("LightGBM V2 Active-Store 1M WAPE Search")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CURVE_PATH.with_suffix(".png.tmp")
    figure.savefig(temporary, dpi=150, format="png")
    plt.close(figure)
    temporary.replace(CURVE_PATH)


def select_on_complete_valid(
    booster: lgb.Booster, metadata: dict, logger: logging.Logger
) -> tuple[lgb.Booster, dict, pd.DataFrame]:
    experiment = experiment_module()
    if CANDIDATE_PATH.exists():
        candidate_frame = pd.read_csv(CANDIDATE_PATH)
        rows = candidate_frame.to_dict("records")
        validate_candidate_rows(rows, EXPECTED_VALID_ROWS)
        logger.info("Reusing complete-Valid candidate table: %s", CANDIDATE_PATH)
    else:
        pipeline = pipeline_module()
        guard = pipeline.MemoryGuard("active-store-wape-1m-valid", logger)
        started = time.perf_counter()
        rows = experiment.evaluate_quantity_candidates(
            pipeline,
            DATASET_PATH,
            "1m",
            booster,
            metadata,
            extended_candidate_iterations(),
            guard,
            "log1p",
            max_row_groups=None,
        )
        rows = sorted(rows, key=lambda row: int(row["iteration"]))
        validate_candidate_rows(rows, EXPECTED_VALID_ROWS)
        candidate_frame = pd.DataFrame(rows)
        candidate_frame["evaluation_seconds"] = time.perf_counter() - started
        candidate_frame["peak_ram_gib"] = guard.peak / (1024 ** 3)
        CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CANDIDATE_PATH.with_suffix(".csv.tmp")
        candidate_frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(CANDIDATE_PATH)
        logger.info(
            "Complete Valid candidate scan finished in %.1fs peak %.2f GiB",
            candidate_frame["evaluation_seconds"].iloc[0], guard.peak / (1024 ** 3),
        )
    _save_curve(candidate_frame)
    rows = candidate_frame.to_dict("records")
    best = experiment.select_best_candidate(rows, experiment.QUANTITY_CANDIDATE_METRIC_ORDER)
    best_iteration = int(best["iteration"])
    status = wape_search_status(rows, best_iteration)
    final_metadata = dict(metadata)
    final_metadata.update(
        best_iteration=best_iteration,
        candidate_scores=rows,
        search_status=status,
        selection_split="complete_valid_original_scale",
        selection_metric="WAPE = sum(abs(y_true-y_pred)) / sum(y_true)",
        test_used_for_selection=False,
    )
    if FINAL_MODEL_PATH.exists():
        selected, saved_metadata = load_model_bundle(FINAL_MODEL_PATH)
        if int(saved_metadata.get("best_iteration", -1)) != best_iteration:
            raise RuntimeError("Existing final 1M WAPE model conflicts with current candidate selection")
    else:
        selected = experiment._pruned(booster, best_iteration)
        FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_model_bundle(selected, FINAL_MODEL_PATH, final_metadata)
        selected, saved_metadata = load_model_bundle(FINAL_MODEL_PATH)
        if selected.current_iteration() != best_iteration:
            raise RuntimeError("Selected model reload tree count mismatch")
    logger.info(
        "Selected iteration=%d valid WAPE=%.6f%% status=%s",
        best_iteration, float(best["wape"]), status,
    )
    return selected, final_metadata, candidate_frame


def evaluate_test_once(model: lgb.Booster, metadata: dict, logger: logging.Logger) -> tuple[dict, dict]:
    metrics_path = TEST_PREDICTION_PATH.with_suffix(".metrics.json")
    if TEST_PREDICTION_PATH.exists() and metrics_path.exists():
        logger.info("Reusing existing one-time Test prediction and metrics")
        return json.loads(metrics_path.read_text(encoding="utf-8")), {"reused": True}
    if TEST_PREDICTION_PATH.exists() or metrics_path.exists():
        raise RuntimeError("Incomplete Test output pair exists; refusing ambiguous re-evaluation")

    pipeline = pipeline_module()
    experiment = experiment_module()
    config = load_feature_config()
    guard = pipeline.MemoryGuard("active-store-wape-1m-test", logger)
    columns = experiment._physical_eval_columns(config, metadata["feature_names"], "1m")
    accumulator = ExactRangeMetrics()
    temporary = TEST_PREDICTION_PATH.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    started = time.perf_counter()
    rows_written = 0
    try:
        for number, frame in enumerate(
            experiment.iter_horizon_row_groups(DATASET_PATH, "1m", columns, {"test"}), start=1
        ):
            target = clip_target(frame["future_qty_1m"].to_numpy())
            x = experiment._prepare_eval(pipeline, frame, metadata)
            prediction = np.clip(
                np.expm1(model.predict(x, num_iteration=int(metadata["best_iteration"]))), 0.0, None
            )
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise RuntimeError("Invalid active-store Test prediction")
            accumulator.update(target, prediction)
            output = pd.DataFrame(
                {
                    "month": frame["month"].astype("string"),
                    "site_no": frame["site_no"].astype("string"),
                    "item_id": frame["item_id"].astype("string"),
                    "target_qty_1m": target.astype("float32"),
                    "lightgbm_v2_active_store_wape_pred_1m": prediction.astype("float32"),
                }
            )
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                TEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=True)
            writer.write_table(table, row_group_size=500_000)
            rows_written += len(output)
            if number % 10 == 0:
                guard.check("test_stream")
                logger.info("Test prediction rows=%s", f"{rows_written:,}")
            del frame, x, output, table, target, prediction
    finally:
        if writer is not None:
            writer.close()
    if writer is None or rows_written != EXPECTED_TEST_ROWS:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {EXPECTED_TEST_ROWS:,} Test rows, wrote {rows_written:,}")
    temporary.replace(TEST_PREDICTION_PATH)
    elapsed = time.perf_counter() - started
    metrics = accumulator.compute()
    runtime = {
        "rows": rows_written,
        "evaluation_seconds": elapsed,
        "peak_ram_gib": guard.peak / (1024 ** 3),
        "prediction_file_bytes": TEST_PREDICTION_PATH.stat().st_size,
        "reused": False,
    }
    temp_metrics = metrics_path.with_suffix(".json.tmp")
    temp_metrics.write_text(json.dumps({"metrics": metrics, "runtime": runtime}, ensure_ascii=False), encoding="utf-8")
    temp_metrics.replace(metrics_path)
    return {"metrics": metrics, "runtime": runtime}, runtime


def _fair_comparison() -> pd.DataFrame:
    if FAIR_COMPARISON_PATH.exists():
        return pd.read_csv(FAIR_COMPARISON_PATH)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = __import__("duckdb").connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='6GB'")
    connection.execute(f"PRAGMA temp_directory='{TEMP_DIR.resolve().as_posix()}'")
    new_path = TEST_PREDICTION_PATH.resolve().as_posix().replace("'", "''")
    old_path = OLD_PREDICTION_PATH.resolve().as_posix().replace("'", "''")
    try:
        frame = connection.execute(
            f"""
            WITH common AS MATERIALIZED (
              SELECT n.target_qty_1m AS target,
                     n.lightgbm_v2_active_store_wape_pred_1m AS new_pred,
                     o.lightgbm_v2_logl2_pred_1m AS old_pred,
                     ABS(n.target_qty_1m-o.target_qty_1m) AS target_difference
              FROM read_parquet('{new_path}') n
              JOIN read_parquet('{old_path}') o USING(month,site_no,item_id)
            ), expanded AS (
              SELECT target, new_pred AS prediction, 'active_store_wape' AS model, target_difference FROM common
              UNION ALL
              SELECT target, old_pred AS prediction, 'old_v2_logl2' AS model, target_difference FROM common
            ), segmented AS (
              SELECT *, segment FROM expanded
              CROSS JOIN (VALUES ('overall'),('nonzero'),('5-20'),('20+'),('0')) s(segment)
              WHERE segment='overall'
                 OR (segment='nonzero' AND target>0)
                 OR (segment='5-20' AND target>=5 AND target<20)
                 OR (segment='20+' AND target>=20)
                 OR (segment='0' AND target<=0)
            )
            SELECT model, segment, COUNT(*)::BIGINT AS count,
                   AVG(ABS(target-prediction)) AS mae,
                   SQRT(AVG(POW(target-prediction,2))) AS rmse,
                   CASE WHEN SUM(target)>0 THEN 100.0*SUM(ABS(target-prediction))/SUM(target) ELSE NULL END AS wape,
                   SUM(target) AS target_sum, SUM(prediction) AS prediction_sum,
                   CASE WHEN SUM(target)>0 THEN 100.0*(SUM(prediction)-SUM(target))/SUM(target) ELSE NULL END AS total_bias_rate,
                   CASE WHEN segment='0' THEN AVG(prediction) ELSE NULL END AS zero_mean_prediction,
                   CASE WHEN segment='0' THEN AVG(CASE WHEN prediction>0.5 THEN 1.0 ELSE 0.0 END) ELSE NULL END AS zero_pred_gt_0_5_rate,
                   CASE WHEN segment='0' THEN AVG(CASE WHEN prediction>1 THEN 1.0 ELSE 0.0 END) ELSE NULL END AS zero_pred_gt_1_rate,
                   MAX(target_difference) AS max_target_difference
            FROM segmented GROUP BY model,segment ORDER BY model,segment
            """
        ).df()
    finally:
        connection.close()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    if frame.empty or frame["max_target_difference"].max() != 0:
        raise RuntimeError("Old/new common-Test target alignment failed")
    overall_counts = frame[frame["segment"].eq("overall")]["count"].unique()
    if len(overall_counts) != 1 or int(overall_counts[0]) != EXPECTED_TEST_ROWS:
        raise RuntimeError("Old/new common-Test intersection does not match active Test")
    temporary = FAIR_COMPARISON_PATH.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(FAIR_COMPARISON_PATH)
    return frame


def _metric_table(metrics: dict) -> list[str]:
    lines = [
        "| Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20"):
        row = metrics[segment]
        wape = "N/A" if row.get("wape") is None or not math.isfinite(float(row.get("wape", np.nan))) else f"{row['wape']:.4f}%"
        bias = "N/A" if row.get("total_bias_rate") is None or not math.isfinite(float(row.get("total_bias_rate", np.nan))) else f"{row['total_bias_rate']:.4f}%"
        lines.append(
            f"| {segment} | {int(row['count']):,} | {row['mae']:.6f} | {row['rmse']:.6f} | {wape} | "
            f"{row['target_sum']:.2f} | {row['prediction_sum']:.2f} | {bias} |"
        )
    return lines


def write_report(candidate_frame: pd.DataFrame, metadata: dict, test_payload: dict, fair: pd.DataFrame, log_path: Path) -> None:
    best_iteration = int(metadata["best_iteration"])
    best = candidate_frame[candidate_frame["iteration"].eq(best_iteration)].iloc[0]
    training = metadata.get("training_summary", {})
    test_metrics = test_payload["metrics"]
    zero = test_metrics["0"]
    candidate_lines = [
        "| Iteration | Valid rows | WAPE | MAE | RMSE | True total | Predicted total | Total bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_frame.sort_values("iteration").itertuples(index=False):
        candidate_lines.append(
            f"| {int(row.iteration)} | {int(row.count):,} | {row.wape:.6f}% | {row.mae:.6f} | {row.rmse:.6f} | "
            f"{row.target_sum:.2f} | {row.prediction_sum:.2f} | {row.total_bias_rate:.6f}% |"
        )
    fair_lines = [
        "| Model | Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fair.itertuples(index=False):
        fair_lines.append(
            f"| {row.model} | {row.segment} | {int(row.count):,} | {row.mae:.6f} | {row.rmse:.6f} | "
            f"{('N/A' if pd.isna(row.wape) else f'{row.wape:.6f}%')} | {row.target_sum:.2f} | {row.prediction_sum:.2f} | "
            f"{('N/A' if pd.isna(row.total_bias_rate) else f'{row.total_bias_rate:.6f}%')} |"
        )
    search_note = (
        "The best candidate is an internal minimum in the evaluated range."
        if metadata["search_status"] == "internal_minimum"
        else "The best candidate remains on a search boundary; it is not claimed as a proven interior optimum."
    )
    lines = [
        "# LightGBM V2 Active-Store 1M WAPE Report", "",
        "## Experiment", "",
        f"- Dataset: `{DATASET_PATH.relative_to(ROOT).as_posix()}`",
        "- Target: `max(future_qty_1m, 0)`; training target is `log1p(target)`.",
        "- Objective: `regression_l2`; prediction is `clip(expm1(log_prediction), 0, None)`.",
        f"- Features: {len(metadata['feature_names'])}; feature order and Train-only category maps are stored in the model metadata.",
        "- Formal selection metric: complete Valid WAPE on the original quantity scale.",
        "- WAPE is accumulated globally as total absolute error divided by total true quantity; row-group WAPEs are never averaged.",
        "- Test is not used for iteration, parameter, threshold, or model selection.",
        f"- Model: `{FINAL_MODEL_PATH.relative_to(ROOT).as_posix()}`",
        f"- Prediction: `{TEST_PREDICTION_PATH.relative_to(ROOT).as_posix()}`",
        f"- Log: `{log_path.relative_to(ROOT).as_posix()}`", "",
        "## Training And Search", "",
        f"- Base checkpoint: 700 trees; extended training limit: {MAX_ITERATION} trees.",
        f"- Train sample: {int(training.get('train_rows', 0)):,}; sampled Valid: {int(training.get('sampled_valid_rows', 0)):,}.",
        f"- Sampling: {float(training.get('sampling_seconds', 0)):.1f}s; continuation training: {float(training.get('continuation_training_seconds', 0)):.1f}s.",
        f"- Training peak RAM: {float(training.get('peak_ram_gib', 0)):.2f} GiB.",
        f"- Selected iteration: {best_iteration}; search status: `{metadata['search_status']}`.",
        f"- Selected Valid WAPE: {best.wape:.6f}%.",
        f"- {search_note}", "",
        "## Candidate Valid WAPE", "",
        *candidate_lines, "",
        f"Curve: `{CURVE_PATH.relative_to(ROOT).as_posix()}`", "",
        "## Active-Store Test", "",
        *_metric_table(test_metrics), "",
        "### Zero-Sales Diagnostics", "",
        f"- Mean prediction: {zero['mean_prediction']:.6f}.",
        f"- Prediction > 0.5: {zero['prediction_gt_0_5_rate']:.4%}.",
        f"- Prediction > 1: {zero['prediction_gt_1_rate']:.4%}.", "",
        "## Fair Old/New Comparison", "",
        "Both models are evaluated only on the `month + site_no + item_id` Test intersection. Targets are required to match exactly.", "",
        *fair_lines, "",
        "## Scope Notes", "",
        "- Existing LightGBM V2 models, predictions, and reports were not overwritten.",
        "- MC remains a post-processing evaluation using the provisional five-level project rule; it is not treated as a confirmed Wenxuan business definition.",
        "- No 2M, two-stage, independent MC, trend, random forest, MLP, or baseline training is performed by this script.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    _required_inputs()
    logger, log_path = setup_logging()
    started = time.perf_counter()
    booster, extended_metadata = continue_to_2000(logger)
    selected, final_metadata, candidates = select_on_complete_valid(booster, extended_metadata, logger)
    test_payload, _ = evaluate_test_once(selected, final_metadata, logger)
    if "metrics" not in test_payload:
        test_payload = test_payload
    fair = _fair_comparison()
    write_report(candidates, final_metadata, test_payload, fair, log_path)
    elapsed = time.perf_counter() - started
    logger.info("Active-store WAPE 1M experiment completed in %.1fs", elapsed)
    return {
        "best_iteration": int(final_metadata["best_iteration"]),
        "search_status": final_metadata["search_status"],
        "valid_wape": float(candidates.loc[candidates["iteration"].eq(final_metadata["best_iteration"]), "wape"].iloc[0]),
        "test_wape": float(test_payload["metrics"]["overall"]["wape"]),
        "elapsed_seconds": elapsed,
        "model": str(FINAL_MODEL_PATH),
        "report": str(REPORT_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend and select LightGBM V2 active-store 1M by complete Valid WAPE")
    parser.parse_args()
    result = run()
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
