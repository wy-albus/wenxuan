from __future__ import annotations

import gc
import importlib.util
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.mc_metrics import TrendAccumulator  # noqa: E402
from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402
from src.evaluation.two_stage_gate import GateATradeoffAccumulator  # noqa: E402
from src.features.preprocessing import load_feature_config  # noqa: E402
from src.models.lightgbm_model import (  # noqa: E402
    MODEL_METADATA_SENTINEL,
    load_model_bundle,
    save_model_bundle,
)
from src.models.two_stage_model import (  # noqa: E402
    combine_predictions,
    select_component_candidates,
    select_pair_candidates,
    weighted_binary_component_metrics,
    weighted_regression_component_metrics,
)


DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
REFERENCE_MODEL_PATH = ROOT / "models/final/lightgbm_v2_logl2_active_store_wape_1m.txt"
FINAL_CLASSIFIER_PATH = ROOT / "models/final/two_stage_classifier_active_store_1m.txt"
FINAL_REGRESSOR_PATH = ROOT / "models/final/two_stage_regressor_active_store_1m.txt"
CHECKPOINT_DIR = ROOT / "models/checkpoints/two_stage_active_store_1m"
CLASSIFIER_CHECKPOINT = CHECKPOINT_DIR / "classifier_full.txt"
REGRESSOR_CHECKPOINT = CHECKPOINT_DIR / "regressor_full.txt"
COMPONENT_PATH = ROOT / "reports/two_stage_active_store_1m_component_candidates.csv"
SAMPLED_PAIR_PATH = ROOT / "reports/two_stage_active_store_1m_sampled_pair_candidates.csv"
COMPLETE_PAIR_PATH = ROOT / "reports/two_stage_active_store_1m_complete_valid_pairs.csv"
GATE_PATH = ROOT / "reports/two_stage_active_store_1m_gate_tradeoff.csv"
REPORT_PATH = ROOT / "reports/two_stage_active_store_1m_training_report.md"
RUNTIME_PATH = ROOT / "reports/two_stage_active_store_1m_runtime.json"
LOG_DIR = ROOT / "logs/lightgbm/two_stage_active_store_1m"
TEMP_DIR = ROOT / "data/temp/two_stage_active_store_1m"

MAX_COMPONENT_CANDIDATES = 4
MAX_COMPLETE_VALID_PAIRS = 3
MAX_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 150
EXPECTED_COMPLETE_VALID_ROWS = 18_563_573


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("wenxuan_two_stage_active_store_1m")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    shared = logging.getLogger("wenxuan_active_store_experiment")
    shared.handlers = logger.handlers
    shared.setLevel(logging.INFO)
    return logger, path


def refuse_existing_outputs() -> None:
    paths = [
        FINAL_CLASSIFIER_PATH, FINAL_REGRESSOR_PATH, COMPONENT_PATH, SAMPLED_PAIR_PATH,
        COMPLETE_PAIR_PATH, GATE_PATH, REPORT_PATH, RUNTIME_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite formal Active-Store 1M outputs: " + ", ".join(existing))


def read_model_metadata(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if MODEL_METADATA_SENTINEL not in text:
        raise ValueError(f"Model metadata not found in {path}")
    return json.loads(text.split(MODEL_METADATA_SENTINEL, maxsplit=1)[1].strip())


def candidate_iterations(booster: lgb.Booster, proxy: int) -> list[int]:
    maximum = int(booster.current_iteration())
    values = {1, maximum, min(maximum, max(1, int(proxy)))}
    values.update(range(50, maximum + 1, 100))
    for offset in (-100, -50, 50, 100):
        values.add(min(maximum, max(1, int(proxy) + offset)))
    return sorted(values)


def gate_thresholds() -> np.ndarray:
    required = np.array(
        [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
         0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        dtype="float64",
    )
    return np.unique(np.concatenate([np.round(np.arange(0.00, 0.951, 0.01), 2), required]))


def iter_complete_valid(columns: list[str]):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=2")
    connection.execute("PRAGMA memory_limit='512MB'")
    connection.execute(f"PRAGMA temp_directory='{TEMP_DIR.resolve().as_posix()}'")
    source = DATASET_PATH.resolve().as_posix().replace("'", "''")
    projection = ",".join(f'"{column}"' for column in dict.fromkeys(columns))
    try:
        connection.execute(
            f"SELECT {projection} FROM read_parquet('{source}') "
            "WHERE split='valid' AND target_available_1m=1"
        )
        while True:
            frame = connection.fetch_df_chunk(vectors_per_chunk=64)
            if frame.empty:
                break
            yield frame
    finally:
        connection.close()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


def collect_train_valid_samples_duckdb(
    pipeline,
    experiment,
    config: dict,
    category_maps: dict,
    guard,
    logger: logging.Logger,
) -> dict[str, dict]:
    """Stream the formal deterministic sample without invoking PyArrow row-group reads."""
    features = experiment._horizon_features(config, "1m")
    allowed = set(features) | set(pipeline.IDENTIFIER_COLUMNS) | {
        "gds_ctgry_3_lvel", *experiment.CATEGORY_FEATURES,
    }
    columns = [
        column for column in pipeline.physical_source_columns(config) if column in allowed
    ]
    target_column = "future_qty_1m"
    settings = config["lightgbm_training"]
    rates = {
        "train": settings["sampling_rates"],
        "valid": settings["valid_sampling_rates"],
    }
    states = {
        split: {
            "x_parts": [], "y_parts": [], "weight_parts": [], "rows": 0,
            "before": pipeline.DistributionStats(),
            "after_raw": pipeline.DistributionStats(),
            "after_weighted": pipeline.DistributionStats(),
        }
        for split in ("train", "valid")
    }

    sampling_temp = TEMP_DIR / "sampling"
    sampling_temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='1GB'")
    connection.execute(f"PRAGMA temp_directory='{sampling_temp.resolve().as_posix()}'")
    source = DATASET_PATH.resolve().as_posix().replace("'", "''")
    projection = ",".join(f'"{column}"' for column in columns)
    try:
        connection.execute(
            f"SELECT {projection} FROM read_parquet('{source}') "
            "WHERE split IN ('train', 'valid') AND target_available_1m=1"
        )
        for chunk_number in range(1, 1_000_000):
            frame = connection.fetch_df_chunk(vectors_per_chunk=32)
            if frame.empty:
                break
            for split in ("train", "valid"):
                split_frame = frame.loc[frame["split"].eq(split)]
                if split_frame.empty:
                    continue
                target = clip_target(split_frame[target_column].to_numpy())
                state = states[split]
                state["before"].update(split_frame, target)
                sampled, raw_weights, _ = pipeline.sample_by_target(
                    split_frame, "1m", rates[split], seed=42
                )
                if sampled.empty:
                    continue
                sampled_target = clip_target(sampled[target_column].to_numpy())
                state["after_raw"].update(sampled, sampled_target)
                state["after_weighted"].update(sampled, sampled_target, raw_weights)
                state["x_parts"].append(
                    pipeline.prepare_feature_frame(
                        sampled,
                        features,
                        category_maps,
                        config["time_feature_settings"]["base_month"],
                    )
                )
                state["y_parts"].append(np.log1p(sampled_target).astype("float32"))
                state["weight_parts"].append(raw_weights)
                state["rows"] += len(sampled)
            if chunk_number % 20 == 0:
                guard.check("duckdb_sampling")
                logger.info(
                    "DuckDB sampled train=%s valid=%s after %d chunks",
                    f"{states['train']['rows']:,}", f"{states['valid']['rows']:,}", chunk_number,
                )
            del frame
    finally:
        connection.close()
        shutil.rmtree(sampling_temp, ignore_errors=True)

    results = {}
    for split, state in states.items():
        if not state["x_parts"]:
            raise RuntimeError(f"No sampled rows for {split} 1m")
        x = pd.concat(state["x_parts"], ignore_index=True, copy=False)
        y = np.concatenate(state["y_parts"])
        weights = np.concatenate(state["weight_parts"]).astype("float32")
        weights /= weights.mean()
        shifts = {
            "month": pipeline.max_distribution_shift(
                state["before"], state["after_weighted"], "month_shares"
            ),
            "site": pipeline.max_distribution_shift(
                state["before"], state["after_weighted"], "site_shares"
            ),
            "category": pipeline.max_distribution_shift(
                state["before"], state["after_weighted"], "category_shares"
            ),
        }
        if max(shifts.values()) > 0.02:
            raise RuntimeError(
                f"{split} weighted sampling distribution shift exceeds 2 percentage points: {shifts}"
            )
        results[split] = {
            "x": x, "y": y, "weight": weights,
            "distribution": {
                "before": state["before"].summary(),
                "sample_raw": state["after_raw"].summary(),
                "sample_weighted": state["after_weighted"].summary(),
                "max_shifts": shifts,
            },
        }
    guard.check("duckdb_sample_concat")
    return results


def component_metric_rows(
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    classifier_iterations: list[int],
    regressor_iterations: list[int],
    valid: dict,
    valid_target: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    binary_target = (valid_target > 0).astype("int32")
    classifier_rows = []
    for iteration in classifier_iterations:
        probability = np.clip(classifier.predict(valid["x"], num_iteration=iteration), 0.0, 1.0)
        classifier_rows.append({
            "component": "classifier", "iteration": iteration,
            **weighted_binary_component_metrics(binary_target, probability, valid["weight"]),
        })

    positive = valid_target > 0
    positive_x = valid["x"].loc[positive]
    positive_target = valid_target[positive]
    positive_weight = valid["weight"][positive]
    regressor_rows = []
    for iteration in regressor_iterations:
        prediction = np.clip(regressor.predict(positive_x, num_iteration=iteration), 0.0, None)
        regressor_rows.append({
            "component": "regressor", "iteration": iteration,
            **weighted_regression_component_metrics(positive_target, prediction, positive_weight),
        })
    return classifier_rows, regressor_rows


def sampled_pair_rows(
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    classifier_iterations: list[int],
    regressor_iterations: list[int],
    valid: dict,
    valid_target: np.ndarray,
) -> list[dict]:
    probabilities = {
        iteration: np.clip(classifier.predict(valid["x"], num_iteration=iteration), 0.0, 1.0)
        for iteration in classifier_iterations
    }
    conditionals = {
        iteration: np.clip(regressor.predict(valid["x"], num_iteration=iteration), 0.0, None)
        for iteration in regressor_iterations
    }
    rows = []
    for classifier_iteration in classifier_iterations:
        for regressor_iteration in regressor_iterations:
            prediction = combine_predictions(
                probabilities[classifier_iteration], conditionals[regressor_iteration]
            )
            metrics = weighted_regression_component_metrics(
                valid_target, prediction, valid["weight"]
            )
            rows.append({
                "iteration": (classifier_iteration, regressor_iteration),
                "classifier_iteration": classifier_iteration,
                "regressor_iteration": regressor_iteration,
                **metrics,
            })
    return rows


def evaluate_complete_valid(
    pipeline,
    experiment,
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    metadata: dict,
    pairs: list[tuple[int, int]],
    guard,
    logger: logging.Logger,
) -> tuple[list[dict], dict, dict]:
    config = load_feature_config()
    columns = experiment._physical_eval_columns(config, metadata["feature_names"], "1m")
    metrics = {pair: LongTailStreamingMetrics() for pair in pairs}
    trends = {pair: TrendAccumulator(flat_tolerance=0.0) for pair in pairs}
    gates = {pair: GateATradeoffAccumulator(gate_thresholds()) for pair in pairs}
    classifier_iterations = sorted({pair[0] for pair in pairs})
    regressor_iterations = sorted({pair[1] for pair in pairs})
    rows = 0
    started = time.perf_counter()
    for number, frame in enumerate(iter_complete_valid(columns), start=1):
        target = clip_target(frame["future_qty_1m"].to_numpy())
        current = clip_target(frame["total_qty"].to_numpy())
        x = experiment._prepare_eval(pipeline, frame, metadata)
        probabilities = {
            iteration: np.clip(classifier.predict(x, num_iteration=iteration), 0.0, 1.0)
            for iteration in classifier_iterations
        }
        conditionals = {
            iteration: np.clip(regressor.predict(x, num_iteration=iteration), 0.0, None)
            for iteration in regressor_iterations
        }
        for pair in pairs:
            prediction = combine_predictions(probabilities[pair[0]], conditionals[pair[1]])
            metrics[pair].update(target, prediction)
            trends[pair].update(current, target, prediction)
            gates[pair].update(target, probabilities[pair[0]], prediction)
        rows += len(frame)
        if number % 10 == 0:
            guard.check("complete_valid")
            logger.info("Complete Valid rows=%s", f"{rows:,}")
        del frame, x, target, current, probabilities, conditionals
        gc.collect()
    if rows != EXPECTED_COMPLETE_VALID_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETE_VALID_ROWS:,} Complete Valid rows, got {rows:,}"
        )

    pair_rows = []
    for pair in pairs:
        quantity = metrics[pair].compute()
        trend = trends[pair].compute()
        overall = quantity["overall"]
        pair_rows.append({
            "iteration": pair,
            "classifier_iteration": pair[0],
            "regressor_iteration": pair[1],
            "count": overall["count"],
            "wape": overall["wape"],
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "target_sum": overall["target_sum"],
            "prediction_sum": overall["prediction_sum"],
            "total_bias": overall["total_bias_rate"],
            "nonzero_wape": quantity["nonzero"]["wape"],
            "5_20_wape": quantity["5-20"]["wape"],
            "20plus_wape": quantity["20+"]["wape"],
            "trend_macro_f1": trend["macro_f1"],
        })
    runtime = {"rows": rows, "evaluation_seconds": time.perf_counter() - started}
    return pair_rows, metrics, {"gates": gates, "runtime": runtime}


def select_complete_pair(rows: list[dict]) -> dict:
    return min(
        rows,
        key=lambda row: (
            float(row["wape"]), abs(float(row["total_bias"])),
            -float(row["trend_macro_f1"]),
            int(row["classifier_iteration"]) + int(row["regressor_iteration"]),
        ),
    )


def model_metadata(
    kind: str,
    features: list[str],
    category_maps: dict,
    params: dict,
    best_iteration: int,
    train_rows: int,
    valid_rows: int,
    positive_train_rows: int,
    positive_valid_rows: int,
    complete_pair_rows: list[dict],
) -> dict:
    return {
        "experiment": "two_stage_active_store_1m_v2",
        "model_kind": kind,
        "horizon": "1m",
        "feature_names": features,
        "categorical_features": [
            "site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"
        ],
        "category_maps": category_maps,
        "time_base": "2023-01",
        "params": params,
        "best_iteration": int(best_iteration),
        "selected_iteration": int(best_iteration),
        "selection_split": "complete_valid_original_scale",
        "selection_metric": "overall_wape",
        "combination": "p_sale * conditional_quantity",
        "train_rows": int(train_rows),
        "valid_rows": int(valid_rows),
        "positive_train_rows": int(positive_train_rows),
        "positive_valid_rows": int(positive_valid_rows),
        "complete_valid_pair_candidates": complete_pair_rows,
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
    }


def write_report(
    component: pd.DataFrame,
    sampled_pairs: pd.DataFrame,
    complete_pairs: pd.DataFrame,
    gate: pd.DataFrame,
    selected: dict,
    original_metrics: dict,
    runtime: dict,
) -> None:
    quantity = original_metrics
    zero = quantity["0"]
    attention = gate[
        gate["threshold"].isin([0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
    ]
    lines = [
        "# Active-Store Two-stage 1M 训练与 Valid Gate-A 报告", "",
        "## 正式范围", "",
        "- Dataset: `data/processed/model_dataset_monthly_active_store.parquet`.",
        "- Components: binary classifier and positive-sales Tweedie regressor.",
        "- Combination: `p_sale * conditional_qty`.",
        "- 组件候选先在 Sampled Valid 筛选，Complete Valid 只评价 3 个组合。",
        "- 正式组合仅按 Complete Valid 原始销量尺度 Overall WAPE 选择；接近时依次比较绝对总量偏差、趋势 Macro-F1 和轮次。",
        "- Gate-A 在组合冻结后评价；本报告不自动选定最终 tau，也未访问留出评估集。", "",
        "## 训练资源", "",
        f"- Features: {runtime['feature_count']}.",
        f"- Classifier train/valid: {runtime['train_rows']:,} / {runtime['valid_rows']:,}.",
        f"- Regressor positive train/valid: {runtime['positive_train_rows']:,} / {runtime['positive_valid_rows']:,}.",
        f"- Sampling: {runtime['sampling_seconds']:.1f}s; classifier training: {runtime['classifier_training_seconds']:.1f}s; regressor training: {runtime['regressor_training_seconds']:.1f}s.",
        f"- Complete Valid inference: {runtime['complete_valid_seconds']:.1f}s.",
        f"- Peak RAM: {runtime['peak_ram_gib']:.3f} GiB.", "",
        "## 组件诊断", "",
        "Classifier 记录 PR-AUC、ROC-AUC、Logloss 和 threshold=0.5 Recall；Regressor 仅在真实正销量样本记录 WAPE、MAE、RMSE。完整明细见 `reports/two_stage_active_store_1m_component_candidates.csv`。", "",
        "## 组合选择", "",
        f"最终选择 classifier={int(selected['classifier_iteration'])} 轮，regressor={int(selected['regressor_iteration'])} 轮。",
        f"Complete Valid Overall WAPE={selected['wape']:.6f}%，Total Bias={selected['total_bias']:.6f}%，Trend Macro-F1={100 * selected['trend_macro_f1']:.4f}%。",
        "选择原因是该组合在受控的 3 个 Complete Valid 候选中 Overall WAPE 最低；未使用组件单独损失替代组合选模。", "",
        "## Original Complete Valid 业务指标", "",
        "| Metric | Value |", "|---|---:|",
        f"| Overall WAPE | {quantity['overall']['wape']:.6f}% |",
        f"| Nonzero WAPE | {quantity['nonzero']['wape']:.6f}% |",
        f"| 5-20 WAPE | {quantity['5-20']['wape']:.6f}% |",
        f"| 20+ WAPE | {quantity['20+']['wape']:.6f}% |",
        f"| MAE | {quantity['overall']['mae']:.6f} |",
        f"| RMSE | {quantity['overall']['rmse']:.6f} |",
        f"| True total | {quantity['overall']['target_sum']:.2f} |",
        f"| Predicted total | {quantity['overall']['prediction_sum']:.2f} |",
        f"| Total Bias | {quantity['overall']['total_bias_rate']:.6f}% |",
        f"| Zero sample count | {int(zero['count']):,} |",
        f"| Zero mean prediction | {zero['mean_prediction']:.6f} |",
        f"| Zero predicted total | {zero['prediction_sum']:.2f} |",
        f"| Zero > 0.5 | {100 * zero['prediction_gt_0_5_rate']:.6f}% |",
        f"| Zero > 1 | {100 * zero['prediction_gt_1_rate']:.6f}% |", "",
        "## Gate-A 值得关注的观察点", "",
        "以下只是便于阅读的观察点，不是自动选择结果。完整 97 个阈值见 `reports/two_stage_active_store_1m_gate_tradeoff.csv`。", "",
        "| tau | Overall WAPE | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | Zero total | Zero >0.5 | Zero >1 | Total Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in attention.iterrows():
        lines.append(
            f"| {row['threshold']:.3f} | {row['overall_wape']:.4f}% | {row['nonzero_wape']:.4f}% | "
            f"{row['5_20_wape']:.4f}% | {row['20plus_wape']:.4f}% | {row['zero_predicted_total']:.2f} | "
            f"{100 * row['zero_prediction_gt_0_5_rate']:.4f}% | {100 * row['zero_prediction_gt_1_rate']:.4f}% | {row['total_bias']:.4f}% |"
        )
    lines.extend([
        "", "## 正式产物", "",
        "- `models/final/two_stage_classifier_active_store_1m.txt`",
        "- `models/final/two_stage_regressor_active_store_1m.txt`",
        "- `reports/two_stage_active_store_1m_complete_valid_pairs.csv`",
        "- `reports/two_stage_active_store_1m_gate_tradeoff.csv`",
        "", "流程在 Complete Valid Gate-A trade-off 后停止。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_storage_manifest() -> None:
    manifest = ROOT / "reports/storage_manifest.md"
    marker = "<!-- TWO_STAGE_ACTIVE_STORE_1M_V2 -->"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else "# Storage Manifest\n"
    existing = existing.split(marker, maxsplit=1)[0].rstrip()
    paths = [
        Path(__file__), FINAL_CLASSIFIER_PATH, FINAL_REGRESSOR_PATH, COMPONENT_PATH,
        SAMPLED_PAIR_PATH, COMPLETE_PAIR_PATH, GATE_PATH, REPORT_PATH, RUNTIME_PATH,
    ]
    lines = [marker, "", "## Active-Store Two-stage 1M V2", "", "| File | Size MiB | Regenerable |", "|---|---:|---|"]
    for path in paths:
        size = path.stat().st_size / 2**20 if path.exists() else 0.0
        regenerable = "No (formal model)" if path in {FINAL_CLASSIFIER_PATH, FINAL_REGRESSOR_PATH} else "Yes"
        lines.append(f"| `{path.relative_to(ROOT).as_posix()}` | {size:.2f} | {regenerable} |")
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(existing + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(manifest)


def main() -> None:
    refuse_existing_outputs()
    logger, log_path = setup_logging()
    started = time.perf_counter()
    pipeline = _load_script("wenxuan_two_stage_active_pipeline", ROOT / "scripts/10_train_lightgbm.py")
    experiment = _load_script("wenxuan_two_stage_active_experiment", ROOT / "scripts/18_train_active_store_experiment.py")
    config = load_feature_config()
    reference = read_model_metadata(REFERENCE_MODEL_PATH)
    features = experiment._horizon_features(config, "1m")
    if reference["feature_names"] != features or reference.get("time_base") != "2023-01":
        raise RuntimeError("Active-Store LightGBM metadata does not match the frozen 1M feature scope")
    category_maps = reference["category_maps"]
    guard = pipeline.MemoryGuard("two-stage-active-store-1m-v2", logger)

    sampling_started = time.perf_counter()
    samples = collect_train_valid_samples_duckdb(
        pipeline, experiment, config, category_maps, guard, logger
    )
    sampling_seconds = time.perf_counter() - sampling_started
    train_target = np.expm1(samples["train"]["y"].astype("float64"))
    valid_target = np.expm1(samples["valid"]["y"].astype("float64"))
    train_rows = len(train_target)
    valid_rows = len(valid_target)
    positive_train = train_target > 0
    positive_valid = valid_target > 0
    positive_train_rows = int(positive_train.sum())
    positive_valid_rows = int(positive_valid.sum())
    base = experiment._base_params(config)

    classifier_params = {**base, "objective": "binary", "metric": "binary_logloss"}
    classifier_started = time.perf_counter()
    classifier, classifier_evaluations = experiment._train_full_booster(
        samples["train"], samples["valid"], positive_train.astype("float32"),
        positive_valid.astype("float32"), classifier_params, experiment.CATEGORY_FEATURES,
        MAX_ROUNDS, guard, EARLY_STOPPING_ROUNDS,
    )
    classifier_training_seconds = time.perf_counter() - classifier_started
    classifier_proxy = experiment._proxy_best_iteration(
        classifier_evaluations, "binary_logloss"
    )

    reg_train = {
        "x": samples["train"]["x"].loc[positive_train].reset_index(drop=True),
        "weight": samples["train"]["weight"][positive_train].copy(),
    }
    reg_valid = {
        "x": samples["valid"]["x"].loc[positive_valid].reset_index(drop=True),
        "weight": samples["valid"]["weight"][positive_valid].copy(),
    }
    reg_train["weight"] /= reg_train["weight"].mean()
    reg_valid["weight"] /= reg_valid["weight"].mean()
    regressor_params = {
        **base, "objective": "tweedie", "metric": "tweedie", "tweedie_variance_power": 1.4,
    }
    regressor_started = time.perf_counter()
    regressor, regressor_evaluations = experiment._train_full_booster(
        reg_train, reg_valid, train_target[positive_train].astype("float32"),
        valid_target[positive_valid].astype("float32"), regressor_params,
        experiment.CATEGORY_FEATURES, MAX_ROUNDS, guard, EARLY_STOPPING_ROUNDS,
    )
    regressor_training_seconds = time.perf_counter() - regressor_started
    regressor_proxy = experiment._proxy_best_iteration(regressor_evaluations, "tweedie")
    guard.check("components_trained")

    classifier_grid = candidate_iterations(classifier, classifier_proxy)
    regressor_grid = candidate_iterations(regressor, regressor_proxy)
    classifier_rows, regressor_rows = component_metric_rows(
        classifier, regressor, classifier_grid, regressor_grid, samples["valid"], valid_target
    )
    classifier_candidates = select_component_candidates(
        classifier_rows,
        (("logloss", "min"), ("pr_auc", "max"), ("roc_auc", "max"), ("recall", "max")),
        MAX_COMPONENT_CANDIDATES,
    )
    regressor_candidates = select_component_candidates(
        regressor_rows, (("wape", "min"), ("mae", "min"), ("rmse", "min")),
        MAX_COMPONENT_CANDIDATES,
    )
    component_frame = pd.DataFrame([*classifier_rows, *regressor_rows])
    component_frame["selected_for_pair_screen"] = (
        (component_frame["component"].eq("classifier") & component_frame["iteration"].isin(classifier_candidates))
        | (component_frame["component"].eq("regressor") & component_frame["iteration"].isin(regressor_candidates))
    )

    sampled_rows = sampled_pair_rows(
        classifier, regressor, classifier_candidates, regressor_candidates,
        samples["valid"], valid_target,
    )
    complete_candidates = select_pair_candidates(sampled_rows, MAX_COMPLETE_VALID_PAIRS)
    candidate_pairs = [tuple(row["iteration"]) for row in complete_candidates]
    logger.info("Complete Valid candidate pairs: %s", candidate_pairs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_common = {
        "feature_names": features, "categorical_features": experiment.CATEGORY_FEATURES,
        "category_maps": category_maps, "time_base": "2023-01", "horizon": "1m",
        "checkpoint_only": True,
    }
    save_model_bundle(
        classifier, CLASSIFIER_CHECKPOINT,
        {**checkpoint_common, "params": classifier_params, "best_iteration": classifier.current_iteration(),
         "selected_iteration": classifier.current_iteration()},
    )
    save_model_bundle(
        regressor, REGRESSOR_CHECKPOINT,
        {**checkpoint_common, "params": regressor_params, "best_iteration": regressor.current_iteration(),
         "selected_iteration": regressor.current_iteration()},
    )

    check_x = samples["valid"]["x"].iloc[:1000].copy()
    del samples["train"], reg_train, reg_valid, train_target
    gc.collect()
    guard.check("training_matrices_released")

    evaluation_metadata = {
        "feature_names": features, "category_maps": category_maps, "time_base": "2023-01",
    }
    complete_rows, complete_metrics, evaluation = evaluate_complete_valid(
        pipeline, experiment, classifier, regressor, evaluation_metadata,
        candidate_pairs, guard, logger,
    )
    selected = select_complete_pair(complete_rows)
    selected_pair = (
        int(selected["classifier_iteration"]), int(selected["regressor_iteration"])
    )
    gate_frame = pd.DataFrame(evaluation["gates"][selected_pair].rows())
    gate_frame.insert(0, "classifier_iteration", selected_pair[0])
    gate_frame.insert(1, "regressor_iteration", selected_pair[1])
    original_metrics = complete_metrics[selected_pair].compute()

    classifier_metadata = model_metadata(
        "two_stage_classifier", features, category_maps, classifier_params, selected_pair[0],
        train_rows, valid_rows, positive_train_rows, positive_valid_rows, complete_rows,
    )
    regressor_metadata = model_metadata(
        "two_stage_positive_regressor", features, category_maps, regressor_params, selected_pair[1],
        train_rows, valid_rows, positive_train_rows, positive_valid_rows, complete_rows,
    )
    regressor_metadata.update(tweedie_variance_power=1.4, conditional_on_positive=True)
    save_model_bundle(classifier, FINAL_CLASSIFIER_PATH, classifier_metadata)
    save_model_bundle(regressor, FINAL_REGRESSOR_PATH, regressor_metadata)

    before_probability = classifier.predict(check_x, num_iteration=selected_pair[0])
    before_conditional = regressor.predict(check_x, num_iteration=selected_pair[1])
    loaded_classifier, loaded_classifier_meta = load_model_bundle(FINAL_CLASSIFIER_PATH)
    loaded_regressor, loaded_regressor_meta = load_model_bundle(FINAL_REGRESSOR_PATH)
    np.testing.assert_allclose(
        before_probability,
        loaded_classifier.predict(check_x, num_iteration=loaded_classifier_meta["best_iteration"]),
        rtol=1e-7, atol=1e-8,
    )
    np.testing.assert_allclose(
        before_conditional,
        loaded_regressor.predict(check_x, num_iteration=loaded_regressor_meta["best_iteration"]),
        rtol=1e-7, atol=1e-8,
    )

    component_frame.to_csv(COMPONENT_PATH, index=False, encoding="utf-8-sig")
    sampled_frame = pd.DataFrame(sampled_rows).drop(columns="iteration")
    sampled_frame["selected_for_complete_valid"] = [
        (int(row.classifier_iteration), int(row.regressor_iteration)) in candidate_pairs
        for row in sampled_frame.itertuples()
    ]
    sampled_frame.to_csv(SAMPLED_PAIR_PATH, index=False, encoding="utf-8-sig")
    complete_frame = pd.DataFrame(complete_rows).drop(columns="iteration")
    complete_frame["selected_formal_pair"] = (
        (complete_frame["classifier_iteration"].eq(selected_pair[0]))
        & (complete_frame["regressor_iteration"].eq(selected_pair[1]))
    )
    complete_frame.to_csv(COMPLETE_PAIR_PATH, index=False, encoding="utf-8-sig")
    gate_frame.to_csv(GATE_PATH, index=False, encoding="utf-8-sig")

    runtime = {
        "feature_count": len(features), "train_rows": train_rows, "valid_rows": valid_rows,
        "positive_train_rows": positive_train_rows, "positive_valid_rows": positive_valid_rows,
        "sampling_seconds": sampling_seconds,
        "classifier_training_seconds": classifier_training_seconds,
        "regressor_training_seconds": regressor_training_seconds,
        "complete_valid_seconds": evaluation["runtime"]["evaluation_seconds"],
        "complete_valid_rows": evaluation["runtime"]["rows"],
        "peak_ram_gib": guard.peak / (1024 ** 3),
        "total_seconds": time.perf_counter() - started,
        "classifier_trained_iterations": classifier.current_iteration(),
        "regressor_trained_iterations": regressor.current_iteration(),
        "classifier_selected_iteration": selected_pair[0],
        "regressor_selected_iteration": selected_pair[1],
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "classifier_candidates": classifier_candidates,
        "regressor_candidates": regressor_candidates,
        "complete_valid_pairs": candidate_pairs,
        "log_path": str(log_path.relative_to(ROOT)),
    }
    RUNTIME_PATH.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        component_frame, sampled_frame, complete_frame, gate_frame,
        selected, original_metrics, runtime,
    )
    update_storage_manifest()
    logger.info(
        "Active-Store Two-stage 1M complete: classifier=%d regressor=%d WAPE=%.6f%%",
        selected_pair[0], selected_pair[1], selected["wape"],
    )


if __name__ == "__main__":
    main()
