from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # Load the Pandas/PyArrow native stack before LightGBM on Python 3.13.
import pyarrow.parquet as pq
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import DEMAND_BUCKETS, LongTailStreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import load_feature_config, project_path  # noqa: E402
from src.models.lightgbm_model import load_model_bundle, save_model_bundle, train_booster  # noqa: E402
from src.models.lightgbm_objectives import (  # noqa: E402
    ObjectiveSpec,
    best_ranked_candidate,
    candidate_specs,
    inverse_prediction,
    select_business_candidate,
    transform_training_target,
)


HORIZONS = ("1m", "2m")
V1_PROTECTED_FILES = (
    "models/final/lightgbm_1m.txt",
    "models/final/lightgbm_2m.txt",
    "reports/lightgbm_model_report.md",
    "reports/lightgbm_feature_importance_1m.csv",
    "reports/lightgbm_feature_importance_2m.csv",
    "reports/storage_manifest.md",
    "data/outputs/lightgbm_test_predictions.parquet",
    "logs/lightgbm/formal_run_summary.json",
)


def load_v1_pipeline():
    path = ROOT / "scripts/10_train_lightgbm.py"
    spec = importlib.util.spec_from_file_location("wenxuan_v1_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V1 pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    log_dir = ROOT / "logs/lightgbm/v2_pilot"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"objective_pilot_{run_id}.log"
    logger = logging.getLogger("wenxuan_lightgbm_v2_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def protected_snapshot() -> dict[str, dict[str, int | str]]:
    snapshot = {}
    for relative in V1_PROTECTED_FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"Required V1 artifact is missing: {path}")
        snapshot[relative] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return snapshot


def assert_protected_unchanged(before: dict) -> None:
    after = protected_snapshot()
    if before != after:
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise RuntimeError(f"V1 protected artifacts changed: {changed}")


def control_spec() -> ObjectiveSpec:
    return ObjectiveSpec("v1_log_l1", "V1 Log-L1", "log1p", {})


def train_candidates_for_horizon(
    pipeline,
    dataset_path: Path,
    horizon: str,
    config: dict,
    bucket_counts: dict,
    category_maps: dict,
    specs: dict[str, ObjectiveSpec],
    checkpoint_dir: Path,
    logger: logging.Logger,
) -> dict:
    sampling_guard = pipeline.MemoryGuard(f"v2-pilot-sampling-{horizon}", logger)
    settings = config["lightgbm_training"]
    train_rates = pipeline.scaled_rates(settings["sampling_rates"], bucket_counts["train"][horizon], 1_000_000)
    valid_rates = pipeline.scaled_rates(settings["valid_sampling_rates"], bucket_counts["valid"][horizon], 200_000)
    features = pipeline.horizon_features(config, horizon)
    base_month = config["time_feature_settings"]["base_month"]
    sampling_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        dataset_path,
        horizon,
        {"train": train_rates, "valid": valid_rates},
        category_maps,
        features,
        base_month,
        sampling_guard,
        logger,
        {"train": 1_000_000, "valid": 200_000},
        None,
        True,
    )
    sampling_seconds = time.perf_counter() - sampling_started
    raw_train = np.expm1(samples["train"]["y"].astype("float64"))
    raw_valid = np.expm1(samples["valid"]["y"].astype("float64"))
    results = {}
    for name, objective in specs.items():
        guard = pipeline.MemoryGuard(f"v2-pilot-{name}-{horizon}", logger)
        guard.check("candidate_start")
        train_y = transform_training_target(raw_train, objective)
        valid_y = transform_training_target(raw_valid, objective)
        estimated = int((len(train_y) + len(valid_y)) * (len(features) * 3 + 64))
        guard.ensure_capacity(estimated, "before_dataset_construction")
        logger.info(
            "Training %s %s train=%s valid=%s objective=%s",
            name, horizon, f"{len(train_y):,}", f"{len(valid_y):,}", objective.params["objective"],
        )
        started = time.perf_counter()
        booster, evaluations = train_booster(
            samples["train"]["x"], train_y, samples["train"]["weight"],
            samples["valid"]["x"], valid_y, samples["valid"]["weight"],
            categorical_features=pipeline.CATEGORY_FEATURES,
            params=objective.params,
            num_boost_round=600,
            early_stopping_rounds=100,
            callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)],
            construction_check=lambda label: guard.check(label),
        )
        training_seconds = time.perf_counter() - started
        sample_prediction = inverse_prediction(
            booster.predict(samples["valid"]["x"], num_iteration=booster.best_iteration), objective
        )
        sample_metrics = LongTailStreamingMetrics()
        sample_metrics.update(raw_valid, sample_prediction)
        model_path = checkpoint_dir / f"{name}_{horizon}.txt"
        metadata = {
            "experiment": "lightgbm_v2_objective_pilot",
            "candidate": name,
            "display_name": objective.display_name,
            "target_transform": objective.target_transform,
            "horizon": horizon,
            "feature_names": features,
            "categorical_features": pipeline.CATEGORY_FEATURES,
            "category_maps": category_maps,
            "time_base": base_month,
            "params": objective.params,
            "best_iteration": int(booster.best_iteration),
            "sampling_rates": train_rates,
            "valid_sampling_rates": valid_rates,
            "lightgbm_version": lgb.__version__,
            "python_version": platform.python_version(),
        }
        save_model_bundle(booster, model_path, metadata)
        loaded, loaded_metadata = load_model_bundle(model_path)
        check_x = samples["valid"]["x"].iloc[:1000]
        original = inverse_prediction(booster.predict(check_x, num_iteration=booster.best_iteration), objective)
        reloaded = inverse_prediction(loaded.predict(check_x, num_iteration=loaded_metadata["best_iteration"]), objective)
        np.testing.assert_allclose(original, reloaded, rtol=1e-7, atol=1e-8)
        results[name] = {
            "display_name": objective.display_name,
            "model_path": str(model_path),
            "best_iteration": int(booster.best_iteration),
            "sampling_seconds": sampling_seconds,
            "training_seconds": training_seconds,
            "peak_memory_bytes": guard.peak,
            "memory_limit_bytes": guard.limit,
            "sample_metrics": sample_metrics.compute(),
            "evaluations": evaluations,
            "params": objective.params,
            "train_rows": len(train_y),
            "valid_rows": len(valid_y),
        }
        logger.info(
            "Completed %s %s best_iteration=%d training=%.1fs peak=%.2fGiB sample_bias=%.2f%%",
            name, horizon, booster.best_iteration, training_seconds, guard.peak / 2**30,
            results[name]["sample_metrics"]["overall"]["total_bias_rate"],
        )
        del booster, loaded, train_y, valid_y, sample_prediction, sample_metrics, check_x, original, reloaded
        gc.collect()
        guard.check("candidate_released")
    distribution = {
        "train": samples["train"]["distribution"],
        "valid": samples["valid"]["distribution"],
    }
    sample_peak = sampling_guard.peak
    del samples, raw_train, raw_valid
    gc.collect()
    sampling_guard.check("horizon_samples_released")
    return {
        "candidates": results,
        "sampling_seconds": sampling_seconds,
        "sampling_peak_memory_bytes": sample_peak,
        "released_rss_bytes": pipeline.process_rss(),
        "distribution": distribution,
    }


def evaluate_complete_valid(
    pipeline,
    dataset_path: Path,
    model_paths: dict[str, dict[str, Path]],
    specs: dict[str, ObjectiveSpec],
    logger: logging.Logger,
) -> tuple[dict, float, int]:
    guard = pipeline.MemoryGuard("v2-pilot-full-valid", logger)
    bundles = {h: {name: load_model_bundle(path) for name, path in paths.items()} for h, paths in model_paths.items()}
    metrics = {
        horizon: {name: LongTailStreamingMetrics() for name in paths}
        for horizon, paths in model_paths.items()
    }
    runtime = set(load_feature_config().get("runtime_time_features", []))
    physical = set(pipeline.IDENTIFIER_COLUMNS)
    for horizon_bundles in bundles.values():
        for _, metadata in horizon_bundles.values():
            physical.update(column for column in metadata["feature_names"] if column not in runtime)
    parquet = pq.ParquetFile(dataset_path)
    started = time.perf_counter()
    valid_rows = 0
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=list(physical)).to_pandas()
            frame = frame[frame["split"] == "valid"]
            if frame.empty:
                continue
            valid_rows += len(frame)
            for horizon in HORIZONS:
                target = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                reference_metadata = next(iter(bundles[horizon].values()))[1]
                x = pipeline.prepare_evaluation_frame(frame, reference_metadata)
                for name, (booster, metadata) in bundles[horizon].items():
                    if metadata["feature_names"] != reference_metadata["feature_names"]:
                        raise RuntimeError(f"Feature order mismatch in {name} {horizon}")
                    objective = control_spec() if name == "v1_log_l1" else specs[name]
                    prediction = inverse_prediction(
                        booster.predict(x, num_iteration=metadata["best_iteration"]), objective
                    )
                    metrics[horizon][name].update(target, prediction)
                del x, target
            del frame
            if (row_group + 1) % 10 == 0:
                guard.check("full_valid_evaluation")
                logger.info("Full valid row groups %d/%d rows=%s", row_group + 1, parquet.num_row_groups, f"{valid_rows:,}")
    finally:
        parquet.close()
    computed = {
        horizon: {name: accumulator.compute() for name, accumulator in candidates.items()}
        for horizon, candidates in metrics.items()
    }
    return computed, time.perf_counter() - started, guard.peak


def fmt(value, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def render_report(summary: dict, destination: Path) -> None:
    display = summary["display_names"]
    lines = [
        "# LightGBM V2 目标函数 Pilot 报告",
        "",
        "本实验只比较目标函数，不修改特征、时间切分、确定性分层采样、逆概率权重或类别编码。V1 正式模型及结果未被覆盖。",
        "",
        "## 实验设置",
        "",
        "- V1 对照：`log1p(target) + regression_l1`。",
        "- Tweedie：原始非负目标，`variance_power = 1.2 / 1.4 / 1.6`。",
        "- Log-L2：`log1p(target) + regression_l2`，预测后 `expm1`。",
        "- 每个 horizon 约 100 万 train、20 万 early-stopping valid；完整 valid 不采样、按 row group 流式评价。",
        "- 晋级护栏：零销量误报与头部 WAPE 不得明显恶化，整体 MAE 相对 V1 恶化超过 35% 视为不可接受。",
        "",
        "## Pilot 训练",
        "",
        "| Horizon | 候选 | Train | Early Valid | 最佳轮次 | 训练耗时(秒) | 峰值RAM(GiB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for name, result in summary["training"][horizon]["candidates"].items():
            lines.append(
                f"| {horizon} | {display[name]} | {result['train_rows']:,} | {result['valid_rows']:,} | "
                f"{result['best_iteration']} | {result['training_seconds']:.1f} | {result['peak_memory_bytes'] / 2**30:.2f} |"
            )
    lines.extend([
        "",
        "## 完整 Valid 整体指标",
        "",
        "| Horizon | 候选 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    ordered = ["v1_log_l1", "tweedie_1_2", "tweedie_1_4", "tweedie_1_6", "log_l2"]
    for horizon in HORIZONS:
        for name in ordered:
            value = summary["full_valid_metrics"][horizon][name]["overall"]
            lines.append(
                f"| {horizon} | {display[name]} | {value['count']:,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
                f"{fmt(value['smape'], 2)}% | {fmt(value['wape'], 2)}% | {value['target_sum']:.0f} | "
                f"{value['prediction_sum']:.0f} | {fmt(value['total_bias_rate'], 2)}% |"
            )
    lines.extend([
        "",
        "## 零销量与关键层级",
        "",
        "| Horizon | 候选 | 零均值预测 | 零>0.5 | 零>1 | Nonzero WAPE | >=5 WAPE | >=20 WAPE | 1 MAE/WAPE | 2-5 MAE/WAPE | 5-20 MAE/WAPE | 20+ MAE/WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        for name in ordered:
            value = summary["full_valid_metrics"][horizon][name]
            segment = lambda key: f"{fmt(value[key]['mae'])}/{fmt(value[key]['wape'], 2)}%"
            lines.append(
                f"| {horizon} | {display[name]} | {fmt(value['0']['mean_prediction'])} | "
                f"{fmt(100 * value['0']['prediction_gt_0_5_rate'], 2)}% | {fmt(100 * value['0']['prediction_gt_1_rate'], 2)}% | "
                f"{fmt(value['nonzero']['wape'], 2)}% | {fmt(value['ge_5']['wape'], 2)}% | {fmt(value['ge_20']['wape'], 2)}% | "
                f"{segment('1')} | {segment('2-5')} | {segment('5-20')} | {segment('20+')} |"
            )
    lines.extend(["", "## 选择结果", ""])
    for horizon in HORIZONS:
        power = summary["selection"][horizon]["best_tweedie"]
        recommendation = summary["selection"][horizon]["final"]["selected"]
        lines.append(f"- **{horizon} 最佳 Tweedie power**：{display.get(power, '无法排序')}（按业务优先级排序，不代表已通过晋级护栏）。")
        lines.append(f"- **{horizon} 正式训练建议**：{display.get(recommendation, '不建议晋级任何 V2 候选')}。")
        if recommendation:
            candidate = summary["full_valid_metrics"][horizon][recommendation]
            control = summary["full_valid_metrics"][horizon]["v1_log_l1"]
            lines.append(
                f"  总量偏差由 {control['overall']['total_bias_rate']:.2f}% 变为 {candidate['overall']['total_bias_rate']:.2f}%；"
                f"nonzero WAPE 由 {control['nonzero']['wape']:.2f}% 变为 {candidate['nonzero']['wape']:.2f}%；"
                f"20+ WAPE 由 {control['20+']['wape']:.2f}% 变为 {candidate['20+']['wape']:.2f}%。"
            )
    lines.extend([
        "",
        "## 运行与保护",
        "",
        f"- 完整 valid 样本量：{summary['valid_rows']:,}。",
        f"- 完整 valid 流式评价耗时：{summary['full_valid_seconds']:.1f} 秒。",
        f"- 评价峰值 RAM：{summary['full_valid_peak_memory_bytes'] / 2**30:.2f} GiB。",
        f"- Python：{summary['python_version']}；LightGBM：{summary['lightgbm_version']}。",
        "- V1 保护文件在实验前后 SHA-256 完全一致。",
        "- 本次只完成 pilot，没有执行正式训练，也没有写入新的大型训练数据文件。",
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded LightGBM V2 objective pilots.")
    parser.add_argument("--dataset", default="data/processed/model_dataset_monthly.parquet")
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logging(run_id)
    checkpoint_dir = ROOT / "models/checkpoints/lightgbm_v2_pilot" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    summary_path = ROOT / "logs/lightgbm/v2_pilot" / f"objective_pilot_{run_id}.json"
    report_path = ROOT / "reports/lightgbm_v2_objective_pilot_report.md"
    dataset_path = project_path(args.dataset)
    pipeline = load_v1_pipeline()
    protected_before = protected_snapshot()
    public_signature = {"size": dataset_path.stat().st_size, "mtime_ns": dataset_path.stat().st_mtime_ns}
    config = load_feature_config()
    base_params = pipeline.training_params(config)
    specs = candidate_specs(base_params)
    display_names = {"v1_log_l1": "V1 Log-L1", **{name: spec.display_name for name, spec in specs.items()}}
    logger.info("Starting V2 objective pilot Python=%s LightGBM=%s", platform.python_version(), lgb.__version__)
    logger.info("V1 protected artifact hashes recorded; checkpoint_dir=%s", checkpoint_dir)
    bucket_counts, category_maps = pipeline.load_or_build_scan_metadata(dataset_path, logger)
    training = {}
    try:
        for horizon in HORIZONS:
            training[horizon] = train_candidates_for_horizon(
                pipeline, dataset_path, horizon, config, bucket_counts, category_maps, specs, checkpoint_dir, logger
            )
        model_paths = {}
        for horizon in HORIZONS:
            model_paths[horizon] = {"v1_log_l1": ROOT / f"models/final/lightgbm_{horizon}.txt"}
            model_paths[horizon].update({
                name: Path(result["model_path"])
                for name, result in training[horizon]["candidates"].items()
            })
        full_metrics, evaluation_seconds, evaluation_peak = evaluate_complete_valid(
            pipeline, dataset_path, model_paths, specs, logger
        )
        expected_valid_rows = int(bucket_counts["valid"]["1m"].total())
        if any(full_metrics[h][name]["overall"]["count"] != expected_valid_rows for h in HORIZONS for name in model_paths[h]):
            raise RuntimeError("Complete-valid evaluation row count mismatch")
        selection = {}
        tweedie_names = ["tweedie_1_2", "tweedie_1_4", "tweedie_1_6"]
        for horizon in HORIZONS:
            best_tweedie_result = select_business_candidate(full_metrics[horizon], tweedie_names)
            best_tweedie = best_ranked_candidate(best_tweedie_result)
            finalists = [name for name in (best_tweedie, "log_l2") if name]
            final_result = select_business_candidate(full_metrics[horizon], finalists)
            selection[horizon] = {
                "best_tweedie": best_tweedie,
                "tweedie": best_tweedie_result,
                "final": final_result,
            }
        assert_protected_unchanged(protected_before)
        if dataset_path.stat().st_size != public_signature["size"] or dataset_path.stat().st_mtime_ns != public_signature["mtime_ns"]:
            raise RuntimeError("Public model dataset changed during V2 pilot")
        summary = {
            "run_id": run_id,
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__,
            "dataset_signature": public_signature,
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "display_names": display_names,
            "training": training,
            "full_valid_metrics": full_metrics,
            "full_valid_seconds": evaluation_seconds,
            "full_valid_peak_memory_bytes": evaluation_peak,
            "valid_rows": expected_valid_rows,
            "selection": selection,
            "v1_protected_artifacts": protected_before,
            "v1_unchanged": True,
            "formal_training_started": False,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        render_report(summary, report_path)
        logger.info("V2 objective pilot completed; report=%s", report_path)
    except Exception:
        logger.exception("V2 objective pilot failed; V1 formal artifacts must remain unchanged")
        assert_protected_unchanged(protected_before)
        raise


if __name__ == "__main__":
    main()
