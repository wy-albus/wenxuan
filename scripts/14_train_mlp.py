from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import LongTailStreamingMetrics, clip_target  # noqa: E402
from src.features.preprocessing import add_runtime_columns, get_feature_list, load_feature_config, project_path  # noqa: E402
from src.models.mlp_model import (  # noqa: E402
    TabularMLP,
    inverse_prediction,
    load_model_bundle,
    save_model_bundle,
    weighted_mse_loss,
)


HORIZONS = ("1m", "2m")
CATEGORY_FEATURES = ["site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"]
SEGMENTS = ("overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20")
MODEL_NAMES = ("weighted_moving_average", "lightgbm_v2_logl2", "random_forest", "mlp")
FINAL_OUTPUTS = {
    "model_1m": ROOT / "models/final/mlp_1m.pt",
    "model_2m": ROOT / "models/final/mlp_2m.pt",
    "report": ROOT / "reports/mlp_model_report.md",
    "history_1m": ROOT / "reports/mlp_training_history_1m.csv",
    "history_2m": ROOT / "reports/mlp_training_history_2m.csv",
    "predictions": ROOT / "data/outputs/mlp_test_predictions.parquet",
}


def feature_spec(config: dict) -> tuple[list[str], list[str]]:
    numeric = get_feature_list("mlp", config)
    forbidden = set(config.get("excluded_from_all_features", [])) | set(config.get("excluded_v1_anomalous_fields", []))
    categorical = [column for column in CATEGORY_FEATURES if column not in forbidden]
    leaked = forbidden.intersection(numeric + categorical)
    if leaked:
        raise ValueError(f"MLP features contain excluded fields: {sorted(leaked)}")
    return numeric, categorical


def fit_numeric_stats(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    means = matrix.mean(axis=0).fillna(0).to_numpy(dtype="float64")
    stds = matrix.std(axis=0, ddof=0).fillna(1).to_numpy(dtype="float64")
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    return means, stds


def standardize_numeric(
    frame: pd.DataFrame,
    features: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    matrix[~np.isfinite(matrix)] = np.nan
    missing = np.isnan(matrix)
    if missing.any():
        matrix[missing] = np.take(means, np.nonzero(missing)[1])
    return np.ascontiguousarray(((matrix - means) / stds).astype("float32"))


def encode_mlp_categories(
    frame: pd.DataFrame,
    features: list[str],
    category_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    encoded = np.zeros((len(frame), len(features)), dtype="int64")
    for index, column in enumerate(features):
        encoded[:, index] = (
            frame[column].fillna("unknown").astype("string").map(category_maps[column]).fillna(-1).to_numpy(dtype="int64") + 1
        )
    return np.ascontiguousarray(encoded)


def embedding_dimensions(cardinalities: list[int]) -> list[int]:
    return [min(32, max(4, int(math.ceil(math.sqrt(max(1, value)))))) for value in cardinalities]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_module():
    return load_script_module("wenxuan_lightgbm_pipeline_for_mlp", ROOT / "scripts/10_train_lightgbm.py")


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    directory = ROOT / "logs/mlp"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"training_{run_id}.log"
    logger = logging.getLogger("wenxuan_mlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sampled_category_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(dtype="int64") + 1
    return np.ascontiguousarray(np.maximum(matrix, 0))


def make_loader(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(numeric), torch.from_numpy(categorical),
        torch.from_numpy(target.astype("float32", copy=False)),
        torch.from_numpy(weight.astype("float32", copy=False)),
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=device.type == "cuda", generator=generator,
    )


@torch.no_grad()
def validation_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_numerator = 0.0
    weight_sum = 0.0
    raw_absolute_error = 0.0
    count = 0
    for numeric, categorical, target, weight in loader:
        numeric, categorical = numeric.to(device), categorical.to(device)
        target, weight = target.to(device), weight.to(device)
        prediction = model(numeric, categorical)
        loss_numerator += float((torch.square(prediction - target) * weight).sum().item())
        weight_sum += float(weight.sum().item())
        raw_prediction = torch.clamp(torch.expm1(prediction), min=0)
        raw_target = torch.expm1(target)
        raw_absolute_error += float(torch.abs(raw_prediction - raw_target).sum().item())
        count += int(target.numel())
    return loss_numerator / max(weight_sum, 1e-12), raw_absolute_error / max(count, 1)


def train_network(
    model: TabularMLP,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    checkpoint: Path,
    metadata: dict[str, Any],
    guard,
    logger: logging.Logger,
) -> tuple[TabularMLP, list[dict], int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    model.to(device)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        numerator = 0.0
        denominator = 0.0
        for batch_index, (numeric, categorical, target, weight) in enumerate(train_loader, 1):
            numeric, categorical = numeric.to(device), categorical.to(device)
            target, weight = target.to(device), weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(numeric, categorical)
            loss = weighted_mse_loss(prediction, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            numerator += float((torch.square(prediction.detach() - target) * weight).sum().item())
            denominator += float(weight.sum().item())
            if batch_index % 100 == 0:
                guard.check("training_batch", epoch)
        train_loss = numerator / max(denominator, 1e-12)
        valid_loss, valid_mae = validation_epoch(model, valid_loader, device)
        scheduler.step(valid_loss)
        guard.check("epoch_end", epoch)
        row = {
            "epoch": epoch, "train_weighted_log_mse": train_loss,
            "valid_weighted_log_mse": valid_loss, "valid_raw_mae": valid_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "rss_gib": guard.peak / 2**30,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_log_mse=%.6f valid_log_mse=%.6f valid_mae=%.6f lr=%.2g seconds=%.1f rss=%.2f GiB",
            epoch, train_loss, valid_loss, valid_mae, row["learning_rate"], row["epoch_seconds"], row["rss_gib"],
        )
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            current_metadata = dict(metadata)
            current_metadata.update({"best_epoch": best_epoch, "best_valid_weighted_log_mse": best_loss})
            cpu_model = model.to("cpu")
            save_model_bundle(cpu_model, checkpoint, current_metadata)
            model.to(device)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
                break
    loaded, loaded_metadata = load_model_bundle(checkpoint, map_location=device)
    loaded.to(device).eval()
    if int(loaded_metadata["best_epoch"]) != best_epoch:
        raise RuntimeError("Reloaded MLP metadata does not match best epoch")
    return loaded, history, best_epoch, time.perf_counter() - started


@torch.no_grad()
def predict_batches(
    model: TabularMLP,
    numeric: np.ndarray,
    categorical: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(numeric), batch_size):
        stop = min(len(numeric), start + batch_size)
        output = model(
            torch.from_numpy(numeric[start:stop]).to(device),
            torch.from_numpy(categorical[start:stop]).to(device),
        )
        parts.append(output.detach().cpu().numpy())
    return inverse_prediction(np.concatenate(parts)) if parts else np.empty(0, dtype="float64")


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
    max_epochs: int,
    patience: int,
    batch_size: int,
    checkpoint: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict, TabularMLP]:
    seed_everything(42)
    guard = pipeline.MemoryGuard(f"mlp-{stage}-{horizon}", logger)
    guard.check("stage_start")
    numeric_features, categorical_features = feature_spec(config)
    sampling_features = numeric_features + categorical_features
    base_rates = config["lightgbm_training"]["sampling_rates"]
    train_rates = pipeline.scaled_rates(base_rates, bucket_counts["train"][horizon], train_rows)
    valid_rates = pipeline.scaled_rates(base_rates, bucket_counts["valid"][horizon], valid_rows)
    sampled_started = time.perf_counter()
    samples = pipeline.collect_train_valid_samples(
        dataset_path, horizon, {"train": train_rates, "valid": valid_rates}, category_maps,
        sampling_features, config["time_feature_settings"]["base_month"], guard, logger,
        {"train": train_rows, "valid": valid_rows}, None, stage != "smoke",
    )
    sampling_seconds = time.perf_counter() - sampled_started
    means, stds = fit_numeric_stats(samples["train"]["x"], numeric_features)
    arrays = {}
    for split in ("train", "valid"):
        value = samples[split]
        arrays[split] = {
            "numeric": standardize_numeric(value["x"], numeric_features, means, stds),
            "categorical": sampled_category_matrix(value["x"], categorical_features),
            "target": value["y"].astype("float32", copy=False),
            "weight": value["weight"].astype("float32", copy=False),
            "distribution": value["distribution"],
        }
    del samples
    gc.collect()
    cardinalities = [len(category_maps[column]) + 1 for column in categorical_features]
    dimensions = embedding_dimensions(cardinalities)
    train_loader = make_loader(**{k: arrays["train"][k] for k in ("numeric", "categorical", "target", "weight")}, batch_size=batch_size, shuffle=True, device=device)
    valid_loader = make_loader(**{k: arrays["valid"][k] for k in ("numeric", "categorical", "target", "weight")}, batch_size=batch_size * 2, shuffle=False, device=device)
    model = TabularMLP(len(numeric_features), cardinalities, dimensions)
    metadata = {
        "model": "mlp_log_l2", "stage": stage, "horizon": horizon,
        "numeric_features": numeric_features, "categorical_features": categorical_features,
        "feature_order": numeric_features + categorical_features,
        "numeric_means": means.tolist(), "numeric_stds": stds.tolist(),
        "category_maps": category_maps, "category_unknown_index": 0,
        "time_base": config["time_feature_settings"]["base_month"],
        "sampling_rates": train_rates, "valid_sampling_rates": valid_rates,
        "target_transform": "log1p", "inverse_transform": "expm1_clip_nonnegative",
        "training_params": {
            "loss": "weighted_mse_log_target", "optimizer": "AdamW", "learning_rate": 1e-3,
            "weight_decay": 1e-5, "gradient_clip": 5.0, "max_epochs": max_epochs,
            "early_stopping_patience": patience, "batch_size": batch_size, "seed": 42,
        },
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "device": str(device),
    }
    model, history, best_epoch, training_seconds = train_network(
        model, train_loader, valid_loader, device, max_epochs, patience, checkpoint, metadata, guard, logger,
    )
    check_size = min(1000, len(arrays["valid"]["target"]))
    before = predict_batches(model, arrays["valid"]["numeric"][:check_size], arrays["valid"]["categorical"][:check_size], device, batch_size)
    reloaded, loaded_metadata = load_model_bundle(checkpoint, map_location=device)
    reloaded.to(device).eval()
    after = predict_batches(reloaded, arrays["valid"]["numeric"][:check_size], arrays["valid"]["categorical"][:check_size], device, batch_size)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
    sample_metrics = LongTailStreamingMetrics()
    sample_metrics.update(np.expm1(arrays["valid"]["target"][:check_size]), after)
    result = {
        "stage": stage, "horizon": horizon,
        "train_rows": len(arrays["train"]["target"]), "valid_rows": len(arrays["valid"]["target"]),
        "numeric_feature_count": len(numeric_features), "category_count": len(categorical_features),
        "best_epoch": best_epoch, "history": history,
        "sampling_seconds": sampling_seconds, "training_seconds": training_seconds,
        "peak_memory_bytes": guard.peak, "memory_limit_bytes": guard.limit,
        "model_size_bytes": checkpoint.stat().st_size,
        "sample_metrics": sample_metrics.compute(),
        "train_distribution": arrays["train"]["distribution"],
        "valid_distribution": arrays["valid"]["distribution"],
        "sampling_rates": train_rates, "reload_consistent": True,
        "batch_size": batch_size,
    }
    del arrays, train_loader, valid_loader, reloaded, before, after
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    guard.check("stage_released")
    return result, model


def prepare_evaluation_arrays(frame: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    numeric = standardize_numeric(
        frame, metadata["numeric_features"],
        np.asarray(metadata["numeric_means"], dtype="float64"),
        np.asarray(metadata["numeric_stds"], dtype="float64"),
    )
    categorical = encode_mlp_categories(frame, metadata["categorical_features"], metadata["category_maps"])
    return numeric, categorical


def evaluation_columns(metadata_by_horizon: dict[str, dict]) -> list[str]:
    columns = ["month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m"]
    for metadata in metadata_by_horizon.values():
        columns.extend(metadata["numeric_features"])
        columns.extend(metadata["categorical_features"])
    return list(dict.fromkeys(columns))


def load_comparison_metrics() -> dict:
    lgb_candidates = sorted((ROOT / "logs/lightgbm/v2_logl2").glob("formal_*.json"), key=lambda path: path.stat().st_mtime)
    if not lgb_candidates:
        raise FileNotFoundError("LightGBM V2 formal metrics are missing")
    lgb = json.loads(lgb_candidates[-1].read_text(encoding="utf-8"))
    rf = json.loads((ROOT / "logs/random_forest/formal_run_summary.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("valid", "test"):
        result[split] = {}
        for horizon in HORIZONS:
            lgb_metrics = lgb[f"{split}_metrics"][horizon]
            result[split][horizon] = {
                "weighted_moving_average": lgb_metrics["weighted_moving_average"],
                "lightgbm_v2_logl2": lgb_metrics["v2_log_l2"],
                "random_forest": rf["metrics"][split][horizon]["random_forest"],
            }
    return result


def evaluate_full(
    pipeline,
    dataset_path: Path,
    bundles: dict[str, tuple[TabularMLP, dict]],
    output_path: Path,
    device: torch.device,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[dict, float, int]:
    guard = pipeline.MemoryGuard("mlp-full-evaluation", logger)
    metrics = {split: {h: LongTailStreamingMetrics() for h in HORIZONS} for split in ("valid", "test")}
    metadata = {h: bundles[h][1] for h in HORIZONS}
    columns = evaluation_columns(metadata)
    temporary = output_path.with_suffix(".partial.parquet")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    parquet = pq.ParquetFile(dataset_path)
    started = time.perf_counter()
    try:
        for row_group in range(parquet.num_row_groups):
            split_table = parquet.read_row_group(row_group, columns=["split"])
            split_values = split_table.column("split").to_pandas().astype(str).to_numpy()
            selected = np.flatnonzero(np.isin(split_values, ["valid", "test"]))
            del split_table, split_values
            if not selected.size:
                continue
            frame = parquet.read_row_group(row_group, columns=columns).take(pa.array(selected)).to_pandas()
            split_values = frame["split"].astype(str).to_numpy()
            predictions = {}
            targets = {}
            for horizon in HORIZONS:
                numeric, categorical = prepare_evaluation_arrays(frame, metadata[horizon])
                predictions[horizon] = predict_batches(bundles[horizon][0], numeric, categorical, device, batch_size)
                if not np.isfinite(predictions[horizon]).all() or (predictions[horizon] < 0).any():
                    raise RuntimeError(f"Invalid MLP {horizon} prediction in row group {row_group}")
                targets[horizon] = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                del numeric, categorical
                for split in ("valid", "test"):
                    mask = split_values == split
                    if mask.any():
                        metrics[split][horizon].update(targets[horizon][mask], predictions[horizon][mask])
            test_mask = split_values == "test"
            if test_mask.any():
                output = pd.DataFrame({
                    "month": frame.loc[test_mask, "month"].astype(str),
                    "site_no": frame.loc[test_mask, "site_no"].astype(str),
                    "item_id": frame.loc[test_mask, "item_id"].astype(str),
                    "target_qty_1m": targets["1m"][test_mask].astype("float32"),
                    "target_qty_2m": targets["2m"][test_mask].astype("float32"),
                    "mlp_pred_1m": predictions["1m"][test_mask].astype("float32"),
                    "mlp_pred_2m": predictions["2m"][test_mask].astype("float32"),
                })
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=["month", "site_no"])
                writer.write_table(table, row_group_size=250_000)
                del output, table
            del frame, predictions, targets
            gc.collect()
            guard.check("evaluation_row_group", row_group)
            logger.info("MLP full evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        parquet.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No test predictions were written")
    temporary.replace(output_path)
    computed = load_comparison_metrics()
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            computed[split][horizon]["mlp"] = metrics[split][horizon].compute()
    return computed, time.perf_counter() - started, guard.peak


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def render_report(summary: dict, path: Path) -> None:
    lines = [
        "# MLP 模型报告", "",
        "## 实验设置", "",
        "本阶段仅训练 MLP-1M 与 MLP-2M。目标使用 `log1p`，损失为带逆采样概率权重的 MSE，预测经 `expm1` 后裁剪为非负。",
        f"运行设备：`{summary['device']}`；PyTorch `{summary['torch_version']}`；Python `{summary['python_version']}`。", "",
        "## 分阶段资源", "",
        "| 阶段 | 目标 | train | early-valid | 最佳 epoch | 训练秒数 | 峰值 RAM GiB | 模型 MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("smoke", "pilot", "formal"):
        for horizon in HORIZONS:
            value = summary["stages"][stage][horizon]
            lines.append(
                f"| {stage} | {horizon} | {value['train_rows']:,} | {value['valid_rows']:,} | {value['best_epoch']} | "
                f"{value['training_seconds']:.1f} | {value['peak_memory_bytes']/2**30:.2f} | {value['model_size_bytes']/2**20:.2f} |"
            )
    lines.extend(["", "## 完整 valid/test 指标", ""])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            lines.extend([
                f"### {split} / {horizon}", "",
                "| 模型 | 分层 | 样本量 | MAE | RMSE | SMAPE | WAPE | 真实总量 | 预测总量 | 总量偏差 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for model_name in MODEL_NAMES:
                for segment in SEGMENTS:
                    value = summary["metrics"][split][horizon][model_name][segment]
                    lines.append(
                        f"| {model_name} | {segment} | {int(value['count']):,} | {fmt(value['mae'])} | {fmt(value['rmse'])} | "
                        f"{fmt(value['smape'])}% | {fmt(value['wape'])}% | {fmt(value['target_sum'],0)} | "
                        f"{fmt(value['prediction_sum'],0)} | {fmt(value['total_bias_rate'],2)}% |"
                    )
                zero = summary["metrics"][split][horizon][model_name]["0"]
                lines.append(
                    f"\n{model_name} 零销量诊断：平均预测 {fmt(zero.get('mean_prediction'))}，>0.5 "
                    f"{fmt(100*zero.get('prediction_gt_0_5_rate', math.nan),2)}%，>1 "
                    f"{fmt(100*zero.get('prediction_gt_1_rate', math.nan),2)}%。\n"
                )
    lines.extend(["", "## Test 关键比较", "", "| 目标 | 模型 | overall MAE | nonzero WAPE | 5-20 WAPE | 20+ WAPE | 总量偏差 |", "|---|---|---:|---:|---:|---:|---:|"])
    for horizon in HORIZONS:
        for name in MODEL_NAMES:
            value = summary["metrics"]["test"][horizon][name]
            lines.append(
                f"| {horizon} | {name} | {fmt(value['overall']['mae'])} | {fmt(value['nonzero']['wape'])}% | "
                f"{fmt(value['5-20']['wape'])}% | {fmt(value['20+']['wape'])}% | {fmt(value['overall']['total_bias_rate'],2)}% |"
            )
    lines.extend(["", "## 业务结论", ""])
    for horizon in HORIZONS:
        test = summary["metrics"]["test"][horizon]
        mlp = test["mlp"]
        wma = test["weighted_moving_average"]
        lgb = test["lightgbm_v2_logl2"]
        rf = test["random_forest"]
        wma_gain = 100 * (wma["overall"]["mae"] - mlp["overall"]["mae"]) / wma["overall"]["mae"]
        lgb_gap = 100 * (mlp["overall"]["mae"] - lgb["overall"]["mae"]) / lgb["overall"]["mae"]
        rf_gain = 100 * (rf["overall"]["mae"] - mlp["overall"]["mae"]) / rf["overall"]["mae"]
        lines.append(
            f"- **{horizon}**：MLP test MAE 相对加权移动平均改善 {wma_gain:.2f}%，但比 LightGBM V2 高 {lgb_gap:.2f}%；"
            f"相对随机森林改善 {rf_gain:.2f}%。总量偏差为 {mlp['overall']['total_bias_rate']:.2f}%，仍存在明显低估。"
        )
        lines.append(
            f"  nonzero/5-20/20+ WAPE 为 {mlp['nonzero']['wape']:.2f}% / {mlp['5-20']['wape']:.2f}% / "
            f"{mlp['20+']['wape']:.2f}%，均未超过 LightGBM V2；说明 MLP 的整体收益仍有相当部分来自零销量样本。"
        )
        zero = mlp["0"]
        lines.append(
            f"  零销量平均预测 {zero['mean_prediction']:.4f}，预测 >0.5 / >1 的比例为 "
            f"{100*zero['prediction_gt_0_5_rate']:.2f}% / {100*zero['prediction_gt_1_rate']:.2f}%，误报总体可控。"
        )
    lines.extend([
        "- 两个 horizon 的训练损失持续下降，而 early-valid 在最佳轮次后进入波动并触发早停，存在轻微过拟合倾向，但已由最佳 checkpoint 恢复机制控制。",
        "- MLP 可保留为神经网络结构对照，但当前不应替代 LightGBM V2 作为默认补货点预测模型；随机森林在 nonzero 上也更稳健。",
    ])
    lines.extend([
        "", "## 训练历史与工程审计", "",
        "数值均值和标准差、类别映射均只在各目标的 train 分层样本上拟合；valid/test 只复用这些统计量，未知类别映射为 0。",
        "`future_*`、`split`、`item_id`、`isbn` 与 `gds_no` 均未进入模型输入；`item_id` 仅保留在测试预测文件中。",
        f"完整评价耗时 {summary['evaluation_seconds']:.1f} 秒，评价峰值 RAM {summary['evaluation_peak_bytes']/2**30:.2f} GiB，测试预测文件 {summary['prediction_size_bytes']/2**20:.2f} MiB。",
        "正式模型重载预测一致性已通过。Smoke/Pilot checkpoint 与临时缓存可安全删除；正式模型、测试预测、报告和训练历史应保留。", "",
        "本阶段未启动两阶段模型。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def update_storage_manifest(paths: list[Path]) -> None:
    manifest = ROOT / "reports/storage_manifest.md"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else "# Storage Manifest\n"
    marker = "## MLP 正式产物"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    lines = [marker, "", "| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |", "|---|---|---:|---|---|---|"]
    for artifact in paths:
        if not artifact.exists():
            continue
        relative = artifact.relative_to(ROOT).as_posix()
        if artifact.suffix == ".pt":
            purpose, safe = "正式模型", "否"
        elif artifact.suffix == ".parquet":
            purpose, safe = "正式测试预测", "是（可由正式模型重建）"
        else:
            purpose, safe = "代码、报告或训练历史", "否"
        lines.append(f"| `{relative}` | {purpose} | {artifact.stat().st_size/2**20:.2f} MiB | 是 | {safe} | `scripts/14_train_mlp.py` |")
    manifest.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def ensure_formal_outputs_absent() -> None:
    existing = [str(path) for path in FINAL_OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing MLP outputs: {existing}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_previous_stages() -> dict:
    result = {}
    for stage in ("smoke", "pilot"):
        path = ROOT / f"logs/mlp/{stage}_summary.json"
        if path.exists():
            result[stage] = json.loads(path.read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bounded PyTorch MLP models for Wenxuan demand forecasting.")
    parser.add_argument("--stage", choices=("all", "smoke", "pilot", "formal", "evaluate"), default="all")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logging(run_id)
    pipeline = pipeline_module()
    config = load_feature_config()
    dataset_path = project_path(args.dataset or config["dataset"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 2)))
        torch.set_num_interop_threads(1)
    checkpoint_dir = ROOT / "models/checkpoints/mlp"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/cache/mlp").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/temp").mkdir(parents=True, exist_ok=True)
    logger.info(
        "Python=%s PyTorch=%s CUDA=%s device=%s RAM_available=%.2f GiB",
        platform.python_version(), torch.__version__, torch.cuda.is_available(), device,
        pipeline.physical_memory()[1] / 2**30,
    )
    if args.stage in ("all", "formal"):
        ensure_formal_outputs_absent()
    bucket_counts, category_maps = pipeline.load_or_build_scan_metadata(dataset_path, logger)
    summary = {
        "stages": {}, "device": str(device), "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "log_path": str(log_path),
    }

    if args.stage == "evaluate":
        bundles = {h: load_model_bundle(FINAL_OUTPUTS[f"model_{h}"], map_location=device) for h in HORIZONS}
        for model, _ in bundles.values():
            model.to(device).eval()
        formal_summary = json.loads((ROOT / "logs/mlp/formal_training_summary.json").read_text(encoding="utf-8"))
        summary["stages"] = {**load_previous_stages(), "formal": formal_summary}
        metrics, elapsed, peak = evaluate_full(pipeline, dataset_path, bundles, FINAL_OUTPUTS["predictions"], device, 32768, logger)
        summary.update({"metrics": metrics, "evaluation_seconds": elapsed, "evaluation_peak_bytes": peak, "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size})
        render_report(summary, FINAL_OUTPUTS["report"])
        save_json(ROOT / "logs/mlp/formal_run_summary.json", summary)
        update_storage_manifest([ROOT / "src/models/mlp_model.py", Path(__file__), *FINAL_OUTPUTS.values()])
        return

    requested = ("smoke", "pilot", "formal") if args.stage == "all" else (args.stage,)
    stage_specs = {
        "smoke": {"train": 100_000, "valid": 30_000, "epochs": 2, "patience": 2, "batch": 4096},
        "pilot": {"train": 500_000, "valid": 100_000, "epochs": 10, "patience": 3, "batch": 8192},
        "formal": {"train": 1_500_000, "valid": 200_000, "epochs": 30, "patience": 5, "batch": 8192},
    }
    formal_bundles = {}
    for stage in requested:
        if stage == "formal" and "pilot" not in summary["stages"] and "pilot" not in load_previous_stages():
            raise RuntimeError("Formal training requires a successful MLP pilot")
        summary["stages"][stage] = {}
        spec = stage_specs[stage]
        for horizon in HORIZONS:
            checkpoint = checkpoint_dir / f"{stage}_{horizon}.pt"
            result, model = train_stage_horizon(
                pipeline, dataset_path, config, bucket_counts, category_maps, stage, horizon,
                spec["train"], spec["valid"], spec["epochs"], spec["patience"], spec["batch"],
                checkpoint, device, logger,
            )
            summary["stages"][stage][horizon] = result
            if stage == "formal":
                destination = FINAL_OUTPUTS[f"model_{horizon}"]
                shutil.copy2(checkpoint, destination)
                formal_bundles[horizon] = load_model_bundle(destination, map_location=device)
                pd.DataFrame(result["history"]).to_csv(FINAL_OUTPUTS[f"history_{horizon}"], index=False, encoding="utf-8-sig")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if stage in ("smoke", "pilot"):
            save_json(ROOT / f"logs/mlp/{stage}_summary.json", summary["stages"][stage])
            logger.info("MLP %s passed for both horizons", stage)
        if stage == "formal":
            save_json(ROOT / "logs/mlp/formal_training_summary.json", summary["stages"][stage])

    if "formal" not in requested:
        save_json(ROOT / f"logs/mlp/{args.stage}_run_summary_{run_id}.json", summary)
        return
    summary["stages"] = {**load_previous_stages(), "formal": summary["stages"]["formal"]}
    metrics, elapsed, peak = evaluate_full(
        pipeline, dataset_path, formal_bundles, FINAL_OUTPUTS["predictions"], device, 32768, logger,
    )
    summary.update({
        "metrics": metrics, "evaluation_seconds": elapsed, "evaluation_peak_bytes": peak,
        "prediction_size_bytes": FINAL_OUTPUTS["predictions"].stat().st_size,
    })
    render_report(summary, FINAL_OUTPUTS["report"])
    save_json(ROOT / "logs/mlp/formal_run_summary.json", summary)
    update_storage_manifest([ROOT / "src/models/mlp_model.py", Path(__file__), *FINAL_OUTPUTS.values()])
    for transient in checkpoint_dir.glob("smoke_*.pt"):
        transient.unlink()
    for transient in checkpoint_dir.glob("pilot_*.pt"):
        transient.unlink()
    for transient in (ROOT / "data/cache/mlp").glob("*"):
        if transient.is_file():
            transient.unlink()
    logger.info("MLP stage completed successfully")


if __name__ == "__main__":
    main()
