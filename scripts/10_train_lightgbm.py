from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import DEMAND_BUCKETS, LongTailStreamingMetrics, clip_target, demand_bucket  # noqa: E402
from src.features.preprocessing import add_runtime_columns, get_feature_list, load_feature_config, project_path  # noqa: E402
from src.models.lightgbm_model import (  # noqa: E402
    encode_categories,
    load_model_bundle,
    sample_by_target,
    save_model_bundle,
    train_booster,
)


GIB = 1024**3
HORIZONS = ("1m", "2m")
CATEGORY_FEATURES = ["site_no", "blt_site_no", "gds_ctgry_3_lvel", "gds_ctgry_4_lvel", "gds_ctgry_5_lvel"]
IDENTIFIER_COLUMNS = ["month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m"]
SEGMENT_ORDER = ["overall", "0", "1", "2-5", "5-20", "20+", "nonzero", "ge_5", "ge_20"]


class MemoryLimitExceeded(RuntimeError):
    pass


def physical_memory() -> tuple[int, int]:
    if os.name != "nt":
        import resource

        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        return int(total), int(available)

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def process_rss() -> int:
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(current(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


class MemoryGuard:
    def __init__(self, stage: str, logger: logging.Logger):
        _, available = physical_memory()
        self.stage = stage
        self.logger = logger
        self.limit = int(min(11.5 * GIB, available - 4 * GIB))
        if self.limit <= 2 * GIB:
            raise MemoryLimitExceeded(f"{stage}: insufficient free physical memory; available={available / GIB:.2f} GiB")
        self.peak = process_rss()
        self.history: list[dict] = []
        logger.info("%s memory limit %.2f GiB; available at start %.2f GiB", stage, self.limit / GIB, available / GIB)

    def check(self, label: str, iteration: int | None = None) -> None:
        rss = process_rss()
        _, available = physical_memory()
        self.peak = max(self.peak, rss)
        record = {"label": label, "iteration": iteration, "rss_bytes": rss, "available_bytes": available, "time": time.time()}
        self.history.append(record)
        if rss >= self.limit or available < 4 * GIB:
            raise MemoryLimitExceeded(
                f"{self.stage}: memory safety threshold reached at {label}; rss={rss / GIB:.2f} GiB, "
                f"available={available / GIB:.2f} GiB, limit={self.limit / GIB:.2f} GiB"
            )

    def ensure_capacity(self, additional_bytes: int, label: str) -> None:
        self.check(label)
        projected = process_rss() + int(additional_bytes)
        if projected >= self.limit:
            raise MemoryLimitExceeded(
                f"{self.stage}: projected allocation at {label} exceeds limit; "
                f"projected={projected / GIB:.2f} GiB, limit={self.limit / GIB:.2f} GiB"
            )

    def callback(self, check_period: int = 10, log_period: int = 100):
        guard = self

        def _callback(env):
            iteration = env.iteration + 1
            if iteration == 1 or iteration % check_period == 0:
                guard.check("boosting", iteration)
                if iteration == 1 or iteration % log_period == 0:
                    metrics = ", ".join(f"{name}:{value:.6f}" for _, name, value, _ in env.evaluation_result_list)
                    guard.logger.info("%s iteration=%d %s rss=%.2f GiB", guard.stage, iteration, metrics, process_rss() / GIB)

        _callback.order = 20
        _callback.before_iteration = False
        return _callback


def setup_logging() -> tuple[logging.Logger, Path]:
    log_dir = ROOT / "logs" / "lightgbm"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("wenxuan_lightgbm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, path


def horizon_features(config: dict, horizon: str) -> list[str]:
    return get_feature_list("lightgbm", config) + list(config["lightgbm_horizon_features"][horizon])


def run_context(dataset_path: Path, config_path: Path) -> dict:
    config_bytes = config_path.read_bytes()
    return {
        "dataset_size": dataset_path.stat().st_size,
        "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
    }


def physical_source_columns(config: dict) -> list[str]:
    runtime = set(config.get("runtime_time_features", []))
    features = set()
    for horizon in HORIZONS:
        features.update(horizon_features(config, horizon))
    return list(dict.fromkeys([column for column in features if column not in runtime] + IDENTIFIER_COLUMNS))


def fit_category_maps_streaming(dataset_path: Path, logger: logging.Logger, guard: MemoryGuard) -> dict[str, dict[str, int]]:
    parquet = pq.ParquetFile(dataset_path)
    active_store_dataset = "target_available_1m" in parquet.schema.names
    values = {column: set() for column in CATEGORY_FEATURES}
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=["split", *CATEGORY_FEATURES]).to_pandas()
            frame = frame[frame["split"] == "train"]
            for column in CATEGORY_FEATURES:
                values[column].update(frame[column].fillna("unknown").astype("string").unique().tolist())
            if (row_group + 1) % 25 == 0:
                guard.check("category_scan")
    finally:
        parquet.close()
    if active_store_dataset:
        maps = {
            column: {"__UNKNOWN__": 0, **{str(value): index + 1 for index, value in enumerate(sorted(items))}}
            for column, items in values.items()
        }
    else:
        maps = {column: {str(value): index for index, value in enumerate(sorted(items))} for column, items in values.items()}
    logger.info("Category cardinalities: %s", {column: len(mapping) for column, mapping in maps.items()})
    return maps


def scan_bucket_counts(dataset_path: Path, logger: logging.Logger) -> dict:
    counts = {split: {horizon: Counter() for horizon in HORIZONS} for split in ("train", "valid", "test")}
    parquet = pq.ParquetFile(dataset_path)
    availability_columns = [column for column in ("target_available_1m", "target_available_2m") if column in parquet.schema.names]
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(
                row_group, columns=["split", "future_qty_1m", "future_qty_2m", *availability_columns]
            ).to_pandas()
            for split in counts:
                split_frame = frame[frame["split"] == split]
                for horizon in HORIZONS:
                    availability = f"target_available_{horizon}"
                    horizon_frame = split_frame[
                        split_frame[availability].eq(1)
                    ] if availability in split_frame.columns else split_frame
                    target = clip_target(horizon_frame[f"future_qty_{horizon}"].to_numpy())
                    labels, amounts = np.unique(demand_bucket(target), return_counts=True)
                    counts[split][horizon].update({str(label): int(amount) for label, amount in zip(labels, amounts)})
            if (row_group + 1) % 50 == 0:
                logger.info("Bucket scan row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        parquet.close()
    return counts


def load_or_build_scan_metadata(dataset_path: Path, logger: logging.Logger) -> tuple[dict, dict]:
    cache_path = ROOT / "data/cache/lightgbm_scan_metadata.json"
    signature = {"size": dataset_path.stat().st_size, "mtime_ns": dataset_path.stat().st_mtime_ns}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("dataset_signature") == signature:
            counts = {
                split: {horizon: Counter(values) for horizon, values in horizons.items()}
                for split, horizons in cached["bucket_counts"].items()
            }
            logger.info("Loaded bucket counts and category maps from %s", cache_path)
            return counts, cached["category_maps"]
    counts = scan_bucket_counts(dataset_path, logger)
    map_guard = MemoryGuard("category-map", logger)
    category_maps = fit_category_maps_streaming(dataset_path, logger, map_guard)
    payload = {
        "dataset_signature": signature,
        "bucket_counts": {split: {horizon: dict(values) for horizon, values in horizons.items()} for split, horizons in counts.items()},
        "category_maps": category_maps,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return counts, category_maps


def scaled_rates(base_rates: dict[str, float], counts: Counter, target_rows: int | None) -> dict[str, float]:
    base = {str(key): float(value) for key, value in base_rates.items()}
    if target_rows is None:
        return base
    low, high = 0.0, 100.0
    for _ in range(80):
        scale = (low + high) / 2
        expected = sum(counts[bucket] * min(1.0, base[bucket] * scale) for bucket in DEMAND_BUCKETS)
        if expected < target_rows:
            low = scale
        else:
            high = scale
    return {bucket: min(1.0, base[bucket] * high) for bucket in DEMAND_BUCKETS}


class DistributionStats:
    def __init__(self):
        self.count = 0.0
        self.target_sum = 0.0
        self.buckets = Counter()
        self.months = Counter()
        self.sites = Counter()
        self.categories = Counter()

    def update(self, frame: pd.DataFrame, target: np.ndarray, weights: np.ndarray | None = None) -> None:
        if len(frame) == 0:
            return
        w = np.ones(len(frame), dtype="float64") if weights is None else np.asarray(weights, dtype="float64")
        self.count += float(w.sum())
        self.target_sum += float(np.sum(target * w))
        for labels, destination in (
            (demand_bucket(target), self.buckets),
            (frame["month"].fillna("unknown").astype(str).to_numpy(), self.months),
            (frame["site_no"].fillna("unknown").astype(str).to_numpy(), self.sites),
            (frame["gds_ctgry_3_lvel"].fillna("unknown").astype(str).to_numpy(), self.categories),
        ):
            unique, inverse = np.unique(labels, return_inverse=True)
            sums = np.bincount(inverse, weights=w)
            destination.update({str(label): float(value) for label, value in zip(unique, sums)})

    def summary(self) -> dict:
        def share(counter: Counter) -> dict:
            total = sum(counter.values())
            return {key: value / total for key, value in counter.items()} if total else {}

        return {
            "count": self.count,
            "target_mean": self.target_sum / self.count if self.count else 0.0,
            "target_sum": self.target_sum,
            "nonzero_rate": 1.0 - share(self.buckets).get("0", 0.0),
            "bucket_counts": dict(self.buckets), "bucket_shares": share(self.buckets),
            "month_shares": share(self.months), "site_shares": share(self.sites),
            "category_shares": share(self.categories),
        }


def max_distribution_shift(before: DistributionStats, after_weighted: DistributionStats, field: str) -> float:
    left = before.summary()[field]
    right = after_weighted.summary()[field]
    return max((abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right)), default=0.0)


def prepare_feature_frame(frame: pd.DataFrame, features: list[str], category_maps: dict, base_month: str) -> pd.DataFrame:
    prepared = add_runtime_columns(frame, base_month=base_month)
    prepared = encode_categories(prepared, category_maps)
    for column in features:
        if column not in category_maps:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return prepared[features].copy()


def collect_sample(
    dataset_path: Path,
    split: str,
    horizon: str,
    rates: dict[str, float],
    category_maps: dict,
    features: list[str],
    base_month: str,
    guard: MemoryGuard,
    logger: logging.Logger,
    max_rows: int | None = None,
    max_row_groups: int | None = None,
    enforce_distribution: bool = True,
) -> dict:
    columns = list(dict.fromkeys([column for column in physical_source_columns(load_feature_config()) if column in set(features) | set(IDENTIFIER_COLUMNS) | {"gds_ctgry_3_lvel", *CATEGORY_FEATURES}]))
    parquet = pq.ParquetFile(dataset_path)
    x_parts, y_parts, weight_parts = [], [], []
    before, after_raw, after_weighted = DistributionStats(), DistributionStats(), DistributionStats()
    row_group_count = parquet.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.num_row_groups)
    target_column = f"future_qty_{horizon}"
    availability = f"target_available_{horizon}"
    has_availability = availability in parquet.schema.names
    selection_columns = ["split", "month", "site_no", "item_id", target_column, "gds_ctgry_3_lvel"]
    if has_availability:
        selection_columns.append(availability)
    try:
        for row_group in range(row_group_count):
            selection_frame = parquet.read_row_group(row_group, columns=selection_columns).to_pandas()
            split_selection = selection_frame[selection_frame["split"] == split]
            if has_availability:
                split_selection = split_selection[split_selection[availability].eq(1)]
            if split_selection.empty:
                continue
            target = clip_target(split_selection[target_column].to_numpy())
            before.update(split_selection, target)
            sampled_keys, raw_weights, _ = sample_by_target(split_selection, horizon, rates, seed=42)
            if max_rows is not None:
                remaining = max_rows - sum(len(part) for part in y_parts)
                sampled_keys = sampled_keys.iloc[:remaining]
                raw_weights = raw_weights[:remaining]
            if not sampled_keys.empty:
                selected_indices = pa.array(sampled_keys.index.to_numpy(dtype="int64"))
                sampled = parquet.read_row_group(row_group, columns=columns).take(selected_indices).to_pandas()
                sampled_target = clip_target(sampled[f"future_qty_{horizon}"].to_numpy())
                after_raw.update(sampled, sampled_target)
                after_weighted.update(sampled, sampled_target, raw_weights)
                x_parts.append(prepare_feature_frame(sampled, features, category_maps, base_month))
                y_parts.append(np.log1p(sampled_target).astype("float32"))
                weight_parts.append(raw_weights)
                del sampled
            del selection_frame, split_selection, sampled_keys
            if (row_group + 1) % 10 == 0:
                guard.check("sampling")
                logger.info("%s %s sampled %s rows after %d row groups", guard.stage, split, f"{sum(len(x) for x in y_parts):,}", row_group + 1)
            if max_rows is not None and sum(len(part) for part in y_parts) >= max_rows:
                break
    finally:
        parquet.close()
    if not x_parts:
        raise RuntimeError(f"No sampled rows for {split} {horizon}")
    x = pd.concat(x_parts, ignore_index=True, copy=False)
    y = np.concatenate(y_parts)
    weights = np.concatenate(weight_parts).astype("float32")
    weights /= weights.mean()
    guard.check("sample_concat")
    shifts = {
        "month": max_distribution_shift(before, after_weighted, "month_shares"),
        "site": max_distribution_shift(before, after_weighted, "site_shares"),
        "category": max_distribution_shift(before, after_weighted, "category_shares"),
    }
    if enforce_distribution and max(shifts.values()) > 0.02:
        raise RuntimeError(f"Weighted sampling distribution shift exceeds 2 percentage points: {shifts}")
    return {
        "x": x, "y": y, "weight": weights,
        "distribution": {"before": before.summary(), "sample_raw": after_raw.summary(), "sample_weighted": after_weighted.summary(), "max_shifts": shifts},
    }


def collect_train_valid_samples(
    dataset_path: Path,
    horizon: str,
    rates_by_split: dict[str, dict[str, float]],
    category_maps: dict,
    features: list[str],
    base_month: str,
    guard: MemoryGuard,
    logger: logging.Logger,
    max_rows_by_split: dict[str, int | None],
    max_row_groups: int | None = None,
    enforce_distribution: bool = True,
) -> dict[str, dict]:
    config = load_feature_config()
    allowed = set(features) | set(IDENTIFIER_COLUMNS) | {"gds_ctgry_3_lvel", *CATEGORY_FEATURES}
    columns = [column for column in physical_source_columns(config) if column in allowed]
    target_column = f"future_qty_{horizon}"
    availability = f"target_available_{horizon}"
    schema = pq.ParquetFile(dataset_path).schema.names
    has_availability = availability in schema
    selection_columns = ["split", "month", "site_no", "item_id", target_column, "gds_ctgry_3_lvel"]
    if has_availability:
        selection_columns.append(availability)
    states = {
        split: {
            "x_parts": [], "y_parts": [], "weight_parts": [],
            "before": DistributionStats(), "after_raw": DistributionStats(), "after_weighted": DistributionStats(),
            "rows": 0,
        }
        for split in ("train", "valid")
    }
    parquet = pq.ParquetFile(dataset_path)
    row_group_count = parquet.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.num_row_groups)
    try:
        for row_group in range(row_group_count):
            selection_frame = parquet.read_row_group(row_group, columns=selection_columns).to_pandas()
            selected = {}
            for split in ("train", "valid"):
                split_frame = selection_frame[selection_frame["split"] == split]
                if has_availability:
                    split_frame = split_frame[split_frame[availability].eq(1)]
                if split_frame.empty:
                    continue
                target = clip_target(split_frame[target_column].to_numpy())
                states[split]["before"].update(split_frame, target)
                sampled_keys, raw_weights, _ = sample_by_target(split_frame, horizon, rates_by_split[split], seed=42)
                maximum = max_rows_by_split[split]
                if maximum is not None:
                    remaining = maximum - states[split]["rows"]
                    sampled_keys = sampled_keys.iloc[:remaining]
                    raw_weights = raw_weights[:remaining]
                if not sampled_keys.empty:
                    selected[split] = (sampled_keys.index.to_numpy(dtype="int64"), raw_weights)
            if selected:
                full_table = parquet.read_row_group(row_group, columns=columns)
                for split, (indices, raw_weights) in selected.items():
                    sampled = full_table.take(pa.array(indices)).to_pandas()
                    sampled_target = clip_target(sampled[target_column].to_numpy())
                    state = states[split]
                    state["after_raw"].update(sampled, sampled_target)
                    state["after_weighted"].update(sampled, sampled_target, raw_weights)
                    state["x_parts"].append(prepare_feature_frame(sampled, features, category_maps, base_month))
                    state["y_parts"].append(np.log1p(sampled_target).astype("float32"))
                    state["weight_parts"].append(raw_weights)
                    state["rows"] += len(sampled)
                    del sampled
                del full_table
            del selection_frame
            if (row_group + 1) % 10 == 0:
                guard.check("sampling")
                logger.info(
                    "%s sampled train=%s valid=%s after %d row groups",
                    guard.stage, f"{states['train']['rows']:,}", f"{states['valid']['rows']:,}", row_group + 1,
                )
            done = all(max_rows_by_split[split] is not None and states[split]["rows"] >= max_rows_by_split[split] for split in ("train", "valid"))
            if done:
                break
    finally:
        parquet.close()

    results = {}
    for split, state in states.items():
        if not state["x_parts"]:
            raise RuntimeError(f"No sampled rows for {split} {horizon}")
        x = pd.concat(state["x_parts"], ignore_index=True, copy=False)
        y = np.concatenate(state["y_parts"])
        weights = np.concatenate(state["weight_parts"]).astype("float32")
        weights /= weights.mean()
        shifts = {
            "month": max_distribution_shift(state["before"], state["after_weighted"], "month_shares"),
            "site": max_distribution_shift(state["before"], state["after_weighted"], "site_shares"),
            "category": max_distribution_shift(state["before"], state["after_weighted"], "category_shares"),
        }
        if enforce_distribution and max(shifts.values()) > 0.02:
            raise RuntimeError(f"{split} weighted sampling distribution shift exceeds 2 percentage points: {shifts}")
        results[split] = {
            "x": x, "y": y, "weight": weights,
            "distribution": {
                "before": state["before"].summary(), "sample_raw": state["after_raw"].summary(),
                "sample_weighted": state["after_weighted"].summary(), "max_shifts": shifts,
            },
        }
    guard.check("sample_concat")
    return results


def training_params(config: dict) -> dict:
    params = dict(config["lightgbm_training"]["params"])
    params["num_threads"] = max(1, (os.cpu_count() or 2) - 2)
    return params


def train_stage_model(
    stage: str,
    horizon: str,
    dataset_path: Path,
    category_maps: dict,
    bucket_counts: dict,
    config: dict,
    logger: logging.Logger,
    train_rows: int | None,
    valid_rows: int | None,
    num_boost_round: int,
    early_stopping_rounds: int,
    output_path: Path,
    max_row_groups: int | None = None,
) -> dict:
    guard = MemoryGuard(f"{stage}-{horizon}", logger)
    guard.check("stage_start")
    settings = config["lightgbm_training"]
    if stage == "smoke":
        train_rates = {bucket: 0.5 for bucket in DEMAND_BUCKETS}
        valid_rates = {bucket: 0.5 for bucket in DEMAND_BUCKETS}
    else:
        train_rates = scaled_rates(settings["sampling_rates"], bucket_counts["train"][horizon], train_rows)
        valid_rates = scaled_rates(settings["valid_sampling_rates"], bucket_counts["valid"][horizon], valid_rows)
    features = horizon_features(config, horizon)
    base_month = config["time_feature_settings"]["base_month"]
    sample_started = time.perf_counter()
    samples = collect_train_valid_samples(
        dataset_path, horizon, {"train": train_rates, "valid": valid_rates}, category_maps, features, base_month,
        guard, logger, {"train": train_rows, "valid": valid_rows}, max_row_groups, stage != "smoke",
    )
    train, valid = samples["train"], samples["valid"]
    sampling_seconds = time.perf_counter() - sample_started
    params = training_params(config)
    train_count, valid_count = len(train["y"]), len(valid["y"])
    logger.info("Training %s %s with train=%s valid=%s features=%d", stage, horizon, f"{train_count:,}", f"{valid_count:,}", len(features))
    estimated_dataset_bytes = int((train_count + valid_count) * (len(features) * 3 + 64))
    guard.ensure_capacity(estimated_dataset_bytes, "before_dataset_construction")
    train_started = time.perf_counter()
    booster, evaluations = train_booster(
        train["x"], train["y"], train["weight"],
        valid["x"], valid["y"], valid["weight"],
        categorical_features=CATEGORY_FEATURES,
        params=params, num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[lgb.log_evaluation(100), guard.callback(10, 100)],
        construction_check=lambda label: guard.check(label),
    )
    training_seconds = time.perf_counter() - train_started
    check_x = valid["x"].iloc[:1000]
    before = np.clip(np.expm1(booster.predict(check_x, num_iteration=booster.best_iteration)), 0, None)
    if not np.isfinite(before).all() or (before < 0).any():
        raise RuntimeError(f"{stage}-{horizon}: invalid predictions")
    metadata = {
        "feature_names": features, "categorical_features": CATEGORY_FEATURES,
        "category_maps": category_maps, "time_base": base_month, "horizon": horizon,
        "params": params, "best_iteration": booster.best_iteration,
        "sampling_rates": train_rates, "valid_sampling_rates": valid_rates,
        "lightgbm_version": lgb.__version__, "python_version": platform.python_version(),
    }
    save_model_bundle(booster, output_path, metadata)
    loaded, loaded_metadata = load_model_bundle(output_path)
    after = np.clip(np.expm1(loaded.predict(check_x, num_iteration=loaded.best_iteration)), 0, None)
    np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-8)
    if loaded_metadata["feature_names"] != features:
        raise RuntimeError("Reloaded feature order mismatch")
    sample_metric = LongTailStreamingMetrics()
    sample_metric.update(np.expm1(valid["y"][: len(after)]), after)
    result = {
        "stage": stage, "horizon": horizon, "train_rows": train_count, "valid_rows": valid_count,
        "feature_count": len(features), "category_count": len(CATEGORY_FEATURES),
        "best_iteration": int(booster.best_iteration), "sampling_seconds": sampling_seconds,
        "training_seconds": training_seconds, "peak_memory_bytes": guard.peak,
        "memory_limit_bytes": guard.limit, "memory_history": guard.history,
        "evaluations": evaluations, "sample_metrics": sample_metric.compute(),
        "train_distribution": train["distribution"], "valid_distribution": valid["distribution"],
        "params": params, "model_path": str(output_path),
    }
    del train, valid, booster, loaded, check_x, before, after
    gc.collect()
    guard.check("stage_released")
    result["released_rss_bytes"] = process_rss()
    return result


def prepare_evaluation_frame(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    return prepare_feature_frame(frame, metadata["feature_names"], metadata["category_maps"], metadata["time_base"])


def evaluate_full(
    dataset_path: Path,
    model_paths: dict[str, Path],
    output_path: Path,
    logger: logging.Logger,
    guard: MemoryGuard,
) -> tuple[dict, float]:
    bundles = {horizon: load_model_bundle(path) for horizon, path in model_paths.items()}
    physical = set(IDENTIFIER_COLUMNS)
    runtime = set(load_feature_config().get("runtime_time_features", []))
    for _, metadata in bundles.values():
        physical.update(column for column in metadata["feature_names"] if column not in runtime)
    columns = list(physical)
    metrics = {split: {horizon: LongTailStreamingMetrics() for horizon in HORIZONS} for split in ("valid", "test")}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    parquet = pq.ParquetFile(dataset_path)
    started = time.perf_counter()
    try:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
            frame = frame[frame["split"].isin(["valid", "test"])]
            predictions = {}
            targets = {}
            for horizon, (booster, metadata) in bundles.items():
                x = prepare_evaluation_frame(frame, metadata)
                pred = np.clip(np.expm1(booster.predict(x, num_iteration=metadata["best_iteration"])), 0, None)
                if not np.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite {horizon} predictions in row group {row_group}")
                predictions[horizon] = pred
                targets[horizon] = clip_target(frame[f"future_qty_{horizon}"].to_numpy())
                for split in ("valid", "test"):
                    mask = frame["split"].eq(split).to_numpy()
                    metrics[split][horizon].update(targets[horizon][mask], pred[mask])
                del x
            test_mask = frame["split"].eq("test").to_numpy()
            if test_mask.any():
                output = frame.loc[test_mask, ["month", "site_no", "item_id"]].copy()
                output["target_qty_1m"] = targets["1m"][test_mask].astype("float32")
                output["target_qty_2m"] = targets["2m"][test_mask].astype("float32")
                output["lightgbm_pred_1m"] = predictions["1m"][test_mask].astype("float32")
                output["lightgbm_pred_2m"] = predictions["2m"][test_mask].astype("float32")
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=["month", "site_no"])
                writer.write_table(table, row_group_size=250_000)
            if (row_group + 1) % 10 == 0:
                guard.check("full_evaluation")
                logger.info("Full evaluation row groups %d/%d", row_group + 1, parquet.num_row_groups)
    finally:
        if writer is not None:
            writer.close()
        parquet.close()
    temporary.replace(output_path)
    return {split: {horizon: value.compute() for horizon, value in horizons.items()} for split, horizons in metrics.items()}, time.perf_counter() - started


def write_importance(model_paths: dict[str, Path]) -> dict:
    top = {}
    for horizon, path in model_paths.items():
        booster, metadata = load_model_bundle(path)
        frame = pd.DataFrame({
            "feature": metadata["feature_names"],
            "gain_importance": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }).sort_values("gain_importance", ascending=False)
        destination = ROOT / "reports" / f"lightgbm_feature_importance_{horizon}.csv"
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        top[horizon] = frame.head(20).to_dict("records")
    return top


def parse_baseline_report(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| valid") and not line.startswith("| test"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 8 and cells[1] in {"历史均值法", "加权移动平均法"}:
            split, model, horizon = cells[:3]
            result.setdefault(split, {}).setdefault(model, {}).setdefault(horizon, {})["overall"] = {
                "count": int(cells[3].replace(",", "")), "mae": float(cells[4]), "rmse": float(cells[5]),
                "smape": float(cells[6]), "wape": float(cells[7]),
            }
        elif len(cells) == 9 and cells[1] in {"历史均值法", "加权移动平均法"}:
            split, model, horizon, segment = cells[:4]
            def baseline_value(value: str) -> float:
                return math.inf if value == "Inf" else float(value)
            result.setdefault(split, {}).setdefault(model, {}).setdefault(horizon, {}).setdefault("segments", {})[segment] = {
                "count": int(cells[4].replace(",", "")), "mae": float(cells[5]), "rmse": float(cells[6]),
                "smape": float(cells[7]), "wape": baseline_value(cells[8]),
            }
    return result


def assess_zero_sensitivity(metrics: dict, baseline: dict) -> dict:
    reasons = []
    details = {}
    for horizon in HORIZONS:
        zero = metrics["valid"][horizon]["0"]
        overall = metrics["valid"][horizon]["overall"]
        baseline_zero = baseline["valid"]["加权移动平均法"][horizon]["segments"]["0"]["mae"]
        thresholds = {
            "mean_prediction": max(0.1, 1.2 * baseline_zero),
            "prediction_gt_0_5_rate": 0.05,
            "total_bias_rate": 20.0,
        }
        horizon_reasons = []
        if zero["mean_prediction"] > thresholds["mean_prediction"]:
            horizon_reasons.append("zero_mean_prediction_high")
        if zero["prediction_gt_0_5_rate"] > thresholds["prediction_gt_0_5_rate"]:
            horizon_reasons.append("zero_false_positive_rate_high")
        if overall["total_bias_rate"] > thresholds["total_bias_rate"]:
            horizon_reasons.append("total_prediction_overestimated")
        if zero["mae"] > 1.2 * baseline_zero:
            horizon_reasons.append("zero_mae_worse_than_baseline")
        reasons.extend(f"{horizon}:{reason}" for reason in horizon_reasons)
        details[horizon] = {
            "mean_prediction": zero["mean_prediction"],
            "prediction_gt_0_5_rate": zero["prediction_gt_0_5_rate"],
            "prediction_gt_1_rate": zero["prediction_gt_1_rate"],
            "total_bias_rate": overall["total_bias_rate"],
            "baseline_zero_mae": baseline_zero,
            "thresholds": thresholds,
            "reasons": horizon_reasons,
        }
    return {"triggered": bool(reasons), "reasons": reasons, "details": details}


def build_conclusion(metrics: dict, baseline: dict) -> str:
    return (
        "LightGBM 显著降低整体误差，但存在明显总量低估。1M 的总体改善主要来自零销量样本；"
        "销量 1、2-5 和 20+ 的 MAE 未优于加权移动平均。2M 对销量 1、2-5、5-20 有改善，"
        "但 20+ 仍略差。模型适合作为低误报基线，暂不宜直接把点预测当作完整补货量。"
    )


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict, report_path: Path) -> None:
    metrics = summary["full_metrics"]
    baseline = summary["baseline"]
    lines = [
        "# LightGBM 第一版模型报告", "", "## 训练方案", "",
        f"- LightGBM：{summary['lightgbm_version']}；Python：{summary['python_version']}。",
        "- 目标：非负补货需求经 `log1p` 训练，预测后 `expm1` 并截断为非负。",
        "- 1M 与 2M 分别按自身目标分层采样，使用确定性哈希和归一化逆采样概率权重。",
        "- early-stopping valid 使用分层样本；最终 valid/test 指标均使用完整数据流式计算。",
        "- `time_index` 固定以 2023-01 为 0；所有时间特征仅在运行时生成。", "",
        "## 阶段结果", "",
        "| 阶段 | 模型 | train | early-stop valid | 最佳轮数 | 采样耗时(s) | 训练耗时(s) | 峰值内存(GiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("smoke", "pilot", "formal"):
        for horizon in HORIZONS:
            item = summary["stages"].get(stage, {}).get(horizon)
            if item:
                lines.append(f"| {stage} | {horizon} | {item['train_rows']:,} | {item['valid_rows']:,} | {item['best_iteration']} | {item['sampling_seconds']:.1f} | {item['training_seconds']:.1f} | {item['peak_memory_bytes']/GIB:.2f} |")
    lines.extend(["", "## 完整 Valid/Test 指标", "", "| split | 模型 | 分组 | 样本量 | MAE | RMSE | SMAPE(%) | WAPE(%) | 真实总量 | 预测总量 | 总量偏差(%) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            for segment in SEGMENT_ORDER:
                value = metrics[split][horizon][segment]
                lines.append(f"| {split} | {horizon} | {segment} | {fmt(value['count'])} | {fmt(value['mae'])} | {fmt(value['rmse'])} | {fmt(value['smape'])} | {fmt(value['wape'])} | {fmt(value['target_sum'],2)} | {fmt(value['prediction_sum'],2)} | {fmt(value['total_bias_rate'])} |")
            zero = metrics[split][horizon]["0"]
            lines.append(f"\n{split} / {horizon} 零销量诊断：平均预测 {fmt(zero['mean_prediction'])}，预测 >0.5 比例 {fmt(100*zero['prediction_gt_0_5_rate'])}%，预测 >1 比例 {fmt(100*zero['prediction_gt_1_rate'])}%。\n")
    lines.extend(["", "## 与加权移动平均比较", "", "| split | 模型 | 指标 | LightGBM | 加权移动平均 | 相对改善(%) |", "|---|---|---|---:|---:|---:|"])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            overall = metrics[split][horizon]["overall"]
            base = baseline[split]["加权移动平均法"][horizon]["overall"]
            for key in ("mae", "rmse", "smape", "wape"):
                improvement = 100 * (base[key] - overall[key]) / base[key]
                lines.append(f"| {split} | {horizon} | {key.upper()} | {overall[key]:.4f} | {base[key]:.4f} | {improvement:.2f} |")
    lines.extend(["", "## 各销量层级相对加权移动平均", "", "| split | 模型 | 分组 | LightGBM MAE | Baseline MAE | MAE改善(%) | LightGBM WAPE(%) | Baseline WAPE(%) |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for split in ("valid", "test"):
        for horizon in HORIZONS:
            baseline_segments = baseline[split]["加权移动平均法"][horizon]["segments"]
            for segment in DEMAND_BUCKETS:
                current = metrics[split][horizon][segment]
                base = baseline_segments[segment]
                improvement = 100 * (base["mae"] - current["mae"]) / base["mae"]
                lines.append(f"| {split} | {horizon} | {segment} | {current['mae']:.4f} | {base['mae']:.4f} | {improvement:.2f} | {fmt(current['wape'])} | {fmt(base['wape'])} |")
    lines.extend(["", "## 前 20 个特征", ""])
    for horizon in HORIZONS:
        lines.extend([f"### {horizon}", "", "| 排名 | 特征 | Gain | Split |", "|---:|---|---:|---:|"])
        for rank, row in enumerate(summary["feature_importance"][horizon], 1):
            lines.append(f"| {rank} | {row['feature']} | {row['gain_importance']:.2f} | {int(row['split_importance'])} |")
        lines.append("")
    lines.extend(["## 采样分布检查", ""])
    for horizon in HORIZONS:
        distribution = summary["stages"]["formal"][horizon]["train_distribution"]
        lines.append(f"- {horizon} 加权后最大月份偏移 {100*distribution['max_shifts']['month']:.3f} 个百分点，门店偏移 {100*distribution['max_shifts']['site']:.3f}，三级类目偏移 {100*distribution['max_shifts']['category']:.3f}。")
        lines.append(f"- {horizon} 抽样前 {distribution['before']['count']:.0f} 条，抽样后 {distribution['sample_raw']['count']:.0f} 条；抽样前非零比例 {100*distribution['before']['nonzero_rate']:.2f}%，抽样后 {100*distribution['sample_raw']['nonzero_rate']:.2f}%，逆概率加权后 {100*distribution['sample_weighted']['nonzero_rate']:.2f}%。")
        lines.extend(["", f"{horizon} 训练集销量层级：", "", "| 层级 | 抽样前数量 | 抽样前占比 | 抽样后数量 | 抽样后占比 | 加权后占比 |", "|---|---:|---:|---:|---:|---:|"])
        for bucket in DEMAND_BUCKETS:
            before = distribution["before"]
            raw = distribution["sample_raw"]
            weighted = distribution["sample_weighted"]
            lines.append(f"| {bucket} | {before['bucket_counts'].get(bucket,0):.0f} | {100*before['bucket_shares'].get(bucket,0):.2f}% | {raw['bucket_counts'].get(bucket,0):.0f} | {100*raw['bucket_shares'].get(bucket,0):.2f}% | {100*weighted['bucket_shares'].get(bucket,0):.2f}% |")
    sensitivity = summary["sensitivity"]
    sensitivity_lines = []
    for horizon in HORIZONS:
        detail = sensitivity["details"][horizon]
        sensitivity_lines.append(
            f"- {horizon}：零销量平均预测 {detail['mean_prediction']:.4f}，预测 >0.5 比例 "
            f"{100*detail['prediction_gt_0_5_rate']:.2f}%，预测 >1 比例 {100*detail['prediction_gt_1_rate']:.2f}%，"
            f"总量偏差 {detail['total_bias_rate']:.2f}%。"
        )
    sensitivity_status = "已触发" if sensitivity["triggered"] else "未触发"
    lines.extend([
        "", "## 0 层敏感性实验判断", "",
        f"{sensitivity_status} 0 层 10% 敏感性实验。",
        *sensitivity_lines,
        "", "## 风险与结论", "", summary.get("conclusion", "完整结果见以上分层指标；不能只依据整体 MAE 判断业务价值。"), "",
    ])
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)


def write_storage_manifest(summary: dict, script_path: Path) -> None:
    paths = [
        ROOT / "src/models/lightgbm_model.py", script_path,
        ROOT / "models/final/lightgbm_1m.txt", ROOT / "models/final/lightgbm_2m.txt",
        ROOT / "reports/lightgbm_model_report.md", ROOT / "reports/lightgbm_feature_importance_1m.csv",
        ROOT / "reports/lightgbm_feature_importance_2m.csv", ROOT / "data/outputs/lightgbm_test_predictions.parquet",
        Path(summary["log_path"]),
    ]
    lines = ["# Storage Manifest", "", "| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |", "|---|---|---:|---|---|---|"]
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(ROOT)
        final = str(relative).startswith("models\\final") or str(relative).startswith("models/final")
        prediction = "lightgbm_test_predictions" in path.name
        report = path.suffix in {".md", ".csv", ".log"}
        purpose = "最终模型" if final else "正式测试预测" if prediction else "报告或日志" if report else "训练代码"
        safe_delete = "否（正式产物或可复现代码，应保留）"
        lines.append(f"| `{relative}` | {purpose} | {path.stat().st_size/1024**2:.2f} MiB | 是 | {safe_delete} | `scripts/10_train_lightgbm.py` |")
    manifest = ROOT / "reports/storage_manifest.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_transient_files() -> None:
    for directory in (ROOT / "data/temp", ROOT / "data/cache", ROOT / "models/checkpoints"):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-version LightGBM demand models.")
    parser.add_argument("--dataset", default="data/processed/model_dataset_monthly.parquet")
    parser.add_argument("--stage", choices=["all", "smoke", "pilot", "formal"], default="all")
    args = parser.parse_args()
    logger, log_path = setup_logging()
    config = load_feature_config()
    dataset_path = project_path(args.dataset)
    context = run_context(dataset_path, ROOT / "config/model_features.yaml")
    for directory in ("data/temp", "data/cache", "logs/lightgbm", "models/checkpoints", "models/final", "reports", "data/outputs"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    logger.info("Python %s LightGBM %s dataset=%s", platform.python_version(), lgb.__version__, dataset_path)
    logger.info("Configured feature counts: 1m=%d 2m=%d", len(horizon_features(config, "1m")), len(horizon_features(config, "2m")))
    bucket_counts, category_maps = load_or_build_scan_metadata(dataset_path, logger)
    stages: dict = {}
    if args.stage == "formal":
        for prior_stage in ("smoke", "pilot"):
            prior_summary = ROOT / "logs/lightgbm" / f"{prior_stage}_summary.json"
            if not prior_summary.exists():
                raise RuntimeError(f"Formal training requires a completed {prior_stage} summary: {prior_summary}")
            payload = json.loads(prior_summary.read_text(encoding="utf-8"))
            if not payload.get("passed") or payload.get("context") != context:
                raise RuntimeError(f"Formal training blocked: {prior_stage} summary is stale or did not pass")
            stages.update(payload["stages"])

    plans = {
        "smoke": {"train": 200_000, "valid": 50_000, "rounds": 80, "early": 20, "max_row_groups": 2},
        "pilot": {"train": 1_000_000, "valid": 200_000, "rounds": 600, "early": 100, "max_row_groups": None},
        "formal": {"train": None, "valid": None, "rounds": 2000, "early": 100, "max_row_groups": None},
    }
    requested = ["smoke", "pilot", "formal"] if args.stage == "all" else [args.stage]
    try:
        for stage in requested:
            stages[stage] = {}
            for horizon in HORIZONS:
                output = (ROOT / "models/final" if stage == "formal" else ROOT / "models/checkpoints") / f"{'lightgbm' if stage == 'formal' else stage + '_lightgbm'}_{horizon}.txt"
                stages[stage][horizon] = train_stage_model(
                    stage, horizon, dataset_path, category_maps, bucket_counts, config, logger,
                    plans[stage]["train"], plans[stage]["valid"], plans[stage]["rounds"], plans[stage]["early"], output,
                    plans[stage]["max_row_groups"],
                )
                if stage == "pilot":
                    item = stages[stage][horizon]
                    if item["peak_memory_bytes"] >= item["memory_limit_bytes"]:
                        raise RuntimeError(f"Pilot {horizon} exceeded memory limit")
            if stage == "smoke":
                logger.info("Smoke test passed for both horizons")
            if stage == "pilot":
                logger.info("Pilot admission checks passed for both horizons")
        if "formal" not in requested:
            summary_path = ROOT / "logs/lightgbm" / f"{args.stage}_summary.json"
            payload = {"passed": True, "context": context, "stages": stages}
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            return

        model_paths = {horizon: ROOT / "models/final" / f"lightgbm_{horizon}.txt" for horizon in HORIZONS}
        evaluation_guard = MemoryGuard("full-evaluation", logger)
        full_metrics, evaluation_seconds = evaluate_full(
            dataset_path, model_paths, ROOT / "data/outputs/lightgbm_test_predictions.parquet", logger, evaluation_guard
        )
        importance = write_importance(model_paths)
        baseline = parse_baseline_report(ROOT / "reports/baseline_model_report.md")
        sensitivity = assess_zero_sensitivity(full_metrics, baseline)
        if sensitivity["triggered"]:
            raise RuntimeError(f"0-layer 10% sensitivity experiment required before completion: {sensitivity['reasons']}")
        summary = {
            "lightgbm_version": lgb.__version__, "python_version": platform.python_version(),
            "log_path": str(log_path), "stages": stages, "full_metrics": full_metrics,
            "full_evaluation_seconds": evaluation_seconds, "evaluation_peak_memory_bytes": evaluation_guard.peak,
            "feature_importance": importance, "baseline": baseline,
            "sensitivity": sensitivity,
            "conclusion": build_conclusion(full_metrics, baseline),
        }
        render_report(summary, ROOT / "reports/lightgbm_model_report.md")
        summary_path = ROOT / "logs/lightgbm" / "formal_run_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        clean_transient_files()
        write_storage_manifest(summary, Path(__file__))
        logger.info("LightGBM training and full evaluation completed in %.1f seconds", evaluation_seconds)
    except Exception:
        logger.exception("LightGBM pipeline failed; final public dataset was not modified")
        raise


if __name__ == "__main__":
    main()
