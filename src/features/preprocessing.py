from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_CONFIG = PROJECT_ROOT / "config" / "model_features.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "model_dataset_monthly.parquet"
TIME_INDEX_BASE_MONTH = "2023-01"

MODEL_FEATURE_KEYS = {
    "historical_mean": "historical_mean_features",
    "weighted_moving_average": "weighted_moving_average_features",
    "random_forest": "random_forest_features",
    "lightgbm": "lightgbm_features",
    "mlp": "mlp_features",
    "two_stage": "two_stage_features",
}

TREE_MODELS = {"random_forest", "lightgbm", "two_stage"}


@dataclass
class PreprocessingState:
    model_name: str
    feature_columns: list[str]
    categorical_features: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    category_maps: dict[str, dict[Any, int]] = field(default_factory=dict)
    mlp_means: dict[str, float] = field(default_factory=dict)
    mlp_stds: dict[str, float] = field(default_factory=dict)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_feature_config(config_path: str | Path = DEFAULT_FEATURE_CONFIG) -> dict:
    with project_path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("-", "_")
    aliases = {
        "mean": "historical_mean",
        "historical_mean_features": "historical_mean",
        "weighted_ma": "weighted_moving_average",
        "weighted_moving_average_features": "weighted_moving_average",
        "rf": "random_forest",
        "random_forest_features": "random_forest",
        "lgbm": "lightgbm",
        "lightgbm_features": "lightgbm",
        "mlp_features": "mlp",
        "two_stage_features": "two_stage",
    }
    key = aliases.get(key, key)
    if key not in MODEL_FEATURE_KEYS:
        supported = ", ".join(sorted(MODEL_FEATURE_KEYS))
        raise ValueError(f"Unsupported model_name={model_name!r}. Supported names: {supported}")
    return key


def get_feature_list(model_name: str, config: dict | None = None) -> list[str]:
    config = config or load_feature_config()
    normalized = normalize_model_name(model_name)
    feature_key = MODEL_FEATURE_KEYS[normalized]
    features = list(config.get(feature_key, []))
    excluded = set(config.get("excluded_from_all_features", []))
    excluded.update(config.get("excluded_v1_anomalous_fields", []))
    leaked = [feature for feature in features if feature in excluded]
    if leaked:
        raise ValueError(f"{feature_key} contains excluded fields: {leaked}")
    return features


def required_columns_for_model(model_name: str, config: dict | None = None) -> list[str]:
    config = config or load_feature_config()
    features = get_feature_list(model_name, config)
    runtime_columns = set(config.get("runtime_time_features", []))
    required = set(features) - runtime_columns
    required.add("month")
    required.add(config.get("split_field", "split"))
    for target in config.get("target_fields", {}).get("regression", []):
        required.add(target)
    for target in config.get("target_fields", {}).get("classification", []):
        required.add(target)
    return sorted(required)


def read_model_columns(
    model_name: str,
    dataset_path: str | Path = DEFAULT_DATASET,
    config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    filters: list[tuple] | list[list[tuple]] | None = None,
) -> pd.DataFrame:
    config = load_feature_config(config_path)
    columns = required_columns_for_model(model_name, config)
    return pd.read_parquet(project_path(dataset_path), columns=columns, filters=filters)


def add_runtime_columns(df: pd.DataFrame, base_month: str = TIME_INDEX_BASE_MONTH) -> pd.DataFrame:
    df = df.copy()
    month_period = pd.PeriodIndex(df["month"].astype(str), freq="M")
    future_1m = month_period + 1
    future_2m = month_period + 2
    base_ordinal = pd.Period(base_month, freq="M").ordinal
    if "future_qty_1m" in df.columns:
        df["target_qty_1m"] = np.maximum(pd.to_numeric(df["future_qty_1m"], errors="coerce"), 0)
    if "future_qty_2m" in df.columns:
        df["target_qty_2m"] = np.maximum(pd.to_numeric(df["future_qty_2m"], errors="coerce"), 0)
    df["year"] = month_period.year.astype("int16")
    df["month_of_year"] = month_period.month.astype("int8")
    df["quarter"] = month_period.quarter.astype("int8")
    df["time_index"] = (month_period.asi8 - base_ordinal).astype("int16")
    df["future_1m_year"] = future_1m.year.astype("int16")
    df["future_1m_month_of_year"] = future_1m.month.astype("int8")
    df["future_1m_quarter"] = future_1m.quarter.astype("int8")
    df["future_1m_time_index"] = (future_1m.asi8 - base_ordinal).astype("int16")
    df["future_2m_month_of_year"] = future_2m.month.astype("int8")
    df["future_2m_quarter"] = future_2m.quarter.astype("int8")
    df["future_2m_time_index"] = (future_2m.asi8 - base_ordinal).astype("int16")
    return df


def _feature_types(model_name: str, features: list[str], config: dict, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    normalized = normalize_model_name(model_name)
    configured_categoricals = set(config.get("categorical_features", []))
    if normalized == "mlp":
        categorical_features: list[str] = []
    else:
        categorical_features = [feature for feature in features if feature in configured_categoricals]
    numeric_features = [feature for feature in features if feature not in categorical_features]
    non_numeric = [
        feature
        for feature in numeric_features
        if feature in df.columns and not pd.api.types.is_numeric_dtype(df[feature])
    ]
    if non_numeric:
        raise ValueError(f"Numeric feature list contains non-numeric columns: {non_numeric}")
    return numeric_features, categorical_features


def _fit_category_maps(df: pd.DataFrame, categorical_features: list[str]) -> dict[str, dict[Any, int]]:
    maps: dict[str, dict[Any, int]] = {}
    for col in categorical_features:
        values = df[col].fillna("unknown").astype("string")
        uniques = sorted(value for value in values.unique().tolist() if value is not pd.NA)
        maps[col] = {value: idx for idx, value in enumerate(uniques)}
    return maps


def _apply_category_maps(df: pd.DataFrame, category_maps: dict[str, dict[Any, int]]) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in category_maps.items():
        df[col] = df[col].fillna("unknown").astype("string").map(mapping).fillna(-1).astype("int32")
    return df


def _fit_mlp_scaler(df: pd.DataFrame, numeric_features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
        if not np.isfinite(std) or std == 0:
            std = 1.0
        means[col] = mean
        stds[col] = std
    return means, stds


def _apply_mlp_scaler(df: pd.DataFrame, means: dict[str, float], stds: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for col, mean in means.items():
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) - mean) / stds[col]
    return df


def preprocess_frame(
    df: pd.DataFrame,
    model_name: str,
    config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    state: PreprocessingState | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingState]:
    config = load_feature_config(config_path)
    normalized = normalize_model_name(model_name)
    features = get_feature_list(normalized, config)
    df = add_runtime_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_features, categorical_features = _feature_types(normalized, features, config, df)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in categorical_features:
        df[col] = df[col].fillna("unknown").astype("string")

    if state is None:
        train_mask = df[config.get("split_field", "split")].eq("train") if "split" in df.columns else pd.Series(True, index=df.index)
        fit_df = df.loc[train_mask]
        state = PreprocessingState(
            model_name=normalized,
            feature_columns=features,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
        if normalized in TREE_MODELS:
            state.category_maps = _fit_category_maps(fit_df, categorical_features)
        if normalized == "mlp":
            state.mlp_means, state.mlp_stds = _fit_mlp_scaler(fit_df, numeric_features)
    elif fit:
        raise ValueError("Pass either state or fit=True, not both.")

    if normalized in TREE_MODELS:
        df = _apply_category_maps(df, state.category_maps)
    if normalized == "mlp":
        df = _apply_mlp_scaler(df, state.mlp_means, state.mlp_stds)

    targets = df[
        [
            "future_qty_1m",
            "future_qty_2m",
            "future_has_sales_1m",
            "future_has_sales_2m",
            "target_qty_1m",
            "target_qty_2m",
            "split",
        ]
    ].copy()
    return df[features].copy(), targets, state


def load_preprocessed_data(
    model_name: str,
    dataset_path: str | Path = DEFAULT_DATASET,
    config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    filters: list[tuple] | list[list[tuple]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingState]:
    raw = read_model_columns(model_name, dataset_path=dataset_path, config_path=config_path, filters=filters)
    return preprocess_frame(raw, model_name=model_name, config_path=config_path)
