from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class MCLevelConfig:
    codes: tuple[int, ...]
    names: tuple[str, ...]
    lower_bounds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.codes) == len(self.names) == len(self.lower_bounds)):
            raise ValueError("MC codes, names and lower bounds must have equal length")
        if tuple(sorted(self.lower_bounds)) != self.lower_bounds or self.lower_bounds[0] != 0:
            raise ValueError("MC lower bounds must be sorted and start at zero")


def load_mc_level_config(path: str | Path) -> MCLevelConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    levels = payload["levels"]
    return MCLevelConfig(
        codes=tuple(int(level["code"]) for level in levels),
        names=tuple(str(level["name"]) for level in levels),
        lower_bounds=tuple(int(level["min_qty"]) for level in levels),
    )


def rounded_horizon_quantity(values, horizon: str) -> np.ndarray:
    quantity = np.asarray(values, dtype="float64")
    if horizon == "2m":
        quantity = quantity / 2.0
    elif horizon != "1m":
        raise ValueError(f"Unsupported horizon: {horizon}")
    quantity = np.nan_to_num(quantity, nan=0.0, posinf=0.0, neginf=0.0)
    return np.floor(np.clip(quantity, 0.0, None) + 0.5).astype("int64")


def quantity_to_mc(values, config: MCLevelConfig, horizon: str) -> np.ndarray:
    rounded = rounded_horizon_quantity(values, horizon)
    boundaries = np.asarray(config.lower_bounds[1:], dtype="int64")
    indices = np.searchsorted(boundaries, rounded, side="right")
    codes = np.asarray(config.codes, dtype="int32")
    return codes[indices]


def classification_metrics_from_matrix(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype="int64")
    count = int(matrix.sum())
    true_count = matrix.sum(axis=1)
    pred_count = matrix.sum(axis=0)
    tp = np.diag(matrix).astype("float64")
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count != 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    weights = np.divide(true_count, count, out=np.zeros_like(true_count, dtype="float64"), where=count != 0)
    distance = np.abs(np.arange(matrix.shape[0])[:, None] - np.arange(matrix.shape[1])[None, :])
    return {
        "count": count,
        "accuracy": float(tp.sum() / count) if count else np.nan,
        "macro_f1": float(f1.mean()) if len(f1) else np.nan,
        "weighted_f1": float(np.sum(f1 * weights)) if count else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix.copy(),
        "adjacent_error_rate": float(matrix[distance == 1].sum() / count) if count else np.nan,
        "severe_cross_level_error_rate": float(matrix[distance >= 2].sum() / count) if count else np.nan,
    }


class MCClassificationAccumulator:
    def __init__(self, config: MCLevelConfig) -> None:
        self.config = config
        self.matrix = np.zeros((len(config.codes), len(config.codes)), dtype="int64")
        self._code_to_index = {code: index for index, code in enumerate(config.codes)}

    def update(self, true_codes, pred_codes) -> None:
        true = np.asarray(true_codes, dtype="int32")
        pred = np.asarray(pred_codes, dtype="int32")
        if true.shape != pred.shape:
            raise ValueError("MC target and prediction must have the same shape")
        true_idx = np.array([self._code_to_index.get(int(value), -1) for value in true], dtype="int32")
        pred_idx = np.array([self._code_to_index.get(int(value), -1) for value in pred], dtype="int32")
        valid = (true_idx >= 0) & (pred_idx >= 0)
        flat = true_idx[valid] * len(self.config.codes) + pred_idx[valid]
        self.matrix += np.bincount(flat, minlength=self.matrix.size).reshape(self.matrix.shape)

    def compute(self) -> dict:
        result = classification_metrics_from_matrix(self.matrix)
        high_index = len(self.config.codes) - 1
        denom = self.matrix[high_index].sum()
        result["high_level_recall"] = (
            float(self.matrix[high_index, high_index] / denom) if denom else np.nan
        )
        return result


class TrendAccumulator:
    LABELS = (-1, 0, 1)

    def __init__(self, flat_tolerance: float = 0.0) -> None:
        self.flat_tolerance = float(flat_tolerance)
        self.matrix = np.zeros((3, 3), dtype="int64")

    def _direction(self, current, future) -> np.ndarray:
        delta = np.asarray(future, dtype="float64") - np.asarray(current, dtype="float64")
        return np.where(delta > self.flat_tolerance, 1, np.where(delta < -self.flat_tolerance, -1, 0))

    def update(self, current, future_true, future_pred) -> None:
        true = self._direction(current, future_true)
        pred = self._direction(current, future_pred)
        true_idx = true + 1
        pred_idx = pred + 1
        flat = true_idx * 3 + pred_idx
        self.matrix += np.bincount(flat, minlength=9).reshape((3, 3))

    def compute(self) -> dict:
        result = classification_metrics_from_matrix(self.matrix)
        total = int(self.matrix.sum())
        severe = int(self.matrix[0, 2] + self.matrix[2, 0])
        result["up_recall"] = float(result["recall"][2])
        result["down_recall"] = float(result["recall"][0])
        result["severe_direction_error_rate"] = severe / total if total else np.nan
        return result
