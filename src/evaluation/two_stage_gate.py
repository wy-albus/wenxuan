from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SEGMENTS = ("overall", "zero", "qty_1", "qty_2_5", "nonzero", "5_20", "20plus")


def _segment_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "overall": np.ones(target.shape, dtype=bool),
        "zero": target <= 0,
        "qty_1": target == 1,
        "qty_2_5": (target >= 2) & (target < 5),
        "nonzero": target > 0,
        "5_20": (target >= 5) & (target < 20),
        "20plus": target >= 20,
    }


@dataclass
class _BinnedStats:
    size: int
    count: np.ndarray = field(init=False)
    target: np.ndarray = field(init=False)
    target_sq: np.ndarray = field(init=False)
    original_abs: np.ndarray = field(init=False)
    original_sq: np.ndarray = field(init=False)
    original_pred: np.ndarray = field(init=False)
    gate_b_abs: np.ndarray = field(init=False)
    gate_b_sq: np.ndarray = field(init=False)
    gate_b_pred: np.ndarray = field(init=False)
    original_gt_0_5: np.ndarray = field(init=False)
    original_gt_1: np.ndarray = field(init=False)
    gate_b_gt_0_5: np.ndarray = field(init=False)
    gate_b_gt_1: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "count", "target", "target_sq", "original_abs", "original_sq", "original_pred",
            "gate_b_abs", "gate_b_sq", "gate_b_pred", "original_gt_0_5", "original_gt_1",
            "gate_b_gt_0_5", "gate_b_gt_1",
        ):
            setattr(self, name, np.zeros(self.size, dtype="float64"))

    def add(self, bins: np.ndarray, target: np.ndarray, original: np.ndarray, conditional: np.ndarray) -> None:
        values = {
            "count": np.ones(target.shape, dtype="float64"),
            "target": target,
            "target_sq": np.square(target),
            "original_abs": np.abs(target - original),
            "original_sq": np.square(target - original),
            "original_pred": original,
            "gate_b_abs": np.abs(target - conditional),
            "gate_b_sq": np.square(target - conditional),
            "gate_b_pred": conditional,
            "original_gt_0_5": original > 0.5,
            "original_gt_1": original > 1.0,
            "gate_b_gt_0_5": conditional > 0.5,
            "gate_b_gt_1": conditional > 1.0,
        }
        for name, value in values.items():
            getattr(self, name)[:] += np.bincount(bins, weights=np.asarray(value, dtype="float64"), minlength=self.size)


class ThresholdGateAccumulator:
    """Accumulate every gate threshold after one classifier/regressor inference pass."""

    def __init__(self, thresholds: np.ndarray) -> None:
        self.thresholds = np.asarray(thresholds, dtype="float64")
        if self.thresholds.ndim != 1 or len(self.thresholds) == 0:
            raise ValueError("thresholds must be a non-empty vector")
        if not np.all(np.diff(self.thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing")
        self.stats = {name: _BinnedStats(len(self.thresholds) + 1) for name in SEGMENTS}

    def update(self, target, probability, conditional) -> None:
        target = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
        probability = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
        conditional = np.clip(np.asarray(conditional, dtype="float64"), 0.0, None)
        if not (target.shape == probability.shape == conditional.shape):
            raise ValueError("target, probability and conditional must have matching shapes")
        if not (np.isfinite(target).all() and np.isfinite(probability).all() and np.isfinite(conditional).all()):
            raise ValueError("gate inputs must be finite")
        bins = np.searchsorted(self.thresholds, probability, side="right")
        original = probability * conditional
        for name, mask in _segment_masks(target).items():
            if mask.any():
                self.stats[name].add(bins[mask], target[mask], original[mask], conditional[mask])

    @staticmethod
    def _suffix(values: np.ndarray) -> np.ndarray:
        return np.cumsum(values[::-1])[::-1]

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    def _prediction_row(self, method: str, threshold_index: int | None) -> dict[str, float | int | str]:
        threshold = None if threshold_index is None else float(self.thresholds[threshold_index])
        metrics: dict[str, dict[str, float]] = {}
        for segment, stats in self.stats.items():
            count = float(stats.count.sum())
            target_sum = float(stats.target.sum())
            if method == "original":
                absolute = float(stats.original_abs.sum())
                squared = float(stats.original_sq.sum())
                prediction_sum = float(stats.original_pred.sum())
            else:
                assert threshold_index is not None
                gated_absolute = float(stats.target[: threshold_index + 1].sum())
                gated_squared = float(stats.target_sq[: threshold_index + 1].sum())
                suffix_start = threshold_index + 1
                if method == "gate_a":
                    absolute = gated_absolute + float(stats.original_abs[suffix_start:].sum())
                    squared = gated_squared + float(stats.original_sq[suffix_start:].sum())
                    prediction_sum = float(stats.original_pred[suffix_start:].sum())
                else:
                    absolute = gated_absolute + float(stats.gate_b_abs[suffix_start:].sum())
                    squared = gated_squared + float(stats.gate_b_sq[suffix_start:].sum())
                    prediction_sum = float(stats.gate_b_pred[suffix_start:].sum())
            metrics[segment] = {
                "count": count,
                "target_sum": target_sum,
                "absolute": absolute,
                "squared": squared,
                "prediction_sum": prediction_sum,
                "wape": 100.0 * self._safe_ratio(absolute, target_sum),
            }

        overall = metrics["overall"]
        row: dict[str, float | int | str] = {
            "method": method,
            "threshold": np.nan if threshold is None else threshold,
            "count": int(overall["count"]),
            "mae": self._safe_ratio(overall["absolute"], overall["count"]),
            "rmse": float(np.sqrt(self._safe_ratio(overall["squared"], overall["count"]))),
            "wape": overall["wape"],
            "target_sum": overall["target_sum"],
            "prediction_sum": overall["prediction_sum"],
            "total_bias": 100.0 * self._safe_ratio(overall["prediction_sum"] - overall["target_sum"], overall["target_sum"]),
            "nonzero_wape": metrics["nonzero"]["wape"],
            "qty_1_wape": metrics["qty_1"]["wape"],
            "qty_1_mae": self._safe_ratio(metrics["qty_1"]["absolute"], metrics["qty_1"]["count"]),
            "qty_2_5_wape": metrics["qty_2_5"]["wape"],
            "qty_2_5_mae": self._safe_ratio(metrics["qty_2_5"]["absolute"], metrics["qty_2_5"]["count"]),
            "5_20_wape": metrics["5_20"]["wape"],
            "20plus_wape": metrics["20plus"]["wape"],
            "count_qty_1": int(metrics["qty_1"]["count"]),
            "count_qty_2_5": int(metrics["qty_2_5"]["count"]),
            "count_5_20": int(metrics["5_20"]["count"]),
            "count_20plus": int(metrics["20plus"]["count"]),
        }

        classification_index = int(np.searchsorted(self.thresholds, 0.5, side="left"))
        classification_index = min(classification_index, len(self.thresholds) - 1)
        index = classification_index if threshold_index is None else threshold_index
        positive_pass = float(self.stats["nonzero"].count[index + 1 :].sum())
        zero_pass = float(self.stats["zero"].count[index + 1 :].sum())
        positive_count = float(self.stats["nonzero"].count.sum())
        row["precision"] = self._safe_ratio(positive_pass, positive_pass + zero_pass)
        row["recall"] = self._safe_ratio(positive_pass, positive_count)
        row["5_20_gate_recall"] = self._safe_ratio(
            float(self.stats["5_20"].count[index + 1 :].sum()), float(self.stats["5_20"].count.sum())
        )
        row["20plus_gate_recall"] = self._safe_ratio(
            float(self.stats["20plus"].count[index + 1 :].sum()), float(self.stats["20plus"].count.sum())
        )

        zero = self.stats["zero"]
        if method == "original":
            row["zero_pred_total"] = float(zero.original_pred.sum())
            row["zero_pred_gt_0_5"] = self._safe_ratio(float(zero.original_gt_0_5.sum()), float(zero.count.sum()))
            row["zero_pred_gt_1"] = self._safe_ratio(float(zero.original_gt_1.sum()), float(zero.count.sum()))
        else:
            assert threshold_index is not None
            start = threshold_index + 1
            pred = zero.original_pred if method == "gate_a" else zero.gate_b_pred
            gt_0_5 = zero.original_gt_0_5 if method == "gate_a" else zero.gate_b_gt_0_5
            gt_1 = zero.original_gt_1 if method == "gate_a" else zero.gate_b_gt_1
            row["zero_pred_total"] = float(pred[start:].sum())
            row["zero_pred_gt_0_5"] = self._safe_ratio(float(gt_0_5[start:].sum()), float(zero.count.sum()))
            row["zero_pred_gt_1"] = self._safe_ratio(float(gt_1[start:].sum()), float(zero.count.sum()))
        return row

    def rows(self) -> list[dict[str, float | int | str]]:
        rows = [self._prediction_row("original", None)]
        for method in ("gate_a", "gate_b"):
            rows.extend(self._prediction_row(method, index) for index in range(len(self.thresholds)))
        return rows


@dataclass
class _GateAStats:
    size: int
    count: np.ndarray = field(init=False)
    target: np.ndarray = field(init=False)
    target_sq: np.ndarray = field(init=False)
    absolute: np.ndarray = field(init=False)
    squared: np.ndarray = field(init=False)
    prediction: np.ndarray = field(init=False)
    prediction_gt_0_5: np.ndarray = field(init=False)
    prediction_gt_1: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "count", "target", "target_sq", "absolute", "squared", "prediction",
            "prediction_gt_0_5", "prediction_gt_1",
        ):
            setattr(self, name, np.zeros(self.size, dtype="float64"))

    def add(self, bins: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
        values = {
            "count": np.ones(target.shape, dtype="float64"),
            "target": target,
            "target_sq": np.square(target),
            "absolute": np.abs(target - prediction),
            "squared": np.square(target - prediction),
            "prediction": prediction,
            "prediction_gt_0_5": prediction > 0.5,
            "prediction_gt_1": prediction > 1.0,
        }
        for name, value in values.items():
            getattr(self, name)[:] += np.bincount(
                bins, weights=np.asarray(value, dtype="float64"), minlength=self.size
            )


class GateATradeoffAccumulator:
    """Accumulate Original and Gate-A business metrics without evaluating Gate-B."""

    SEGMENTS = ("overall", "zero", "nonzero", "5_20", "20plus")

    def __init__(self, thresholds: np.ndarray) -> None:
        self.thresholds = np.asarray(thresholds, dtype="float64")
        if self.thresholds.ndim != 1 or len(self.thresholds) == 0:
            raise ValueError("thresholds must be a non-empty vector")
        if not np.all(np.diff(self.thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing")
        self.stats = {
            name: _GateAStats(len(self.thresholds) + 1) for name in self.SEGMENTS
        }

    @staticmethod
    def _masks(target: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "overall": np.ones(target.shape, dtype=bool),
            "zero": target == 0,
            "nonzero": target > 0,
            "5_20": (target >= 5) & (target < 20),
            "20plus": target >= 20,
        }

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    def update(self, target, probability, prediction) -> None:
        target = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
        probability = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
        prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, None)
        if not (target.shape == probability.shape == prediction.shape):
            raise ValueError("target, probability and prediction must have matching shapes")
        if not (
            np.isfinite(target).all()
            and np.isfinite(probability).all()
            and np.isfinite(prediction).all()
        ):
            raise ValueError("gate inputs must be finite")
        bins = np.searchsorted(self.thresholds, probability, side="right")
        for name, mask in self._masks(target).items():
            if mask.any():
                self.stats[name].add(bins[mask], target[mask], prediction[mask])

    def _segment_metrics(self, name: str, threshold_index: int | None) -> dict[str, float]:
        stats = self.stats[name]
        count = float(stats.count.sum())
        target_sum = float(stats.target.sum())
        if threshold_index is None:
            absolute = float(stats.absolute.sum())
            squared = float(stats.squared.sum())
            prediction_sum = float(stats.prediction.sum())
            gt_0_5 = float(stats.prediction_gt_0_5.sum())
            gt_1 = float(stats.prediction_gt_1.sum())
        else:
            start = threshold_index + 1
            absolute = float(stats.target[:start].sum() + stats.absolute[start:].sum())
            squared = float(stats.target_sq[:start].sum() + stats.squared[start:].sum())
            prediction_sum = float(stats.prediction[start:].sum())
            gt_0_5 = float(stats.prediction_gt_0_5[start:].sum())
            gt_1 = float(stats.prediction_gt_1[start:].sum())
        return {
            "count": count,
            "target_sum": target_sum,
            "absolute": absolute,
            "squared": squared,
            "prediction_sum": prediction_sum,
            "wape": 100.0 * self._ratio(absolute, target_sum),
            "gt_0_5": gt_0_5,
            "gt_1": gt_1,
        }

    def _row(self, threshold_index: int | None) -> dict:
        metrics = {
            name: self._segment_metrics(name, threshold_index) for name in self.SEGMENTS
        }
        overall = metrics["overall"]
        zero = metrics["zero"]
        row = {
            "method": "original" if threshold_index is None else "gate_a",
            "threshold": np.nan if threshold_index is None else float(self.thresholds[threshold_index]),
            "count": int(overall["count"]),
            "mae": self._ratio(overall["absolute"], overall["count"]),
            "rmse": float(np.sqrt(self._ratio(overall["squared"], overall["count"]))),
            "overall_wape": overall["wape"],
            "nonzero_wape": metrics["nonzero"]["wape"],
            "5_20_wape": metrics["5_20"]["wape"],
            "20plus_wape": metrics["20plus"]["wape"],
            "target_sum": overall["target_sum"],
            "prediction_sum": overall["prediction_sum"],
            "total_bias": 100.0 * self._ratio(
                overall["prediction_sum"] - overall["target_sum"], overall["target_sum"]
            ),
            "zero_sample_count": int(zero["count"]),
            "zero_mean_prediction": self._ratio(zero["prediction_sum"], zero["count"]),
            "zero_predicted_total": zero["prediction_sum"],
            "zero_prediction_gt_0_5_rate": self._ratio(zero["gt_0_5"], zero["count"]),
            "zero_prediction_gt_1_rate": self._ratio(zero["gt_1"], zero["count"]),
        }
        if threshold_index is None:
            row["5_20_gate_pass_rate"] = np.nan
            row["20plus_gate_pass_rate"] = np.nan
        else:
            start = threshold_index + 1
            for name in ("5_20", "20plus"):
                stats = self.stats[name]
                row[f"{name}_gate_pass_rate"] = self._ratio(
                    float(stats.count[start:].sum()), float(stats.count.sum())
                )
        return row

    def rows(self) -> list[dict]:
        return [self._row(None), *(self._row(index) for index in range(len(self.thresholds)))]


class ProbabilityDiagnostic:
    def __init__(self, resolution: int = 10_000) -> None:
        self.resolution = resolution
        self.histograms = {
            (basis, segment): np.zeros(resolution + 1, dtype="int64")
            for basis in ("target_qty_1m", "qty_lag_1m")
            for segment in ("0", "1", "2-5", "5-20", "20+")
        }
        self.sums = {key: 0.0 for key in self.histograms}

    @staticmethod
    def _demand_masks(values: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "0": values <= 0,
            "1": values == 1,
            "2-5": (values >= 2) & (values < 5),
            "5-20": (values >= 5) & (values < 20),
            "20+": values >= 20,
        }

    def update(self, target, lag, probability) -> None:
        probability = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
        bins = np.minimum((probability * self.resolution).astype("int64"), self.resolution)
        for basis, values in (("target_qty_1m", target), ("qty_lag_1m", lag)):
            values = np.asarray(values, dtype="float64")
            for segment, mask in self._demand_masks(values).items():
                if mask.any():
                    key = (basis, segment)
                    self.histograms[key] += np.bincount(bins[mask], minlength=self.resolution + 1)
                    self.sums[key] += float(probability[mask].sum())

    def _quantile(self, histogram: np.ndarray, quantile: float) -> float:
        count = int(histogram.sum())
        if not count:
            return float("nan")
        rank = max(1, int(np.ceil(quantile * count)))
        index = int(np.searchsorted(np.cumsum(histogram), rank, side="left"))
        return index / self.resolution

    def rows(self) -> list[dict[str, float | int | str]]:
        rows = []
        for (basis, segment), histogram in self.histograms.items():
            count = int(histogram.sum())
            cumulative = np.cumsum(histogram)
            below = lambda threshold: int(cumulative[min(int(np.ceil(threshold * self.resolution)) - 1, self.resolution)]) if count else 0
            rows.append({
                "basis": basis,
                "segment": segment,
                "count": count,
                "mean": self.sums[(basis, segment)] / count if count else np.nan,
                "p10": self._quantile(histogram, 0.10),
                "p25": self._quantile(histogram, 0.25),
                "median": self._quantile(histogram, 0.50),
                "p75": self._quantile(histogram, 0.75),
                "p90": self._quantile(histogram, 0.90),
                "p_sale_lt_0_20_rate": below(0.20) / count if count else np.nan,
                "p_sale_lt_0_30_rate": below(0.30) / count if count else np.nan,
                "p_sale_lt_0_50_rate": below(0.50) / count if count else np.nan,
                "p_sale_ge_0_50_rate": 1.0 - below(0.50) / count if count else np.nan,
            })
        return rows


def select_gate_candidate(rows: list[dict]) -> dict:
    gated = [row for row in rows if row["method"] in {"gate_a", "gate_b"}]
    finalists = []
    for method in ("gate_a", "gate_b"):
        method_rows = [row for row in gated if row["method"] == method]
        coarse = [row for row in method_rows if np.isclose((float(row["threshold"]) * 100) % 5, 0.0, atol=1e-8)]
        if not coarse:
            coarse = method_rows
        coarse_best = min(coarse, key=lambda row: float(row["wape"]))
        center = float(coarse_best["threshold"])
        local = [row for row in method_rows if abs(float(row["threshold"]) - center) <= 0.0500001]
        finalists.append(min(local, key=lambda row: (float(row["wape"]), float(row["threshold"]))))
    return min(finalists, key=lambda row: (float(row["wape"]), row["method"], float(row["threshold"])))


def select_safe_gate_candidate(rows: list[dict], min_head_recall: float = 0.95) -> dict:
    safe = [
        row for row in rows
        if row["method"] in {"gate_a", "gate_b"}
        and float(row["5_20_gate_recall"]) >= min_head_recall
        and float(row["20plus_gate_recall"]) >= min_head_recall
    ]
    if not safe:
        raise RuntimeError("No gated threshold satisfies the head-demand safety constraint")
    return min(safe, key=lambda row: (float(row["wape"]), row["method"], float(row["threshold"])))


def should_evaluate_test(
    original: dict,
    selected: dict,
    min_relative_wape_gain: float = 0.005,
    min_head_recall: float = 0.95,
) -> bool:
    relative_gain = (float(original["wape"]) - float(selected["wape"])) / float(original["wape"])
    return (
        relative_gain >= min_relative_wape_gain
        and float(selected["5_20_gate_recall"]) >= min_head_recall
        and float(selected["20plus_gate_recall"]) >= min_head_recall
    )


def protection_candidates(rows: list[dict], levels: list[float]) -> list[dict]:
    result = []
    for level in levels:
        for method in ("gate_a", "gate_b"):
            eligible = [
                row for row in rows
                if row["method"] == method and float(row["20plus_gate_recall"]) >= float(level)
            ]
            if not eligible:
                continue
            best = dict(min(eligible, key=lambda row: (float(row["wape"]), float(row["threshold"]))))
            best["min_20plus_pass_rate"] = float(level)
            result.append(best)
    return result


def classify_history_pattern(lag_1m: float, lag_2m: float, lag_3m: float) -> str:
    history = np.clip(np.asarray([lag_1m, lag_2m, lag_3m], dtype="float64"), 0.0, None)
    if np.all(history >= 5):
        return "persistent_high"
    if float(history.mean()) <= 1.0:
        return "sudden_burst_from_low"
    if float(history.max() - history.min()) >= 5.0:
        return "volatile_history"
    return "other"
