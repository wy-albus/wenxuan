from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEMAND_BUCKETS = ("0", "1", "2-5", "5-20", "20+")


def clip_target(values) -> np.ndarray:
    numeric = np.nan_to_num(np.asarray(values, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(numeric, 0.0)


def demand_bucket(target) -> np.ndarray:
    target = clip_target(target)
    return np.select(
        [target == 0, target < 2, target < 5, target < 20],
        DEMAND_BUCKETS[:4],
        default=DEMAND_BUCKETS[4],
    )


@dataclass
class StreamingMetrics:
    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_smape: float = 0.0
    sum_abs_target: float = 0.0
    target_sum: float = 0.0
    prediction_sum: float = 0.0

    def update(self, target, prediction) -> None:
        target = clip_target(target)
        prediction = clip_target(prediction)
        if target.shape != prediction.shape:
            raise ValueError("target and prediction must have the same shape")

        error = prediction - target
        absolute_error = np.abs(error)
        denominator = np.abs(target) + np.abs(prediction)
        smape = np.divide(
            2.0 * absolute_error,
            denominator,
            out=np.zeros_like(absolute_error),
            where=denominator != 0,
        )
        self.count += int(target.size)
        self.sum_abs_error += float(absolute_error.sum(dtype="float64"))
        self.sum_squared_error += float(np.square(error).sum(dtype="float64"))
        self.sum_smape += float(smape.sum(dtype="float64"))
        self.sum_abs_target += float(np.abs(target).sum(dtype="float64"))
        self.target_sum += float(target.sum(dtype="float64"))
        self.prediction_sum += float(prediction.sum(dtype="float64"))

    def merge(self, other: "StreamingMetrics") -> None:
        self.count += other.count
        self.sum_abs_error += other.sum_abs_error
        self.sum_squared_error += other.sum_squared_error
        self.sum_smape += other.sum_smape
        self.sum_abs_target += other.sum_abs_target
        self.target_sum += other.target_sum
        self.prediction_sum += other.prediction_sum

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0, "mae": np.nan, "rmse": np.nan, "smape": np.nan, "wape": np.nan,
                "target_sum": 0.0, "prediction_sum": 0.0, "total_bias_rate": np.nan,
            }
        if self.sum_abs_target == 0:
            wape = 0.0 if self.sum_abs_error == 0 else np.inf
        else:
            wape = 100.0 * self.sum_abs_error / self.sum_abs_target
        return {
            "count": self.count,
            "mae": self.sum_abs_error / self.count,
            "rmse": np.sqrt(self.sum_squared_error / self.count),
            "smape": 100.0 * self.sum_smape / self.count,
            "wape": wape,
            "target_sum": self.target_sum,
            "prediction_sum": self.prediction_sum,
            "total_bias_rate": (
                100.0 * (self.prediction_sum - self.target_sum) / self.target_sum
                if self.target_sum != 0
                else np.nan
            ),
        }


class GroupedStreamingMetrics:
    def __init__(self) -> None:
        self.overall = StreamingMetrics()
        self.by_bucket = {bucket: StreamingMetrics() for bucket in DEMAND_BUCKETS}

    def update(self, target, prediction) -> None:
        target = clip_target(target)
        prediction = clip_target(prediction)
        self.overall.update(target, prediction)
        buckets = demand_bucket(target)
        for bucket, accumulator in self.by_bucket.items():
            mask = buckets == bucket
            if mask.any():
                accumulator.update(target[mask], prediction[mask])

    def compute(self) -> dict[str, dict[str, float | int]]:
        return {
            "overall": self.overall.compute(),
            **{bucket: accumulator.compute() for bucket, accumulator in self.by_bucket.items()},
        }


class LongTailStreamingMetrics:
    """Streaming metrics for business demand buckets and head/tail segments."""

    def __init__(self) -> None:
        self.segments = {
            "overall": StreamingMetrics(),
            **{bucket: StreamingMetrics() for bucket in DEMAND_BUCKETS},
            "nonzero": StreamingMetrics(),
            "ge_5": StreamingMetrics(),
            "ge_20": StreamingMetrics(),
        }
        self.zero_prediction_gt_0_5 = 0
        self.zero_prediction_gt_1 = 0

    def update(self, target, prediction) -> None:
        target = clip_target(target)
        prediction = clip_target(prediction)
        if target.shape != prediction.shape:
            raise ValueError("target and prediction must have the same shape")
        self.segments["overall"].update(target, prediction)
        buckets = demand_bucket(target)
        for bucket in DEMAND_BUCKETS:
            mask = buckets == bucket
            if mask.any():
                self.segments[bucket].update(target[mask], prediction[mask])
        for name, mask in (
            ("nonzero", target > 0),
            ("ge_5", target >= 5),
            ("ge_20", target >= 20),
        ):
            if mask.any():
                self.segments[name].update(target[mask], prediction[mask])
        zero_prediction = prediction[target == 0]
        self.zero_prediction_gt_0_5 += int((zero_prediction > 0.5).sum())
        self.zero_prediction_gt_1 += int((zero_prediction > 1.0).sum())

    def compute(self) -> dict[str, dict[str, float | int]]:
        result = {name: accumulator.compute() for name, accumulator in self.segments.items()}
        zero = result["0"]
        zero_count = int(zero["count"])
        zero["wape"] = np.nan
        zero["mean_prediction"] = zero["prediction_sum"] / zero_count if zero_count else np.nan
        zero["prediction_gt_0_5_rate"] = self.zero_prediction_gt_0_5 / zero_count if zero_count else np.nan
        zero["prediction_gt_1_rate"] = self.zero_prediction_gt_1 / zero_count if zero_count else np.nan
        return result
