from __future__ import annotations

from src.evaluation.mc_metrics import (
    MCClassificationAccumulator,
    MCLevelConfig,
    TrendAccumulator,
    quantity_to_mc,
)
from src.evaluation.metrics import LongTailStreamingMetrics, clip_target


class BusinessEvaluationAccumulator:
    """Accumulate the project's quantity, MC and trend metrics in one pass."""

    def __init__(self, mc_config: MCLevelConfig, horizon: str) -> None:
        if horizon not in {"1m", "2m"}:
            raise ValueError(f"Unsupported horizon: {horizon}")
        self.mc_config = mc_config
        self.horizon = horizon
        self.quantity = LongTailStreamingMetrics()
        self.mc = MCClassificationAccumulator(mc_config)
        self.trend = TrendAccumulator(flat_tolerance=0.0)

    def update(self, target, prediction, current) -> None:
        target_values = clip_target(target)
        prediction_values = clip_target(prediction)
        current_values = clip_target(current)
        if not (target_values.shape == prediction_values.shape == current_values.shape):
            raise ValueError("target, prediction and current must have the same shape")

        self.quantity.update(target_values, prediction_values)
        self.mc.update(
            quantity_to_mc(target_values, self.mc_config, self.horizon),
            quantity_to_mc(prediction_values, self.mc_config, self.horizon),
        )
        if self.horizon == "2m":
            target_values = target_values / 2.0
            prediction_values = prediction_values / 2.0
        self.trend.update(current_values, target_values, prediction_values)

    def compute(self) -> dict:
        return {
            "quantity": self.quantity.compute(),
            "mc": self.mc.compute(),
            "trend": self.trend.compute(),
        }
