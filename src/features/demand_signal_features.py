from __future__ import annotations

import numpy as np
import pandas as pd


DIFF_FEATURES = [
    "qty_diff_current_lag1",
    "qty_diff_lag1_lag2",
    "log_qty_diff_current_lag1",
    "sales_days_diff_current_lag1",
    "qty_diff_recent3_previous3",
    "qty_diff_6m_history_available",
]

CROSS_STORE_FEATURES = [
    "xstore_qty_current",
    "xstore_positive_sites_current",
    "xstore_qty_per_positive_site_current",
    "xstore_qty_mean_last_3m",
    "xstore_qty_max_last_3m",
    "xstore_qty_diff_1m",
]


def cross_store_lookup_months(month: str) -> tuple[str, str, str]:
    observed = pd.Period(str(month), freq="M")
    return tuple(str(observed - offset) for offset in range(3))


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64", copy=False)
    )


def add_diff_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return leakage-safe demand-change features derived from existing history."""
    current = _numeric(frame, "total_qty")
    lag1 = _numeric(frame, "qty_lag_1m")
    lag2 = _numeric(frame, "qty_lag_2m")
    recent_sum = _numeric(frame, "qty_sum_last_3m")
    last_six_sum = _numeric(frame, "qty_sum_last_6m")
    sales_days = _numeric(frame, "sales_days")
    sales_days_lag1 = _numeric(frame, "sales_days_lag_1m")
    observed_six = _numeric(frame, "active_months_last_6m") + _numeric(
        frame, "zero_sales_months_last_6m"
    )
    history_available = np.isclose(observed_six, 6.0)
    previous_sum = last_six_sum - recent_sum

    return pd.DataFrame(
        {
            "qty_diff_current_lag1": current - lag1,
            "qty_diff_lag1_lag2": lag1 - lag2,
            "log_qty_diff_current_lag1": np.log1p(np.clip(current, 0.0, None))
            - np.log1p(np.clip(lag1, 0.0, None)),
            "sales_days_diff_current_lag1": sales_days - sales_days_lag1,
            "qty_diff_recent3_previous3": np.where(
                history_available, recent_sum / 3.0 - previous_sum / 3.0, 0.0
            ),
            "qty_diff_6m_history_available": history_available.astype("int8"),
        },
        index=frame.index,
    ).astype({feature: "float32" for feature in DIFF_FEATURES[:-1]})


def add_cross_store_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the target store month by month before rolling aggregation."""
    other_qty: list[np.ndarray] = []
    other_sites: list[np.ndarray] = []
    for suffix, self_column in (
        ("t", "total_qty"),
        ("t1", "qty_lag_1m"),
        ("t2", "qty_lag_2m"),
    ):
        self_qty = _numeric(frame, self_column)
        group_qty = _numeric(frame, f"group_positive_qty_{suffix}")
        group_sites = _numeric(frame, f"group_positive_sites_{suffix}")
        other_qty.append(np.clip(group_qty - np.clip(self_qty, 0.0, None), 0.0, None))
        other_sites.append(np.clip(group_sites - (self_qty > 0.0), 0.0, None))

    current_qty, lag1_qty, lag2_qty = other_qty
    current_sites = other_sites[0]
    per_site = np.divide(
        current_qty,
        current_sites,
        out=np.zeros_like(current_qty),
        where=current_sites > 0,
    )
    stacked_qty = np.vstack(other_qty)
    result = pd.DataFrame(
        {
            "xstore_qty_current": current_qty,
            "xstore_positive_sites_current": current_sites,
            "xstore_qty_per_positive_site_current": per_site,
            "xstore_qty_mean_last_3m": stacked_qty.mean(axis=0),
            "xstore_qty_max_last_3m": stacked_qty.max(axis=0),
            "xstore_qty_diff_1m": current_qty - lag1_qty,
        },
        index=frame.index,
    )
    return result.astype("float32")


def threshold_for_top_fraction(scores, fraction: float) -> float:
    values = np.asarray(scores, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("scores must contain at least one finite value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return float(np.quantile(values, 1.0 - fraction, method="higher"))


def weighted_threshold_for_top_fraction(scores, weights, fraction: float) -> float:
    values = np.asarray(scores, dtype="float64")
    sample_weights = np.asarray(weights, dtype="float64")
    if values.shape != sample_weights.shape:
        raise ValueError("scores and weights must have matching shapes")
    valid = np.isfinite(values) & np.isfinite(sample_weights) & (sample_weights > 0)
    values, sample_weights = values[valid], sample_weights[valid]
    if values.size == 0:
        raise ValueError("scores and weights must contain positive finite mass")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(sample_weights[order])
    boundary = int(np.searchsorted(cumulative, fraction * sample_weights.sum(), side="left"))
    return float(values[order[min(boundary, values.size - 1)]])


def candidate_b_multiplier(target, prediction, weights, maximum: float = 1.5) -> float:
    target = np.clip(np.asarray(target, dtype="float64"), 0.0, None)
    prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, None)
    weights = np.asarray(weights, dtype="float64")
    if target.shape != prediction.shape or target.shape != weights.shape:
        raise ValueError("target, prediction and weights must have matching shapes")
    if maximum < 1.0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("maximum and weights are invalid")
    predicted_total = float(np.sum(weights * prediction))
    true_total = float(np.sum(weights * target))
    if predicted_total <= 0.0 or true_total <= predicted_total:
        return 1.0
    ratio = true_total / predicted_total
    return float(np.clip(np.sqrt(ratio), 1.0, maximum))
