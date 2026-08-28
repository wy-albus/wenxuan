from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.demand_signal_features import (
    add_diff_features,
    add_cross_store_features,
    candidate_b_multiplier,
    cross_store_lookup_months,
    threshold_for_top_fraction,
    weighted_threshold_for_top_fraction,
)


def test_diff_features_use_true_recent_three_minus_previous_three():
    frame = pd.DataFrame(
        {
            "total_qty": [8.0],
            "qty_lag_1m": [5.0],
            "qty_lag_2m": [2.0],
            "qty_sum_last_3m": [12.0],
            "qty_sum_last_6m": [18.0],
            "sales_days": [4.0],
            "sales_days_lag_1m": [2.0],
            "active_months_last_6m": [4.0],
            "zero_sales_months_last_6m": [2.0],
        }
    )

    result = add_diff_features(frame)

    assert result.loc[0, "qty_diff_current_lag1"] == 3.0
    assert result.loc[0, "qty_diff_lag1_lag2"] == 3.0
    assert np.isclose(result.loc[0, "log_qty_diff_current_lag1"], np.log1p(8) - np.log1p(5))
    assert result.loc[0, "sales_days_diff_current_lag1"] == 2.0
    assert result.loc[0, "qty_diff_recent3_previous3"] == 2.0
    assert result.loc[0, "qty_diff_6m_history_available"] == 1


def test_diff_features_zero_unavailable_six_month_comparison():
    frame = pd.DataFrame(
        {
            "total_qty": [1.0], "qty_lag_1m": [0.0], "qty_lag_2m": [0.0],
            "qty_sum_last_3m": [1.0], "qty_sum_last_6m": [1.0],
            "sales_days": [1.0], "sales_days_lag_1m": [0.0],
            "active_months_last_6m": [1.0], "zero_sales_months_last_6m": [4.0],
        }
    )

    result = add_diff_features(frame)

    assert result.loc[0, "qty_diff_recent3_previous3"] == 0.0
    assert result.loc[0, "qty_diff_6m_history_available"] == 0


def test_cross_store_unique_selling_store_has_zero_other_store_signals():
    frame = pd.DataFrame(
        {
            "total_qty": [5.0], "qty_lag_1m": [3.0], "qty_lag_2m": [2.0],
            "group_positive_qty_t": [5.0], "group_positive_qty_t1": [3.0], "group_positive_qty_t2": [2.0],
            "group_positive_sites_t": [1], "group_positive_sites_t1": [1], "group_positive_sites_t2": [1],
        }
    )

    result = add_cross_store_features(frame)

    assert (result.iloc[0] == 0).all()


def test_cross_store_keeps_only_other_store_and_does_not_subtract_zero_self():
    frame = pd.DataFrame(
        {
            "total_qty": [4.0, 0.0], "qty_lag_1m": [1.0, 0.0], "qty_lag_2m": [0.0, 0.0],
            "group_positive_qty_t": [10.0, 6.0], "group_positive_qty_t1": [3.0, 0.0], "group_positive_qty_t2": [0.0, 0.0],
            "group_positive_sites_t": [2, 1], "group_positive_sites_t1": [2, 0], "group_positive_sites_t2": [0, 0],
        }
    )

    result = add_cross_store_features(frame)

    assert result.loc[0, "xstore_qty_current"] == 6.0
    assert result.loc[0, "xstore_positive_sites_current"] == 1
    assert result.loc[0, "xstore_qty_per_positive_site_current"] == 6.0
    assert result.loc[1, "xstore_qty_current"] == 6.0
    assert result.loc[1, "xstore_positive_sites_current"] == 1


def test_cross_store_missing_history_is_zero_before_rolling_and_max():
    frame = pd.DataFrame(
        {
            "total_qty": [1.0], "qty_lag_1m": [0.0], "qty_lag_2m": [2.0],
            "group_positive_qty_t": [4.0], "group_positive_qty_t1": [np.nan], "group_positive_qty_t2": [10.0],
            "group_positive_sites_t": [2], "group_positive_sites_t1": [np.nan], "group_positive_sites_t2": [2],
        }
    )

    result = add_cross_store_features(frame)

    assert result.loc[0, "xstore_qty_mean_last_3m"] == (3.0 + 0.0 + 8.0) / 3.0
    assert result.loc[0, "xstore_qty_max_last_3m"] == 8.0
    assert result.loc[0, "xstore_qty_diff_1m"] == 3.0


def test_cross_store_lookup_never_uses_a_month_after_observation():
    assert cross_store_lookup_months("2025-01") == ("2025-01", "2024-12", "2024-11")


def test_candidate_b_top_fraction_threshold_and_bounded_shrunk_multiplier():
    scores = np.arange(1, 101, dtype="float64")
    assert threshold_for_top_fraction(scores, 0.05) == 96.0

    target = np.array([20.0, 30.0])
    prediction = np.array([5.0, 5.0])
    weights = np.array([1.0, 1.0])
    assert candidate_b_multiplier(target, prediction, weights, maximum=1.5) == 1.5

    assert candidate_b_multiplier(target, np.array([30.0, 30.0]), weights, maximum=1.5) == 1.0


def test_weighted_top_fraction_restores_original_distribution():
    scores = np.array([0.1, 0.9])
    weights = np.array([99.0, 1.0])

    assert weighted_threshold_for_top_fraction(scores, weights, 0.10) == 0.1
