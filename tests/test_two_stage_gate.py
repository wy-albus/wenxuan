import numpy as np

from src.evaluation.two_stage_gate import (
    GateATradeoffAccumulator,
    ProbabilityDiagnostic,
    ThresholdGateAccumulator,
    classify_history_pattern,
    protection_candidates,
    select_gate_candidate,
    select_safe_gate_candidate,
    should_evaluate_test,
)


def gate_script_source():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "22_diagnose_two_stage_gate_1m.py"
    return path.read_text(encoding="utf-8")


def tradeoff_script_source():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "23_analyze_two_stage_gate_tradeoff_1m.py"
    return path.read_text(encoding="utf-8")


def brute_metrics(y, p, q, threshold, method):
    original = p * q
    if method == "gate_a":
        pred = np.where(p < threshold, 0.0, original)
    elif method == "gate_b":
        pred = np.where(p < threshold, 0.0, q)
    else:
        pred = original
    error = y - pred
    return {
        "count": len(y),
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt(np.square(error).mean()),
        "wape": 100 * np.abs(error).sum() / y.sum(),
        "prediction_sum": pred.sum(),
    }


def test_threshold_accumulator_matches_brute_force():
    y = np.array([0, 0, 1, 3, 8, 25], dtype=float)
    p = np.array([0.02, 0.4, 0.1, 0.3, 0.7, 0.9], dtype=float)
    q = np.array([1, 2, 2, 4, 9, 30], dtype=float)
    thresholds = np.array([0.05, 0.2, 0.5, 0.8], dtype=float)
    accumulator = ThresholdGateAccumulator(thresholds)
    accumulator.update(y, p, q)
    rows = accumulator.rows()

    for method in ("original", "gate_a", "gate_b"):
        method_rows = [row for row in rows if row["method"] == method]
        expected_thresholds = [None] if method == "original" else thresholds
        assert len(method_rows) == len(expected_thresholds)
        for row, threshold in zip(method_rows, expected_thresholds):
            expected = brute_metrics(y, p, q, 0.0 if threshold is None else threshold, method)
            for key in ("mae", "rmse", "wape", "prediction_sum"):
                assert np.isclose(row[key], expected[key])


def test_gate_metrics_use_exact_business_ranges_and_zero_diagnostics():
    y = np.array([0, 1, 2, 4, 5, 19, 20], dtype=float)
    p = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    q = np.ones_like(y)
    accumulator = ThresholdGateAccumulator(np.array([0.5]))
    accumulator.update(y, p, q)
    gate = next(row for row in accumulator.rows() if row["method"] == "gate_b")
    assert gate["count_5_20"] == 2
    assert gate["count_20plus"] == 1
    assert gate["count_qty_1"] == 1
    assert gate["count_qty_2_5"] == 2
    assert np.isclose(gate["qty_1_mae"], 0.0)
    assert np.isclose(gate["qty_2_5_mae"], 2.0)
    assert gate["zero_pred_total"] == 1.0
    assert gate["zero_pred_gt_0_5"] == 1.0
    assert gate["zero_pred_gt_1"] == 0.0
    assert gate["5_20_gate_recall"] == 1.0
    assert gate["20plus_gate_recall"] == 1.0


def test_probability_diagnostic_reports_quantiles_and_threshold_shares():
    diagnostic = ProbabilityDiagnostic()
    diagnostic.update(
        np.array([0, 1, 3, 8, 25], dtype=float),
        np.array([0, 1, 3, 8, 25], dtype=float),
        np.array([0.1, 0.2, 0.4, 0.8, 0.9], dtype=float),
    )
    rows = diagnostic.rows()
    head = next(row for row in rows if row["basis"] == "target_qty_1m" and row["segment"] == "20+")
    assert head["count"] == 1
    assert np.isclose(head["mean"], 0.9)
    assert head["p_sale_ge_0_50_rate"] == 1.0


def test_candidate_selection_uses_coarse_then_local_fine_wape():
    rows = [
        {"method": "original", "threshold": np.nan, "wape": 120.0},
        {"method": "gate_a", "threshold": 0.35, "wape": 100.0},
        {"method": "gate_a", "threshold": 0.40, "wape": 98.0},
        {"method": "gate_a", "threshold": 0.41, "wape": 97.0},
        {"method": "gate_a", "threshold": 0.45, "wape": 99.0},
        {"method": "gate_b", "threshold": 0.40, "wape": 101.0},
    ]
    selected = select_gate_candidate(rows)
    assert selected["method"] == "gate_a"
    assert selected["threshold"] == 0.41


def test_accumulator_supports_zero_and_nonstandard_thresholds():
    thresholds = np.array([0.0, 0.01, 0.075, 0.10])
    accumulator = ThresholdGateAccumulator(thresholds)
    accumulator.update(np.array([0.0, 3.0]), np.array([0.05, 0.075]), np.array([2.0, 4.0]))
    rows = accumulator.rows()
    gate_a_zero = next(row for row in rows if row["method"] == "gate_a" and row["threshold"] == 0.0)
    original = next(row for row in rows if row["method"] == "original")
    assert np.isclose(gate_a_zero["wape"], original["wape"])
    gate_b = next(row for row in rows if row["method"] == "gate_b" and row["threshold"] == 0.075)
    assert np.isclose(gate_b["prediction_sum"], 4.0)


def test_protection_candidates_select_each_method_at_each_level():
    rows = [
        {"method": "gate_a", "threshold": 0.1, "wape": 100.0, "20plus_gate_recall": 0.99, "5_20_gate_recall": 0.98},
        {"method": "gate_a", "threshold": 0.2, "wape": 90.0, "20plus_gate_recall": 0.90, "5_20_gate_recall": 0.91},
        {"method": "gate_b", "threshold": 0.1, "wape": 110.0, "20plus_gate_recall": 0.99, "5_20_gate_recall": 0.98},
        {"method": "gate_b", "threshold": 0.2, "wape": 80.0, "20plus_gate_recall": 0.90, "5_20_gate_recall": 0.91},
    ]
    selected = protection_candidates(rows, [0.98, 0.90])
    assert len(selected) == 4
    assert next(row for row in selected if row["method"] == "gate_a" and row["min_20plus_pass_rate"] == 0.90)["threshold"] == 0.2
    assert next(row for row in selected if row["method"] == "gate_b" and row["min_20plus_pass_rate"] == 0.98)["threshold"] == 0.1


def test_history_pattern_separates_persistent_burst_and_volatile():
    assert classify_history_pattern(8, 7, 9) == "persistent_high"
    assert classify_history_pattern(0, 0, 1) == "sudden_burst_from_low"
    assert classify_history_pattern(0, 8, 1) == "volatile_history"
    assert classify_history_pattern(2, 2, 3) == "other"


def test_test_gate_requires_wape_gain_and_head_recall():
    original = {"wape": 120.0}
    assert should_evaluate_test(
        original,
        {"wape": 110.0, "5_20_gate_recall": 0.97, "20plus_gate_recall": 0.99},
    )
    assert not should_evaluate_test(
        original,
        {"wape": 119.8, "5_20_gate_recall": 0.99, "20plus_gate_recall": 0.99},
    )
    assert not should_evaluate_test(
        original,
        {"wape": 105.0, "5_20_gate_recall": 0.80, "20plus_gate_recall": 0.99},
    )


def test_test_candidate_is_lowest_wape_among_head_safe_gates():
    rows = [
        {"method": "gate_a", "threshold": 0.65, "wape": 92.0, "5_20_gate_recall": 0.55, "20plus_gate_recall": 0.60},
        {"method": "gate_a", "threshold": 0.02, "wape": 130.0, "5_20_gate_recall": 0.98, "20plus_gate_recall": 0.97},
        {"method": "gate_b", "threshold": 0.01, "wape": 700.0, "5_20_gate_recall": 0.99, "20plus_gate_recall": 0.99},
    ]
    selected = select_safe_gate_candidate(rows)
    assert selected["method"] == "gate_a"
    assert selected["threshold"] == 0.02


def test_memory_guard_is_created_before_large_models_are_loaded():
    source = gate_script_source().split("def evaluate_valid_once", 1)[1].split("def _build_test_prediction", 1)[0]
    assert source.index("MemoryGuard") < source.index("load_model_bundle")


def test_valid_scan_uses_bounded_duckdb_chunks_instead_of_pyarrow_row_groups():
    source = gate_script_source().split("def iter_valid_chunks", 1)[1].split("def _validate_model_pair", 1)[0]
    assert "memory_limit='512MB'" in source
    assert "fetch_df_chunk" in source
    assert "target_available_1m" in source


def test_tradeoff_stage_is_valid_only_and_does_not_reference_test_predictions():
    source = tradeoff_script_source()
    assert "split='valid'" in source
    assert "target_available_1m=1" in source
    assert "two_stage_test_predictions" not in source
    assert "_test_predictions" not in source


def test_tradeoff_valid_scan_keeps_four_gib_free_with_a_small_inference_budget():
    source = tradeoff_script_source()
    assert "MIN_VALID_SCAN_BUDGET_GIB = 0.5" in source
    assert "SYSTEM_RESERVE_GIB = 4.0" in source
    assert "available < SYSTEM_RESERVE_GIB * pipeline.GIB" in source


def test_gate_a_only_accumulator_reports_business_tradeoff_without_gate_b():
    target = np.array([0, 1, 3, 8, 25], dtype=float)
    probability = np.array([0.01, 0.2, 0.4, 0.6, 0.8], dtype=float)
    prediction = np.array([0.5, 0.8, 2.5, 6.0, 20.0], dtype=float)
    accumulator = GateATradeoffAccumulator(np.array([0.1, 0.5]))
    accumulator.update(target, probability, prediction)
    rows = accumulator.rows()

    assert [row["method"] for row in rows] == ["original", "gate_a", "gate_a"]
    threshold = rows[1]
    brute = np.where(probability < 0.1, 0.0, prediction)
    assert np.isclose(threshold["overall_wape"], 100 * np.abs(target - brute).sum() / target.sum())
    assert threshold["zero_predicted_total"] == 0.0
    assert threshold["5_20_gate_pass_rate"] == 1.0
    assert threshold["20plus_gate_pass_rate"] == 1.0
