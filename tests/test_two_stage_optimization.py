from __future__ import annotations

import importlib.util
import inspect
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/26_optimize_two_stage_1m.py"
_MODULE = None


def load_script():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    spec = importlib.util.spec_from_file_location("two_stage_optimization_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return _MODULE


def test_join_uses_current_and_two_prior_months_only():
    module = load_script()
    sql = module._joined_sql(module.load_optimization_config(), "b.split='train'", ["month", "item_id"])

    assert "INTERVAL 1 MONTH" in sql
    assert "INTERVAL 2 MONTH" in sql
    assert "+ INTERVAL" not in sql


def test_high_demand_classifiers_do_not_use_q_as_input():
    module = load_script()
    config = module.load_optimization_config()
    configured = set(config["diff_features"] + config["cross_store_features"] + module.base_features())

    assert "q" not in configured
    assert "conditional_qty" not in configured
    assert "future_qty_1m" not in configured


def test_validation_sampling_is_restorable_by_inverse_probability():
    module = load_script()
    config = module.load_optimization_config()

    assert config["sampling_rates"]["validation"] == {
        "0": 0.02, "1": 0.10, "2-5": 0.25, "5-20": 0.75, "20+": 1.0
    }
    assert "sample_weight" in SCRIPT.read_text(encoding="utf-8")


def test_candidate_b_oof_uses_forward_sale_classifier_not_formal_classifier():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "classifier_sale_train" in source
    assert '"E0_sale"' in source


def test_physical_projection_satisfies_shared_runtime_target_builder():
    module = load_script()

    assert {"future_qty_1m", "future_qty_2m"}.issubset(module.physical_columns())


def test_fold_cache_reader_avoids_python313_pyarrow_native_path():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def _read_fold_part", 1)[1].split("\ndef ", 1)[0]

    assert "duckdb.connect" in function_source
    assert "pd.read_parquet" not in function_source
    assert "fetch_df" not in function_source
    assert "fetchnumpy" in function_source


def test_explicit_stage_stop_preserves_artifacts_when_gate_passed():
    module = load_script()

    assert module.should_clean_stage_artifacts(stop_requested=True, passed=True) is False
    assert module.should_clean_stage_artifacts(stop_requested=False, passed=False) is True


def test_forward_fold_categories_use_fixed_unknown_for_unseen_valid_values():
    module = load_script()
    train = pd.DataFrame({"site_no": [10, 20, 10], "value": [1.0, 2.0, 3.0]})
    valid = pd.DataFrame({"site_no": [20, 30], "value": [4.0, 5.0]})

    seen = module.fit_seen_category_codes(train, ["site_no"])
    encoded = module.apply_seen_category_codes(valid, seen)

    assert seen == {"site_no": [10, 20]}
    assert encoded["site_no"].tolist() == [20, -1]


def test_oof_result_reader_avoids_pyarrow_dataframe_path():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def _read_oof_frame", 1)[1].split("\ndef ", 1)[0]

    assert "duckdb.connect" in function_source
    assert "pd.read_parquet" not in function_source


def test_total_bias_moving_toward_zero_is_not_worsening():
    module = load_script()

    assert module.absolute_bias_worsening(-10.0, -5.0) == -5.0
    assert module.absolute_bias_worsening(5.0, 8.0) == 3.0


def test_flattened_ablation_row_contains_no_unhashable_values():
    module = load_script()
    row = module._flatten_training_result({
        "fold": "q1", "component": "classifier_5plus",
        "valid_unknown_rates": {"site_no": 0.01},
        "metrics": {"average_precision": 0.2},
    })

    assert isinstance(row["valid_unknown_rates"], str)


def test_final_cleanup_removes_rebuildable_cache_and_failed_checkpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def clean_rebuildable_artifacts", 1)[1].split("\ndef ", 1)[0]

    assert "shutil.rmtree(cache_dir" in function_source
    assert 'config["outputs"]["checkpoint_dir"]' in function_source


def test_cross_store_formal_rounds_use_four_forward_folds_only():
    module = load_script()
    frame = pd.DataFrame({
        "fold": ["q3_2024", "q4_2024", "q1_2025", "q2_2025", "test"],
        "experiment": ["E2"] * 5,
        "component": ["regressor"] * 5,
        "best_iteration": [382, 657, 1136, 1357, 1],
        "valid_rows": [122836, 107822, 97798, 121023, 999999],
    })

    selection = module.select_cross_store_formal_rounds(frame)

    assert selection["rounds"] == 657
    assert {row["fold"] for row in selection["folds"]} == {
        "q3_2024", "q4_2024", "q1_2025", "q2_2025"
    }


def test_cross_store_formal_cache_samples_regressor_only():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def _build_cross_store_regressor_train_cache", 1)[1].split("\ndef ", 1)[0]

    assert '"regressor_train"' in function_source
    assert "classifier_5plus_train" not in function_source
    assert "classifier_20plus_train" not in function_source
    assert "pa.Table" not in function_source
    assert "_write_sample_rows" not in function_source
    assert "_write_duckdb_parquet_part" in function_source


def test_cross_store_complete_valid_is_valid_only_and_has_no_candidate_b():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def evaluate_cross_store_complete_valid", 1)[1].split("\ndef ", 1)[0]

    assert "b.split='valid'" in function_source
    assert "Candidate_B" not in function_source
    assert "split='test'" not in function_source


def test_high_demand_median_ratio_uses_only_twenty_plus_rows():
    module = load_script()

    ratio = module.twenty_plus_median_ratio(
        target=[4.0, 20.0, 40.0], prediction=[100.0, 10.0, 30.0]
    )

    assert ratio == 0.625


def test_cross_store_formal_cli_branches_before_completed_stages():
    source = SCRIPT.read_text(encoding="utf-8")
    main_source = source.split("def main()", 1)[1]

    assert main_source.index("args.cross_store_formal_only") < main_source.index("build_cross_store_table")
    assert main_source.index("args.cross_store_formal_only") < main_source.index("run_statistical_audit")


def test_cross_store_formal_report_is_appended_without_rewriting_prior_report():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def append_cross_store_formal_report", 1)[1].split("\ndef ", 1)[0]

    assert 'report_path.read_text' in function_source
    assert '"## Cross-store E2独立正式验证"' in function_source
    assert "write_outputs(" not in function_source


def test_cross_store_diagnostic_rebuild_is_locked_to_657_rounds():
    module = load_script()

    assert module.CROSS_STORE_DIAGNOSTIC_ROUNDS == 657
    assert len(module._cross_store_formal_features()) == 63


def test_cross_store_rebuild_consistency_detects_material_difference():
    module = load_script()
    expected = {
        "q_20_plus_wape": 68.7368,
        "final_20_plus_recall": 27.6692,
        "zero_prediction_total": 1_449_851.97,
    }
    matching = {
        "q_20_plus_wape": 68.73681,
        "final_20_plus_recall": 27.66919,
        "zero_prediction_total": 1_449_851.975,
    }
    different = dict(matching, q_20_plus_wape=68.80)

    assert module.compare_rebuild_summary(matching, expected)["passed"] is True
    result = module.compare_rebuild_summary(different, expected)
    assert result["passed"] is False
    assert result["checks"]["q_20_plus_wape"]["passed"] is False


def test_cross_store_rebuild_can_be_diagnostic_equivalent_after_order_variance():
    module = load_script()
    expected = {
        "q_20_plus_wape": 68.7368,
        "final_20_plus_recall": 27.6692,
        "zero_prediction_total": 1_449_851.97,
    }
    rebuilt = {
        "q_20_plus_wape": 68.8195,
        "final_20_plus_recall": 27.5270,
        "zero_prediction_total": 1_443_365.92,
    }

    result = module.compare_rebuild_summary(rebuilt, expected)

    assert result["passed"] is False
    assert result["diagnostic_equivalent"] is True
    assert result["equivalence_checks"]["zero_prediction_total"]["passed"] is True


def test_mc_codes_use_business_rounding_and_existing_boundaries():
    module = load_script()

    codes = module.business_mc_codes([-1.0, 0.49, 0.50, 1.49, 1.50, 4.49, 4.50, 19.49, 19.50])

    assert codes.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_diagnostic_cli_is_valid_only_and_preserves_rebuilt_checkpoint():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def run_cross_store_diagnostic_only", 1)[1].split("\ndef ", 1)[0]

    assert "CROSS_STORE_DIAGNOSTIC_ROUNDS" in function_source
    assert "evaluate_cross_store_error_diagnostic" in function_source
    assert "checkpoint.unlink" not in function_source
    assert "models/final" not in function_source
    assert "split='test'" not in function_source


def _diff_result(fold, experiment, nonzero, wape5, wape20, recall5, recall20):
    return {
        "fold": fold,
        "experiment": experiment,
        "component": "regressor",
        "metrics": {
            "nonzero": {"wape": nonzero},
            "5-19": {"wape": wape5},
            "20+": {"wape": wape20},
            "recall_5_19": recall5,
            "recall_20_plus": recall20,
        },
    }


def test_diff_pilot_feature_scope_adds_only_six_leakage_safe_fields():
    module = load_script()

    features = module._diff_pilot_features()

    assert len(features) == 63
    assert features[-6:] == module.DIFF_FEATURES
    assert not {"future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m"}.intersection(features)
    assert not {"target_qty_1m", "split", "item_id", "isbn", "gds_no"}.intersection(features)


def test_diff_pilot_advances_only_when_both_forward_folds_pass():
    module = load_script()
    config = module.load_optimization_config()
    rows = []
    for fold in ("q4_2024", "q2_2025"):
        rows.extend([
            _diff_result(fold, "E0_DIFF_CONTROL", 55.0, 60.0, 75.0, 25.0, 10.0),
            _diff_result(fold, "E1_DIFF", 54.8, 59.4, 74.4, 26.1, 11.1),
        ])

    decision = module.assess_diff_regressor_pilot(rows, config)

    assert decision["status"] == "consistent_gain"
    assert decision["allow_complete_valid"] is True


def test_diff_pilot_marks_one_fold_gain_as_time_conditional():
    module = load_script()
    config = module.load_optimization_config()
    rows = [
        _diff_result("q4_2024", "E0_DIFF_CONTROL", 55.0, 60.0, 75.0, 25.0, 10.0),
        _diff_result("q4_2024", "E1_DIFF", 54.8, 59.4, 74.4, 26.1, 11.1),
        _diff_result("q2_2025", "E0_DIFF_CONTROL", 53.0, 57.0, 74.0, 26.0, 10.0),
        _diff_result("q2_2025", "E1_DIFF", 53.2, 57.2, 74.2, 25.8, 9.8),
    ]

    decision = module.assess_diff_regressor_pilot(rows, config)

    assert decision["status"] == "time_conditional"
    assert decision["allow_complete_valid"] is False


def test_diff_pilot_has_dedicated_cli_cache_and_regressor_only_path():
    module = load_script()
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def run_diff_pilot_only", 1)[1].split("\ndef ", 1)[0]

    assert "--diff-pilot-only" in source
    assert "diff_pilot" in str(module._diff_pilot_cache_path(module.load_optimization_config(), "q4_2024"))
    assert "classifier_5plus" not in function_source
    assert "classifier_20plus" not in function_source
    assert "split='test'" not in function_source
    assert "shared_train" in function_source
    assert "shared_valid" in function_source


def test_diff_cache_projects_both_targets_required_by_runtime_preprocessing():
    source = SCRIPT.read_text(encoding="utf-8")
    function_source = source.split("def build_diff_pilot_fold_cache", 1)[1].split("\ndef ", 1)[0]

    assert '"future_qty_1m", "future_qty_2m"' in function_source


def test_train_valid_protocol_uses_locked_feature_scopes_without_leakage():
    module = load_script()

    scopes = module.train_valid_protocol_feature_sets()

    assert list(scopes) == [
        "E0_VALID_SELECTED", "E1_DIFF_VALID_SELECTED", "E2_CROSS_STORE_VALID_SELECTED",
        "E3_DIFF_CROSS_STORE_VALID_SELECTED",
    ]
    assert len(scopes["E0_VALID_SELECTED"]) == 57
    assert scopes["E1_DIFF_VALID_SELECTED"][-6:] == module.DIFF_FEATURES
    assert scopes["E2_CROSS_STORE_VALID_SELECTED"][-6:] == module.CROSS_STORE_FEATURES
    assert len(scopes["E3_DIFF_CROSS_STORE_VALID_SELECTED"]) == 69
    assert scopes["E3_DIFF_CROSS_STORE_VALID_SELECTED"][-12:-6] == module.DIFF_FEATURES
    assert scopes["E3_DIFF_CROSS_STORE_VALID_SELECTED"][-6:] == module.CROSS_STORE_FEATURES
    forbidden = {
        "future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m",
        "target_qty_1m", "split", "item_id", "isbn", "gds_no",
    }
    assert all(not forbidden.intersection(features) for features in scopes.values())


def test_stable_training_sort_and_hash_are_independent_of_scan_order():
    module = load_script()
    first = pd.DataFrame({
        "month": ["2025-02", "2025-01", "2025-01"],
        "site_no": [2, 1, 1],
        "item_id": ["b", "c", "a"],
        "target_qty": [2.0, 3.0, 1.0],
        "sample_weight": [1.0, 2.0, 3.0],
    })
    second = first.sample(frac=1.0, random_state=7).reset_index(drop=True)

    sorted_first = module.stable_sort_training_rows(first)
    sorted_second = module.stable_sort_training_rows(second)
    first_hashes = module.training_frame_hashes(sorted_first)
    second_hashes = module.training_frame_hashes(sorted_second)

    assert sorted_first[["month", "site_no", "item_id"]].equals(
        sorted_second[["month", "site_no", "item_id"]]
    )
    assert first_hashes == second_hashes
    assert first_hashes["row_count"] == 3


def test_train_valid_lightgbm_params_enable_deterministic_training():
    module = load_script()
    params = module.train_valid_lightgbm_params(module.load_optimization_config(), "regressor")

    assert params["seed"] == 42
    assert params["bagging_seed"] == 42
    assert params["feature_fraction_seed"] == 42
    assert params["data_random_seed"] == 42
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True


def test_incremental_tweedie_predictions_match_direct_predictions():
    module = load_script()
    x = np.arange(400, dtype="float32").reshape(200, 2)
    y = np.arange(1, 201, dtype="float32")
    booster = lgb.train(
        {"objective": "tweedie", "verbosity": -1, "seed": 42, "num_threads": 1},
        lgb.Dataset(x, label=y),
        num_boost_round=12,
    )

    predictions = module.incremental_tweedie_predictions(booster, x, [3, 7, 12])

    for iteration in (3, 7, 12):
        np.testing.assert_allclose(
            predictions[iteration], booster.predict(x, num_iteration=iteration), rtol=1e-12, atol=1e-12
        )


def test_valid_wape_iteration_selection_uses_wape_then_bias_then_smaller_round():
    module = load_script()
    rows = [
        {"iteration": 100, "wape": 90.0, "total_bias": 5.0},
        {"iteration": 200, "wape": 89.0, "total_bias": 8.0},
        {"iteration": 300, "wape": 89.0, "total_bias": -3.0},
        {"iteration": 400, "wape": 89.0, "total_bias": 3.0},
    ]

    selected = module.select_valid_wape_iteration(rows)

    assert selected["iteration"] == 300


def test_candidate_b_train_valid_protocol_keeps_exactly_original_25_pairs():
    module = load_script()

    pairs = module.candidate_b_fraction_pairs(module.load_optimization_config())

    assert len(pairs) == 25
    assert len(set(pairs)) == 25
    assert pairs[0] == (0.0025, 0.0005)
    assert pairs[-1] == (0.05, 0.01)


def test_train_valid_protocol_cli_runs_before_historical_forward_workflow_and_never_uses_test():
    source = SCRIPT.read_text(encoding="utf-8")
    main_source = source.split("def main()", 1)[1]
    function_source = source.split("def run_train_valid_protocol", 1)[1].split("\ndef ", 1)[0]

    assert "--train-valid-protocol" in source
    assert main_source.index("args.train_valid_protocol") < main_source.index("build_cross_store_table")
    assert "split='test'" not in function_source


def test_e3_valid_protocol_is_isolated_from_test_and_historical_workflow():
    source = SCRIPT.read_text(encoding="utf-8")
    main_source = source.split("def main()", 1)[1]
    function_source = source.split("def run_e3_train_valid_protocol", 1)[1].split("\ndef ", 1)[0]

    assert "--e3-valid-protocol-only" in source
    assert main_source.index("args.e3_valid_protocol_only") < main_source.index("build_cross_store_table")
    assert "split='test'" not in function_source
    assert "Candidate_B" not in function_source


def test_iteration_wape_accumulator_matches_direct_complete_valid_metric():
    module = load_script()
    target = np.array([0.0, 1.0, 5.0, 20.0])
    predictions = {
        10: np.array([0.2, 1.2, 4.0, 18.0]),
        20: np.array([0.1, 1.0, 5.0, 20.0]),
    }
    accumulator = module.IterationWapeAccumulator([10, 20])
    accumulator.update(target[:2], {key: value[:2] for key, value in predictions.items()})
    accumulator.update(target[2:], {key: value[2:] for key, value in predictions.items()})

    rows = accumulator.compute()

    expected = {
        iteration: module.weighted_business_metrics(target, prediction, np.ones(len(target)))["overall"]
        for iteration, prediction in predictions.items()
    }
    for row in rows:
        metric = expected[row["iteration"]]
        assert row["wape"] == metric["wape"]
        assert row["total_bias"] == metric["total_bias"]


def test_streaming_binary_ranking_prefers_perfect_scores_to_reversed_scores():
    module = load_script()
    target = np.array([0, 0, 1, 1], dtype="uint8")
    perfect = module.BinaryRankingAccumulator(bins=100)
    reversed_scores = module.BinaryRankingAccumulator(bins=100)
    perfect.update(target, np.array([0.01, 0.1, 0.9, 0.99]))
    reversed_scores.update(target, np.array([0.99, 0.9, 0.1, 0.01]))

    perfect_result = perfect.compute()
    reversed_result = reversed_scores.compute()

    assert perfect_result["average_precision"] > reversed_result["average_precision"]
    assert perfect_result["logloss"] < reversed_result["logloss"]


def test_protocol_iteration_grids_cover_cap_and_refine_around_winner():
    module = load_script()

    coarse = module.protocol_coarse_iterations(max_rounds=103, step=25)
    refine = module.protocol_refinement_iterations(best_iteration=50, max_rounds=103, radius=2)

    assert coarse == [1, 25, 50, 75, 100, 103]
    assert refine == [48, 49, 50, 51, 52]


def test_candidate_b_development_decision_requires_head_gain_and_protection():
    module = load_script()
    config = module.load_optimization_config()
    baseline = {
        "20+": {"wape": 75.0}, "5-19": {"wape": 64.0}, "nonzero": {"wape": 72.0},
        "overall": {"total_bias": 8.0}, "recall_20_plus": 24.0,
        "zero_prediction_total": 1000.0,
    }
    useful = {
        "20+": {"wape": 74.0}, "5-19": {"wape": 64.1}, "nonzero": {"wape": 72.1},
        "overall": {"total_bias": 8.5}, "recall_20_plus": 25.5,
        "zero_prediction_total": 1005.0,
    }
    weak = dict(useful, **{"20+": {"wape": 74.8}, "recall_20_plus": 24.5})

    assert module.candidate_b_development_decision(baseline, useful, config)["promoted"] is True
    assert module.candidate_b_development_decision(baseline, weak, config)["promoted"] is False


def test_boundary_models_are_extended_without_retraining_interior_models():
    module = load_script()
    selected = {
        "E0": {"iteration": 1950},
        "E1": {"iteration": 2000},
        "E2": {"iteration": 2000},
    }
    trained = {
        "E0": {"metadata": {"max_rounds": 2000, "component": "regressor"}},
        "E1": {"metadata": {"max_rounds": 2000, "component": "regressor"}},
        "E2": {"metadata": {"max_rounds": 2000, "component": "regressor"}},
    }

    boundary = module.boundary_selected_regressors(selected, trained)

    assert boundary == ["E1", "E2"]


def test_train_memory_limit_is_fixed_before_loading_the_training_matrix():
    module = load_script()
    source = inspect.getsource(module.train_train_valid_max_component)

    guard_position = source.index("guard = PIPELINE.MemoryGuard")
    sample_position = source.index("sample = read_train_valid_protocol_sample")

    assert guard_position < sample_position


def test_completed_iteration_history_merges_extensions_and_respects_model_cap(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "_protocol_root", lambda config: tmp_path)
    base = {
        "rows": 100,
        "regressors": {"E0": [{"iteration": 2000}], "E1": [{"iteration": 2000}]},
        "classifiers": {"S5": [{"iteration": 675}]},
    }
    extension = {
        "rows": 100,
        "regressors": {
            "E1": [{"iteration": 2025}, {"iteration": 3000}, {"iteration": 4000}]
        },
        "classifiers": {},
    }
    (tmp_path / "coarse_iteration_metrics.json").write_text(
        json.dumps(base), encoding="utf-8"
    )
    (tmp_path / "coarse_extension_4000_iteration_metrics.json").write_text(
        json.dumps(extension), encoding="utf-8"
    )
    trained = {
        "E0": {"metadata": {"max_rounds": 2000, "component": "regressor"}},
        "E1": {"metadata": {"max_rounds": 3000, "component": "regressor"}},
        "S5": {"metadata": {"max_rounds": 2000, "component": "classifier_5plus"}},
    }

    result = module.load_completed_protocol_iteration_history(
        {}, trained, logging.getLogger("test")
    )

    assert [row["iteration"] for row in result["regressors"]["E1"]] == [2000, 2025, 3000]
    assert [row["iteration"] for row in result["classifiers"]["S5"]] == [675]
