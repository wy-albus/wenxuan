from pathlib import Path


def script_source():
    path = Path(__file__).resolve().parents[1] / "scripts" / "25_train_two_stage_active_store_1m.py"
    return path.read_text(encoding="utf-8")


def test_training_is_active_store_one_month_valid_only_selection():
    source = script_source()

    assert "model_dataset_monthly_active_store.parquet" in source
    assert "target_available_1m=1" in source
    assert "split='valid'" in source
    assert "GateATradeoffAccumulator" in source
    assert "Gate-B" not in source
    assert "gate_b" not in source
    assert "target_available_2m" not in source
    assert "split='test'" not in source


def test_training_refuses_to_overwrite_formal_models_and_bounds_complete_valid_pairs():
    source = script_source()

    assert "two_stage_classifier_active_store_1m.txt" in source
    assert "two_stage_regressor_active_store_1m.txt" in source
    assert "MAX_COMPLETE_VALID_PAIRS = 3" in source
    assert "refuse_existing_outputs" in source


def test_training_report_template_is_valid_utf8_chinese():
    source = script_source()

    assert "# Active-Store Two-stage 1M 训练与 Valid Gate-A 报告" in source
    assert "璁" not in source


def test_training_uses_duckdb_sampling_to_avoid_pyarrow_native_crash():
    source = script_source()

    assert "collect_train_valid_samples_duckdb" in source
    assert "experiment._sample_horizon(" not in source
