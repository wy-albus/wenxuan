import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "20_train_lightgbm_v2_active_store_1m_wape.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("active_store_wape_1m", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extended_candidates_cover_both_sides_of_previous_boundary():
    module = _load_module()
    candidates = module.extended_candidate_iterations()

    assert candidates == sorted(set(candidates))
    assert candidates[0] < 700
    assert 700 in candidates
    assert {800, 900, 1000, 1200, 1400, 1600, 1800, 2000}.issubset(candidates)
    assert candidates[-1] == 2000


def test_search_status_distinguishes_internal_minimum_from_upper_boundary():
    module = _load_module()
    rows = [
        {"iteration": 600, "wape": 120.0},
        {"iteration": 700, "wape": 110.0},
        {"iteration": 800, "wape": 111.0},
    ]
    assert module.wape_search_status(rows, 700) == "internal_minimum"
    assert module.wape_search_status(rows, 800) == "best_at_search_upper_bound"


def test_candidate_table_requires_same_complete_valid_count():
    module = _load_module()
    rows = [
        {"iteration": 700, "count": 10, "wape": 1.0},
        {"iteration": 800, "count": 9, "wape": 0.9},
    ]
    try:
        module.validate_candidate_rows(rows, expected_count=10)
    except ValueError as error:
        assert "complete Valid count" in str(error)
    else:
        raise AssertionError("Expected inconsistent candidate counts to fail")


def test_exact_test_segments_use_five_and_twenty_as_lower_bounds():
    module = _load_module()
    metrics = module.ExactRangeMetrics()
    target = np.array([0, 1, 2, 4, 5, 19, 20, 30], dtype="float64")
    metrics.update(target, target)
    result = metrics.compute()

    assert result["2-5"]["count"] == 2
    assert result["5-20"]["count"] == 2
    assert result["20+"]["count"] == 2


def test_continuation_datasets_retain_raw_data_for_init_model():
    module = _load_module()
    features = ["value"]
    train = {
        "x": pd.DataFrame({"value": np.arange(8, dtype="float32")}),
        "y": np.arange(8, dtype="float32"),
        "weight": np.ones(8, dtype="float32"),
    }
    valid = {
        "x": pd.DataFrame({"value": np.arange(4, dtype="float32")}),
        "y": np.arange(4, dtype="float32"),
        "weight": np.ones(4, dtype="float32"),
    }
    train_set, valid_set = module.build_continuation_datasets(
        train, valid, features, [], {"objective": "regression_l2", "verbosity": -1}
    )
    train_set.construct()
    valid_set.construct()

    assert train_set.get_data() is not None
    assert valid_set.get_data() is not None
