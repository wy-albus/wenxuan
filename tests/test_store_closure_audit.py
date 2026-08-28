from pathlib import Path

import pandas as pd
import pytest

from src.data.store_closures import classify_closure_exclusion, load_store_closures


def test_closure_boundary_classification_matches_forecast_horizons():
    closure = "2025-07"

    assert classify_closure_exclusion("2025-07", closure) == "after_closure"
    assert classify_closure_exclusion("2025-06", closure) == "cross_1m"
    assert classify_closure_exclusion("2025-05", closure) == "cross_2m_extra"
    assert classify_closure_exclusion("2025-04", closure) == "unaffected"


def test_store_closure_config_has_28_unique_valid_sites():
    path = Path(__file__).resolve().parents[1] / "config" / "active_store_closures.csv"
    frame = load_store_closures(path)

    assert len(frame) == 28
    assert frame["site_no"].is_unique
    assert frame["closure_month"].str.fullmatch(r"\d{4}-\d{2}").all()


def test_store_closure_config_rejects_duplicate_sites(tmp_path):
    path = tmp_path / "closures.csv"
    pd.DataFrame(
        {"site_no": ["S1", "S1"], "closure_month": ["2025-01", "2025-02"]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate site_no"):
        load_store_closures(path)
