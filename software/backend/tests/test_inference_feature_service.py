from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _monthly(months: list[str]) -> pd.DataFrame:
    rows = []
    for month_index, month in enumerate(months, start=1):
        for site_no in ("8000", "8001"):
            for item_id in ("BOOK-1", "BOOK-2"):
                rows.append(
                    {
                        "month": month,
                        "site_no": site_no,
                        "blt_site_no": site_no,
                        "item_id": item_id,
                        "isbn": item_id,
                        "gds_no": item_id,
                        "gds_ctgry_3_lvel": "C3",
                        "gds_ctgry_4_lvel": "C4",
                        "gds_ctgry_5_lvel": "C5",
                        "price": 20.0,
                        "total_qty": float(month_index),
                        "offline_qty": float(month_index),
                        "online_qty": 0.0,
                        "unknown_channel_qty": 0.0,
                        "total_tlp": 20.0 * month_index,
                        "total_tsp": 18.0 * month_index,
                        "avg_real_price": 18.0,
                        "discount_rate": 0.9,
                        "sales_days": 1,
                        "sales_count": 1,
                        "return_count": 0,
                        "return_qty": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_inference_builder_uses_latest_history_without_future_targets() -> None:
    from software.backend.services.inference_feature_service import InferenceFeatureService

    service = InferenceFeatureService()
    result = service.build_from_monthly(
        _monthly(["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]),
        model_ids=["E0", "E2", "E3"],
        target_month="2026-07",
        prediction_run_id="run-test",
    )

    assert result.metadata["observation_month"] == "2026-06"
    assert result.metadata["target_month"] == "2026-07"
    assert result.metadata["source_months"] == ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    assert set(result.features["month"].astype(str)) == {"2026-06"}
    assert "future_qty_1m" not in result.features.columns
    assert "future_qty_2m" not in result.features.columns
    assert "target_qty_1m" not in result.features.columns
    assert result.features["qty_lag_6m"].notna().all()
    assert result.features["xstore_qty_current"].notna().all()
    assert result.features["qty_diff_current_lag1"].notna().all()


def test_inference_builder_reports_missing_history_instead_of_fallback() -> None:
    from software.backend.services.data_requirement_service import DataNotReady
    from software.backend.services.inference_feature_service import InferenceFeatureService

    service = InferenceFeatureService()

    with pytest.raises(DataNotReady) as exc_info:
        service.build_from_monthly(
            _monthly(["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]),
            model_ids=["E0"],
            target_month="2026-07",
            prediction_run_id="run-test",
        )

    error = exc_info.value
    assert error.missing_months == ["2025-12"]
    assert "qty_lag_6m" in error.affected_features
    assert error.observation_month == "2026-06"
    assert error.target_month == "2026-07"
