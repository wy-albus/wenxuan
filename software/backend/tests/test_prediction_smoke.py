from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook
from fastapi.testclient import TestClient


def test_ready_dataset_creates_e0_e2_e3_prediction_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WENXUAN_SOFTWARE_RUNTIME", str(tmp_path / "runtime"))
    from src.models.lightgbm_model import load_model_bundle
    from software.backend.api.main import create_app
    from software.backend.services.dataset_registry import DatasetRegistry

    model_paths = [
        "models/final/two_stage_classifier_active_store_1m.txt",
        "models/checkpoints/two_stage_optimization/train_valid_protocol/selected/E0_VALID_SELECTED.txt",
        "models/checkpoints/two_stage_optimization/train_valid_protocol/selected/E2_CROSS_STORE_VALID_SELECTED.txt",
        "models/checkpoints/two_stage_optimization/train_valid_protocol/selected/E3_DIFF_CROSS_STORE_VALID_SELECTED.txt",
    ]
    features: set[str] = set()
    for path in model_paths:
        _, metadata = load_model_bundle(path)
        features.update(metadata["feature_names"])
    row = {column: 0.0 for column in features}
    row.update({
        "month": "2026-03", "site_no": "S1", "item_id": "BOOK-1", "future_qty_1m": 0.0,
        "future_qty_2m": 0.0, "blt_site_no": "S1", "gds_ctgry_3_lvel": "unknown",
        "gds_ctgry_4_lvel": "unknown", "gds_ctgry_5_lvel": "unknown", "book_name": "Smoke Book",
    })
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame([row]).to_parquet(feature_path, index=False)
    dataset = DatasetRegistry().register({
        "dataset_name": "prediction-smoke", "source_type": "csv", "source_files": [],
        "date_range": {"start": "2026-03", "end": "2026-03"}, "store_count": 1, "item_count": 1, "row_count": 1,
        "monthly_parquet_path": str(feature_path), "active_store_parquet_path": str(feature_path), "feature_parquet_path": str(feature_path),
        "has_active_store": True, "has_diff_features": True, "has_cross_store_features": True,
    })
    client = TestClient(create_app())

    created = client.post("/api/predictions", json={"dataset_id": dataset["dataset_id"], "model_ids": ["E0", "E2", "E3"]})
    assert created.status_code == 201, created.text
    payload = created.json()
    assert client.get(f"/api/jobs/{payload['job_id']}").json()["status"] == "SUCCESS"
    run = client.get(f"/api/predictions/{payload['prediction_run_id']}").json()
    assert run["status"] == "SUCCESS"
    assert Path(run["prediction_dir"], "predictions.parquet").is_file()
    summary = client.get(f"/api/predictions/{payload['prediction_run_id']}/summary").json()
    assert set(summary["model_summaries"]) == {"E0", "E2", "E3"}
    top_books = client.get(f"/api/predictions/{payload['prediction_run_id']}/results?kind=top_books").json()
    assert top_books["total"] == 3
    assert {row["model_id"] for row in top_books["items"]} == {"E0", "E2", "E3"}
    downloaded = client.get(f"/api/predictions/{payload['prediction_run_id']}/download-parquet")
    assert downloaded.status_code == 200

    filtered = client.get(
        f"/api/predictions/{payload['prediction_run_id']}/results?kind=predictions&model_id=E2&site_no=S1&mc=MC0&page=1&page_size=10"
    )
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["model_id"] == "E2"

    store_summary = client.get(f"/api/predictions/{payload['prediction_run_id']}/store/S1/summary?model_id=E2")
    assert store_summary.status_code == 200, store_summary.text
    assert store_summary.json()["prediction_total"] >= 0

    exported = client.get(f"/api/predictions/{payload['prediction_run_id']}/export-excel?model_id=E2&site_no=S1")
    assert exported.status_code == 200, exported.text
    assert exported.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    workbook = load_workbook(BytesIO(exported.content), read_only=True)
    assert {"Summary", "Store_Summary", "Top_Books", "Predictions", "Model_Info"}.issubset(workbook.sheetnames)
