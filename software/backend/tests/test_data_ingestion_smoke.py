from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_upload_processes_csv_and_registers_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WENXUAN_SOFTWARE_RUNTIME", str(tmp_path / "runtime"))
    from software.backend.api.main import create_app

    csv_body = "门店编码,ISBN,销售日期,数量,价格\n" + "\n".join(
        [
            "S1,9780000000001,2026-01-01,2,20",
            "S2,9780000000001,2026-01-01,1,20",
            "S1,9780000000001,2026-02-01,3,20",
            "S2,9780000000001,2026-02-01,0,20",
            "S1,9780000000001,2026-03-01,1,20",
            "S2,9780000000001,2026-03-01,4,20",
        ]
    )
    client = TestClient(create_app())

    assert client.get("/api/health").json()["status"] == "ok"
    upload = client.post(
        "/api/uploads",
        files={"file": ("sales.csv", csv_body.encode("utf-8"), "text/csv")},
    )
    assert upload.status_code == 201
    upload_id = upload.json()["upload_id"]
    assert upload.json()["file_size_bytes"] == len(csv_body.encode("utf-8"))
    assert upload.json()["detected_date_start"] == "2026-01-01"
    assert upload.json()["detected_date_end"] == "2026-03-01"
    assert upload.json()["detected_year_months"] == ["2026-01", "2026-02", "2026-03"]
    assert upload.json()["schema_status"] == "READY"
    assert upload.json()["field_mapping"]["site_no"] == "门店编码"
    assert client.get(f"/api/jobs/{upload.json()['job_id']}").json()["status"] == "SUCCESS"

    uploads_before_processing = client.get("/api/uploads").json()["items"]
    assert uploads_before_processing[0]["upload_id"] == upload_id
    assert uploads_before_processing[0]["detected_year_months"] == ["2026-01", "2026-02", "2026-03"]
    assert uploads_before_processing[0]["processing_status"] == "未处理"
    assert uploads_before_processing[0]["related_dataset"] is None

    processed = client.post(
        "/api/datasets/process",
        json={"upload_id": upload_id, "dataset_name": "smoke-sales", "process_mode": "create"},
    )
    assert processed.status_code == 201, processed.text
    job = client.get(f"/api/jobs/{processed.json()['job_id']}").json()
    assert job["status"] == "SUCCESS"
    assert Path(job["log_path"]).exists()

    datasets = client.get("/api/datasets").json()["items"]
    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset["dataset_name"] == "smoke-sales"
    assert dataset["has_active_store"] is True
    assert dataset["has_diff_features"] is True
    assert dataset["has_cross_store_features"] is True
    assert Path(dataset["monthly_parquet_path"]).exists()
    assert Path(dataset["active_store_parquet_path"]).exists()
    assert Path(dataset["feature_parquet_path"]).exists()

    uploads_after_processing = client.get("/api/uploads").json()["items"]
    assert uploads_after_processing[0]["processing_status"] == "已处理"
    assert uploads_after_processing[0]["related_dataset"]["dataset_name"] == "smoke-sales"
