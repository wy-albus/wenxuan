from __future__ import annotations

import json


def test_email_service_sends_test_message_and_records_result(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WENXUAN_SOFTWARE_RUNTIME", str(tmp_path / "runtime"))
    from software.backend.services.email_service import EmailConfig, EmailService

    sent: list[object] = []

    class FakeSMTP:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            assert (host, port, timeout) == ("smtp.example.test", 587, 20)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def starttls(self) -> None:
            sent.append("tls")

        def login(self, user: str, password: str) -> None:
            assert (user, password) == ("user", "password")

        def send_message(self, message) -> None:
            sent.append(message)

    service = EmailService(
        config=EmailConfig(host="smtp.example.test", port=587, user="user", password="password", sender="noreply@example.test", use_tls=True),
        smtp_factory=FakeSMTP,
    )
    record = service.send_test_email("business@example.test")

    assert record["type"] == "TEST"
    assert record["status"] == "SUCCESS"
    assert sent[0] == "tls"
    assert "测试邮件" in sent[1].get_content()


def test_notification_api_records_missing_smtp_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WENXUAN_SOFTWARE_RUNTIME", str(tmp_path / "runtime"))
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_USE_TLS"):
        monkeypatch.delenv(name, raising=False)
    from fastapi.testclient import TestClient
    from software.backend.api.main import create_app

    client = TestClient(create_app())
    response = client.post("/api/notifications/test-email", json={"target_email": "business@example.test"})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "FAILED"
    assert "SMTP" in response.json()["error_message"]
    records = client.get("/api/notifications").json()["items"]
    assert records[0]["type"] == "TEST"
    assert records[0]["status"] == "FAILED"


def test_prediction_success_email_contains_summary_and_download_links(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WENXUAN_SOFTWARE_RUNTIME", str(tmp_path / "runtime"))
    from software.backend.services.email_service import EmailConfig, EmailService

    prediction_dir = tmp_path / "prediction"
    prediction_dir.mkdir()
    (prediction_dir / "summary.json").write_text(json.dumps({"observation_month": "2026-03", "model_summaries": {"E2": {"prediction_total": 10, "predicted_nonzero_book_count": 3, "mc_counts": {"MC3": 1, "MC4": 0}, "store_count": 2, "item_count": 3}}}), encoding="utf-8")
    captured = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs) -> None: pass
        def __enter__(self): return self
        def __exit__(self, *args) -> None: return None
        def starttls(self) -> None: return None
        def send_message(self, message) -> None: captured.append(message)

    service = EmailService(config=EmailConfig(host="smtp.example.test", port=587, user=None, password=None, sender="noreply@example.test", use_tls=True, public_base_url="https://forecast.example.test"), smtp_factory=FakeSMTP)
    record = service.send_prediction_success({"prediction_run_id": "run-1", "dataset_id": "dataset-1", "prediction_dir": str(prediction_dir)}, "business@example.test", "E2")

    assert record["status"] == "SUCCESS"
    content = captured[0].get_content()
    assert "prediction_run_id: run-1" in content
    assert "预测总销量: 10" in content
    assert "/export-excel?model_id=E2" in content
    assert "/download-parquet" in content
