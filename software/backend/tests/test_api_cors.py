from fastapi.testclient import TestClient


def test_api_allows_vite_development_origin() -> None:
    from software.backend.api.main import create_app

    response = TestClient(create_app()).options(
        "/api/uploads",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_api_allows_configured_github_pages_origin(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://example.github.io/wenxuan-forecast")
    from software.backend.api.main import create_app

    response = TestClient(create_app()).options(
        "/api/health",
        headers={"Origin": "https://example.github.io/wenxuan-forecast", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.github.io/wenxuan-forecast"
