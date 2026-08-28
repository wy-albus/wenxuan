from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_datasets import router as datasets_router
from .routes_jobs import router as jobs_router
from .routes_predictions import router as predictions_router
from .routes_notifications import router as notifications_router
from .routes_uploads import router as uploads_router


def cors_origins() -> list[str]:
    defaults = ["http://localhost:5173", "http://127.0.0.1:5173"]
    configured = [origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
    return list(dict.fromkeys([*defaults, *configured]))


def create_app() -> FastAPI:
    app = FastAPI(title="Wenxuan Data Ingestion API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(uploads_router)
    app.include_router(datasets_router)
    app.include_router(jobs_router)
    app.include_router(predictions_router)
    app.include_router(notifications_router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
