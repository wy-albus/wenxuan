from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from software.backend.services.email_service import EmailService
from software.backend.services.email_service import EmailConfig
from software.backend.services.notification_service import NotificationService


router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class TestEmailRequest(BaseModel):
    target_email: str


class NotificationSettingsRequest(BaseModel):
    target_email: str
    enabled_types: list[str]


@router.post("/test-email")
def send_test_email(request: TestEmailRequest) -> dict:
    return EmailService().send_test_email(request.target_email)


@router.get("/settings")
def get_notification_settings() -> dict:
    return NotificationService().get_settings()


@router.post("/settings")
def save_notification_settings(request: NotificationSettingsRequest) -> dict:
    return NotificationService().save_settings(target_email=request.target_email, enabled_types=request.enabled_types)


@router.get("/smtp-status")
def smtp_status() -> dict:
    config = EmailConfig.from_sources()
    configured = bool(config.host and config.sender)
    return {"status": "已配置" if configured else "未配置", "configured": configured}


@router.get("")
def list_notifications() -> dict:
    return {"items": NotificationService().list()}
