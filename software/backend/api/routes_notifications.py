from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from software.backend.services.email_service import EmailService
from software.backend.services.notification_service import NotificationService


router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class TestEmailRequest(BaseModel):
    target_email: str


@router.post("/test-email")
def send_test_email(request: TestEmailRequest) -> dict:
    return EmailService().send_test_email(request.target_email)


@router.get("")
def list_notifications() -> dict:
    return {"items": NotificationService().list()}
