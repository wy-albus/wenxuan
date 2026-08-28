from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime

from .runtime import database_path


class NotificationService:
    def __init__(self) -> None:
        self.db_path = database_path()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY, type TEXT, target_email TEXT,
                status TEXT, related_run_id TEXT, created_at TEXT, error_message TEXT)""")

    def record(self, *, notification_type: str, target_email: str, status: str, related_run_id: str | None = None, error_message: str | None = None) -> dict:
        record = {"notification_id": uuid.uuid4().hex, "type": notification_type, "target_email": target_email,
                  "status": status, "related_run_id": related_run_id, "created_at": datetime.now(UTC).isoformat(), "error_message": error_message}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO notifications VALUES (?,?,?,?,?,?,?)", tuple(record.values()))
        return record

    def list(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT notification_id,type,target_email,status,related_run_id,created_at,error_message FROM notifications ORDER BY rowid DESC").fetchall()
        keys = ("notification_id", "type", "target_email", "status", "related_run_id", "created_at", "error_message")
        return [dict(zip(keys, row, strict=True)) for row in rows]
