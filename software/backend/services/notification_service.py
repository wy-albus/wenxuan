from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
import json

from .runtime import database_path


class NotificationService:
    def __init__(self) -> None:
        self.db_path = database_path()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY, type TEXT, target_email TEXT,
                status TEXT, related_run_id TEXT, created_at TEXT, error_message TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS notification_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1), target_email TEXT,
                enabled_types TEXT, updated_at TEXT)""")

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

    def get_settings(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT target_email, enabled_types, updated_at FROM notification_settings WHERE id=1").fetchone()
        if row is None:
            return {"target_email": "", "enabled_types": ["PREDICTION_SUCCESS", "PREDICTION_FAILED", "DATA_PROCESSING_FAILED"], "updated_at": None}
        return {"target_email": row[0] or "", "enabled_types": json.loads(row[1] or "[]"), "updated_at": row[2]}

    def save_settings(self, *, target_email: str, enabled_types: list[str]) -> dict:
        record = {"target_email": target_email, "enabled_types": enabled_types, "updated_at": datetime.now(UTC).isoformat()}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO notification_settings (id, target_email, enabled_types, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET target_email=excluded.target_email,
                enabled_types=excluded.enabled_types, updated_at=excluded.updated_at""",
                (target_email, json.dumps(enabled_types, ensure_ascii=False), record["updated_at"]),
            )
        return record
