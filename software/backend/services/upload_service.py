from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from .field_mapping_service import mapping_status
from .job_service import JobService
from .runtime import database_path, runtime_root
from .schema_detection_service import detect_field_mapping, read_csv_headers


class UploadService:
    def __init__(self) -> None:
        self.db_path = database_path()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY, filename TEXT, stored_path TEXT, csv_files TEXT,
                headers TEXT, field_mapping TEXT, mapping_status TEXT, created_at TEXT)""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(uploads)").fetchall()}
            if "file_size_bytes" not in columns:
                conn.execute("ALTER TABLE uploads ADD COLUMN file_size_bytes INTEGER")
            if "job_id" not in columns:
                conn.execute("ALTER TABLE uploads ADD COLUMN job_id TEXT")

    async def save(self, upload: UploadFile) -> dict:
        filename = Path(upload.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".zip"}:
            raise ValueError("Only .csv and .zip uploads are supported")
        upload_id = uuid.uuid4().hex
        destination = runtime_root() / "uploads" / f"{upload_id}_{filename}"
        jobs = JobService()
        job = jobs.create("UPLOAD", [filename])
        jobs.update(job["job_id"], status="RUNNING", progress=5)
        try:
            with destination.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            file_size = destination.stat().st_size
            csv_files = self._csv_files(upload_id, destination)
            headers = read_csv_headers(csv_files[0])
            mapping = detect_field_mapping(headers)
            record = {"upload_id": upload_id, "filename": filename, "stored_path": str(destination),
                      "csv_files": [str(path) for path in csv_files], "headers": headers, "field_mapping": mapping,
                      "mapping_status": mapping_status(mapping), "created_at": datetime.now(UTC).isoformat(),
                      "file_size_bytes": file_size, "job_id": job["job_id"]}
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""INSERT INTO uploads (
                    upload_id, filename, stored_path, csv_files, headers, field_mapping, mapping_status,
                    created_at, file_size_bytes, job_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                    record["upload_id"], record["filename"], record["stored_path"], json.dumps(record["csv_files"]),
                    json.dumps(record["headers"]), json.dumps(record["field_mapping"]), record["mapping_status"],
                    record["created_at"], record["file_size_bytes"], record["job_id"],
                ))
            jobs.update(job["job_id"], status="SUCCESS", progress=100, output_files=record["csv_files"])
            jobs.log(job["job_id"], f"stored upload {destination}")
            return record
        except Exception as exc:
            jobs.update(job["job_id"], status="FAILED", progress=100, error_message=str(exc))
            jobs.log(job["job_id"], f"ERROR {exc}")
            raise

    def _csv_files(self, upload_id: str, stored: Path) -> list[Path]:
        if stored.suffix.lower() == ".csv":
            return [stored]
        destination = runtime_root() / "extracted" / upload_id
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(stored) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                target = (destination / member_path).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe ZIP member path")
                if not member.is_dir():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        files = sorted(path for path in destination.rglob("*") if path.suffix.lower() == ".csv")
        if not files:
            raise ValueError("ZIP does not contain a CSV file")
        return files

    def get(self, upload_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM uploads WHERE upload_id=?", (upload_id,)).fetchone()
        if row is None:
            raise KeyError(upload_id)
        keys = ("upload_id", "filename", "stored_path", "csv_files", "headers", "field_mapping", "mapping_status", "created_at", "file_size_bytes", "job_id")
        record = dict(zip(keys, row, strict=True))
        for key in ("csv_files", "headers", "field_mapping"):
            record[key] = json.loads(record[key])
        return record

    def list(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            ids = [row[0] for row in conn.execute("SELECT upload_id FROM uploads ORDER BY rowid DESC")]
            has_datasets = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'").fetchone()
            dataset_rows = conn.execute("SELECT dataset_id, dataset_name, source_files FROM datasets ORDER BY rowid DESC").fetchall() if has_datasets else []
        datasets = []
        for dataset_id, dataset_name, source_files in dataset_rows:
            datasets.append({"dataset_id": dataset_id, "dataset_name": dataset_name, "source_files": set(json.loads(source_files))})
        records = []
        for upload_id in ids:
            record = self.get(upload_id)
            related = next((dataset for dataset in datasets if set(record["csv_files"]).intersection(dataset["source_files"])), None)
            record["processing_status"] = "已处理" if related else "未处理"
            record["related_dataset"] = {"dataset_id": related["dataset_id"], "dataset_name": related["dataset_name"]} if related else None
            records.append(record)
        return records
