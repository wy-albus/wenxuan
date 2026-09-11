from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from software.backend.services.data_processing_service import DataProcessingService
from software.backend.services.dataset_registry import DatasetRegistry
from software.backend.services.job_service import JobService
from software.backend.services.monthly_data_service import MonthlyDataService
from software.backend.services.upload_service import UploadService


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class ProcessRequest(BaseModel):
    upload_id: str
    dataset_name: str
    process_mode: str = "create"
    target_dataset_id: str | None = None


@router.post("/process", status_code=201)
def process_dataset(request: ProcessRequest) -> dict:
    try:
        upload = UploadService().get(request.upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Upload not found") from exc
    jobs = JobService()
    job = jobs.create("DATA_PROCESSING", upload["csv_files"])
    jobs.update(job["job_id"], status="RUNNING", progress=5)
    jobs.log(job["job_id"], f"processing upload {request.upload_id}")
    try:
        result = DataProcessingService().process(upload, request.dataset_name, jobs, job["job_id"])
        jobs.update(job["job_id"], status="RUNNING", progress=85)
        dataset = DatasetRegistry().register({"dataset_name": request.dataset_name, **{key: value for key, value in result.items() if key != "output_files"}})
        jobs.update(job["job_id"], status="SUCCESS", progress=100, output_files=result["output_files"])
        jobs.log(job["job_id"], f"registered dataset {dataset['dataset_id']}")
        return {"job_id": job["job_id"], "dataset_id": dataset["dataset_id"]}
    except Exception as exc:
        jobs.update(job["job_id"], status="FAILED", progress=100, error_message=str(exc))
        jobs.log(job["job_id"], f"ERROR {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def list_datasets() -> dict:
    return {"items": DatasetRegistry().list()}


@router.get("/months")
def list_data_months() -> dict:
    monthly, lineage = MonthlyDataService().read_standard_history()
    if monthly.empty:
        return {"items": [], "summary": {"month_count": 0, "store_count": 0, "item_count": 0, "date_range": None}, "lineage": lineage}
    grouped = []
    for month, rows in monthly.groupby("month", sort=True):
        sources = []
        if "source_dataset_id" in rows.columns:
            sources = sorted(rows["source_dataset_id"].dropna().astype(str).unique().tolist())
        grouped.append({
            "month": str(month),
            "year": str(month)[:4],
            "status": "READY",
            "store_count": int(rows["site_no"].nunique()),
            "item_count": int(rows["item_id"].nunique()),
            "row_count": int(len(rows)),
            "source_dataset_ids": sources,
        })
    months = [item["month"] for item in grouped]
    return {
        "items": grouped,
        "summary": {
            "month_count": len(grouped),
            "store_count": int(monthly["site_no"].nunique()),
            "item_count": int(monthly["item_id"].nunique()),
            "date_range": {"start": months[0], "end": months[-1]} if months else None,
        },
        "lineage": lineage,
    }


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    try:
        return DatasetRegistry().get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
