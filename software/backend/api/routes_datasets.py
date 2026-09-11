from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from software.backend.services.data_processing_service import DataProcessingService
from software.backend.services.dataset_registry import DatasetRegistry
from software.backend.services.job_service import JobService
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


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    try:
        return DatasetRegistry().get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
