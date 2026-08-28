from __future__ import annotations

from fastapi import APIRouter, HTTPException

from software.backend.services.job_service import JobService


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs() -> dict:
    return {"items": JobService().list()}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return JobService().get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
