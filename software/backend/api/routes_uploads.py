from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from software.backend.services.upload_service import UploadService


router = APIRouter(prefix="/api/uploads", tags=["uploads"])


@router.post("", status_code=201)
async def upload_file(file: UploadFile = File(...)) -> dict:
    try:
        return await UploadService().save(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{upload_id}")
def get_upload(upload_id: str) -> dict:
    try:
        return UploadService().get(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Upload not found") from exc
