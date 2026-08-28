from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from software.backend.services.dataset_registry import DatasetRegistry
from software.backend.services.job_service import JobService
from software.backend.services.prediction_registry import PredictionRegistry
from software.backend.services.prediction_service import PredictionService
from software.backend.services.export_service import export_prediction_excel
from software.backend.services.result_query_service import page_rows, read_result_rows
from software.backend.services.email_service import EmailService


router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class PredictionRequest(BaseModel):
    dataset_id: str
    model_ids: list[str] = Field(min_length=1)
    observation_month: str | None = None
    store_ids: list[str] | None = None


class PredictionNotificationRequest(BaseModel):
    target_email: str
    model_id: str | None = None
    include_excel_link: bool = True
    notification_type: str = "PREDICTION_SUCCESS"
    site_no: str | None = None


@router.post("", status_code=201)
def create_prediction(request: PredictionRequest) -> dict:
    try:
        dataset = DatasetRegistry().get(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    run_id = uuid.uuid4().hex
    jobs = JobService()
    job = jobs.create("PREDICTION", [dataset["feature_parquet_path"]])
    registry = PredictionRegistry()
    registry.create({"prediction_run_id": run_id, "dataset_id": request.dataset_id, "model_ids": request.model_ids,
                     "observation_month": request.observation_month, "store_ids": request.store_ids, "job_id": job["job_id"]})
    jobs.update(job["job_id"], status="RUNNING", progress=5)
    jobs.log(job["job_id"], f"starting prediction run {run_id}")
    registry.update(run_id, status="RUNNING")
    try:
        result = PredictionService().run(dataset=dataset, model_ids=request.model_ids, observation_month=request.observation_month,
                                         store_ids=request.store_ids, prediction_run_id=run_id, job_service=jobs, job_id=job["job_id"])
        registry.update(run_id, status="SUCCESS", prediction_dir=result["prediction_dir"])
        jobs.update(job["job_id"], status="SUCCESS", progress=100, output_files=result["output_files"])
        return {"prediction_run_id": run_id, "job_id": job["job_id"], "status": "SUCCESS"}
    except Exception as exc:
        registry.update(run_id, status="FAILED", error_message=str(exc))
        jobs.update(job["job_id"], status="FAILED", progress=100, error_message=str(exc))
        jobs.log(job["job_id"], f"ERROR {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def list_predictions() -> dict:
    return {"items": PredictionRegistry().list()}


@router.get("/{run_id}")
def get_prediction(run_id: str) -> dict:
    try:
        return PredictionRegistry().get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction run not found") from exc


def _completed_run(run_id: str) -> dict:
    try:
        run = PredictionRegistry().get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction run not found") from exc
    if run["status"] != "SUCCESS" or not run["prediction_dir"]:
        raise HTTPException(status_code=409, detail="Prediction run has not completed successfully")
    return run


@router.get("/{run_id}/summary")
def get_summary(run_id: str) -> dict:
    run = _completed_run(run_id)
    return json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))


@router.post("/{run_id}/notify")
def notify_prediction(run_id: str, request: PredictionNotificationRequest) -> dict:
    try:
        run = PredictionRegistry().get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction run not found") from exc
    service = EmailService()
    if run["status"] == "FAILED":
        try:
            log_path = JobService().get(run["job_id"])["log_path"]
        except KeyError:
            log_path = None
        return service.send_prediction_failure(run, request.target_email, log_path)
    if run["status"] != "SUCCESS" or not run["prediction_dir"]:
        raise HTTPException(status_code=409, detail="Prediction run has not completed")
    model_id = request.model_id or run["model_ids"][0]
    if model_id not in run["model_ids"]:
        raise HTTPException(status_code=422, detail="model_id is not part of this prediction run")
    if request.notification_type == "EXPORT_SUCCESS":
        excel_path = export_prediction_excel(run, model_id=model_id, site_no=request.site_no)
        return service.send_export_success(run, request.target_email, model_id=model_id, site_no=request.site_no, excel_path=str(excel_path))
    if request.notification_type != "PREDICTION_SUCCESS":
        raise HTTPException(status_code=422, detail="notification_type must be PREDICTION_SUCCESS or EXPORT_SUCCESS")
    return service.send_prediction_success(run, request.target_email, model_id, request.include_excel_link)


@router.get("/{run_id}/results")
def get_results(
    run_id: str,
    kind: str = Query("predictions", pattern="^(predictions|top_books|store_summary)$"),
    model_id: str | None = None,
    site_no: str | None = None,
    mc: str | None = Query(None, pattern="^MC[0-4]$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
) -> dict:
    run = _completed_run(run_id)
    try:
        result = page_rows(read_result_rows(run["prediction_dir"], kind=kind, model_id=model_id, site_no=site_no, mc=mc), page=page, page_size=page_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "kind": kind}


@router.get("/{run_id}/store/{site_no}/summary")
def get_store_summary(run_id: str, site_no: str, model_id: str | None = None) -> dict:
    run = _completed_run(run_id)
    rows = read_result_rows(run["prediction_dir"], model_id=model_id, site_no=site_no)
    if rows.empty:
        raise HTTPException(status_code=404, detail="No predictions found for this store")
    summaries = {}
    for selected_model, selected_rows in rows.groupby("model_id", sort=False):
        summaries[str(selected_model)] = {
            "prediction_total": int(selected_rows["pred_qty_int"].sum()),
            "pred_nonzero_count": int((selected_rows["pred_qty_int"] > 0).sum()),
            "pred_mc3_count": int((selected_rows["pred_mc"] == "MC3").sum()),
            "pred_mc4_count": int((selected_rows["pred_mc"] == "MC4").sum()),
            "item_count": int(len(selected_rows)),
        }
    if model_id:
        return {"prediction_run_id": run_id, "site_no": site_no, "model_id": model_id, **summaries[model_id]}
    return {"prediction_run_id": run_id, "site_no": site_no, "model_summaries": summaries}


@router.get("/{run_id}/store/{site_no}/results")
def get_store_results(run_id: str, site_no: str, model_id: str | None = None, mc: str | None = Query(None, pattern="^MC[0-4]$"), page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=1000)) -> dict:
    return get_results(run_id, kind="predictions", model_id=model_id, site_no=site_no, mc=mc, page=page, page_size=page_size)


@router.get("/{run_id}/export-excel")
def export_excel(run_id: str, model_id: str | None = None, site_no: str | None = None, mc: str | None = Query(None, pattern="^MC[0-4]$"), include_predictions: bool = False, top_n: int = Query(100, ge=1, le=10000)):
    run = _completed_run(run_id)
    try:
        path = export_prediction_excel(run, model_id=model_id, site_no=site_no, mc=mc, include_predictions=include_predictions, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=path.name)


@router.get("/{run_id}/download-parquet")
def download_parquet(run_id: str):
    run = _completed_run(run_id)
    path = Path(run["prediction_dir"]) / "predictions.parquet"
    return FileResponse(path, media_type="application/octet-stream", filename=f"{run_id}_predictions.parquet")
