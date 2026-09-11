from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from software.backend.services.dataset_registry import DatasetRegistry
from software.backend.services.difficult_books_service import build_difficult_books, difficult_books_summary, export_difficult_books, page_difficult_books
from software.backend.services.job_service import JobService
from software.backend.services.prediction_registry import PredictionRegistry
from software.backend.services.prediction_service import PredictionService, target_month_for
from software.backend.services.export_service import export_prediction_excel
from software.backend.services.result_query_service import page_rows, read_result_rows, summarize_prediction_rows
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
    mc: str | None = Field(None, pattern="^MC[0-4]$")


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
def get_summary(
    run_id: str,
    model_id: str | None = None,
    site_no: str | None = None,
    mc: str | None = Query(None, pattern="^MC[0-4]$"),
) -> dict:
    run = _completed_run(run_id)
    artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    artifact.setdefault("target_month", target_month_for(artifact.get("observation_month")))
    rows = read_result_rows(run["prediction_dir"], model_id=model_id, site_no=site_no, mc=mc)
    return {**artifact, "filtered_summary": summarize_prediction_rows(rows, model_id=model_id, site_no=site_no, mc=mc)}


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
    return service.send_prediction_success(run, request.target_email, model_id, request.include_excel_link, site_no=request.site_no, mc=request.mc)


@router.get("/{run_id}/results")
def get_results(
    run_id: str,
    kind: str = Query("predictions", pattern="^(predictions|top_books|store_summary)$"),
    model_id: str | None = None,
    site_no: str | None = None,
    mc: str | None = Query(None, pattern="^MC[0-4]$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    sort_by: str = Query("pred_qty_int", pattern="^(pred_qty_int|p_sale|item_id)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict:
    run = _completed_run(run_id)
    try:
        result = page_rows(read_result_rows(run["prediction_dir"], kind=kind, model_id=model_id, site_no=site_no, mc=mc, sort_by=sort_by, sort_order=sort_order), page=page, page_size=page_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "kind": kind, "sort": {"sort_by": sort_by, "sort_order": sort_order}}


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


@router.get("/{run_id}/historical-series")
def get_historical_series(
    run_id: str,
    model_id: str | None = None,
    site_no: str | None = None,
    item_id: str | None = None,
) -> dict:
    run = _completed_run(run_id)
    dataset = DatasetRegistry().get(run["dataset_id"])
    monthly_path = Path(dataset["monthly_parquet_path"])
    if not monthly_path.is_file():
        raise HTTPException(status_code=404, detail="Historical monthly sales file not found")
    monthly = pd.read_parquet(monthly_path)
    if "total_qty" not in monthly.columns:
        raise HTTPException(status_code=422, detail="Historical monthly sales file has no total_qty column")
    if site_no and "site_no" in monthly.columns:
        monthly = monthly.loc[monthly["site_no"].astype(str).eq(str(site_no))]
    if item_id:
        monthly_item_column = "item_id" if "item_id" in monthly.columns else "gds_no" if "gds_no" in monthly.columns else None
        if not monthly_item_column:
            raise HTTPException(status_code=422, detail="Historical monthly sales file has no item_id column")
        monthly = monthly.loc[monthly[monthly_item_column].astype(str).eq(str(item_id))]
    actual = monthly.groupby("month", as_index=False).agg(actual_qty=("total_qty", "sum")).sort_values("month")

    artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    selected_model = model_id or run["model_ids"][0]
    predictions = read_result_rows(run["prediction_dir"], kind="predictions", model_id=selected_model, site_no=site_no)
    if item_id:
        predictions = predictions.loc[predictions["item_id"].astype(str).eq(str(item_id))]
    pred_total = int(predictions["pred_qty_int"].sum()) if not predictions.empty else 0
    target_month = None
    if artifact.get("observation_month"):
        target_month = str(pd.Period(artifact["observation_month"], freq="M") + 1)
    items = [
        {"month": str(row["month"]), "actual_qty": int(row["actual_qty"]), "pred_qty": None, "is_prediction": False}
        for row in actual.where(actual.notna(), None).to_dict(orient="records")
    ]
    if target_month:
        items.append({"month": target_month, "actual_qty": None, "pred_qty": pred_total, "is_prediction": True})
    return {"prediction_run_id": run_id, "model_id": selected_model, "site_no": site_no, "item_id": item_id, "items": items}


@router.get("/{run_id}/difficult-books")
def get_difficult_books(
    run_id: str,
    model_id: str | None = None,
    site_no: str | None = None,
    difficulty_level: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
) -> dict:
    run = _completed_run(run_id)
    artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    rows = build_difficult_books(
        run["prediction_dir"],
        observation_month=artifact.get("observation_month"),
        model_id=model_id,
        site_no=site_no,
        difficulty_level=difficulty_level,
    )
    return page_difficult_books(rows, page=page, page_size=page_size)


@router.get("/{run_id}/difficult-books/summary")
def get_difficult_books_summary(
    run_id: str,
    model_id: str | None = None,
    site_no: str | None = None,
    difficulty_level: str | None = None,
) -> dict:
    run = _completed_run(run_id)
    artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    rows = build_difficult_books(
        run["prediction_dir"],
        observation_month=artifact.get("observation_month"),
        model_id=model_id,
        site_no=site_no,
        difficulty_level=difficulty_level,
    )
    all_rows = read_result_rows(run["prediction_dir"], kind="predictions", model_id=model_id, site_no=site_no)
    return difficult_books_summary(rows, total_prediction_rows=len(all_rows))


@router.get("/{run_id}/difficult-books/export")
def export_difficult_books_excel(
    run_id: str,
    model_id: str | None = None,
    site_no: str | None = None,
    difficulty_level: str | None = None,
):
    run = _completed_run(run_id)
    artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
    rows = build_difficult_books(
        run["prediction_dir"],
        observation_month=artifact.get("observation_month"),
        model_id=model_id,
        site_no=site_no,
        difficulty_level=difficulty_level,
    )
    if rows.empty:
        raise HTTPException(status_code=404, detail="No difficult books found for the selected filters")
    path = export_difficult_books(run_id, rows)
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=path.name)


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
