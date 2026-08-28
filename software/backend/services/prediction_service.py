from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocessing import add_runtime_columns
from src.models.active_store_common import encode_categories_with_unknown
from src.models.two_stage_model import combine_predictions

from .model_registry import ModelRegistry
from .postprocess import apply_prediction_postprocess
from .runtime import runtime_root


IDENTITY_COLUMNS = ("site_no", "item_id")
OPTIONAL_DISPLAY_COLUMNS = ("isbn", "book_name", "category")


def _prepare_model_frame(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    features = list(metadata["feature_names"])
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature parquet is missing required model fields: {missing}")
    prepared = frame.replace([np.inf, -np.inf], np.nan).copy()
    category_maps = metadata.get("category_maps", {})
    missing_categories = [column for column in category_maps if column not in prepared.columns]
    if missing_categories:
        raise ValueError(f"Feature parquet is missing required categorical fields: {missing_categories}")
    prepared, _ = encode_categories_with_unknown(prepared, category_maps)
    categorical = set(category_maps)
    for column in features:
        if column not in categorical:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
    return prepared[features]


class PredictionService:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    @staticmethod
    def available_months(feature_path: str | Path) -> list[str]:
        frame = pd.read_parquet(feature_path, columns=["month"])
        return sorted(frame["month"].dropna().astype(str).unique().tolist())

    def run(
        self,
        *,
        dataset: dict,
        model_ids: list[str],
        observation_month: str | None = None,
        store_ids: list[str] | None = None,
        prediction_run_id: str | None = None,
        job_service=None,
        job_id: str | None = None,
    ) -> dict:
        if dataset.get("status") != "READY":
            raise ValueError("Only READY datasets can be used for prediction")
        if not model_ids:
            raise ValueError("At least one model_id is required")
        unsupported = sorted(set(model_ids) - set(self.model_registry.supported_model_ids()))
        if unsupported:
            raise ValueError(f"Unsupported model_ids: {unsupported}")
        feature_path = Path(dataset["feature_parquet_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature parquet not found: {feature_path}")
        frame = pd.read_parquet(feature_path)
        missing_identity = [column for column in ("month", *IDENTITY_COLUMNS) if column not in frame.columns]
        if missing_identity:
            raise ValueError(f"Feature parquet is missing identity fields: {missing_identity}")
        months = sorted(frame["month"].dropna().astype(str).unique().tolist())
        if not months:
            raise ValueError("Feature parquet has no observation month")
        observation_month = observation_month or months[-1]
        if observation_month not in months:
            raise ValueError(f"observation_month is not available in dataset: {observation_month}")
        selected = frame.loc[frame["month"].astype(str).eq(observation_month)].copy()
        if store_ids:
            selected = selected.loc[selected["site_no"].astype(str).isin({str(value) for value in store_ids})].copy()
        if selected.empty:
            raise ValueError("No feature rows matched the requested observation month and stores")
        if job_service and job_id:
            job_service.update(job_id, status="RUNNING", progress=25)
            job_service.log(job_id, f"loaded {len(selected)} feature rows for {observation_month}")

        selected = add_runtime_columns(selected)
        classifier, classifier_metadata = self.model_registry.load("classifier")
        classifier_x = _prepare_model_frame(selected, classifier_metadata)
        p_sale = classifier.predict(classifier_x, num_iteration=int(classifier_metadata["best_iteration"]))

        run_id = prediction_run_id or uuid.uuid4().hex
        result_parts: list[pd.DataFrame] = []
        for index, model_id in enumerate(model_ids, start=1):
            regressor, metadata = self.model_registry.load(model_id)
            conditional_qty = regressor.predict(
                _prepare_model_frame(selected, metadata), num_iteration=int(metadata["best_iteration"])
            )
            pred_raw, pred_int, pred_mc = apply_prediction_postprocess(combine_predictions(p_sale, conditional_qty))
            output = pd.DataFrame({
                "dataset_id": dataset["dataset_id"], "prediction_run_id": run_id, "model_id": model_id,
                "site_no": selected["site_no"].astype(str).to_numpy(), "item_id": selected["item_id"].astype(str).to_numpy(),
                "p_sale": np.asarray(p_sale, dtype="float64"), "conditional_qty": np.asarray(conditional_qty, dtype="float64"),
                "pred_qty_raw": pred_raw, "pred_qty_int": pred_int, "pred_mc": pred_mc,
            })
            for column in OPTIONAL_DISPLAY_COLUMNS:
                if column in selected.columns:
                    output[column] = selected[column].to_numpy()
            result_parts.append(output)
            if job_service and job_id:
                job_service.update(job_id, status="RUNNING", progress=25 + int(index / len(model_ids) * 60))
                job_service.log(job_id, f"completed {model_id} prediction")

        predictions = pd.concat(result_parts, ignore_index=True)
        prediction_dir = runtime_root() / "predictions" / run_id
        prediction_dir.mkdir(parents=True, exist_ok=False)
        prediction_path = prediction_dir / "predictions.parquet"
        top_books_path = prediction_dir / "top_books.parquet"
        store_summary_path = prediction_dir / "store_summary.parquet"
        summary_path = prediction_dir / "summary.json"
        predictions.to_parquet(prediction_path, index=False)
        top_books = predictions.sort_values(["model_id", "pred_qty_int", "pred_qty_raw"], ascending=[True, False, False]).groupby("model_id", group_keys=False).head(100).reset_index(drop=True)
        top_books.to_parquet(top_books_path, index=False)
        store_summary = predictions.groupby(["model_id", "site_no"], as_index=False).agg(
            pred_total=("pred_qty_int", "sum"),
            pred_nonzero_count=("pred_qty_int", lambda values: int((values > 0).sum())),
            pred_20_plus_count=("pred_qty_int", lambda values: int((values >= 20).sum())),
        )
        store_summary.to_parquet(store_summary_path, index=False)
        model_summaries: dict[str, dict] = {}
        for model_id, rows in predictions.groupby("model_id", sort=False):
            model_summaries[str(model_id)] = {
                "prediction_total": int(rows["pred_qty_int"].sum()),
                "predicted_nonzero_book_count": int((rows["pred_qty_int"] > 0).sum()),
                "mc_counts": {f"MC{level}": int((rows["pred_mc"] == f"MC{level}").sum()) for level in range(5)},
                "predicted_20_plus_book_count": int((rows["pred_qty_int"] >= 20).sum()),
                "store_count": int(rows["site_no"].nunique()),
                "item_count": int(rows["item_id"].nunique()),
                "row_count": int(len(rows)),
            }
        summary = {
            "prediction_run_id": run_id, "dataset_id": dataset["dataset_id"], "observation_month": observation_month,
            "model_ids": model_ids, "model_summaries": model_summaries,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if job_service and job_id:
            job_service.log(job_id, f"wrote prediction files to {prediction_dir}")
        return {
            "prediction_run_id": run_id, "prediction_dir": str(prediction_dir), "observation_month": observation_month,
            "output_files": [str(prediction_path), str(summary_path), str(top_books_path), str(store_summary_path)],
            "summary": summary,
        }
