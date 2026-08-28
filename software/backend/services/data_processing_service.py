from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.active_store_dataset import build_active_store_site_dataset
from src.data.aggregate_sales import build_daily_sales, build_monthly_sales
from src.data.clean_sales import clean_sales_chunk
from src.features.demand_signal_features import add_cross_store_features, add_diff_features

from .data_quality_service import summarize_monthly
from .field_mapping_service import validate_mapping
from .runtime import runtime_root


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV: {path.name}")


def _to_research_schema(frame: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    output = pd.DataFrame()
    output["site_no"] = frame[mapping["site_no"]]
    output["period"] = frame[mapping["sale_date"]]
    output["qty"] = frame[mapping["qty"]]
    item_source = mapping["item_id"]
    isbn_source = mapping.get("isbn")
    output["isbn"] = frame[isbn_source] if isbn_source else frame[item_source]
    output["gds_no"] = frame[item_source]
    for standard, research in (("price", "price"), ("amount", "tsp"), ("channel", "oln_or_ofln")):
        source = mapping.get(standard)
        if source:
            output[research] = frame[source]
    output["tlp"] = output.get("tsp", output.get("price", 0))
    return output


def _cross_store_frame(active: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    groups = monthly.copy()
    groups["total_qty"] = pd.to_numeric(groups["total_qty"], errors="coerce").fillna(0.0)
    aggregate = groups.groupby(["month", "item_id"], as_index=False).agg(
        group_positive_qty=("total_qty", lambda series: series.clip(lower=0).sum()),
        group_positive_sites=("total_qty", lambda series: int((series > 0).sum())),
    )
    result = active.copy()
    periods = pd.PeriodIndex(result["month"].astype(str), freq="M")
    for offset, suffix in enumerate(("t", "t1", "t2")):
        lookup = aggregate.copy()
        lookup["month"] = (pd.PeriodIndex(lookup["month"].astype(str), freq="M") + offset).astype(str)
        lookup = lookup.rename(columns={"group_positive_qty": f"group_positive_qty_{suffix}", "group_positive_sites": f"group_positive_sites_{suffix}"})
        result = result.merge(lookup, on=["month", "item_id"], how="left")
    required = [column for column in result if column.startswith("group_positive_")]
    result[required] = result[required].fillna(0.0)
    return result


class DataProcessingService:
    def process(self, upload: dict, dataset_name: str, job_service, job_id: str) -> dict:
        mapping = upload["field_mapping"]
        validate_mapping(mapping, upload["headers"])
        job_service.log(job_id, "reading CSV files")
        raw_frames = [_to_research_schema(_read_csv(Path(path)), mapping) for path in upload["csv_files"]]
        cleaned_frames = [clean_sales_chunk(frame)[0] for frame in raw_frames]
        cleaned = pd.concat(cleaned_frames, ignore_index=True)
        if cleaned.empty:
            raise ValueError("No usable rows remain after cleaning")
        daily = build_daily_sales(cleaned)
        monthly = build_monthly_sales(daily)
        active_parts = [build_active_store_site_dataset(part)[0] for _, part in monthly.groupby("site_no", sort=False)]
        active = pd.concat(active_parts, ignore_index=True) if active_parts else pd.DataFrame()
        if active.empty:
            raise ValueError("Active-Store dataset is empty; at least two observed months per store are required")
        enriched = _cross_store_frame(active, monthly)
        diff = add_diff_features(enriched)
        cross = add_cross_store_features(enriched)
        feature = pd.concat([active.reset_index(drop=True), diff.reset_index(drop=True), cross.reset_index(drop=True)], axis=1)
        dataset_dir = runtime_root() / "datasets" / dataset_name.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        paths = {"cleaned": dataset_dir / "cleaned.parquet", "monthly": dataset_dir / "monthly.parquet", "active": dataset_dir / "active_store.parquet", "feature": dataset_dir / "features.parquet"}
        cleaned.to_parquet(paths["cleaned"], index=False)
        monthly.to_parquet(paths["monthly"], index=False)
        active.to_parquet(paths["active"], index=False)
        feature.to_parquet(paths["feature"], index=False)
        job_service.log(job_id, f"wrote feature parquet: {paths['feature']}")
        quality = summarize_monthly(monthly)
        return {**quality, "source_type": Path(upload["filename"]).suffix.lower().lstrip("."), "source_files": upload["csv_files"],
                "monthly_parquet_path": str(paths["monthly"]), "active_store_parquet_path": str(paths["active"]), "feature_parquet_path": str(paths["feature"]),
                "has_active_store": True, "has_diff_features": True, "has_cross_store_features": True,
                "output_files": [str(path) for path in paths.values()]}
