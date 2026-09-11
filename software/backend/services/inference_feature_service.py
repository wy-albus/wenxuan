from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.model_dataset import add_inference_features, expand_site_panel, fill_panel_values, month_to_ord

from .data_processing_service import _cross_store_frame
from .data_requirement_service import DataRequirementService, previous_month
from .model_contract_service import ModelContractService
from .monthly_data_service import MonthlyDataService
from .runtime import runtime_root


@dataclass
class InferenceFeatureBuildResult:
    features: pd.DataFrame
    path: Path | None
    metadata: dict


class InferenceFeatureService:
    def __init__(
        self,
        model_contracts: ModelContractService | None = None,
        requirements: DataRequirementService | None = None,
        monthly_data: MonthlyDataService | None = None,
    ) -> None:
        self.model_contracts = model_contracts or ModelContractService()
        self.requirements = requirements or DataRequirementService()
        self.monthly_data = monthly_data or MonthlyDataService()

    def build(self, *, model_ids: list[str], target_month: str, prediction_run_id: str) -> InferenceFeatureBuildResult:
        monthly, lineage = self.monthly_data.read_standard_history()
        return self.build_from_monthly(
            monthly,
            model_ids=model_ids,
            target_month=target_month,
            prediction_run_id=prediction_run_id,
            lineage=lineage,
            persist=True,
        )

    def build_from_monthly(
        self,
        monthly: pd.DataFrame,
        *,
        model_ids: list[str],
        target_month: str,
        prediction_run_id: str,
        lineage: dict | None = None,
        persist: bool = False,
    ) -> InferenceFeatureBuildResult:
        contracts = self.model_contracts.get_contracts(model_ids)
        readiness = self.requirements.check(monthly, target_month=target_month, contracts=contracts)
        observation_month = readiness["observation_month"]
        observation_ord = int(pd.Period(observation_month, freq="M").ordinal)
        required_months = set(readiness["required_months"])
        source = monthly.loc[monthly["month"].astype(str).isin(required_months)].copy()
        parts = []
        for _, site_frame in source.groupby("site_no", sort=False):
            site_frame = site_frame.copy()
            site_frame["month_ord"] = month_to_ord(site_frame["month"]).to_numpy(dtype="int64")
            panel = expand_site_panel(site_frame, observation_ord)
            panel = fill_panel_values(panel, site_frame)
            part = add_inference_features(panel)
            if not part.empty:
                parts.append(part.loc[part["month"].astype(str).eq(observation_month)])
        features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if features.empty:
            raise ValueError(f"No inference feature rows could be built for observation_month={observation_month}")

        enriched = _cross_store_frame(features, source)
        required_cross_store = any(contract.requires_cross_store for contract in contracts)
        if required_cross_store:
            from src.features.demand_signal_features import add_cross_store_features

            cross_store = add_cross_store_features(enriched)
            enriched = pd.concat([enriched.reset_index(drop=True), cross_store.reset_index(drop=True)], axis=1)
        required_diff = any(contract.requires_diff for contract in contracts)
        if required_diff:
            from src.features.demand_signal_features import add_diff_features

            diff = add_diff_features(enriched)
            enriched = pd.concat([enriched.reset_index(drop=True), diff.reset_index(drop=True)], axis=1)

        banned = {"future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m", "target_qty_1m", "target_qty_2m"}
        enriched = enriched.drop(columns=[column for column in banned if column in enriched.columns])
        metadata = {
            "prediction_run_id": prediction_run_id,
            "model_ids": model_ids,
            "model_contracts": [contract.as_dict() for contract in contracts],
            "observation_month": observation_month,
            "target_month": target_month,
            "data_cutoff_month": observation_month,
            "row_count": int(len(enriched)),
            "feature_names": sorted({feature for contract in contracts for feature in contract.required_feature_names}),
            "source_months": readiness["required_months"],
            "source_file_ids": (lineage or {}).get("source_file_ids", []),
            "clean_data_version": "clean-layer-v1",
            "monthly_data_version": (lineage or {}).get("monthly_data_version", "monthly-standard-v1"),
            "feature_schema_version": "inference-builder-v1",
            "build_time": datetime.now(UTC).isoformat(),
        }
        path = None
        if persist:
            output_dir = runtime_root() / "inference_features" / prediction_run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "inference_features.parquet"
            enriched.to_parquet(path, index=False)
            (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return InferenceFeatureBuildResult(features=enriched, path=path, metadata=metadata)


def target_month_for_observation(observation_month: str) -> str:
    return str(pd.Period(observation_month, freq="M") + 1)


def observation_month_for_target(target_month: str) -> str:
    return previous_month(target_month)
