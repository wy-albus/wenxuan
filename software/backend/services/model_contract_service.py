from __future__ import annotations

from dataclasses import dataclass

from .model_registry import ModelRegistry


LEAKAGE_PREFIXES = ("future_qty", "future_has_sales", "target_qty", "target_mc")


@dataclass(frozen=True)
class ModelInputContract:
    model_id: str
    model_name: str
    model_version: str
    forecast_horizon: str
    required_feature_names: list[str]
    feature_schema_version: str
    requires_active_store: bool
    requires_cross_store: bool
    requires_diff: bool
    minimum_history_months: int
    required_base_fields: list[str]
    category_encoding_version: str
    supported_task: str

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "forecast_horizon": self.forecast_horizon,
            "required_feature_names": self.required_feature_names,
            "feature_schema_version": self.feature_schema_version,
            "requires_active_store": self.requires_active_store,
            "requires_cross_store": self.requires_cross_store,
            "requires_diff": self.requires_diff,
            "minimum_history_months": self.minimum_history_months,
            "required_base_fields": self.required_base_fields,
            "category_encoding_version": self.category_encoding_version,
            "supported_task": self.supported_task,
        }


class ModelContractService:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    def get_contract(self, model_id: str) -> ModelInputContract:
        _, metadata = self.model_registry.load(model_id)
        feature_names = list(metadata["feature_names"])
        leakage = [
            feature
            for feature in feature_names
            if feature.startswith(LEAKAGE_PREFIXES) or feature == "split"
        ]
        if leakage:
            raise ValueError(f"Model {model_id} uses future target fields as input: {leakage}")
        requires_cross_store = any(feature.startswith("xstore_") for feature in feature_names)
        requires_diff = any(
            feature.startswith(prefix)
            for feature in feature_names
            for prefix in ("qty_diff_", "log_qty_diff_", "sales_days_diff_")
        )
        return ModelInputContract(
            model_id=model_id,
            model_name=str(metadata.get("model_name") or model_id),
            model_version=str(metadata.get("model_version") or metadata.get("trained_at") or "metadata"),
            forecast_horizon="1M",
            required_feature_names=feature_names,
            feature_schema_version=str(metadata.get("feature_schema_version") or "model-metadata"),
            requires_active_store=True,
            requires_cross_store=requires_cross_store,
            requires_diff=requires_diff,
            minimum_history_months=self._minimum_history_months(feature_names),
            required_base_fields=["month", "site_no", "item_id", "total_qty"],
            category_encoding_version=str(metadata.get("category_encoding_version") or "model-metadata"),
            supported_task="future_1m_book_sales",
        )

    def get_contracts(self, model_ids: list[str]) -> list[ModelInputContract]:
        return [self.get_contract(model_id) for model_id in model_ids]

    @staticmethod
    def _minimum_history_months(feature_names: list[str]) -> int:
        minimum = 1
        for feature in feature_names:
            for marker, months in (("_last_6m", 6), ("_lag_6m", 6), ("_last_3m", 3), ("_lag_3m", 3), ("_lag_2m", 2), ("_lag_1m", 1)):
                if marker in feature:
                    minimum = max(minimum, months)
        return minimum
