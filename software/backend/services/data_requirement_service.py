from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model_contract_service import ModelInputContract


@dataclass
class DataNotReady(ValueError):
    target_month: str
    observation_month: str
    missing_months: list[str]
    affected_features: list[str]

    def __post_init__(self) -> None:
        super().__init__(
            "DataNotReady: 缺少历史月份 "
            + "、".join(self.missing_months)
            + "，影响特征 "
            + "、".join(self.affected_features)
        )

    def as_dict(self) -> dict:
        return {
            "code": "DataNotReady",
            "target_month": self.target_month,
            "observation_month": self.observation_month,
            "missing_months": self.missing_months,
            "affected_features": self.affected_features,
            "detail": str(self),
        }


def previous_month(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def required_history_months(observation_month: str, contracts: list[ModelInputContract]) -> list[str]:
    minimum = max((contract.minimum_history_months for contract in contracts), default=1)
    observed = pd.Period(observation_month, freq="M")
    return [str(observed - offset) for offset in range(minimum, -1, -1)]


def affected_features_for_missing_history(contracts: list[ModelInputContract]) -> list[str]:
    result: list[str] = []
    for contract in contracts:
        for feature in contract.required_feature_names:
            if any(marker in feature for marker in ("_lag_6m", "_last_6m")) and feature not in result:
                result.append(feature)
    return result or ["historical_window"]


class DataRequirementService:
    def check(self, monthly: pd.DataFrame, *, target_month: str, contracts: list[ModelInputContract]) -> dict:
        observation_month = previous_month(target_month)
        required = required_history_months(observation_month, contracts)
        available = set(monthly["month"].dropna().astype(str).unique().tolist())
        missing = [month for month in required if month not in available]
        if missing:
            raise DataNotReady(
                target_month=target_month,
                observation_month=observation_month,
                missing_months=missing,
                affected_features=affected_features_for_missing_history(contracts),
            )
        return {
            "status": "READY",
            "target_month": target_month,
            "observation_month": observation_month,
            "required_months": required,
            "available_months": sorted(available),
            "contracts": [contract.as_dict() for contract in contracts],
        }
