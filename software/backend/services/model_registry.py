from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.models.lightgbm_model import load_model_bundle

from .runtime import PROJECT_ROOT


class ModelRegistry:
    """Loads only the approved, metadata-bearing LightGBM model bundles."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or PROJECT_ROOT / "software" / "config" / "model_registry.yaml"
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def supported_model_ids(self) -> list[str]:
        return list(self.config.get("models", {}))

    def _entry(self, model_id: str) -> dict:
        if model_id == "classifier":
            return self.config["classifier"]
        try:
            return self.config["models"][model_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported model_id: {model_id}") from exc

    @lru_cache(maxsize=4)
    def load(self, model_id: str):
        entry = self._entry(model_id)
        path = PROJECT_ROOT / entry["model_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Registered model file not found: {path}")
        booster, metadata = load_model_bundle(path)
        expected_iteration = entry.get("best_iteration")
        if expected_iteration is not None and int(metadata.get("best_iteration", -1)) != int(expected_iteration):
            raise ValueError(f"Model iteration mismatch for {model_id}")
        return booster, metadata
