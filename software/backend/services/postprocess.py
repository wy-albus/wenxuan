from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.mc_metrics import load_mc_level_config, quantity_to_mc, rounded_horizon_quantity

from .runtime import PROJECT_ROOT


MC_CONFIG_PATH = PROJECT_ROOT / "config" / "mc_sales_levels.yaml"
MC_LABELS = {0: "MC0", 1: "MC1", 2: "MC2", 3: "MC3", 4: "MC4"}


def apply_prediction_postprocess(values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the research project's fixed 1-month rounding and MC definition."""
    raw = np.nan_to_num(np.asarray(values, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0.0, None)
    quantity = rounded_horizon_quantity(raw, "1m")
    mc_codes = quantity_to_mc(raw, load_mc_level_config(MC_CONFIG_PATH), "1m")
    mc_labels = np.asarray([MC_LABELS[int(code)] for code in mc_codes], dtype=object)
    return raw, quantity, mc_labels
