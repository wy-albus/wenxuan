from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def runtime_root() -> Path:
    root = Path(os.environ.get("WENXUAN_SOFTWARE_RUNTIME", PROJECT_ROOT / "software" / "runtime"))
    for name in ("uploads", "extracted", "jobs", "logs", "datasets", "predictions", "exports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def database_path() -> Path:
    return runtime_root() / "registry.sqlite3"
