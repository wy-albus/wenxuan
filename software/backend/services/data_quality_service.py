from __future__ import annotations

import pandas as pd


def summarize_monthly(frame: pd.DataFrame) -> dict:
    months = frame["month"].astype(str)
    return {
        "date_range": {"start": months.min(), "end": months.max()},
        "store_count": int(frame["site_no"].nunique()),
        "item_count": int(frame["item_id"].nunique()),
        "row_count": int(len(frame)),
    }
