from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.reports import write_markdown  # noqa: E402
from src.utils.io_utils import load_yaml, project_path  # noqa: E402


def _bucket(qty: float) -> str:
    if qty <= 0:
        return "0"
    if qty == 1:
        return "1"
    if qty <= 5:
        return "2-5"
    if qty <= 20:
        return "5-20"
    return "20+"


def _simple_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据_"
    return df.to_markdown(index=False)


def _top_contribution(item_sales: pd.DataFrame, pct: float) -> float:
    if item_sales.empty or item_sales["total_qty"].sum() == 0:
        return 0.0
    sorted_sales = item_sales.sort_values("total_qty", ascending=False)
    top_n = max(1, int(len(sorted_sales) * pct))
    return float(sorted_sales.head(top_n)["total_qty"].sum() / sorted_sales["total_qty"].sum())


def analyze_long_tail() -> dict:
    paths = load_yaml("config/paths.yaml")
    monthly_path = project_path(paths["monthly_sales"])
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly sales file not found: {monthly_path}")

    monthly = pd.read_parquet(monthly_path)
    monthly["qty_bucket"] = monthly["total_qty"].fillna(0).map(_bucket)

    bucket_summary = (
        monthly["qty_bucket"]
        .value_counts()
        .reindex(["0", "1", "2-5", "5-20", "20+"], fill_value=0)
        .rename_axis("bucket")
        .reset_index(name="rows")
    )
    bucket_summary["ratio"] = bucket_summary["rows"] / max(len(monthly), 1)

    monthly_distribution = monthly["total_qty"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    monthly_distribution.columns = ["metric", "value"]

    item_sales = monthly.groupby("item_id", dropna=False)["total_qty"].sum().reset_index()
    top_10 = _top_contribution(item_sales, 0.10)
    top_20 = _top_contribution(item_sales, 0.20)

    category_tail = (
        monthly.groupby("gds_ctgry_3_lvel", dropna=False)
        .agg(
            item_month_rows=("item_id", "size"),
            total_qty=("total_qty", "sum"),
            median_qty=("total_qty", "median"),
            zero_ratio=("total_qty", lambda s: float((s <= 0).mean())),
        )
        .reset_index()
        .sort_values("total_qty", ascending=False)
        .head(30)
    )

    channel_distribution = pd.DataFrame(
        {
            "channel": ["offline", "online", "unknown"],
            "total_qty": [
                monthly["offline_qty"].sum(),
                monthly["online_qty"].sum(),
                monthly["unknown_channel_qty"].sum(),
            ],
            "nonzero_rows": [
                int((monthly["offline_qty"] > 0).sum()),
                int((monthly["online_qty"] > 0).sum()),
                int((monthly["unknown_channel_qty"] > 0).sum()),
            ],
        }
    )
    channel_distribution["qty_ratio"] = channel_distribution["total_qty"] / max(
        channel_distribution["total_qty"].sum(), 1
    )

    zero_ratio = float((monthly["total_qty"] <= 0).mean()) if len(monthly) else 0.0
    low_ratio = float((monthly["total_qty"].between(0, 5, inclusive="right")).mean()) if len(monthly) else 0.0
    core_findings = [
        f"月度门店-图书样本中，销量 <= 0 的比例为 {zero_ratio:.2%}。",
        f"月销量在 1 到 5 本之间的低销量样本比例为 {low_ratio:.2%}。",
        f"头部 10% 图书贡献 {top_10:.2%} 的销量，头部 20% 图书贡献 {top_20:.2%} 的销量。",
    ]

    lines = [
        "# Long Tail Analysis",
        "",
        "> 口径说明：当前月度表由实际销售/退货流水聚合而来，未补全没有任何流水的门店-图书-月份。因此 `0` 桶表示有流水但净销量为 0 的记录；真正的无流水零销量月份，应在下一阶段构造模型训练集时通过完整月份网格补齐。",
        "",
        "## Monthly Quantity Distribution",
        "",
        _simple_table(monthly_distribution),
        "",
        "## Quantity Buckets",
        "",
        _simple_table(bucket_summary),
        "",
        "## Head Item Contribution",
        "",
        f"- Top 10% items contribution: {top_10:.2%}",
        f"- Top 20% items contribution: {top_20:.2%}",
        "",
        "## Category Long Tail Summary",
        "",
        _simple_table(category_tail),
        "",
        "## Online vs Offline Distribution",
        "",
        _simple_table(channel_distribution),
        "",
        "## Modeling Implications",
        "",
        "- 图书销量呈现大量低频、零销量或极低销量样本，序列稀疏且间歇性强。",
        "- LSTM 更依赖连续、稳定、有足够历史信号的序列；在这种长尾低频图书场景中不适合作为优先模型。",
        "- 后续应优先考虑 LightGBM + log1p(y)、Poisson/Tweedie 目标或两阶段模型：先判断是否会卖，再预测会卖多少。",
        "",
        "## Core Findings",
        "",
        "\n".join(f"- {finding}" for finding in core_findings),
        "",
    ]
    write_markdown(paths["long_tail_report"], "\n".join(lines))
    return {
        "zero_ratio": zero_ratio,
        "low_1_to_5_ratio": low_ratio,
        "top_10_contribution": top_10,
        "top_20_contribution": top_20,
        "findings": core_findings,
    }


if __name__ == "__main__":
    print(analyze_long_tail())
