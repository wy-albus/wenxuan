from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.reports import write_markdown  # noqa: E402


DATASET_PATH = ROOT / "data" / "processed" / "model_dataset_monthly.parquet"
REPORT_PATH = ROOT / "reports" / "feature_schema_review.md"

IDENTIFIER_FIELDS = ["month", "site_no", "item_id", "isbn", "gds_no", "blt_site_no"]
CATEGORICAL_FEATURES = [
    "site_no",
    "blt_site_no",
    "gds_ctgry_3_lvel",
    "gds_ctgry_4_lvel",
    "gds_ctgry_5_lvel",
]
TARGET_FIELDS = [
    "future_qty_1m",
    "future_qty_2m",
    "future_has_sales_1m",
    "future_has_sales_2m",
]
SPLIT_FIELDS = ["split"]
HIGH_CARDINALITY_EXCLUDED = ["item_id", "isbn", "gds_no"]

BASELINE_FEATURES_PREFERRED = [
    "qty_lag_1m",
    "qty_lag_2m",
    "qty_lag_3m",
    "qty_mean_last_3m",
    "qty_sum_last_3m",
]

CURRENT_MONTH_NUMERIC_FIELDS = [
    "price",
    "total_qty",
    "offline_qty",
    "online_qty",
    "unknown_channel_qty",
    "total_tlp",
    "total_tsp",
    "avg_real_price",
    "discount_rate",
    "sales_days",
    "sales_count",
    "return_count",
    "return_qty",
]

NON_FEATURE_FIELDS = set(IDENTIFIER_FIELDS + TARGET_FIELDS + SPLIT_FIELDS)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def markdown_list(items: list[str]) -> str:
    if not items:
        return "_无_"
    return "\n".join(f"- `{item}`" for item in items)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无数据_"
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def get_schema() -> tuple[int, list[dict]]:
    pf = pq.ParquetFile(DATASET_PATH)
    fields = []
    for field in pf.schema_arrow:
        fields.append({"field": field.name, "dtype": str(field.type)})
    return pf.metadata.num_rows, fields


def profile_fields(columns: list[str], total_rows: int) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    path = str(DATASET_PATH).replace("\\", "/").replace("'", "''")
    select_parts = []
    for col in columns:
        q = quote_ident(col)
        select_parts.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS {quote_ident(col + '__nulls')}")
        select_parts.append(f"COUNT(DISTINCT {q}) AS {quote_ident(col + '__nunique')}")
    row = con.execute(f"SELECT {', '.join(select_parts)} FROM read_parquet('{path}')").fetchdf().iloc[0].to_dict()
    rows = []
    for col in columns:
        nulls = int(row[f"{col}__nulls"])
        rows.append(
            {
                "field": col,
                "missing_count": nulls,
                "missing_rate": f"{nulls / total_rows:.4%}" if total_rows else "NA",
                "unique_values": int(row[f"{col}__nunique"]),
            }
        )
    return pd.DataFrame(rows)


def numeric_stats(numeric_columns: list[str]) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    path = str(DATASET_PATH).replace("\\", "/").replace("'", "''")
    rows = []
    for col in numeric_columns:
        q = quote_ident(col)
        stats = con.execute(
            f"""
            SELECT
              MIN({q}) AS min_value,
              AVG({q}) AS mean_value,
              STDDEV_SAMP({q}) AS std_value,
              MAX({q}) AS max_value
            FROM read_parquet('{path}')
            """
        ).fetchone()
        rows.append(
            {
                "field": col,
                "min": stats[0],
                "mean": stats[1],
                "std": stats[2],
                "max": stats[3],
            }
        )
    return pd.DataFrame(rows)


def classify_fields(columns: list[str], dtypes: dict[str, str]) -> dict[str, list[str]]:
    numeric_cols = [col for col in columns if dtypes[col] in {"float", "double", "int32", "int64", "int16", "int8"}]
    leakage = [
        col
        for col in columns
        if col.startswith("future_") or col.startswith("target_") or col.startswith("label_")
    ]
    numeric_features = [col for col in numeric_cols if col not in TARGET_FIELDS]
    lag_rolling_and_behavior_features = [
        col
        for col in numeric_features
        if col not in CURRENT_MONTH_NUMERIC_FIELDS
    ]
    return {
        "identifier_fields": [col for col in IDENTIFIER_FIELDS if col in columns],
        "numeric_feature_candidates": numeric_features,
        "lag_rolling_behavior_numeric_features": lag_rolling_and_behavior_features,
        "categorical_feature_candidates": [col for col in CATEGORICAL_FEATURES if col in columns],
        "target_fields": [col for col in TARGET_FIELDS if col in columns],
        "split_fields": [col for col in SPLIT_FIELDS if col in columns],
        "high_cardinality_excluded": [col for col in HIGH_CARDINALITY_EXCLUDED if col in columns],
        "leakage_risk_fields": leakage,
    }


def build_feature_lists(columns: list[str], classified: dict[str, list[str]]) -> dict[str, list[str]]:
    baseline = [col for col in BASELINE_FEATURES_PREFERRED if col in columns]
    numeric_features = [
        col
        for col in classified["numeric_feature_candidates"]
        if col not in TARGET_FIELDS
    ]
    categorical = classified["categorical_feature_candidates"]
    lightgbm = numeric_features + categorical
    lightgbm = [
        col
        for col in lightgbm
        if col not in set(TARGET_FIELDS + SPLIT_FIELDS + HIGH_CARDINALITY_EXCLUDED + ["month"])
    ]
    mlp = [
        col
        for col in numeric_features
        if col not in set(TARGET_FIELDS + SPLIT_FIELDS)
    ]
    two_stage = lightgbm.copy()
    return {
        "baseline_features": baseline,
        "lightgbm_features": lightgbm,
        "mlp_features": mlp,
        "two_stage_features": two_stage,
    }


def write_feature_schema_review() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    total_rows, schema_rows = get_schema()
    schema_df = pd.DataFrame(schema_rows)
    columns = schema_df["field"].tolist()
    dtypes = dict(zip(schema_df["field"], schema_df["dtype"], strict=False))

    profile_df = profile_fields(columns, total_rows)
    full_profile = schema_df.merge(profile_df, on="field", how="left")
    numeric_columns = [
        row["field"]
        for row in schema_rows
        if row["dtype"] in {"float", "double", "int32", "int64", "int16", "int8"}
    ]
    numeric_profile = numeric_stats(numeric_columns)
    classified = classify_fields(columns, dtypes)
    feature_lists = build_feature_lists(columns, classified)

    not_recommended = {
        "future_* target fields": "目标变量，包含未来销量或未来是否有销量，作为输入会直接数据泄露。",
        "split": "只能用于 train / valid / test 时间切分，不能作为模型输入。",
        "item_id / isbn / gds_no": "高基数商品 ID，第一版先不直接输入，避免模型记忆商品编号；后续可考虑频率编码、目标编码或 embedding。",
        "month": "样本时间标识，第一版只用于时间切分和回溯审计，避免模型记忆绝对月份。",
    }

    lines = [
        "# Feature Schema Review",
        "",
        "## Dataset Overview",
        "",
        f"- Dataset: `{DATASET_PATH.as_posix()}`",
        f"- Total rows: {total_rows}",
        f"- Field count: {len(columns)}",
        "",
        "## All Fields",
        "",
        markdown_table(full_profile),
        "",
        "## Numeric Field Statistics",
        "",
        markdown_table(numeric_profile),
        "",
        "## Field Classification",
        "",
        "### Identifier Fields",
        markdown_list(classified["identifier_fields"]),
        "",
        "### Numeric Feature Candidates",
        markdown_list(classified["numeric_feature_candidates"]),
        "",
        "### Categorical Feature Candidates",
        markdown_list(classified["categorical_feature_candidates"]),
        "",
        "### Target Fields",
        markdown_list(classified["target_fields"]),
        "",
        "### Split Fields",
        markdown_list(classified["split_fields"]),
        "",
        "### Temporarily Excluded High-cardinality Fields",
        markdown_list(classified["high_cardinality_excluded"]),
        "",
        "### Potential Leakage Fields",
        markdown_list(classified["leakage_risk_fields"]),
        "",
        "## Fields Not Recommended As Inputs",
        "",
        "| field/group | reason |",
        "|---|---|",
    ]
    for field, reason in not_recommended.items():
        lines.append(f"| `{field}` | {reason} |")
    lines.extend(
        [
            "",
            "## Recommended Feature Lists",
            "",
            f"### baseline_features ({len(feature_lists['baseline_features'])})",
            markdown_list(feature_lists["baseline_features"]),
            "",
            f"### lightgbm_features ({len(feature_lists['lightgbm_features'])})",
            markdown_list(feature_lists["lightgbm_features"]),
            "",
            f"### mlp_features ({len(feature_lists['mlp_features'])})",
            markdown_list(feature_lists["mlp_features"]),
            "",
            f"### two_stage_features ({len(feature_lists['two_stage_features'])})",
            markdown_list(feature_lists["two_stage_features"]),
            "",
            "## Two-stage Model Target Advice",
            "",
            "- Stage 1 classification targets: `future_has_sales_1m`, `future_has_sales_2m`.",
            "- Stage 2 regression targets should be non-negative demand targets.",
            "- Current dataset does not contain `target_qty_1m` or `target_qty_2m`; create them during training as:",
            "  - `target_qty_1m = max(future_qty_1m, 0)`",
            "  - `target_qty_2m = max(future_qty_2m, 0)`",
            "- Reason: negative net sales are useful history, but should not become negative replenishment demand.",
            "",
            "## Leakage Review",
            "",
            "- Detected leakage-risk fields by prefix: "
            + (", ".join(f"`{field}`" for field in classified["leakage_risk_fields"]) or "none"),
            "- These fields are all target fields and are excluded from every feature list.",
            "- `split` is excluded from every feature list.",
            "- `item_id`, `isbn`, and `gds_no` are excluded from first-version model features.",
            "",
            "## Next-stage Modeling Suggestions",
            "",
            "- First run historical mean and weighted moving average baselines using `baseline_features` and the target fields only for evaluation.",
            "- For LightGBM / RandomForest, use `lightgbm_features`; treat category fields as categorical or encode them explicitly.",
            "- For MLP, start with `mlp_features`; add category embeddings only after numeric-feature baseline is stable.",
            "- Evaluate both raw target and clipped demand target for replenishment scenarios, especially because `future_qty_*` contains negative net-sales values from returns.",
            "",
        ]
    )

    write_markdown(REPORT_PATH, "\n".join(lines))
    return {
        "rows": total_rows,
        "fields": len(columns),
        "lightgbm_feature_count": len(feature_lists["lightgbm_features"]),
        "mlp_feature_count": len(feature_lists["mlp_features"]),
        "leakage_risk_fields": classified["leakage_risk_fields"],
        "report": str(REPORT_PATH),
    }


if __name__ == "__main__":
    summary = write_feature_schema_review()
    print(summary)
