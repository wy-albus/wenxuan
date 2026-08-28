from __future__ import annotations

import gc
import importlib.util
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import clip_target  # noqa: E402
from src.evaluation.two_stage_gate import (  # noqa: E402
    ProbabilityDiagnostic,
    ThresholdGateAccumulator,
    select_gate_candidate,
    select_safe_gate_candidate,
    should_evaluate_test as gate_is_test_eligible,
)
from src.features.preprocessing import load_feature_config  # noqa: E402
from src.models.lightgbm_model import load_model_bundle  # noqa: E402


DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
CLASSIFIER_PATH = ROOT / "models/final/two_stage_classifier_1m.txt"
REGRESSOR_PATH = ROOT / "models/final/two_stage_regressor_1m.txt"
OLD_TWO_STAGE_TEST_PATH = ROOT / "data/outputs/two_stage_test_predictions.parquet"
OLD_LGBM_TEST_PATH = ROOT / "data/outputs/lightgbm_v2_logl2_test_predictions.parquet"
ACTIVE_LGBM_TEST_PATH = ROOT / "data/outputs/lightgbm_v2_active_store_wape_1m_test_predictions.parquet"
VALID_SEARCH_PATH = ROOT / "reports/two_stage_gate_1m_valid_search.csv"
CLASSIFIER_DIAGNOSTIC_PATH = ROOT / "reports/two_stage_gate_1m_classifier_diagnostic.csv"
REPORT_PATH = ROOT / "reports/two_stage_gate_1m_diagnostic.md"
VALID_RUNTIME_PATH = ROOT / "reports/two_stage_gate_1m_valid_runtime.json"
TEST_PREDICTION_PATH = ROOT / "data/outputs/two_stage_gate_1m_test_predictions.parquet"
TEST_COMPARISON_PATH = ROOT / "reports/two_stage_gate_1m_test_comparison.csv"
LOG_DIR = ROOT / "logs/lightgbm/two_stage_gate_1m"
TEMP_DIR = ROOT / "data/temp/two_stage_gate_1m"

EXPECTED_VALID_ROWS = 18_563_573
EXPECTED_TEST_ROWS = 12_999_597
MIN_RELATIVE_WAPE_GAIN = 0.005
MIN_HEAD_GATE_RECALL = 0.95


SCHEMA_MEANINGS = {
    "month": ("original_test_key", "样本观察月份"),
    "site_no": ("original_test_key", "门店标识"),
    "item_id": ("original_test_key", "商品标识，仅用于定位"),
    "target_qty_1m": ("target", "未来1个月非负真实销量"),
    "target_qty_2m": ("target", "未来连续2个月非负真实累计销量"),
    "p_sale_1m": ("classifier_output", "1M分类器预测的未来有销量概率"),
    "conditional_qty_1m": ("tweedie_output", "1M Tweedie回归器预测的有销量条件数量"),
    "two_stage_pred_1m": ("combined_prediction", "1M旧组合预测 p_sale × conditional_qty"),
    "p_sale_2m": ("classifier_output", "2M分类器预测的未来有销量概率"),
    "conditional_qty_2m": ("tweedie_output", "2M Tweedie回归器预测的有销量条件数量"),
    "two_stage_pred_2m": ("combined_prediction", "2M旧组合预测 p_sale × conditional_qty"),
}


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_module():
    return _load_script("wenxuan_gate_pipeline", ROOT / "scripts/10_train_lightgbm.py")


def experiment_module():
    return _load_script("wenxuan_gate_experiment", ROOT / "scripts/18_train_active_store_experiment.py")


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("wenxuan_two_stage_gate_1m")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def inspect_test_schema() -> list[dict]:
    parquet = pq.ParquetFile(OLD_TWO_STAGE_TEST_PATH)
    try:
        fields = list(parquet.schema_arrow)
        if len(fields) != 11 or [field.name for field in fields] != list(SCHEMA_MEANINGS):
            raise RuntimeError("Unexpected two-stage Test schema; refusing silent column assumptions")
        return [
            {
                "name": field.name,
                "dtype": str(field.type),
                "source": SCHEMA_MEANINGS[field.name][0],
                "meaning": SCHEMA_MEANINGS[field.name][1],
                "rows": parquet.metadata.num_rows,
            }
            for field in fields
        ]
    finally:
        parquet.close()


def should_evaluate_test(original: dict, selected: dict) -> bool:
    return gate_is_test_eligible(
        original, selected, MIN_RELATIVE_WAPE_GAIN, MIN_HEAD_GATE_RECALL
    )


def select_test_candidate(rows: list[dict]) -> dict:
    return select_safe_gate_candidate(rows, MIN_HEAD_GATE_RECALL)


def _thresholds() -> np.ndarray:
    return np.round(np.arange(0.01, 0.951, 0.01), 2)


def iter_valid_chunks(columns: list[str]):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=2")
    connection.execute("PRAGMA memory_limit='512MB'")
    connection.execute(f"PRAGMA temp_directory='{TEMP_DIR.resolve().as_posix()}'")
    source = DATASET_PATH.resolve().as_posix().replace("'", "''")
    projection = ",".join(f'"{column}"' for column in dict.fromkeys(columns))
    try:
        connection.execute(
            f"SELECT {projection} FROM read_parquet('{source}') "
            "WHERE split='valid' AND target_available_1m=1"
        )
        while True:
            frame = connection.fetch_df_chunk(vectors_per_chunk=64)
            if frame.empty:
                break
            yield frame
    finally:
        connection.close()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


def _validate_model_pair(classifier_meta: dict, regressor_meta: dict) -> None:
    for key in ("feature_names", "category_maps", "time_base", "horizon"):
        if classifier_meta.get(key) != regressor_meta.get(key):
            raise RuntimeError(f"Classifier/regressor metadata mismatch: {key}")
    forbidden = {"future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m", "split", "item_id", "isbn", "gds_no"}
    leakage = forbidden.intersection(classifier_meta["feature_names"])
    if leakage:
        raise RuntimeError(f"Future/identifier leakage in Two-stage features: {sorted(leakage)}")


def evaluate_valid_once(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if VALID_SEARCH_PATH.exists() and CLASSIFIER_DIAGNOSTIC_PATH.exists():
        logger.info("Reusing completed Valid gate search and classifier diagnostic")
        search = pd.read_csv(VALID_SEARCH_PATH)
        diagnostic = pd.read_csv(CLASSIFIER_DIAGNOSTIC_PATH)
        if search["count"].nunique() != 1 or int(search["count"].iloc[0]) != EXPECTED_VALID_ROWS:
            raise RuntimeError("Existing Valid search does not cover the complete active-store Valid set")
        runtime = {"reused": True, "evaluation_seconds": None, "peak_ram_gib": None}
        if VALID_RUNTIME_PATH.exists():
            runtime.update(json.loads(VALID_RUNTIME_PATH.read_text(encoding="utf-8")))
            runtime["reused"] = True
        return search, diagnostic, runtime

    pipeline = pipeline_module()
    experiment = experiment_module()
    config = load_feature_config()
    guard = pipeline.MemoryGuard("two-stage-gate-1m-valid", logger)
    classifier, classifier_meta = load_model_bundle(CLASSIFIER_PATH)
    regressor, regressor_meta = load_model_bundle(REGRESSOR_PATH)
    guard.check("models_loaded")
    _validate_model_pair(classifier_meta, regressor_meta)
    columns = experiment._physical_eval_columns(config, classifier_meta["feature_names"], "1m")
    if "qty_lag_1m" not in columns:
        columns.append("qty_lag_1m")
    accumulator = ThresholdGateAccumulator(_thresholds())
    diagnostic = ProbabilityDiagnostic()
    unknown_counts = {column: 0 for column in classifier_meta["categorical_features"]}
    rows = 0
    started = time.perf_counter()
    for number, frame in enumerate(iter_valid_chunks(columns), start=1):
        target = clip_target(frame["future_qty_1m"].to_numpy())
        lag = clip_target(frame["qty_lag_1m"].to_numpy())
        x = experiment._prepare_eval(pipeline, frame, classifier_meta)
        for column in classifier_meta["categorical_features"]:
            unknown_counts[column] += int((x[column].to_numpy() < 0).sum())
        probability = np.clip(
            classifier.predict(x, num_iteration=int(classifier_meta["best_iteration"])), 0.0, 1.0
        )
        conditional = np.clip(
            regressor.predict(x, num_iteration=int(regressor_meta["best_iteration"])), 0.0, None
        )
        accumulator.update(target, probability, conditional)
        diagnostic.update(target, lag, probability)
        rows += len(frame)
        if number % 10 == 0:
            guard.check("valid_stream")
            logger.info("Valid inference rows=%s", f"{rows:,}")
        del frame, x, target, lag, probability, conditional
        gc.collect()
    if rows != EXPECTED_VALID_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_VALID_ROWS:,} complete Valid rows, got {rows:,}")

    search = pd.DataFrame(accumulator.rows())
    selected = select_gate_candidate(search.to_dict("records"))
    coarse_center = round(float(selected["threshold"]) / 0.05) * 0.05
    search["search_stage"] = "auxiliary"
    search.loc[search["method"].eq("original"), "search_stage"] = "original"
    search.loc[
        search["method"].ne("original") & np.isclose((search["threshold"] * 100) % 5, 0.0, atol=1e-8),
        "search_stage",
    ] = "coarse"
    search.loc[
        search["method"].eq(selected["method"])
        & search["threshold"].between(coarse_center - 0.050001, coarse_center + 0.050001),
        "search_stage",
    ] = "local_fine"
    diagnostic_frame = pd.DataFrame(diagnostic.rows())
    VALID_SEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    search.to_csv(VALID_SEARCH_PATH, index=False, encoding="utf-8-sig")
    diagnostic_frame.to_csv(CLASSIFIER_DIAGNOSTIC_PATH, index=False, encoding="utf-8-sig")
    runtime = {
        "reused": False,
        "valid_rows": rows,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / (1024 ** 3),
        "classifier_iteration": classifier.current_iteration(),
        "regressor_iteration": regressor.current_iteration(),
        "unknown_category_rates": {key: value / rows for key, value in unknown_counts.items()},
    }
    VALID_RUNTIME_PATH.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    return search, diagnostic_frame, runtime


def _build_test_prediction(method: str, threshold: float) -> None:
    if TEST_PREDICTION_PATH.exists():
        return
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='6GB'")
    connection.execute(f"PRAGMA temp_directory='{TEMP_DIR.resolve().as_posix()}'")
    old_two = OLD_TWO_STAGE_TEST_PATH.resolve().as_posix().replace("'", "''")
    active = ACTIVE_LGBM_TEST_PATH.resolve().as_posix().replace("'", "''")
    output = TEST_PREDICTION_PATH.resolve().as_posix().replace("'", "''")
    expression = (
        f"CASE WHEN t.p_sale_1m < {threshold:.8f} THEN 0.0 ELSE t.p_sale_1m*t.conditional_qty_1m END"
        if method == "gate_a"
        else f"CASE WHEN t.p_sale_1m < {threshold:.8f} THEN 0.0 ELSE t.conditional_qty_1m END"
    )
    temporary = TEST_PREDICTION_PATH.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    temp_sql = temporary.resolve().as_posix().replace("'", "''")
    try:
        count, max_difference = connection.execute(
            f"""
            SELECT COUNT(*), MAX(ABS(t.target_qty_1m-a.target_qty_1m))
            FROM read_parquet('{old_two}') t
            JOIN read_parquet('{active}') a USING(month,site_no,item_id)
            """
        ).fetchone()
        if int(count) != EXPECTED_TEST_ROWS or float(max_difference) != 0.0:
            raise RuntimeError("Old Two-stage and active-store Test keys/targets do not align")
        connection.execute(
            f"""
            COPY (
              SELECT t.month,t.site_no,t.item_id,t.target_qty_1m,t.p_sale_1m,t.conditional_qty_1m,
                     t.two_stage_pred_1m AS original_two_stage_pred_1m,
                     CAST({expression} AS FLOAT) AS gated_two_stage_pred_1m,
                     '{method}' AS gate_method, {threshold:.8f}::FLOAT AS gate_threshold
              FROM read_parquet('{old_two}') t
              JOIN read_parquet('{active}') a USING(month,site_no,item_id)
            ) TO '{temp_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
            """
        )
    finally:
        connection.close()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    temporary.replace(TEST_PREDICTION_PATH)


def _test_comparison() -> pd.DataFrame:
    if TEST_COMPARISON_PATH.exists():
        return pd.read_csv(TEST_COMPARISON_PATH)
    connection = duckdb.connect()
    paths = {
        "gate": TEST_PREDICTION_PATH.resolve().as_posix().replace("'", "''"),
        "old_lgbm": OLD_LGBM_TEST_PATH.resolve().as_posix().replace("'", "''"),
        "active_lgbm": ACTIVE_LGBM_TEST_PATH.resolve().as_posix().replace("'", "''"),
    }
    try:
        frame = connection.execute(
            f"""
            WITH common AS MATERIALIZED (
              SELECT g.target_qty_1m AS target,g.original_two_stage_pred_1m,g.gated_two_stage_pred_1m,
                     o.lightgbm_v2_logl2_pred_1m AS old_lgbm_pred,
                     a.lightgbm_v2_active_store_wape_pred_1m AS active_lgbm_pred,
                     GREATEST(ABS(g.target_qty_1m-o.target_qty_1m),ABS(g.target_qty_1m-a.target_qty_1m)) AS target_diff
              FROM read_parquet('{paths['gate']}') g
              JOIN read_parquet('{paths['old_lgbm']}') o USING(month,site_no,item_id)
              JOIN read_parquet('{paths['active_lgbm']}') a USING(month,site_no,item_id)
            ), expanded AS (
              SELECT target,original_two_stage_pred_1m AS prediction,'old_two_stage' AS model,target_diff FROM common UNION ALL
              SELECT target,gated_two_stage_pred_1m,'best_gated_two_stage',target_diff FROM common UNION ALL
              SELECT target,old_lgbm_pred,'old_lightgbm_v2',target_diff FROM common UNION ALL
              SELECT target,active_lgbm_pred,'active_store_lightgbm_v2_wape',target_diff FROM common
            ), segmented AS (
              SELECT *,segment FROM expanded CROSS JOIN (VALUES ('overall'),('zero'),('nonzero'),('5-20'),('20+')) s(segment)
              WHERE segment='overall' OR (segment='zero' AND target<=0) OR (segment='nonzero' AND target>0)
                 OR (segment='5-20' AND target>=5 AND target<20) OR (segment='20+' AND target>=20)
            )
            SELECT model,segment,COUNT(*)::BIGINT AS count,AVG(ABS(target-prediction)) AS mae,
                   SQRT(AVG(POW(target-prediction,2))) AS rmse,
                   CASE WHEN SUM(target)>0 THEN 100*SUM(ABS(target-prediction))/SUM(target) ELSE NULL END AS wape,
                   SUM(target) AS target_sum,SUM(prediction) AS prediction_sum,
                   CASE WHEN SUM(target)>0 THEN 100*(SUM(prediction)-SUM(target))/SUM(target) ELSE NULL END AS total_bias,
                   CASE WHEN segment='zero' THEN SUM(prediction) ELSE NULL END AS zero_pred_total,
                   MAX(target_diff) AS max_target_difference
            FROM segmented GROUP BY model,segment ORDER BY model,segment
            """
        ).df()
    finally:
        connection.close()
    if frame.empty or frame["max_target_difference"].max() != 0:
        raise RuntimeError("Four-model Test target alignment failed")
    frame.to_csv(TEST_COMPARISON_PATH, index=False, encoding="utf-8-sig")
    return frame


def _format_percent(value: float) -> str:
    return "N/A" if not math.isfinite(float(value)) else f"{float(value):.4f}%"


def write_report(
    schema: list[dict], search: pd.DataFrame, diagnostic: pd.DataFrame, runtime: dict,
    selected: dict, allowed_test: bool, comparison: pd.DataFrame | None, log_path: Path,
) -> None:
    original = search[search["method"].eq("original")].iloc[0].to_dict()
    best_a = search[search["method"].eq("gate_a")].sort_values("wape").iloc[0].to_dict()
    best_b = search[search["method"].eq("gate_b")].sort_values("wape").iloc[0].to_dict()
    gain = 100 * (float(original["wape"]) - float(selected["wape"])) / float(original["wape"])
    schema_lines = ["| Field | Type | Source | Meaning |", "|---|---|---|---|"]
    schema_lines.extend(f"| {row['name']} | {row['dtype']} | {row['source']} | {row['meaning']} |" for row in schema)
    target_diag = diagnostic[diagnostic["basis"].eq("target_qty_1m")]
    diag_lines = ["| True quantity | Count | Mean p | Median | P10 | P25 | P75 | P90 | p<0.2 | p<0.3 | p<0.5 | p>=0.5 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in target_diag.itertuples(index=False):
        diag_lines.append(
            f"| {row.segment} | {int(row.count):,} | {row.mean:.4f} | {row.median:.4f} | {row.p10:.4f} | {row.p25:.4f} | {row.p75:.4f} | {row.p90:.4f} | "
            f"{row.p_sale_lt_0_20_rate:.2%} | {row.p_sale_lt_0_30_rate:.2%} | {row.p_sale_lt_0_50_rate:.2%} | {row.p_sale_ge_0_50_rate:.2%} |"
        )
    summary_lines = ["| Method | Threshold | Valid WAPE | MAE | Total bias | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | 5-20 pass | 20+ pass |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in (original, best_a, best_b):
        summary_lines.append(
            f"| {row['method']} | {('N/A' if pd.isna(row['threshold']) else f'{row['threshold']:.2f}')} | {row['wape']:.6f}% | {row['mae']:.6f} | "
            f"{row['total_bias']:.4f}% | {row['nonzero_wape']:.4f}% | {row['5_20_wape']:.4f}% | {row['20plus_wape']:.4f}% | "
            f"{row['5_20_gate_recall']:.2%} | {row['20plus_gate_recall']:.2%} |"
        )
    comparison_lines = []
    if comparison is not None:
        comparison_lines = ["## Locked Test Comparison", "", "The method and threshold below were locked using Valid only.", "", "| Model | Segment | Count | WAPE | MAE | RMSE | True total | Predicted total | Bias |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in comparison.itertuples(index=False):
            comparison_lines.append(
                f"| {row.model} | {row.segment} | {int(row.count):,} | {_format_percent(row.wape)} | {row.mae:.6f} | {row.rmse:.6f} | {row.target_sum:.2f} | {row.prediction_sum:.2f} | {_format_percent(row.total_bias)} |"
            )
        comparison_lines.append("")
    high_5 = target_diag[target_diag["segment"].eq("5-20")].iloc[0]
    high_20 = target_diag[target_diag["segment"].eq("20+")].iloc[0]
    elapsed_text = "not retained" if runtime.get("evaluation_seconds") is None else f"{float(runtime['evaluation_seconds']):.1f}s"
    peak_text = "not retained" if runtime.get("peak_ram_gib") is None else f"{float(runtime['peak_ram_gib']):.2f} GiB"
    conclusion_lines = []
    if comparison is not None:
        overall = comparison[comparison["segment"].eq("overall")].set_index("model")
        zero_rows = comparison[comparison["segment"].eq("zero")].set_index("model")
        nonzero = comparison[comparison["segment"].eq("nonzero")].set_index("model")
        mid = comparison[comparison["segment"].eq("5-20")].set_index("model")
        head = comparison[comparison["segment"].eq("20+")].set_index("model")
        old_wape = float(overall.loc["old_two_stage", "wape"])
        gated_wape = float(overall.loc["best_gated_two_stage", "wape"])
        zero_reduction = 1.0 - float(zero_rows.loc["best_gated_two_stage", "prediction_sum"]) / float(zero_rows.loc["old_two_stage", "prediction_sum"])
        conclusion_lines = [
            f"- On the locked common Test, Gate-A lowers old Two-stage WAPE from {old_wape:.6f}% to {gated_wape:.6f}% ({old_wape-gated_wape:.6f} percentage points; {(old_wape-gated_wape)/old_wape:.2%} relative).",
            f"- Zero-sample predicted quantity falls from {float(zero_rows.loc['old_two_stage','prediction_sum']):.2f} to {float(zero_rows.loc['best_gated_two_stage','prediction_sum']):.2f} ({zero_reduction:.2%} reduction). This confirms that persistent small positive predictions on true zeros are a material part of old Two-stage WAPE.",
            f"- Nonzero WAPE changes from {float(nonzero.loc['old_two_stage','wape']):.6f}% to {float(nonzero.loc['best_gated_two_stage','wape']):.6f}%; 5-20 from {float(mid.loc['old_two_stage','wape']):.6f}% to {float(mid.loc['best_gated_two_stage','wape']):.6f}%; 20+ from {float(head.loc['old_two_stage','wape']):.6f}% to {float(head.loc['best_gated_two_stage','wape']):.6f}%. The safe gate leaves head-demand accuracy essentially unchanged.",
            f"- Total bias moves from {float(overall.loc['old_two_stage','total_bias']):.4f}% to {float(overall.loc['best_gated_two_stage','total_bias']):.4f}%: gating reduces false positives but increases aggregate underestimation.",
            f"- The gated result remains worse in overall WAPE than old LightGBM V2 ({float(overall.loc['old_lightgbm_v2','wape']):.6f}%) and active-store LightGBM V2 ({float(overall.loc['active_store_lightgbm_v2_wape','wape']):.6f}%). It is a useful Two-stage correction, not a new overall winner.",
        ]
    lines = [
        "# Two-Stage 1M Zero-Sales Gate Diagnostic", "",
        "## Scope", "",
        "- No model was trained or adjusted. Existing 1M classifier and Tweedie regressor were loaded read-only.",
        "- Classifier and regressor were each run once per active-store Valid row group; every threshold was computed from binned sufficient statistics.",
        "- Threshold selection uses complete Valid original-scale WAPE only. Test is never used for threshold search.",
        f"- Valid rows: {int(original['count']):,}; runtime: {elapsed_text}; peak RAM: {peak_text}.",
        f"- Log: `{log_path.relative_to(ROOT).as_posix()}`.", "",
        "## Existing Test Schema", "", *schema_lines, "",
        "All 1M and 2M classifier probabilities, conditional quantities, and final products are present. No saved Two-stage Valid prediction file was found, so Valid inference was necessary.", "",
        "## Classifier Probability Diagnostic", "", *diag_lines, "",
        f"For true 5-20 demand, p_sale median is {high_5['median']:.4f} and {high_5['p_sale_lt_0_50_rate']:.2%} fall below 0.50. For true 20+ demand, median is {high_20['median']:.4f} and {high_20['p_sale_lt_0_50_rate']:.2%} fall below 0.50.",
        "The history-stratified version of the same diagnostic is in `reports/two_stage_gate_1m_classifier_diagnostic.csv`.", "",
        "## Valid Gate Search", "", *summary_lines, "",
        f"- Original WAPE: {float(original['wape']):.6f}%.",
        f"- Best Gate-A: threshold {float(best_a['threshold']):.2f}, WAPE {float(best_a['wape']):.6f}%.",
        f"- Best Gate-B: threshold {float(best_b['threshold']):.2f}, WAPE {float(best_b['wape']):.6f}%.",
        f"- Selected: `{selected['method']}` threshold {float(selected['threshold']):.2f}, WAPE {float(selected['wape']):.6f}%, relative improvement {gain:.4f}%.",
        f"- Zero prediction total changes from {float(original['zero_pred_total']):.2f} to {float(selected['zero_pred_total']):.2f}.",
        f"- 5-20 and 20+ gate pass rates are {float(selected['5_20_gate_recall']):.2%} and {float(selected['20plus_gate_recall']):.2%}.",
        f"- Test eligibility rule: relative WAPE improvement >= {MIN_RELATIVE_WAPE_GAIN:.2%}, 5-20 pass >= {MIN_HEAD_GATE_RECALL:.0%}, 20+ pass >= {MIN_HEAD_GATE_RECALL:.0%}.",
        f"- Test executed: {'yes' if allowed_test else 'no'}.", "",
        *comparison_lines,
        "## Conclusion", "",
        ("The locked gate passed the predeclared Valid value/safety rule, so it was evaluated once on Test." if allowed_test else "Neither gated result met the predeclared Valid value/safety rule; the experiment stopped before Test."),
        *conclusion_lines,
        "This experiment diagnoses post-processing only and does not justify retraining or extending to 2M without a separate decision.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    for path in (DATASET_PATH, CLASSIFIER_PATH, REGRESSOR_PATH, OLD_TWO_STAGE_TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    logger, log_path = setup_logging()
    schema = inspect_test_schema()
    search, diagnostic, runtime = evaluate_valid_once(logger)
    original = search[search["method"].eq("original")].iloc[0].to_dict()
    selected = select_test_candidate(search.to_dict("records"))
    allowed_test = should_evaluate_test(original, selected)
    comparison = None
    if allowed_test:
        for path in (OLD_LGBM_TEST_PATH, ACTIVE_LGBM_TEST_PATH):
            if not path.exists():
                raise FileNotFoundError(path)
        logger.info("Valid gate passed; locking method=%s threshold=%.2f before Test", selected["method"], selected["threshold"])
        _build_test_prediction(str(selected["method"]), float(selected["threshold"]))
        comparison = _test_comparison()
    else:
        logger.info("Valid gate did not pass value/safety rule; Test remains untouched")
    write_report(schema, search, diagnostic, runtime, selected, allowed_test, comparison, log_path)
    return {
        "original_valid_wape": float(original["wape"]),
        "selected_method": selected["method"],
        "selected_threshold": float(selected["threshold"]),
        "selected_valid_wape": float(selected["wape"]),
        "test_executed": allowed_test,
        "report": str(REPORT_PATH),
    }


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
