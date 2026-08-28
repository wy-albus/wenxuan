from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import clip_target  # noqa: E402
from src.evaluation.two_stage_gate import (  # noqa: E402
    ThresholdGateAccumulator,
    classify_history_pattern,
    protection_candidates,
)
from src.features.preprocessing import load_feature_config  # noqa: E402
from src.models.lightgbm_model import load_model_bundle  # noqa: E402


DATASET_PATH = ROOT / "data/processed/model_dataset_monthly_active_store.parquet"
CLASSIFIER_PATH = ROOT / "models/final/two_stage_classifier_1m.txt"
REGRESSOR_PATH = ROOT / "models/final/two_stage_regressor_1m.txt"
TRADEOFF_PATH = ROOT / "reports/two_stage_gate_1m_tradeoff.csv"
PROTECTION_PATH = ROOT / "reports/two_stage_gate_1m_protection_candidates.csv"
SAMPLE_PATH = ROOT / "reports/two_stage_gate_1m_high_sales_low_probability_samples.csv"
THRESHOLD_CURVE_PATH = ROOT / "reports/two_stage_gate_1m_wape_threshold_curve.png"
HEAD_CURVE_PATH = ROOT / "reports/two_stage_gate_1m_wape_head_tradeoff.png"
REPORT_PATH = ROOT / "reports/two_stage_gate_1m_tradeoff_report.md"
RUNTIME_PATH = ROOT / "reports/two_stage_gate_1m_tradeoff_runtime.json"
LOG_DIR = ROOT / "logs/lightgbm/two_stage_gate_tradeoff_1m"
TEMP_DIR = ROOT / "data/temp/two_stage_gate_tradeoff_1m"

EXPECTED_VALID_ROWS = 18_563_573
PROTECTION_LEVELS = [0.98, 0.95, 0.90, 0.85, 0.80]
SAMPLE_LIMIT = 100
MIN_VALID_SCAN_BUDGET_GIB = 0.5
SYSTEM_RESERVE_GIB = 4.0


class ValidScanMemoryGuard:
    """Use the measured small inference footprint while preserving system RAM."""

    def __init__(self, pipeline, stage: str, logger: logging.Logger) -> None:
        _, available = pipeline.physical_memory()
        self.pipeline = pipeline
        self.stage = stage
        self.logger = logger
        self.limit = int(min(11.5 * pipeline.GIB, available - SYSTEM_RESERVE_GIB * pipeline.GIB))
        if self.limit < MIN_VALID_SCAN_BUDGET_GIB * pipeline.GIB:
            raise pipeline.MemoryLimitExceeded(
                f"{stage}: insufficient free physical memory; available={available / pipeline.GIB:.2f} GiB"
            )
        self.peak = pipeline.process_rss()
        logger.info(
            "%s inference memory limit %.2f GiB; available at start %.2f GiB",
            stage, self.limit / pipeline.GIB, available / pipeline.GIB,
        )

    def check(self, label: str) -> None:
        pipeline = self.pipeline
        rss = pipeline.process_rss()
        _, available = pipeline.physical_memory()
        self.peak = max(self.peak, rss)
        if rss >= self.limit or available < SYSTEM_RESERVE_GIB * pipeline.GIB:
            raise pipeline.MemoryLimitExceeded(
                f"{self.stage}: memory safety threshold reached at {label}; "
                f"rss={rss / pipeline.GIB:.2f} GiB, available={available / pipeline.GIB:.2f} GiB"
            )


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("wenxuan_two_stage_gate_tradeoff_1m")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def thresholds() -> np.ndarray:
    required = np.array(
        [
            0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90,
        ],
        dtype="float64",
    )
    fine_grid = np.round(np.arange(0.00, 0.951, 0.01), 2)
    return np.unique(np.concatenate([required, fine_grid]))


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


def validate_model_pair(classifier_meta: dict, regressor_meta: dict) -> None:
    for key in ("feature_names", "category_maps", "time_base", "horizon"):
        if classifier_meta.get(key) != regressor_meta.get(key):
            raise RuntimeError(f"Classifier/regressor metadata mismatch: {key}")
    forbidden = {
        "future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m",
        "split", "item_id", "isbn", "gds_no",
    }
    leakage = forbidden.intersection(classifier_meta["feature_names"])
    if leakage:
        raise RuntimeError(f"Future/identifier leakage in Two-stage features: {sorted(leakage)}")


def update_reservoir(
    reservoir: pd.DataFrame | None,
    candidates: pd.DataFrame,
    sample_group: str,
) -> pd.DataFrame | None:
    if candidates.empty:
        return reservoir
    candidates = candidates.copy()
    candidates["sample_group"] = sample_group
    keys = candidates[["month", "site_no", "item_id"]].fillna("unknown").astype("string")
    candidates["_priority"] = (
        pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64") ^ np.uint64(42)
    )
    combined = candidates if reservoir is None else pd.concat([reservoir, candidates], ignore_index=True)
    return combined.nsmallest(SAMPLE_LIMIT, "_priority").reset_index(drop=True)


def collect_diagnostic_samples(
    frame: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    conditional: np.ndarray,
    reservoirs: dict[str, pd.DataFrame | None],
) -> None:
    columns = [
        "month", "site_no", "item_id", "qty_lag_1m", "qty_lag_2m", "qty_lag_3m",
        "qty_mean_last_3m", "total_qty", "sales_days", "sales_count",
    ]
    masks = {
        "A_5_20_low_p": (target >= 5) & (target < 20) & (probability < 0.30),
        "B_20plus_low_p": (target >= 20) & (probability < 0.30),
    }
    for group, mask in masks.items():
        if not mask.any():
            continue
        candidates = frame.loc[mask, columns].copy()
        candidates.insert(3, "target_qty_1m", target[mask])
        candidates.insert(4, "p_sale", probability[mask])
        candidates.insert(5, "conditional_qty", conditional[mask])
        reservoirs[group] = update_reservoir(reservoirs[group], candidates, group)


def finalize_samples(reservoirs: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    frames = [frame for frame in reservoirs.values() if frame is not None]
    if not frames:
        return pd.DataFrame()
    samples = pd.concat(frames, ignore_index=True).drop(columns="_priority")
    samples["history_pattern"] = [
        classify_history_pattern(lag1, lag2, lag3)
        for lag1, lag2, lag3 in samples[["qty_lag_1m", "qty_lag_2m", "qty_lag_3m"]].itertuples(index=False, name=None)
    ]
    return samples.sort_values(["sample_group", "month", "site_no", "item_id"]).reset_index(drop=True)


def plot_curves(tradeoff: pd.DataFrame) -> None:
    gated = tradeoff[tradeoff["method"].isin(["gate_a", "gate_b"])].copy()
    labels = {"gate_a": "Gate-A", "gate_b": "Gate-B"}
    colors = {"gate_a": "#2878B5", "gate_b": "#D95319"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for method in ("gate_a", "gate_b"):
        subset = gated[gated["method"].eq(method)].sort_values("threshold")
        best = subset.loc[subset["wape"].idxmin()]
        for axis in axes:
            axis.plot(subset["threshold"], subset["wape"], label=labels[method], color=colors[method], linewidth=2)
            axis.scatter([best["threshold"]], [best["wape"]], color=colors[method], s=55, zorder=3)
        offset = (-245, 18) if method == "gate_a" else (18, 18)
        axes[1].annotate(
            f"{labels[method]} min: tau={best['threshold']:.3f}, {best['wape']:.2f}%",
            (best["threshold"], best["wape"]), xytext=offset, textcoords="offset points", fontsize=9,
        )
    axes[0].set_title("Two-stage 1M gate threshold trade-off: full range")
    axes[1].set_title("Zoomed range around the useful region")
    axes[1].set_ylim(85, 160)
    axes[1].set_xlabel("Threshold tau")
    for axis in axes:
        axis.set_ylabel("Complete Valid WAPE (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(THRESHOLD_CURVE_PATH, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    points = None
    for method in ("gate_a", "gate_b"):
        subset = gated[gated["method"].eq(method)].sort_values("20plus_gate_recall")
        for axis in axes:
            axis.plot(
                100 * subset["20plus_gate_recall"], subset["wape"],
                label=labels[method], color=colors[method], linewidth=2,
            )
            points = axis.scatter(
                100 * subset["20plus_gate_recall"], subset["wape"],
                c=100 * subset["5_20_gate_recall"], cmap="viridis", s=18, alpha=0.8,
            )
    assert points is not None
    colorbar = fig.colorbar(points, ax=axes, fraction=0.03, pad=0.03)
    colorbar.set_label("5-20 gate pass rate (%)")
    axes[0].set_yscale("log")
    axes[0].set_title("WAPE versus head-demand protection: full range (log scale)")
    axes[1].set_ylim(85, 220)
    axes[1].set_title("Zoomed range for operational candidates")
    axes[1].set_xlabel("20+ gate pass rate (%)")
    for axis in axes:
        axis.set_ylabel("Complete Valid WAPE (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(HEAD_CURVE_PATH, dpi=160)
    plt.close(fig)


def pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{100 * value:.2f}%"


def write_report(
    tradeoff: pd.DataFrame,
    protection: pd.DataFrame,
    samples: pd.DataFrame,
    runtime: dict,
) -> None:
    original = tradeoff[tradeoff["method"].eq("original")].iloc[0]
    best_a = tradeoff[tradeoff["method"].eq("gate_a")].sort_values(["wape", "threshold"]).iloc[0]
    best_b = tradeoff[tradeoff["method"].eq("gate_b")].sort_values(["wape", "threshold"]).iloc[0]
    gate_b_zero = tradeoff[
        tradeoff["method"].eq("gate_b") & np.isclose(tradeoff["threshold"], 0.0)
    ].iloc[0]

    protection_lines = [
        "| 20+最低通过率 | Method | tau | Valid WAPE | 5-20通过率 | 20+通过率 | 总量偏差 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    ordered_protection = protection.sort_values(
        ["min_20plus_pass_rate", "method"], ascending=[False, True]
    )
    for _, row in ordered_protection.iterrows():
        protection_lines.append(
            f"| {100 * row['min_20plus_pass_rate']:.0f}% | {row['method']} | {row['threshold']:.3f} | "
            f"{row['wape']:.4f}% | {100 * row['5_20_gate_recall']:.2f}% | "
            f"{100 * row['20plus_gate_recall']:.2f}% | {row['total_bias']:.2f}% |"
        )
    winners = []
    for level, group in protection.groupby("min_20plus_pass_rate", sort=False):
        winner = group.sort_values(["wape", "threshold"]).iloc[0]
        winners.append(f"{100 * level:.0f}%: {winner['method']} (tau={winner['threshold']:.3f})")

    pattern_lines = ["| 样本组 | 历史形态 | 数量 | 比例 |", "|---|---|---:|---:|"]
    if not samples.empty:
        summary = samples.groupby(["sample_group", "history_pattern"]).size().rename("count").reset_index()
        summary["share"] = summary["count"] / summary.groupby("sample_group")["count"].transform("sum")
        for row in summary.itertuples(index=False):
            pattern_lines.append(f"| {row.sample_group} | {row.history_pattern} | {row.count} | {100 * row.share:.2f}% |")

    difficult_share = 0.0
    persistent_share = 0.0
    if not samples.empty:
        difficult_share = samples["history_pattern"].isin(["sudden_burst_from_low", "volatile_history"]).mean()
        persistent_share = samples["history_pattern"].eq("persistent_high").mean()

    formula_issue = gate_b_zero["20plus_wape"] < original["20plus_wape"]
    classifier_issue = len(samples) > 0
    regressor_issue = best_b["20plus_wape"] > 50.0
    if persistent_share > difficult_share:
        history_conclusion = "低概率高销量样本中，持续高销量形态多于突发/波动形态，分类器漏判更值得优先排查。"
        retrain_conclusion = "已有较强依据优先重训或重新校准 classifier，并用组合 Valid WAPE 重新选择整个 Two-stage。"
    else:
        history_conclusion = "低概率高销量样本更多呈现低基数突发或剧烈波动，样本本身难预测，但分类器仍存在不可忽略的头部漏判。"
        retrain_conclusion = (
            "现有证据不足以把问题归因于 classifier 单一组件；但 q 对真实零样本的巨量高估、p*q 对正销量的幅度衰减，"
            "以及硬门控对头部的误杀共同构成了重新训练或联合校准整个 Two-stage 的充分依据。"
        )
    causes = []
    if formula_issue:
        causes.append("p_sale 与 conditional_qty 相乘造成的幅度衰减")
    if classifier_issue:
        causes.append("classifier 对部分真实中高销量样本给出低概率")
    if regressor_issue:
        causes.append("conditional regressor 对头部数量仍有较大误差")
    cause_text = "、".join(causes) if causes else "当前后处理结果不足以定位单一组件"

    lines = [
        "# Two-stage 1M Gate-A / Gate-B Valid Trade-off",
        "",
        "本报告仅使用 Active-Store Complete Valid。未使用 Test 选择阈值，也未重新训练任何模型。",
        "",
        "## 定义",
        "",
        "- Original: `p_sale * conditional_qty`",
        "- Gate-A: `p_sale < tau` 时为 0，否则为 `p_sale * conditional_qty`",
        "- Gate-B: `p_sale < tau` 时为 0，否则为 `conditional_qty`",
        "",
        "## 无约束结果",
        "",
        f"- Original Valid WAPE: {original['wape']:.4f}%",
        f"- Gate-A 最低 WAPE: {best_a['wape']:.4f}%，tau={best_a['threshold']:.3f}，5-20通过率={pct(best_a['5_20_gate_recall'])}，20+通过率={pct(best_a['20plus_gate_recall'])}",
        f"- Gate-B 最低 WAPE: {best_b['wape']:.4f}%，tau={best_b['threshold']:.3f}，5-20通过率={pct(best_b['5_20_gate_recall'])}，20+通过率={pct(best_b['20plus_gate_recall'])}",
        "",
        "完整阈值结果见 `reports/two_stage_gate_1m_tradeoff.csv`。",
        "",
        "## 保护水平候选",
        "",
        *protection_lines,
        "",
        "各保护水平下最低 WAPE 方案：" + "；".join(winners) + "。",
        "",
        "## Gate-B 分层影响",
        "",
        "| 方案 | qty=1 WAPE / MAE | 2-5 WAPE / MAE | 5-20 WAPE | 20+ WAPE | Overall WAPE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Original | {original['qty_1_wape']:.2f}% / {original['qty_1_mae']:.3f} | {original['qty_2_5_wape']:.2f}% / {original['qty_2_5_mae']:.3f} | {original['5_20_wape']:.2f}% | {original['20plus_wape']:.2f}% | {original['wape']:.2f}% |",
        f"| Gate-B tau=0（全部使用q） | {gate_b_zero['qty_1_wape']:.2f}% / {gate_b_zero['qty_1_mae']:.3f} | {gate_b_zero['qty_2_5_wape']:.2f}% / {gate_b_zero['qty_2_5_mae']:.3f} | {gate_b_zero['5_20_wape']:.2f}% | {gate_b_zero['20plus_wape']:.2f}% | {gate_b_zero['wape']:.2f}% |",
        f"| Gate-B无约束最佳 | {best_b['qty_1_wape']:.2f}% / {best_b['qty_1_mae']:.3f} | {best_b['qty_2_5_wape']:.2f}% / {best_b['qty_2_5_mae']:.3f} | {best_b['5_20_wape']:.2f}% | {best_b['20plus_wape']:.2f}% | {best_b['wape']:.2f}% |",
        "",
        f"Gate-B 在 tau=0 时改善所有正销量分层，但真0样本预测总量达到 {gate_b_zero['zero_pred_total']:,.0f}，使 Overall WAPE 升至 {gate_b_zero['wape']:.2f}%。当阈值提高到整体最优点时，qty=1、2-5、5-20、20+ 又全部恶化，说明它通过大量置零换取整体 WAPE，而不是稳定改善正销量预测。",
        "",
        "## 高销量低概率样本",
        "",
        *pattern_lines,
        "",
        history_conclusion,
        "",
        "## 诊断结论",
        "",
        f"当前现象更符合三者共同作用：{cause_text}。",
        "Gate 后处理能证明组合公式存在可优化空间，但高保护率下的改善幅度和无约束最优点的头部漏杀需要同时考虑。",
        retrain_conclusion,
        "",
        "## 运行信息",
        "",
        f"- Complete Valid rows: {int(original['count']):,}",
        f"- Evaluation seconds: {runtime['evaluation_seconds']:.1f}",
        f"- Peak RAM: {runtime['peak_ram_gib']:.3f} GiB",
        f"- Classifier iteration: {runtime['classifier_iteration']}",
        f"- Regressor iteration: {runtime['regressor_iteration']}",
        f"- Diagnostic sample rows: {len(samples)}",
        "",
        f"![WAPE-threshold curve](two_stage_gate_1m_wape_threshold_curve.png)",
        "",
        f"![WAPE-head protection curve](two_stage_gate_1m_wape_head_tradeoff.png)",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logger, log_path = setup_logging()
    started = time.perf_counter()
    pipeline = _load_script("wenxuan_gate_tradeoff_pipeline", ROOT / "scripts/10_train_lightgbm.py")
    experiment = _load_script("wenxuan_gate_tradeoff_experiment", ROOT / "scripts/18_train_active_store_experiment.py")
    config = load_feature_config()

    guard = ValidScanMemoryGuard(pipeline, "two-stage-gate-tradeoff-1m-valid", logger)
    classifier, classifier_meta = load_model_bundle(CLASSIFIER_PATH)
    regressor, regressor_meta = load_model_bundle(REGRESSOR_PATH)
    guard.check("models_loaded")
    validate_model_pair(classifier_meta, regressor_meta)

    sample_fields = [
        "qty_lag_1m", "qty_lag_2m", "qty_lag_3m", "qty_mean_last_3m",
        "total_qty", "sales_days", "sales_count",
    ]
    columns = experiment._physical_eval_columns(config, classifier_meta["feature_names"], "1m")
    columns.extend(column for column in sample_fields if column not in columns)
    accumulator = ThresholdGateAccumulator(thresholds())
    reservoirs: dict[str, pd.DataFrame | None] = {
        "A_5_20_low_p": None,
        "B_20plus_low_p": None,
    }

    rows = 0
    for number, frame in enumerate(iter_valid_chunks(columns), start=1):
        target = clip_target(frame["future_qty_1m"].to_numpy())
        x = experiment._prepare_eval(pipeline, frame, classifier_meta)
        probability = np.clip(
            classifier.predict(x, num_iteration=int(classifier_meta["best_iteration"])), 0.0, 1.0
        )
        conditional = np.clip(
            regressor.predict(x, num_iteration=int(regressor_meta["best_iteration"])), 0.0, None
        )
        if not (np.isfinite(probability).all() and np.isfinite(conditional).all()):
            raise RuntimeError("Non-finite model output found in Complete Valid")
        accumulator.update(target, probability, conditional)
        collect_diagnostic_samples(frame, target, probability, conditional, reservoirs)
        rows += len(frame)
        if number % 10 == 0:
            guard.check("valid_stream")
            logger.info("Complete Valid inference rows=%s", f"{rows:,}")
        del frame, x, target, probability, conditional
        gc.collect()

    if rows != EXPECTED_VALID_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_VALID_ROWS:,} Complete Valid rows, got {rows:,}")

    tradeoff = pd.DataFrame(accumulator.rows())
    protection = pd.DataFrame(protection_candidates(tradeoff.to_dict("records"), PROTECTION_LEVELS))
    samples = finalize_samples(reservoirs)
    TRADEOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(TRADEOFF_PATH, index=False, encoding="utf-8-sig")
    protection.to_csv(PROTECTION_PATH, index=False, encoding="utf-8-sig")
    samples.to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    plot_curves(tradeoff)

    runtime = {
        "valid_rows": rows,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_ram_gib": guard.peak / (1024 ** 3),
        "classifier_iteration": int(classifier_meta["best_iteration"]),
        "regressor_iteration": int(regressor_meta["best_iteration"]),
        "threshold_count": int(len(thresholds())),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    RUNTIME_PATH.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(tradeoff, protection, samples, runtime)
    logger.info("Trade-off analysis complete; rows=%s elapsed=%.1fs", f"{rows:,}", runtime["evaluation_seconds"])


if __name__ == "__main__":
    main()
