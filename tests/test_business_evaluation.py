import numpy as np
from pathlib import Path

from src.evaluation.business_evaluation import BusinessEvaluationAccumulator
from src.evaluation.mc_metrics import MCLevelConfig
from src.evaluation.metrics import demand_bucket


def mc_config():
    return MCLevelConfig(
        codes=(0, 1, 2, 3, 4),
        names=("no_sales", "low", "normal", "medium_high", "high"),
        lower_bounds=(0, 1, 2, 5, 20),
    )


def test_business_evaluation_reuses_two_month_average_for_mc_and_trend():
    accumulator = BusinessEvaluationAccumulator(mc_config(), "2m")
    accumulator.update(
        target=np.array([0.0, 2.0, 10.0, 40.0]),
        prediction=np.array([0.2, 2.0, 10.0, 40.0]),
        current=np.array([0.0, 1.0, 6.0, 18.0]),
    )

    result = accumulator.compute()

    assert result["quantity"]["overall"]["count"] == 4
    assert result["mc"]["accuracy"] == 1.0
    assert result["trend"]["accuracy"] == 0.75


def test_business_evaluation_reports_zero_risk_and_head_segments():
    accumulator = BusinessEvaluationAccumulator(mc_config(), "1m")
    accumulator.update(
        target=np.array([0.0, 0.0, 7.0, 25.0]),
        prediction=np.array([0.4, 1.2, 6.0, 20.0]),
        current=np.zeros(4),
    )

    result = accumulator.compute()
    zero = result["quantity"]["0"]

    assert zero["count"] == 2
    assert np.isclose(zero["prediction_sum"], 1.6)
    assert zero["prediction_gt_0_5_rate"] == 0.5
    assert zero["prediction_gt_1_rate"] == 0.5
    assert result["quantity"]["5-20"]["count"] == 1
    assert result["quantity"]["20+"]["count"] == 1


def test_demand_buckets_use_exact_business_boundaries():
    values = np.array([0, 1, 2, 4, 5, 19, 20], dtype="float64")
    assert demand_bucket(values).tolist() == ["0", "1", "2-5", "2-5", "5-20", "5-20", "20+"]


def test_business_report_script_reuses_predictions_without_training_models():
    path = Path(__file__).resolve().parents[1] / "scripts" / "24_evaluate_active_store_business.py"
    source = path.read_text(encoding="utf-8")

    assert "lightgbm_v2_active_store_wape_1m_test_predictions.parquet" in source
    assert "lightgbm_v2_active_store_wape_2m_test_predictions.parquet" in source
    assert "BusinessEvaluationAccumulator" in source
    assert "lgb.train" not in source
    assert "load_model_bundle" not in source
