import unittest
from pathlib import Path

import numpy as np


class LightGBMV2ObjectiveTests(unittest.TestCase):
    def test_formal_summary_reads_staged_prediction_before_promotion(self):
        source = Path("scripts/12_train_lightgbm_v2_logl2.py").read_text(encoding="utf-8")

        self.assertNotIn('FINAL_OUTPUTS["predictions"].stat().st_size', source)
        self.assertIn('temporary_predictions.stat().st_size', source)

    def test_calibration_factor_uses_aggregate_valid_totals(self):
        from src.models.lightgbm_objectives import apply_calibration, calibration_factor

        factor = calibration_factor(target_sum=100.0, prediction_sum=80.0)

        self.assertEqual(factor, 1.25)
        np.testing.assert_allclose(apply_calibration(np.array([-1.0, 0.0, 2.0]), factor), [0.0, 0.0, 2.5])

    def test_calibration_factor_is_neutral_when_prediction_total_is_zero(self):
        from src.models.lightgbm_objectives import calibration_factor

        self.assertEqual(calibration_factor(target_sum=100.0, prediction_sum=0.0), 1.0)

    def test_calibration_rejects_non_finite_factor(self):
        from src.models.lightgbm_objectives import apply_calibration

        with self.assertRaises(ValueError):
            apply_calibration(np.array([1.0]), np.inf)

    def test_pilot_imports_pyarrow_before_lightgbm_native_library(self):
        source = Path("scripts/11_lightgbm_objective_pilot.py").read_text(encoding="utf-8")

        self.assertLess(source.index("import pyarrow.parquet as pq"), source.index("import lightgbm as lgb"))

    def test_tweedie_uses_raw_non_negative_target_and_direct_prediction(self):
        from src.models.lightgbm_objectives import candidate_specs, inverse_prediction, transform_training_target

        spec = candidate_specs({"learning_rate": 0.05})["tweedie_1_4"]
        target = np.array([-2.0, 0.0, 3.0, np.nan])

        np.testing.assert_allclose(transform_training_target(target, spec), [0.0, 0.0, 3.0, 0.0])
        np.testing.assert_allclose(inverse_prediction(np.array([-1.0, 0.0, 2.5]), spec), [0.0, 0.0, 2.5])
        self.assertEqual(spec.params["objective"], "tweedie")
        self.assertEqual(spec.params["metric"], ["tweedie", "l1"])
        self.assertEqual(spec.params["tweedie_variance_power"], 1.4)

    def test_log_l2_uses_log1p_target_and_expm1_prediction(self):
        from src.models.lightgbm_objectives import candidate_specs, inverse_prediction, transform_training_target

        spec = candidate_specs({"learning_rate": 0.05})["log_l2"]
        target = np.array([0.0, 1.0, 9.0])

        np.testing.assert_allclose(transform_training_target(target, spec), np.log1p(target))
        np.testing.assert_allclose(inverse_prediction(np.log1p(target), spec), target)
        self.assertEqual(spec.params["objective"], "regression_l2")
        self.assertEqual(spec.params["metric"], "l2")

    def test_inverse_prediction_rejects_non_finite_values(self):
        from src.models.lightgbm_objectives import candidate_specs, inverse_prediction

        with self.assertRaises(ValueError):
            inverse_prediction(np.array([0.0, np.nan]), candidate_specs({})["log_l2"])


class LightGBMV2SelectionTests(unittest.TestCase):
    @staticmethod
    def metrics(bias, mae, nonzero, ge5, twenty, zero_gt=0.01, zero_mean=0.02):
        return {
            "overall": {"total_bias_rate": bias, "mae": mae},
            "nonzero": {"wape": nonzero},
            "ge_5": {"wape": ge5},
            "20+": {"wape": twenty},
            "0": {"prediction_gt_0_5_rate": zero_gt, "mean_prediction": zero_mean},
        }

    def test_selection_prioritizes_bias_then_nonzero_and_head_metrics(self):
        from src.models.lightgbm_objectives import select_business_candidate

        metrics = {
            "v1_log_l1": self.metrics(-79, 0.12, 93, 88, 94),
            "tweedie_1_2": self.metrics(-25, 0.14, 70, 65, 82),
            "tweedie_1_4": self.metrics(-40, 0.13, 68, 63, 80),
            "log_l2": self.metrics(-55, 0.12, 80, 75, 88),
        }

        result = select_business_candidate(metrics, ["tweedie_1_2", "tweedie_1_4", "log_l2"])

        self.assertEqual(result["selected"], "tweedie_1_2")
        self.assertTrue(result["assessments"]["tweedie_1_2"]["eligible"])

    def test_selection_rejects_unacceptable_zero_false_positives(self):
        from src.models.lightgbm_objectives import select_business_candidate

        metrics = {
            "v1_log_l1": self.metrics(-79, 0.12, 93, 88, 94, zero_gt=0.01),
            "tweedie_1_2": self.metrics(-5, 0.13, 70, 65, 82, zero_gt=0.20),
            "log_l2": self.metrics(-35, 0.13, 75, 70, 85, zero_gt=0.02),
        }

        result = select_business_candidate(metrics, ["tweedie_1_2", "log_l2"])

        self.assertEqual(result["selected"], "log_l2")
        self.assertFalse(result["assessments"]["tweedie_1_2"]["eligible"])
        self.assertIn("zero_false_positive_rate", result["assessments"]["tweedie_1_2"]["rejections"])

    def test_selection_allows_bounded_mae_tradeoff_but_rejects_more_than_35_percent(self):
        from src.models.lightgbm_objectives import select_business_candidate

        metrics = {
            "v1_log_l1": self.metrics(-79, 0.10, 93, 88, 94),
            "bounded": self.metrics(-30, 0.134, 78, 75, 88),
            "unacceptable": self.metrics(-5, 0.136, 70, 65, 82),
        }

        result = select_business_candidate(metrics, ["bounded", "unacceptable"])

        self.assertEqual(result["selected"], "bounded")
        self.assertIn("overall_mae", result["assessments"]["unacceptable"]["rejections"])

    def test_best_ranked_candidate_is_reported_even_when_guardrails_reject_it(self):
        from src.models.lightgbm_objectives import best_ranked_candidate, select_business_candidate

        metrics = {
            "v1_log_l1": self.metrics(-79, 0.10, 93, 88, 94),
            "power_1_4": self.metrics(-5, 0.15, 70, 65, 82),
            "power_1_6": self.metrics(-1, 0.16, 71, 66, 83),
        }

        result = select_business_candidate(metrics, ["power_1_4", "power_1_6"])

        self.assertIsNone(result["selected"])
        self.assertEqual(best_ranked_candidate(result), "power_1_6")


if __name__ == "__main__":
    unittest.main()
