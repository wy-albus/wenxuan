import unittest

import numpy as np


class TwoStageModelTests(unittest.TestCase):
    def test_binary_target_is_positive_demand_indicator(self):
        from src.models.two_stage_model import binary_target

        np.testing.assert_array_equal(binary_target([-2, 0, 0.5, 3, np.nan]), [0, 0, 1, 1, 0])

    def test_combined_prediction_is_probability_times_nonnegative_conditional_qty(self):
        from src.models.two_stage_model import combine_predictions

        result = combine_predictions([0.0, 0.25, 1.0, np.nan], [10.0, 8.0, -2.0, 3.0])

        np.testing.assert_allclose(result, [0.0, 2.0, 0.0, 0.0])

    def test_streaming_binary_metrics_handles_degenerate_denominators(self):
        from src.models.two_stage_model import StreamingBinaryMetrics

        metrics = StreamingBinaryMetrics(num_bins=100)
        metrics.update(np.zeros(4), np.zeros(4))
        result = metrics.compute(thresholds=[0.5])

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["positive_rate"], 0.0)
        self.assertTrue(np.isnan(result["roc_auc"]))
        self.assertEqual(result["threshold_metrics"]["0.5"]["recall"], 0.0)
        self.assertTrue(np.isfinite(result["logloss"]))

    def test_streaming_binary_metrics_finds_separating_f1_threshold(self):
        from src.models.two_stage_model import StreamingBinaryMetrics

        metrics = StreamingBinaryMetrics(num_bins=1000)
        metrics.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        result = metrics.compute(thresholds=[0.5])

        self.assertGreater(result["pr_auc"], 0.99)
        self.assertGreater(result["roc_auc"], 0.99)
        self.assertEqual(result["threshold_metrics"]["0.5"]["f1"], 1.0)
        self.assertEqual(result["best_f1"], 1.0)

    def test_weighted_component_metrics_restore_sample_weights(self):
        from src.models.two_stage_model import (
            weighted_binary_component_metrics,
            weighted_regression_component_metrics,
        )

        binary = weighted_binary_component_metrics(
            [0, 1, 1], [0.1, 0.6, 0.4], [1, 2, 1]
        )
        regression = weighted_regression_component_metrics(
            [1, 3], [0, 2], [1, 3]
        )

        self.assertAlmostEqual(binary["recall"], 2 / 3)
        self.assertGreater(binary["pr_auc"], 0.8)
        self.assertAlmostEqual(regression["mae"], 1.0)
        self.assertAlmostEqual(regression["wape"], 40.0)

    def test_candidate_selection_keeps_metric_winners_and_bounds_pairs(self):
        from src.models.two_stage_model import select_component_candidates, select_pair_candidates

        rows = [
            {"iteration": 100, "logloss": 0.2, "pr_auc": 0.7, "recall": 0.6},
            {"iteration": 200, "logloss": 0.3, "pr_auc": 0.9, "recall": 0.7},
            {"iteration": 300, "logloss": 0.4, "pr_auc": 0.8, "recall": 0.95},
            {"iteration": 400, "logloss": 0.25, "pr_auc": 0.85, "recall": 0.8},
        ]
        selected = select_component_candidates(
            rows, (("logloss", "min"), ("pr_auc", "max"), ("recall", "max")), 3
        )
        pairs = select_pair_candidates(
            [
                {"iteration": (100, 200), "wape": 90, "total_bias": 5},
                {"iteration": (200, 200), "wape": 89, "total_bias": 10},
                {"iteration": (300, 300), "wape": 89, "total_bias": 3},
            ],
            2,
        )

        self.assertEqual(set(selected), {100, 200, 300})
        self.assertEqual([row["iteration"] for row in pairs], [(300, 300), (200, 200)])


if __name__ == "__main__":
    unittest.main()
