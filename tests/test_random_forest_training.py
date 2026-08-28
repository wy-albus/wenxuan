import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_training_script():
    path = ROOT / "scripts" / "13_train_random_forest.py"
    spec = importlib.util.spec_from_file_location("random_forest_training_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RandomForestTrainingTests(unittest.TestCase):
    def test_existing_comparison_metrics_cover_all_models_and_splits(self):
        module = load_training_script()

        metrics = module.load_existing_comparison_metrics()

        for split in ("valid", "test"):
            for horizon in ("1m", "2m"):
                self.assertEqual(
                    set(metrics[split][horizon]),
                    {"weighted_moving_average", "lightgbm_v1_logl1", "lightgbm_v2_logl2"},
                )
                self.assertIn("overall", metrics[split][horizon]["lightgbm_v2_logl2"])
                self.assertIn("20+", metrics[split][horizon]["lightgbm_v2_logl2"])

    def test_evaluation_rows_exclude_train_before_prediction(self):
        module = load_training_script()
        frame = pd.DataFrame({"split": ["train", "valid", "test", "other"], "value": [1, 2, 3, 4]})

        selected = module.evaluation_rows_only(frame)

        self.assertEqual(selected["split"].tolist(), ["valid", "test"])
        self.assertEqual(selected["value"].tolist(), [2, 3])

    def test_merge_stage_summaries_keeps_current_results(self):
        module = load_training_script()
        current = {"formal": {"1m": {"train_rows": 10}}}
        previous = {"smoke": {"1m": {"train_rows": 1}}, "pilot": {"1m": {"train_rows": 4}}}

        merged = module.merge_stage_summaries(current, previous)

        self.assertEqual(list(merged), ["smoke", "pilot", "formal"])
        self.assertEqual(merged["formal"]["1m"]["train_rows"], 10)

    def test_configured_dataset_is_the_public_monthly_dataset(self):
        module = load_training_script()
        config = module.load_feature_config()

        self.assertEqual(config["dataset"], "data/processed/model_dataset_monthly.parquet")

    def test_horizon_features_have_expected_order_and_no_leakage(self):
        module = load_training_script()
        config = module.load_feature_config()

        one_month = module.horizon_features(config, "1m")
        two_month = module.horizon_features(config, "2m")
        forbidden = {
            "future_qty_1m", "future_qty_2m", "future_has_sales_1m",
            "future_has_sales_2m", "split", "item_id", "isbn", "gds_no",
        }

        self.assertEqual(len(one_month), 57)
        self.assertEqual(len(two_month), 59)
        self.assertFalse(forbidden.intersection(one_month))
        self.assertFalse(forbidden.intersection(two_month))
        self.assertEqual(one_month[-4:], config["random_forest_horizon_features"]["1m"])
        self.assertEqual(two_month[-6:], config["random_forest_horizon_features"]["2m"])

    def test_scaled_sampling_preserves_relative_base_rates_until_capped(self):
        module = load_training_script()
        base = {"0": 0.05, "1": 0.20, "2-5": 0.40, "5-20": 0.75, "20+": 1.0}
        counts = Counter({bucket: 1_000_000 for bucket in base})
        rates = module.scaled_rates(base, counts, target_rows=240_000)

        scale_factors = [rates[bucket] / base[bucket] for bucket in base]
        self.assertAlmostEqual(max(scale_factors), min(scale_factors), places=8)
        expected = sum(counts[bucket] * rates[bucket] for bucket in base)
        self.assertAlmostEqual(expected, 240_000, delta=1.0)


if __name__ == "__main__":
    unittest.main()
