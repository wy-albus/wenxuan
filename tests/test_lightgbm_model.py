import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.features.preprocessing import add_runtime_columns, get_feature_list, load_feature_config


class LightGBMTimeFeatureTests(unittest.TestCase):
    def test_time_index_uses_fixed_2023_01_base_across_batches(self):
        first = add_runtime_columns(pd.DataFrame({"month": ["2023-01", "2024-01"], "future_qty_1m": [0, 0], "future_qty_2m": [0, 0]}))
        second = add_runtime_columns(pd.DataFrame({"month": ["2024-01"], "future_qty_1m": [0], "future_qty_2m": [0]}))

        self.assertEqual(first["time_index"].tolist(), [0, 12])
        self.assertEqual(second["time_index"].tolist(), [12])

    def test_runtime_columns_include_horizon_calendar_features(self):
        frame = add_runtime_columns(pd.DataFrame({"month": ["2025-12"], "future_qty_1m": [-2], "future_qty_2m": [3]}))

        self.assertEqual(frame.loc[0, "target_qty_1m"], 0)
        self.assertEqual(frame.loc[0, "future_1m_year"], 2026)
        self.assertEqual(frame.loc[0, "future_1m_month_of_year"], 1)
        self.assertEqual(frame.loc[0, "future_1m_quarter"], 1)
        self.assertEqual(frame.loc[0, "future_1m_time_index"], 36)
        self.assertEqual(frame.loc[0, "future_2m_month_of_year"], 2)
        self.assertEqual(frame.loc[0, "future_2m_quarter"], 1)
        self.assertEqual(frame.loc[0, "future_2m_time_index"], 37)

    def test_lightgbm_horizon_feature_lists_include_runtime_time_fields(self):
        config = load_feature_config()
        common = set(get_feature_list("lightgbm", config))
        self.assertTrue({"year", "month_of_year", "quarter", "time_index"}.issubset(common))
        self.assertEqual(
            set(config["lightgbm_horizon_features"]["1m"]),
            {"future_1m_year", "future_1m_month_of_year", "future_1m_quarter", "future_1m_time_index"},
        )
        self.assertEqual(
            set(config["lightgbm_horizon_features"]["2m"]),
            {
                "future_1m_month_of_year", "future_1m_quarter", "future_1m_time_index",
                "future_2m_month_of_year", "future_2m_quarter", "future_2m_time_index",
            },
        )


class LightGBMSamplingTests(unittest.TestCase):
    def test_hash_sampling_is_deterministic_and_horizon_specific(self):
        from src.models.lightgbm_model import deterministic_uniform_hash

        frame = pd.DataFrame({"month": ["2025-01"] * 4, "site_no": ["S1"] * 4, "item_id": ["I1", "I2", "I3", "I4"]})
        first = deterministic_uniform_hash(frame, "1m", 42)
        repeated = deterministic_uniform_hash(frame, "1m", 42)
        other_horizon = deterministic_uniform_hash(frame, "2m", 42)

        np.testing.assert_array_equal(first, repeated)
        self.assertTrue(np.all((first >= 0) & (first < 1)))
        self.assertFalse(np.array_equal(first, other_horizon))

    def test_sampling_uses_target_bucket_rate_and_normalized_inverse_weights(self):
        from src.models.lightgbm_model import sample_by_target

        frame = pd.DataFrame(
            {
                "month": ["2025-01"] * 5,
                "site_no": ["S1"] * 5,
                "item_id": ["I0", "I1", "I2", "I3", "I4"],
                "future_qty_1m": [0, 1, 3, 10, 25],
            }
        )
        rates = {"0": 1.0, "1": 1.0, "2-5": 1.0, "5-20": 1.0, "20+": 1.0}
        sampled, weights, buckets = sample_by_target(frame, "1m", rates, seed=42)

        self.assertEqual(len(sampled), 5)
        self.assertEqual(buckets.tolist(), ["0", "1", "2-5", "5-20", "20+"])
        np.testing.assert_allclose(weights.mean(), 1.0)
        np.testing.assert_allclose(weights, np.ones(5))

    def test_category_maps_fit_on_train_and_unseen_values_become_minus_one(self):
        from src.models.lightgbm_model import encode_categories, fit_category_maps

        train = pd.DataFrame({"site_no": ["S2", "S1", None]})
        maps = fit_category_maps(train, ["site_no"])
        encoded = encode_categories(pd.DataFrame({"site_no": ["S1", "S3", None]}), maps)

        self.assertGreaterEqual(encoded.loc[0, "site_no"], 0)
        self.assertEqual(encoded.loc[1, "site_no"], -1)
        self.assertGreaterEqual(encoded.loc[2, "site_no"], 0)


class LightGBMModelPersistenceTests(unittest.TestCase):
    def test_saved_model_reloads_with_identical_predictions_and_metadata(self):
        from src.models.lightgbm_model import load_model_bundle, save_model_bundle, train_booster

        rng = np.random.default_rng(42)
        features = ["qty_lag_1m", "site_no"]
        train_x = pd.DataFrame({"qty_lag_1m": rng.random(200).astype("float32"), "site_no": rng.integers(0, 3, 200, dtype="int32")})
        valid_x = train_x.iloc[:50].copy()
        train_y = np.log1p(train_x["qty_lag_1m"].to_numpy())
        valid_y = train_y[:50]
        booster, _ = train_booster(
            train_x, train_y, np.ones(200), valid_x, valid_y, np.ones(50),
            categorical_features=["site_no"], params={"objective": "regression_l1", "metric": "l1", "max_bin": 127, "verbosity": -1, "num_threads": 1, "seed": 42},
            num_boost_round=10, early_stopping_rounds=3,
        )
        metadata = {"feature_names": features, "categorical_features": ["site_no"], "category_maps": {"site_no": {"S1": 0}}, "time_base": "2023-01"}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.txt"
            save_model_bundle(booster, path, metadata)
            loaded, loaded_metadata = load_model_bundle(path)

            np.testing.assert_allclose(booster.predict(valid_x), loaded.predict(valid_x))
            self.assertEqual(loaded_metadata, metadata)


class LightGBMLongTailMetricTests(unittest.TestCase):
    def test_long_tail_metrics_include_totals_special_segments_and_zero_diagnostics(self):
        from src.evaluation.metrics import LongTailStreamingMetrics

        metrics = LongTailStreamingMetrics()
        metrics.update(
            np.array([0.0, 0.0, 1.0, 5.0, 20.0, 25.0]),
            np.array([0.0, 2.0, 1.5, 4.0, 18.0, 30.0]),
        )
        result = metrics.compute()

        self.assertEqual(result["overall"]["count"], 6)
        self.assertEqual(result["overall"]["target_sum"], 51.0)
        self.assertEqual(result["overall"]["prediction_sum"], 55.5)
        self.assertEqual(result["nonzero"]["count"], 4)
        self.assertEqual(result["ge_5"]["count"], 3)
        self.assertEqual(result["ge_20"]["count"], 2)
        self.assertTrue(np.isnan(result["0"]["wape"]))
        self.assertEqual(result["0"]["mean_prediction"], 1.0)
        self.assertEqual(result["0"]["prediction_gt_0_5_rate"], 0.5)
        self.assertEqual(result["0"]["prediction_gt_1_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
