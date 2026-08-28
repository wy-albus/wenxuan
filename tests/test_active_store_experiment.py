import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.active_store_dataset import (
    build_active_store_site_dataset,
    build_store_activity,
    filter_existing_model_frame_to_active_store,
)
from src.evaluation.mc_metrics import MCLevelConfig, TrendAccumulator, load_mc_level_config, quantity_to_mc
from src.models.active_store_common import (
    encode_categories_with_unknown,
    fit_category_maps_with_unknown,
)


def _monthly_rows(months, quantities):
    size = len(months)
    return pd.DataFrame(
        {
            "month": months,
            "site_no": ["S1"] * size,
            "blt_site_no": ["B1"] * size,
            "item_id": ["I1"] * size,
            "isbn": ["I1"] * size,
            "gds_no": ["G1"] * size,
            "gds_ctgry_3_lvel": ["C3"] * size,
            "gds_ctgry_4_lvel": ["C4"] * size,
            "gds_ctgry_5_lvel": ["C5"] * size,
            "price": [10.0] * size,
            "total_qty": quantities,
            "offline_qty": quantities,
            "online_qty": [0.0] * size,
            "unknown_channel_qty": [0.0] * size,
            "total_tlp": [10.0 * value for value in quantities],
            "total_tsp": [9.0 * value for value in quantities],
            "avg_real_price": [9.0] * size,
            "discount_rate": [0.9] * size,
            "sales_days": [1] * size,
            "sales_count": [1] * size,
            "return_count": [0] * size,
            "return_qty": [0.0] * size,
        }
    )


class ActiveStoreDatasetTests(unittest.TestCase):
    def test_store_activity_uses_any_monthly_flow_and_keeps_internal_gap(self):
        monthly = pd.concat(
            [
                _monthly_rows(["2023-01", "2023-03"], [2.0, 7.0]),
                _monthly_rows(["2023-02"], [0.0]).assign(site_no="S2", item_id="I2"),
            ],
            ignore_index=True,
        )

        activity = build_store_activity(monthly)
        s1 = activity.set_index("site_no").loc["S1"]

        self.assertEqual(s1["first_active_month"], "2023-01")
        self.assertEqual(s1["last_active_month"], "2023-03")
        self.assertEqual(s1["active_month_count"], 2)
        self.assertEqual(s1["internal_gap_month_count"], 1)

    def test_active_store_panel_stops_at_store_last_month_and_keeps_horizon_flags(self):
        monthly = _monthly_rows(["2023-01", "2023-03"], [2.0, 7.0])
        dataset, panel_rows = build_active_store_site_dataset(monthly)

        self.assertEqual(panel_rows, 3)
        self.assertEqual(dataset["month"].tolist(), ["2023-01", "2023-02"])
        self.assertNotIn("2023-04", dataset["month"].tolist())
        jan = dataset.iloc[0]
        feb = dataset.iloc[1]
        self.assertEqual(jan["future_qty_1m"], 0.0)
        self.assertEqual(jan["future_qty_2m"], 7.0)
        self.assertEqual(jan["target_available_1m"], 1)
        self.assertEqual(jan["target_available_2m"], 1)
        self.assertEqual(feb["future_qty_1m"], 7.0)
        self.assertTrue(np.isnan(feb["future_qty_2m"]))
        self.assertEqual(feb["target_available_1m"], 1)
        self.assertEqual(feb["target_available_2m"], 0)

    def test_existing_leakage_safe_features_can_be_filtered_by_store_horizon(self):
        frame = pd.DataFrame(
            {
                "month": ["2023-01", "2023-02", "2023-03", "2023-04"],
                "site_no": ["S1"] * 4,
                "item_id": ["I1"] * 4,
                "future_qty_1m": [0.0, 7.0, 0.0, 0.0],
                "future_qty_2m": [7.0, 7.0, 0.0, 0.0],
                "future_has_sales_1m": [0, 1, 0, 0],
                "future_has_sales_2m": [1, 1, 0, 0],
            }
        )
        activity = pd.DataFrame(
            {"site_no": ["S1"], "first_active_month": ["2023-01"], "last_active_month": ["2023-03"]}
        )

        filtered = filter_existing_model_frame_to_active_store(frame, activity)

        self.assertEqual(filtered["month"].tolist(), ["2023-01", "2023-02"])
        self.assertEqual(filtered["target_available_2m"].tolist(), [1, 0])
        self.assertTrue(np.isnan(filtered.loc[1, "future_qty_2m"]))


class MCLevelTests(unittest.TestCase):
    def setUp(self):
        self.config = MCLevelConfig(
            codes=(0, 1, 2, 3, 4),
            names=("no_sales", "low", "normal", "medium_high", "high"),
            lower_bounds=(0, 1, 2, 5, 20),
        )

    def test_one_month_quantity_rounding_and_boundaries(self):
        values = np.array([-1.0, 0.49, 0.5, 1.49, 1.5, 4.49, 4.5, 19.49, 19.5])
        result = quantity_to_mc(values, self.config, horizon="1m")
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2, 3, 3, 4])

    def test_two_month_uses_monthly_average_before_rounding_and_mapping(self):
        cumulative = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 39.0, 40.0])
        result = quantity_to_mc(cumulative, self.config, horizon="2m")
        np.testing.assert_array_equal(result, [0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_project_mc_config_matches_current_five_level_rule(self):
        config = load_mc_level_config("config/mc_sales_levels.yaml")

        self.assertEqual(config.codes, (0, 1, 2, 3, 4))
        self.assertEqual(config.lower_bounds, (0, 1, 2, 5, 20))


class TrainOnlyCategoryEncodingTests(unittest.TestCase):
    def test_valid_unknowns_use_reserved_zero_without_refitting(self):
        maps = fit_category_maps_with_unknown(pd.DataFrame({"site_no": ["S1", None]}), ["site_no"])
        encoded, unknown_rates = encode_categories_with_unknown(
            pd.DataFrame({"site_no": ["S1", "S2", None]}), maps
        )

        self.assertGreater(encoded.loc[0, "site_no"], 0)
        self.assertEqual(encoded.loc[1, "site_no"], 0)
        self.assertGreater(encoded.loc[2, "site_no"], 0)
        self.assertAlmostEqual(unknown_rates["site_no"], 1 / 3)
        self.assertNotIn("S2", maps["site_no"])

    def test_shared_lightgbm_encoder_honors_reserved_unknown_code(self):
        from src.models.lightgbm_model import encode_categories

        encoded = encode_categories(
            pd.DataFrame({"site_no": ["S1", "S2"]}),
            {"site_no": {"__UNKNOWN__": 0, "S1": 1}},
        )
        self.assertEqual(encoded["site_no"].tolist(), [1, 0])


class TrendMetricTests(unittest.TestCase):
    def test_trend_metrics_count_severe_opposite_direction_errors(self):
        accumulator = TrendAccumulator(flat_tolerance=0.0)
        accumulator.update(
            current=np.array([1.0, 2.0, 3.0]),
            future_true=np.array([2.0, 2.0, 1.0]),
            future_pred=np.array([2.0, 1.0, 4.0]),
        )
        result = accumulator.compute()

        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["severe_direction_error_rate"], 1 / 3)
        self.assertEqual(result["confusion_matrix"].sum(), 3)


class SelectedIterationPersistenceTests(unittest.TestCase):
    def test_model_bundle_can_persist_full_valid_selected_iteration(self):
        import lightgbm as lgb
        from src.models.lightgbm_model import load_model_bundle, save_model_bundle

        x = pd.DataFrame({"x": np.arange(50, dtype="float32")})
        booster = lgb.train(
            {"objective": "regression", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2},
            lgb.Dataset(x, label=np.arange(50, dtype="float32")),
            num_boost_round=5,
        )
        metadata = {"feature_names": ["x"], "selected_iteration": 2}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"
            save_model_bundle(booster, path, metadata)
            loaded, loaded_metadata = load_model_bundle(path)

        self.assertEqual(loaded.num_trees(), 2)
        self.assertEqual(loaded_metadata["selected_iteration"], 2)


if __name__ == "__main__":
    unittest.main()
