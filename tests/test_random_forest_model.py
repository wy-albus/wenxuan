import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.features.preprocessing import get_feature_list, load_feature_config


class RandomForestFeatureTests(unittest.TestCase):
    def test_random_forest_uses_fixed_runtime_and_horizon_time_features(self):
        config = load_feature_config()
        common = set(get_feature_list("random_forest", config))

        self.assertTrue({"year", "month_of_year", "quarter", "time_index"}.issubset(common))
        self.assertEqual(
            set(config["random_forest_horizon_features"]["1m"]),
            {"future_1m_year", "future_1m_month_of_year", "future_1m_quarter", "future_1m_time_index"},
        )
        self.assertEqual(
            set(config["random_forest_horizon_features"]["2m"]),
            {
                "future_1m_month_of_year", "future_1m_quarter", "future_1m_time_index",
                "future_2m_month_of_year", "future_2m_quarter", "future_2m_time_index",
            },
        )


class RandomForestModelTests(unittest.TestCase):
    def test_log_target_and_inverse_prediction_are_non_negative(self):
        from src.models.random_forest_model import inverse_prediction, transform_target

        target = np.array([-2.0, 0.0, 1.0, 9.0, np.nan])
        transformed = transform_target(target)

        np.testing.assert_allclose(transformed, np.log1p([0.0, 0.0, 1.0, 9.0, 0.0]))
        np.testing.assert_allclose(inverse_prediction(transformed), [0.0, 0.0, 1.0, 9.0, 0.0])

    def test_estimator_uses_bounded_conservative_parameters(self):
        from src.models.random_forest_model import build_estimator

        estimator = build_estimator(n_estimators=100, n_jobs=2)

        self.assertEqual(estimator.criterion, "squared_error")
        self.assertEqual(estimator.max_depth, 18)
        self.assertEqual(estimator.min_samples_leaf, 50)
        self.assertEqual(estimator.min_samples_split, 100)
        self.assertEqual(estimator.max_features, 0.5)
        self.assertEqual(estimator.max_samples, 0.7)
        self.assertTrue(estimator.bootstrap)
        self.assertTrue(estimator.warm_start)
        self.assertEqual(estimator.random_state, 42)

    def test_model_bundle_reloads_with_identical_predictions(self):
        from src.models.random_forest_model import build_estimator, load_model_bundle, save_model_bundle

        rng = np.random.default_rng(42)
        x = rng.normal(size=(300, 4)).astype("float32")
        y = np.log1p(np.maximum(x[:, 0] + 1, 0)).astype("float32")
        model = build_estimator(n_estimators=5, n_jobs=1, max_depth=5, min_samples_leaf=2, min_samples_split=4)
        model.fit(x, y, sample_weight=np.ones(len(y), dtype="float32"))
        metadata = {"feature_names": ["a", "b", "c", "d"], "horizon": "1m"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_model_bundle(model, path, metadata)
            loaded, loaded_metadata = load_model_bundle(path)

            np.testing.assert_allclose(model.predict(x[:20]), loaded.predict(x[:20]))
            self.assertEqual(loaded_metadata, metadata)

    def test_tree_statistics_report_mean_depth_and_nodes(self):
        from src.models.random_forest_model import build_estimator, tree_statistics

        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 3)).astype("float32")
        y = x[:, 0].astype("float32")
        model = build_estimator(n_estimators=3, n_jobs=1, max_depth=4, min_samples_leaf=2, min_samples_split=4)
        model.fit(x, y)
        stats = tree_statistics(model)

        self.assertEqual(stats["tree_count"], 3)
        self.assertGreater(stats["mean_node_count"], 1)
        self.assertLessEqual(stats["mean_depth"], 4)

    def test_stop_rule_requires_two_consecutive_small_improvements(self):
        from src.models.random_forest_model import should_stop_tree_growth

        history = [
            {"mae": 1.0, "nonzero_wape": 80.0},
            {"mae": 0.996, "nonzero_wape": 79.7},
            {"mae": 0.993, "nonzero_wape": 79.4},
        ]

        self.assertTrue(should_stop_tree_growth(history, threshold=0.005))
        history[-1]["mae"] = 0.98
        self.assertFalse(should_stop_tree_growth(history, threshold=0.005))


if __name__ == "__main__":
    unittest.main()
