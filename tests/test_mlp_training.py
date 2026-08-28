import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_training_script():
    path = ROOT / "scripts/14_train_mlp.py"
    spec = importlib.util.spec_from_file_location("mlp_training_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MLPTrainingTests(unittest.TestCase):
    def test_feature_spec_separates_numeric_and_embedding_features(self):
        module = load_training_script()
        numeric, categorical = module.feature_spec(module.load_feature_config())

        self.assertEqual(len(numeric), 44)
        self.assertEqual(categorical, module.CATEGORY_FEATURES)
        forbidden = {"item_id", "isbn", "gds_no", "split", "future_qty_1m", "future_qty_2m"}
        self.assertFalse(forbidden.intersection(numeric + categorical))

    def test_numeric_statistics_are_train_only_and_reused(self):
        module = load_training_script()
        train = pd.DataFrame({"a": [1.0, 3.0], "b": [5.0, 5.0]})
        valid = pd.DataFrame({"a": [5.0], "b": [7.0]})

        means, stds = module.fit_numeric_stats(train, ["a", "b"])
        transformed = module.standardize_numeric(valid, ["a", "b"], means, stds)

        np.testing.assert_allclose(means, [2.0, 5.0])
        np.testing.assert_allclose(stds, [1.0, 1.0])
        np.testing.assert_allclose(transformed, [[3.0, 2.0]])
        self.assertEqual(transformed.dtype, np.float32)

    def test_categories_use_zero_for_unknown_and_one_based_known_ids(self):
        module = load_training_script()
        frame = pd.DataFrame({"site_no": ["A", "B", "missing"]})
        maps = {"site_no": {"A": 0, "B": 1}}

        encoded = module.encode_mlp_categories(frame, ["site_no"], maps)

        np.testing.assert_array_equal(encoded[:, 0], [1, 2, 0])

    def test_embedding_dimensions_are_bounded(self):
        module = load_training_script()

        dimensions = module.embedding_dimensions([2, 10, 1000])

        self.assertEqual(dimensions[0], 4)
        self.assertTrue(all(4 <= value <= 32 for value in dimensions))


if __name__ == "__main__":
    unittest.main()
