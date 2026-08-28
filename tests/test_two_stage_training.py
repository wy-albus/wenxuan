import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_training_script():
    path = ROOT / "scripts/15_train_two_stage.py"
    spec = importlib.util.spec_from_file_location("two_stage_training_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStageTrainingTests(unittest.TestCase):
    def test_pandas_arrow_stack_is_imported_before_lightgbm_on_python_313(self):
        source = (ROOT / "scripts/15_train_two_stage.py").read_text(encoding="utf-8")

        self.assertLess(source.index("import pandas as pd"), source.index("import lightgbm as lgb"))

    def test_two_stage_features_match_lightgbm_information_scope(self):
        module = load_training_script()
        config = module.load_feature_config()

        for horizon in ("1m", "2m"):
            self.assertEqual(module.horizon_features(config, horizon), module.lightgbm_horizon_features(config, horizon))

    def test_feature_lists_exclude_ids_targets_and_split(self):
        module = load_training_script()
        config = module.load_feature_config()
        forbidden = {
            "future_qty_1m", "future_qty_2m", "future_has_sales_1m", "future_has_sales_2m",
            "split", "item_id", "isbn", "gds_no", "unknown_channel_qty",
        }

        for horizon in ("1m", "2m"):
            self.assertFalse(forbidden.intersection(module.horizon_features(config, horizon)))

    def test_regressor_rates_exclude_zero_bucket(self):
        module = load_training_script()

        rates = module.regressor_sampling_rates()

        self.assertEqual(rates, {"0": 0.0, "1": 0.20, "2-5": 0.40, "5-20": 0.75, "20+": 1.0})

    def test_classifier_params_do_not_double_weight_positive_class(self):
        module = load_training_script()

        params = module.classifier_params(module.load_feature_config())

        self.assertEqual(params["objective"], "binary")
        self.assertNotIn("scale_pos_weight", params)
        self.assertNotIn("is_unbalance", params)

    def test_regressor_params_use_raw_tweedie_power_1_4(self):
        module = load_training_script()

        params = module.regressor_params(module.load_feature_config())

        self.assertEqual(params["objective"], "tweedie")
        self.assertEqual(params["tweedie_variance_power"], 1.4)

    def test_only_matching_completed_formal_model_is_resumable(self):
        module = load_training_script()

        self.assertTrue(module.is_resumable_metadata({"run_stage": "formal", "component": "classifier", "horizon": "1m"}, "classifier", "1m"))
        self.assertFalse(module.is_resumable_metadata({"run_stage": "smoke", "component": "classifier", "horizon": "1m"}, "classifier", "1m"))
        self.assertFalse(module.is_resumable_metadata({"run_stage": "formal", "component": "regressor", "horizon": "1m"}, "classifier", "1m"))

    def test_report_renderer_has_readable_chinese_title(self):
        source = (ROOT / "scripts/15_train_two_stage.py").read_text(encoding="utf-8")

        self.assertIn("# 两阶段模型报告", source)


if __name__ == "__main__":
    unittest.main()
