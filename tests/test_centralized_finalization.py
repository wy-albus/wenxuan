import unittest
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CentralizedFinalizationTests(unittest.TestCase):
    @staticmethod
    def _load_script():
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "16_finalize_centralized_stage.py"
        spec = spec_from_file_location("finalize_centralized_stage", script_path)
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=2)

    def test_sales_level_rounding_and_boundaries(self):
        module = self._load_script()

        rounded = module.round_prediction_for_level(np.array([0.0, 0.49, 0.5, 1.49, 1.5, 4.5, 19.5]))
        np.testing.assert_array_equal(rounded, np.array([0, 0, 1, 1, 2, 5, 20]))

        levels = module.sales_level(np.array([0, 1, 2, 4, 5, 19, 20]))
        self.assertEqual(
            levels.tolist(),
            ["no_sales", "low", "normal", "normal", "medium_high", "medium_high", "high"],
        )

    def test_level_metrics_include_business_recalls_and_severe_rates(self):
        module = self._load_script()
        metrics = module.compute_level_metrics(
            np.array([0, 1, 2, 6, 21]),
            np.array([0, 6, 2, 1, 20]),
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["active_recall"], 1.0)
        self.assertAlmostEqual(metrics["medium_high_or_high_recall"], 1 / 2)
        self.assertAlmostEqual(metrics["high_recall"], 1.0)
        self.assertAlmostEqual(metrics["severe_underestimate_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["severe_overestimate_rate"], 1 / 2)

    def test_pipeline_checks_alignment_and_writes_outputs(self):
        module = self._load_script()
        base = pd.DataFrame(
            {
                "month": ["2026-01", "2026-01", "2026-01", "2026-02"],
                "site_no": ["S1", "S1", "S2", "S1"],
                "item_id": ["I1", "I2", "I1", "I3"],
                "target_qty_1m": [0.0, 1.0, 6.0, 21.0],
                "target_qty_2m": [0.0, 2.0, 8.0, 30.0],
            }
        )
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            baseline = base.assign(
                split="test",
                historical_mean_pred_1m=[0.0, 1.0, 5.0, 10.0],
                historical_mean_pred_2m=[0.0, 2.0, 10.0, 20.0],
                weighted_ma_pred_1m=[0.0, 1.0, 6.0, 12.0],
                weighted_ma_pred_2m=[0.0, 2.0, 12.0, 24.0],
            )
            lgb = base.assign(
                lightgbm_v2_logl2_pred_1m=[0.1, 0.8, 5.5, 18.0],
                lightgbm_v2_logl2_calibrated_pred_1m=[0.1, 0.8, 5.5, 18.0],
                lightgbm_v2_logl2_pred_2m=[0.2, 1.5, 7.0, 26.0],
                lightgbm_v2_logl2_calibrated_pred_2m=[0.2, 1.5, 7.0, 26.0],
            )
            rf = base.assign(random_forest_pred_1m=[0.0, 1.1, 4.0, 15.0], random_forest_pred_2m=[0.0, 2.2, 8.0, 24.0])
            mlp = base.assign(mlp_pred_1m=[0.2, 0.5, 3.0, 14.0], mlp_pred_2m=[0.2, 1.0, 6.0, 22.0])
            two = base.assign(
                p_sale_1m=[0.1, 0.5, 0.8, 0.9],
                conditional_qty_1m=[1.0, 2.0, 6.0, 20.0],
                two_stage_pred_1m=[0.1, 1.0, 4.8, 18.0],
                p_sale_2m=[0.1, 0.5, 0.8, 0.9],
                conditional_qty_2m=[1.0, 4.0, 8.0, 30.0],
                two_stage_pred_2m=[0.1, 2.0, 6.4, 27.0],
            )
            paths = {
                "baseline": temp / "baseline.parquet",
                "lightgbm_v2_logl2": temp / "lgb.parquet",
                "random_forest": temp / "rf.parquet",
                "mlp": temp / "mlp.parquet",
                "two_stage": temp / "two.parquet",
            }
            for name, frame in [
                ("baseline", baseline),
                ("lightgbm_v2_logl2", lgb),
                ("random_forest", rf),
                ("mlp", mlp),
                ("two_stage", two),
            ]:
                self._write_parquet(paths[name], frame)

            outputs = {
                "summary": temp / "summary.md",
                "comparison": temp / "comparison.csv",
                "level": temp / "level.csv",
                "topk": temp / "topk.csv",
                "sample": temp / "sample.csv",
                "manifest": temp / "manifest.md",
            }
            summary = module.run_centralized_finalization(
                prediction_paths=paths,
                output_paths=outputs,
                report_paths=[],
                feature_config_path=None,
                storage_manifest_path=outputs["manifest"],
                sample_size=10,
            )

            self.assertTrue(summary["consistency"]["passed"])
            self.assertEqual(summary["rows"], 4)
            for path in outputs.values():
                self.assertGreater(path.stat().st_size, 0)
            comparison = pd.read_csv(outputs["comparison"])
            self.assertEqual(set(comparison["model"]), set(module.MODEL_SPECS))
            self.assertIn("two_stage", pd.read_csv(outputs["topk"])["model"].unique())

            bad_lgb = lgb.copy()
            bad_lgb.loc[0, "item_id"] = "DIFFERENT"
            self._write_parquet(paths["lightgbm_v2_logl2"], bad_lgb)
            with self.assertRaises(ValueError):
                module.run_centralized_finalization(
                    prediction_paths=paths,
                    output_paths=outputs,
                    report_paths=[],
                    feature_config_path=None,
                    storage_manifest_path=outputs["manifest"],
                    sample_size=10,
                )


if __name__ == "__main__":
    unittest.main()
