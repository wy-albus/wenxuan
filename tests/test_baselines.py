import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class BaselineFormulaTests(unittest.TestCase):
    def test_historical_mean_clips_predictions_at_zero(self):
        from src.models.historical_mean import predict_historical_mean

        pred_1m, pred_2m = predict_historical_mean(np.array([2.0, -3.0, np.nan]))

        np.testing.assert_allclose(pred_1m, [2.0, 0.0, 0.0])
        np.testing.assert_allclose(pred_2m, [4.0, 0.0, 0.0])

    def test_weighted_moving_average_uses_configured_formula(self):
        from src.models.weighted_moving_average import predict_weighted_moving_average

        pred_1m, pred_2m = predict_weighted_moving_average(
            np.array([10.0, -10.0, np.nan]),
            np.array([5.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 3.0]),
            weights=[0.5, 0.3, 0.2],
        )

        np.testing.assert_allclose(pred_1m, [6.5, 0.0, 1.2])
        np.testing.assert_allclose(pred_2m, [13.0, 0.0, 2.4])


class StreamingMetricsTests(unittest.TestCase):
    def test_target_clipping_and_demand_bucket_boundaries(self):
        from src.evaluation.metrics import clip_target, demand_bucket

        np.testing.assert_allclose(clip_target([-2, 0, 1, 2, 5, 6, 20, 21]), [0, 0, 1, 2, 5, 6, 20, 21])
        self.assertEqual(
            demand_bucket(np.array([0, 1, 2, 5, 6, 20, 21])).tolist(),
            ["0", "1", "2-5", "5-20", "5-20", "20+", "20+"],
        )

    def test_zero_denominators_do_not_raise(self):
        from src.evaluation.metrics import StreamingMetrics

        exact = StreamingMetrics()
        exact.update(np.zeros(3), np.zeros(3))
        self.assertEqual(exact.compute()["smape"], 0.0)
        self.assertEqual(exact.compute()["wape"], 0.0)

        wrong = StreamingMetrics()
        wrong.update(np.zeros(2), np.ones(2))
        self.assertTrue(np.isfinite(wrong.compute()["smape"]))
        self.assertTrue(np.isinf(wrong.compute()["wape"]))

    def test_streaming_metrics_match_one_shot_metrics(self):
        from src.evaluation.metrics import StreamingMetrics

        target = np.array([0.0, 1.0, 3.0, 8.0, 25.0])
        pred = np.array([0.0, 2.0, 2.0, 10.0, 20.0])
        one_shot = StreamingMetrics()
        one_shot.update(target, pred)
        streamed = StreamingMetrics()
        streamed.update(target[:2], pred[:2])
        streamed.update(target[2:], pred[2:])

        for key in ["count", "mae", "rmse", "smape", "wape"]:
            self.assertAlmostEqual(one_shot.compute()[key], streamed.compute()[key])


class BaselinePipelineTests(unittest.TestCase):
    @staticmethod
    def _load_script():
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "09_train_baselines.py"
        spec = spec_from_file_location("train_baselines", script_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_pipeline_evaluates_valid_and_test_but_writes_only_test(self):
        module = self._load_script()
        self.assertGreater(module._peak_process_memory_bytes(), 0)
        self.assertEqual(len(module.SOURCE_COLUMNS), 11)
        self.assertEqual(
            set(module.SOURCE_COLUMNS),
            {
                "month", "site_no", "item_id", "split", "future_qty_1m", "future_qty_2m",
                "qty_lag_1m", "qty_lag_2m", "qty_lag_3m", "qty_mean_last_3m", "qty_sum_last_3m",
            },
        )
        frame = pd.DataFrame(
            {
                "month": ["2025-06", "2025-07", "2025-08", "2026-01", "2026-02", "2026-03"],
                "site_no": ["S1"] * 6,
                "item_id": ["I1", "I1", "I2", "I1", "I2", "I3"],
                "split": ["train", "valid", "valid", "test", "test", "test"],
                "future_qty_1m": [-1.0, 1.0, 6.0, 0.0, 2.0, 21.0],
                "future_qty_2m": [-2.0, 2.0, 8.0, 1.0, 5.0, 25.0],
                "qty_lag_1m": np.array([1.0, 1.0, 4.0, 0.0, 2.0, 10.0], dtype="float32"),
                "qty_lag_2m": np.array([1.0, 1.0, 7.0, 0.0, 1.0, 8.0], dtype="float32"),
                "qty_lag_3m": np.array([1.0, 1.0, 4.0, 0.0, 0.0, 6.0], dtype="float32"),
                "qty_mean_last_3m": [1.0, 1.0, 4.0, 0.0, 1.0, 8.0],
                "qty_sum_last_3m": [3.0, 3.0, 12.0, 0.0, 3.0, 24.0],
                "unused": [99] * 6,
            }
        )
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset = temp / "input.parquet"
            predictions = temp / "predictions.parquet"
            report = temp / "report.md"
            sample = temp / "sample.csv"
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), dataset, row_group_size=2)

            self.assertTrue(module.run_smoke_test(dataset, row_groups=2, rows_per_group=2)["passed"])
            summary = module.run_baseline_pipeline(dataset, predictions, report, sample)

            output = pd.read_parquet(predictions)
            self.assertEqual(len(output), 3)
            self.assertEqual(set(output["split"]), {"test"})
            self.assertEqual(list(output.columns), list(module.PREDICTION_COLUMNS))
            self.assertEqual(summary["split_rows"], {"valid": 2, "test": 3})
            sample_output = pd.read_csv(sample)
            self.assertEqual(set(sample_output["split"]), {"valid", "test"})
            self.assertGreater(report.stat().st_size, 0)
            self.assertGreater(sample.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
