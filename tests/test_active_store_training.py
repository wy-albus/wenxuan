from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "18_train_active_store_experiment.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("active_store_training", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ActiveStoreTrainingPureFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_training_module()

    def test_pandas_pyarrow_native_stack_loads_before_lightgbm_on_python_313(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertLess(source.index("import pandas as pd"), source.index("import lightgbm as lgb"))
        self.assertLess(source.index("import pyarrow as pa"), source.index("import lightgbm as lgb"))

    def test_evaluation_projection_keeps_both_future_targets_for_runtime_preprocessing(self):
        config = self.module.load_feature_config()
        columns = self.module._physical_eval_columns(config, ["qty_lag_1m"], "1m")
        self.assertIn("future_qty_1m", columns)
        self.assertIn("future_qty_2m", columns)

    def test_train_only_category_maps_reserve_zero_and_report_unknown_rates(self):
        train = pd.DataFrame({"site_no": ["S2", "S1", None]})
        valid = pd.DataFrame({"site_no": ["S1", "S3", None]})

        maps = self.module.fit_train_category_maps(train, ["site_no"])
        encoded, rates = self.module.encode_with_unknown_rates(valid, maps)

        self.assertEqual(maps["site_no"]["__UNKNOWN__"], 0)
        self.assertGreater(encoded.loc[0, "site_no"], 0)
        self.assertEqual(encoded.loc[1, "site_no"], 0)
        self.assertGreater(encoded.loc[2, "site_no"], 0)
        self.assertAlmostEqual(rates["site_no"], 1 / 3)

    def test_two_month_mc_uses_clip_divide_round_then_config_mapping(self):
        config = self.module.MCLevelConfig(
            codes=(0, 1, 2, 3, 4),
            names=("none", "low", "normal", "medium", "high"),
            lower_bounds=(0, 1, 2, 5, 20),
        )
        values = np.array([-2.0, 0.0, 1.0, 2.0, 3.0, 9.0, 10.0, 39.0, 40.0])

        result = self.module.quantity_to_mc_strict(values, config, "2m")

        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 3, 3, 4, 4])

    def test_mc_target_uses_canonical_quantity_mapping_and_validates_matching_field(self):
        config = self.module.MCLevelConfig(
            codes=(0, 1, 2), names=("none", "low", "high"), lower_bounds=(0, 1, 3)
        )
        with_field = pd.DataFrame({"future_qty_1m": [0.0, 7.0], "target_mc_1m": [0, 2]})
        without_field = pd.DataFrame({"future_qty_1m": [0.0, 3.0]})

        np.testing.assert_array_equal(
            self.module.resolve_mc_target(with_field, "1m", config), [0, 2]
        )
        np.testing.assert_array_equal(
            self.module.resolve_mc_target(without_field, "1m", config), [0, 2]
        )

    def test_mc_target_rejects_non_integer_or_quantity_inconsistent_field(self):
        config = self.module.MCLevelConfig(
            codes=(0, 1, 2), names=("none", "low", "high"), lower_bounds=(0, 1, 3)
        )
        non_integer = pd.DataFrame({"future_qty_1m": [3.0], "target_mc_1m": [1.5]})
        inconsistent = pd.DataFrame({"future_qty_1m": [3.0], "target_mc_1m": [1]})

        with self.assertRaisesRegex(ValueError, "integer"):
            self.module.resolve_mc_target(non_integer, "1m", config)
        with self.assertRaisesRegex(ValueError, "canonical quantity mapping"):
            self.module.resolve_mc_target(inconsistent, "1m", config)

    def test_two_stage_prediction_is_probability_times_conditional_quantity(self):
        result = self.module.combine_two_stage(
            np.array([-0.5, 0.25, 2.0]), np.array([10.0, -3.0, 4.0])
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 4.0])

    def test_candidate_selection_uses_secondary_metrics_then_smaller_round(self):
        candidates = [
            {"iteration": 150, "macro_f1": 0.6, "weighted_f1": 0.7, "accuracy": 0.8},
            {"iteration": 100, "macro_f1": 0.6, "weighted_f1": 0.7, "accuracy": 0.8},
            {"iteration": 200, "macro_f1": 0.6, "weighted_f1": 0.69, "accuracy": 0.9},
        ]
        selected = self.module.select_best_candidate(
            candidates,
            metric_order=(("macro_f1", "max"), ("weighted_f1", "max"), ("accuracy", "max")),
        )
        self.assertEqual(selected["iteration"], 100)

    def test_quantity_candidate_selection_minimizes_wape_then_mae(self):
        candidates = [
            {"iteration": 50, "wape": 20.0, "mae": 2.0},
            {"iteration": 100, "wape": 20.0, "mae": 1.5},
            {"iteration": 150, "wape": 20.1, "mae": 1.0},
        ]
        selected = self.module.select_best_candidate(
            candidates, metric_order=(("wape", "min"), ("mae", "min"))
        )
        self.assertEqual(selected["iteration"], 100)

    def test_quantity_candidate_ties_use_absolute_bias_trend_then_round(self):
        candidates = [
            {"iteration": 80, "wape": 20.0, "total_bias_rate": -5.0, "trend_macro_f1": 0.9},
            {"iteration": 100, "wape": 20.0, "total_bias_rate": 2.0, "trend_macro_f1": 0.7},
            {"iteration": 120, "wape": 20.0, "total_bias_rate": -2.0, "trend_macro_f1": 0.8},
        ]

        selected = self.module.select_best_candidate(
            candidates, metric_order=self.module.QUANTITY_CANDIDATE_METRIC_ORDER
        )

        self.assertEqual(selected["iteration"], 120)

    def test_two_stage_final_tie_prefers_fewer_total_component_rounds(self):
        candidates = [
            {"iteration": (80, 80), "wape": 10.0, "total_bias_rate": 1.0, "trend_macro_f1": 0.5},
            {"iteration": (60, 90), "wape": 10.0, "total_bias_rate": -1.0, "trend_macro_f1": 0.5},
        ]

        selected = self.module.select_best_candidate(
            candidates, metric_order=self.module.QUANTITY_CANDIDATE_METRIC_ORDER
        )

        self.assertEqual(selected["iteration"], (60, 90))

    def test_mc_candidate_order_uses_high_recall_weighted_f1_logloss_then_round(self):
        candidates = [
            {"iteration": 40, "macro_f1": 0.6, "high_level_recall": 0.7, "weighted_f1": 0.8, "logloss": 0.5},
            {"iteration": 60, "macro_f1": 0.6, "high_level_recall": 0.8, "weighted_f1": 0.7, "logloss": 0.4},
            {"iteration": 80, "macro_f1": 0.6, "high_level_recall": 0.8, "weighted_f1": 0.7, "logloss": 0.3},
        ]

        selected = self.module.select_best_candidate(
            candidates, metric_order=self.module.MC_CANDIDATE_METRIC_ORDER
        )

        self.assertEqual(selected["iteration"], 80)

    def test_candidate_window_is_capped_by_actual_booster_iterations(self):
        class Booster:
            def current_iteration(self):
                return 37

        candidates = self.module.candidate_window_for_booster(
            Booster(), center=35, radius=100, step=10
        )

        self.assertIn(37, candidates)
        self.assertLessEqual(max(candidates), 37)

    def test_candidate_window_combines_global_grid_with_proxy_local_grid(self):
        class Booster:
            def current_iteration(self):
                return 450

        candidates = self.module.candidate_window_for_booster(
            Booster(), center=350, radius=20, step=10, global_step=100, max_candidates=30
        )

        self.assertEqual(candidates[0], 1)
        self.assertEqual(candidates[-1], 450)
        self.assertIn(100, candidates)
        self.assertIn(350, candidates)
        self.assertLessEqual(len(candidates), 30)

    def test_two_month_trend_inputs_use_monthly_future_and_prediction(self):
        current, target, prediction = self.module.trend_inputs(
            np.array([3.0]), np.array([8.0]), np.array([4.0]), "2m"
        )

        np.testing.assert_array_equal(current, [3.0])
        np.testing.assert_array_equal(target, [4.0])
        np.testing.assert_array_equal(prediction, [2.0])

    def test_multiclass_logloss_uses_true_class_probabilities(self):
        loss = self.module.multiclass_logloss(
            true_codes=np.array([0, 2]),
            probabilities=np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
            codes=(0, 1, 2),
        )

        self.assertAlmostEqual(loss, -(np.log(0.8) + np.log(0.7)) / 2)

    def test_mc_disagreement_diagnostics_partition_every_row(self):
        result = self.module.mc_disagreement_diagnostics(
            target=np.array([0, 1, 2, 2]),
            regression_mapped=np.array([0, 1, 1, 0]),
            direct=np.array([0, 2, 2, 1]),
        )

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["same_count"], 1)
        self.assertEqual(result["regression_correct_direct_wrong_count"], 1)
        self.assertEqual(result["direct_correct_regression_wrong_count"], 1)
        self.assertEqual(result["both_wrong_count"], 1)

    def test_quantity_to_mc_docstates_required_two_month_order(self):
        self.assertIn("qty/2 -> clip -> floor(x + 0.5) -> map", self.module.quantity_to_mc_strict.__doc__)

    def test_training_callbacks_include_early_stopping_safety_limit(self):
        class Guard:
            @staticmethod
            def callback(*args, **kwargs):
                return lambda env: None

        callbacks = self.module.training_callbacks({}, Guard(), early_stopping_rounds=7)

        stopping = [callback for callback in callbacks if hasattr(callback, "stopping_rounds")]
        self.assertEqual(len(stopping), 1)
        self.assertEqual(stopping[0].stopping_rounds, 7)

    def test_fair_comparison_rows_cover_quantity_trend_and_mapped_mc(self):
        quantity_values = {"overall": {"count": 2, "wape": 10.0}}
        trend_values = {"count": 2, "macro_f1": 0.5, "confusion_matrix": np.eye(3, dtype=int)}
        mc_values = {"count": 2, "macro_f1": 0.6, "confusion_matrix": np.eye(3, dtype=int)}
        metrics = {
            "quantity": {"1m": {"lightgbm_v2_logl2": {"old": quantity_values, "active_store": quantity_values}}},
            "trend": {"1m": {"lightgbm_v2_logl2": {"old": trend_values, "active_store": trend_values}}},
            "mc": {"1m": {"lightgbm_v2_logl2": {"old": mc_values, "active_store": mc_values}}},
        }

        rows = self.module._fair_rows(metrics)

        self.assertEqual({row["metric_family"] for row in rows}, {"quantity", "trend", "mapped_mc"})

    def test_experiment_report_lines_include_requested_summaries(self):
        lines = self.module.build_experiment_report_lines(
            structural={"old_rows_after_store_closure": "12", "active_store_rows": "100"},
            core_summary=[{"split": "test", "horizon": "1m", "model": "two_stage", "wape": 10.0}],
            fair_summary=[{"horizon": "1m", "model_family": "two_stage", "metric": "wape", "change": -2.0}],
            disagreement_summary=[{"split": "test", "horizon": "1m", "model": "v2", "same_rate": 0.8}],
            unknown_rates={"valid": {"site_no": 0.1}, "test": {"site_no": 0.2}},
        )
        text = "\n".join(lines)

        self.assertIn("Structural-Zero Correction", text)
        self.assertIn("Core Valid/Test Metrics", text)
        self.assertIn("Common-Sample Old/New Changes", text)
        self.assertIn("Mapped MC vs Direct MC", text)
        self.assertIn("Category Unknown Rates", text)
        self.assertIn("2M eligibility implies 1M eligibility", text)
        self.assertIn("coarse-to-fine full-valid", text)

    def test_refuse_overwrite_checks_every_formal_output(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "already-there.txt"
            missing = Path(directory) / "missing.txt"
            existing.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already-there"):
                self.module.refuse_existing_outputs([missing, existing])

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_promotion_race_fails_without_overwriting_destination(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "result.txt"
            temporary = destination.with_suffix(".txt.tmp")
            temporary.write_text("new", encoding="utf-8")
            destination.write_text("racer", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self.module.promote_formal_artifacts({"result": destination})

            self.assertEqual(destination.read_text(encoding="utf-8"), "racer")
            self.assertEqual(temporary.read_text(encoding="utf-8"), "new")

    def test_formal_unknown_accumulator_uses_training_missing_token_and_reserved_zero(self):
        maps = {"site_no": {"__UNKNOWN__": 0, "unknown": 1, "S1": 2}}
        accumulator = self.module.UnknownRateAccumulator(["site_no"])

        accumulator.update(pd.DataFrame({"site_no": ["S1", "S2"]}), maps)
        accumulator.update(pd.DataFrame({"site_no": [None]}), maps)

        self.assertEqual(maps["site_no"]["__UNKNOWN__"], 0)
        self.assertAlmostEqual(accumulator.compute()["site_no"], 1 / 3)

    def test_joint_output_contract_contains_required_fields(self):
        required = {
            "month", "site_no", "item_id",
            "target_qty_1m", "target_qty_2m",
            "lightgbm_v2_logl2_pred_1m", "lightgbm_v2_logl2_pred_2m",
            "two_stage_pred_1m", "two_stage_pred_2m",
            "target_mc_1m", "target_mc_2m",
            "regression_mapped_mc_1m", "regression_mapped_mc_2m",
            "direct_mc_pred_1m", "direct_mc_pred_2m",
        }
        self.assertTrue(required.issubset(self.module.JOINT_OUTPUT_COLUMNS))


class ActiveStoreParquetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_training_module()

    def test_row_group_reader_projects_columns_and_filters_horizon_availability(self):
        frame = pd.DataFrame(
            {
                "month": ["2025-01", "2025-02", "2025-03", "2025-04"],
                "site_no": ["S1"] * 4,
                "item_id": ["I1", "I2", "I3", "I4"],
                "split": ["train", "valid", "valid", "test"],
                "future_qty_2m": [1.0, 2.0, 3.0, 4.0],
                "target_available_2m": [1, 0, 1, 1],
                "unused_payload": ["x" * 100] * 4,
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.parquet"
            writer = pq.ParquetWriter(path, pa.Table.from_pandas(frame.iloc[:2], preserve_index=False).schema)
            try:
                writer.write_table(pa.Table.from_pandas(frame.iloc[:2], preserve_index=False))
                writer.write_table(pa.Table.from_pandas(frame.iloc[2:], preserve_index=False))
            finally:
                writer.close()

            batches = list(
                self.module.iter_horizon_row_groups(
                    path,
                    horizon="2m",
                    columns=["month", "site_no", "item_id", "split", "future_qty_2m"],
                    splits={"valid", "test"},
                )
            )

        result = pd.concat(batches, ignore_index=True)
        self.assertEqual(result["item_id"].tolist(), ["I3", "I4"])
        self.assertNotIn("unused_payload", result.columns)
        self.assertNotIn("target_available_2m", result.columns)

    def test_quantity_candidate_evaluation_includes_monthly_trend_macro_f1(self):
        class Pipeline:
            @staticmethod
            def prepare_feature_frame(frame, features, category_maps, time_base):
                return frame[features]

        class Booster:
            def predict(self, frame, num_iteration):
                return np.log1p(np.array([4.0, 2.0]))

        class Guard:
            @staticmethod
            def check(*args, **kwargs):
                return None

        frame = pd.DataFrame(
            {
                "month": ["2025-01", "2025-01"], "site_no": ["S1", "S1"],
                "item_id": ["I1", "I2"], "split": ["valid", "valid"],
                "total_qty": [1.0, 2.0], "future_qty_1m": [4.0, 1.0],
                "future_qty_2m": [8.0, 2.0],
                "target_available_2m": [1, 1],
            }
        )
        metadata = {
            "feature_names": ["total_qty"], "category_maps": {}, "time_base": "2023-01"
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.parquet"
            frame.to_parquet(path, index=False)
            rows = self.module.evaluate_quantity_candidates(
                Pipeline(), path, "2m", Booster(), metadata, [1], Guard()
            )

        self.assertIn("trend_macro_f1", rows[0])
        self.assertAlmostEqual(rows[0]["trend_macro_f1"], 2 / 3)

    def test_mc_candidate_evaluation_includes_high_recall_and_logloss(self):
        class Pipeline:
            @staticmethod
            def prepare_feature_frame(frame, features, category_maps, time_base):
                return frame[features]

        class Booster:
            def predict(self, frame, num_iteration):
                return np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])

        class Guard:
            @staticmethod
            def check(*args, **kwargs):
                return None

        config = self.module.MCLevelConfig(
            codes=(0, 1, 2), names=("none", "low", "high"), lower_bounds=(0, 1, 3)
        )
        frame = pd.DataFrame(
            {
                "month": ["2025-01", "2025-01"], "site_no": ["S1", "S1"],
                "item_id": ["I1", "I2"], "split": ["valid", "valid"],
                "total_qty": [0.0, 1.0], "future_qty_1m": [0.0, 3.0],
                "future_qty_2m": [0.0, 6.0],
                "target_mc_1m": [0, 2], "target_available_1m": [1, 1],
            }
        )
        metadata = {
            "feature_names": ["total_qty"], "category_maps": {}, "time_base": "2023-01"
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mc.parquet"
            frame.to_parquet(path, index=False)
            rows = self.module.evaluate_mc_candidates(
                Pipeline(), path, "1m", Booster(), metadata, [1], config, Guard()
            )

        self.assertEqual(rows[0]["high_level_recall"], 1.0)
        self.assertAlmostEqual(rows[0]["logloss"], -(np.log(0.8) + np.log(0.7)) / 2)

    def test_old_prediction_index_joins_by_key_despite_order_and_missing_rows(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            v2_path = directory / "v2.parquet"
            two_path = directory / "two.parquet"
            pd.DataFrame(
                {
                    "month": ["2025-01", "2025-01", "2025-01"],
                    "site_no": ["S1", "S1", "S1"], "item_id": ["I1", "I2", "I3"],
                    "lightgbm_v2_logl2_pred_1m": [1.0, 2.0, 3.0],
                    "lightgbm_v2_logl2_pred_2m": [2.0, 4.0, 6.0],
                }
            ).to_parquet(v2_path, index=False)
            pd.DataFrame(
                {
                    "month": ["2025-01", "2025-01"], "site_no": ["S1", "S1"],
                    "item_id": ["I3", "I1"], "two_stage_pred_1m": [30.0, 10.0],
                    "two_stage_pred_2m": [60.0, 20.0],
                }
            ).to_parquet(two_path, index=False)
            index = self.module.OldPredictionIndex(
                directory / "index.sqlite", v2_path=v2_path, two_path=two_path
            )
            try:
                index.build(logging.getLogger("test-old-index"))
                result = index.lookup(pd.DataFrame(
                    {"month": ["2025-01"] * 3, "site_no": ["S1"] * 3, "item_id": ["I1", "I2", "I3"]}
                ))
            finally:
                index.close()

        self.assertEqual(result["seq"].tolist(), [0, 2])
        self.assertEqual(result["old_two_1m"].tolist(), [10.0, 30.0])

    def test_old_prediction_index_rejects_duplicate_keys(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            v2_path = directory / "v2.parquet"
            two_path = directory / "two.parquet"
            pd.DataFrame(
                {
                    "month": ["2025-01", "2025-01"], "site_no": ["S1", "S1"],
                    "item_id": ["I1", "I1"], "lightgbm_v2_logl2_pred_1m": [1.0, 2.0],
                    "lightgbm_v2_logl2_pred_2m": [2.0, 4.0],
                }
            ).to_parquet(v2_path, index=False)
            pd.DataFrame(
                {
                    "month": ["2025-01"], "site_no": ["S1"], "item_id": ["I1"],
                    "two_stage_pred_1m": [1.0], "two_stage_pred_2m": [2.0],
                }
            ).to_parquet(two_path, index=False)
            index = self.module.OldPredictionIndex(
                directory / "index.sqlite", v2_path=v2_path, two_path=two_path
            )
            try:
                with self.assertRaisesRegex(ValueError, "Duplicate historical prediction key"):
                    index.build(logging.getLogger("test-old-index-duplicate"))
            finally:
                index.close()

    def test_validate_dataset_rejects_2m_available_when_1m_unavailable(self):
        frame = pd.DataFrame(
            {
                "month": ["2025-01"], "site_no": ["S1"], "item_id": ["I1"],
                "split": ["valid"], "total_qty": [1.0], "future_qty_1m": [1.0],
                "future_qty_2m": [2.0], "target_available_1m": [0], "target_available_2m": [1],
                "blt_site_no": ["B1"], "gds_ctgry_3_lvel": ["C3"],
                "gds_ctgry_4_lvel": ["C4"], "gds_ctgry_5_lvel": ["C5"],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.parquet"
            frame.to_parquet(path, index=False)
            with patch.object(self.module, "_horizon_features", return_value=[]):
                with self.assertRaisesRegex(ValueError, "2M eligibility requires 1M eligibility"):
                    self.module.validate_dataset(path, {"runtime_time_features": []})

    def test_validate_dataset_rejects_mc_field_inconsistent_with_quantity_on_train(self):
        frame = pd.DataFrame(
            {
                "month": ["2025-01"], "site_no": ["S1"], "item_id": ["I1"],
                "split": ["train"], "total_qty": [0.0], "future_qty_1m": [0.0],
                "future_qty_2m": [0.0], "target_available_1m": [1], "target_available_2m": [0],
                "target_mc_1m": [2], "target_mc_2m": [-1],
                "blt_site_no": ["B1"], "gds_ctgry_3_lvel": ["C3"],
                "gds_ctgry_4_lvel": ["C4"], "gds_ctgry_5_lvel": ["C5"],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_mc.parquet"
            frame.to_parquet(path, index=False)
            with patch.object(self.module, "_horizon_features", return_value=[]):
                with self.assertRaisesRegex(ValueError, "canonical quantity mapping"):
                    self.module.validate_dataset(path, {"runtime_time_features": []})

    def test_parquet_output_abort_allows_partial_writers_without_masking_error(self):
        with TemporaryDirectory() as directory:
            outputs = self.module.ParquetOutputSet(
                {"written": Path(directory) / "written.parquet", "missing": Path(directory) / "missing.parquet"}
            )
            outputs.write("written", pd.DataFrame({"month": ["2025-01"], "site_no": ["S1"], "value": [1]}))

            try:
                raise RuntimeError("original failure")
            except RuntimeError as original:
                outputs.abort()
                captured = original

        self.assertEqual(str(captured), "original failure")


if __name__ == "__main__":
    unittest.main()
