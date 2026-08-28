import unittest

import numpy as np
import pandas as pd

from src.data.aggregate_sales import build_daily_sales, build_monthly_sales
from src.data.clean_sales import clean_sales_chunk, map_channel_type
from src.data.model_dataset import build_site_model_dataset


class DataPreparationTests(unittest.TestCase):
    def test_item_id_prefers_isbn_and_falls_back_to_gds_no(self):
        raw = pd.DataFrame(
            {
                "period": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "site_no": ["S1", "S1", "S2"],
                "isbn": ["9780001", "", None],
                "gds_no": ["G1", "G2", ""],
                "oln_or_ofln": ["ofln", "oln", "x"],
                "qty": ["1", "2", "3"],
                "price": ["10", "20", "30"],
                "tlp": ["10", "40", "90"],
                "tsp": ["8", "35", "70"],
            }
        )

        cleaned, stats = clean_sales_chunk(raw)

        self.assertEqual(cleaned["item_id"].tolist(), ["9780001", "G2"])
        self.assertEqual(stats["dropped_missing_item_id"], 1)

    def test_channel_type_mapping(self):
        self.assertEqual(map_channel_type("ofln"), "offline")
        self.assertEqual(map_channel_type("offline"), "offline")
        self.assertEqual(map_channel_type("线下"), "offline")
        self.assertEqual(map_channel_type("oln"), "online")
        self.assertEqual(map_channel_type("online"), "online")
        self.assertEqual(map_channel_type("线上"), "online")
        self.assertEqual(map_channel_type("something_else"), "unknown")
        self.assertEqual(map_channel_type(None), "unknown")

    def test_daily_aggregation_sums_net_and_channel_quantities(self):
        cleaned = pd.DataFrame(
            {
                "period": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
                "site_no": ["S1", "S1", "S1"],
                "blt_site_no": ["B1", "B1", "B1"],
                "item_id": ["I1", "I1", "I1"],
                "isbn": ["I1", "I1", "I1"],
                "gds_no": ["G1", "G1", "G1"],
                "gds_ctgry_3_lvel": ["C3", "C3", "C3"],
                "gds_ctgry_4_lvel": ["C4", "C4", "C4"],
                "gds_ctgry_5_lvel": ["C5", "C5", "C5"],
                "price": [10.0, 10.0, 10.0],
                "qty": [3.0, -1.0, 2.0],
                "tlp": [30.0, -10.0, 20.0],
                "tsp": [24.0, -8.0, 18.0],
                "channel_type": ["offline", "offline", "online"],
                "is_return": [False, True, False],
                "return_qty_component": [0.0, 1.0, 0.0],
            }
        )

        daily = build_daily_sales(cleaned)

        row = daily.iloc[0]
        self.assertEqual(len(daily), 1)
        self.assertEqual(row["total_qty"], 4.0)
        self.assertEqual(row["offline_qty"], 2.0)
        self.assertEqual(row["online_qty"], 2.0)
        self.assertEqual(row["sales_count"], 3)
        self.assertEqual(row["return_count"], 1)
        self.assertEqual(row["return_qty"], 1.0)

    def test_monthly_aggregation_derives_prices_without_divide_by_zero(self):
        daily = pd.DataFrame(
            {
                "period": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "site_no": ["S1", "S1", "S2"],
                "blt_site_no": ["B1", "B2", "B3"],
                "item_id": ["I1", "I1", "I2"],
                "isbn": ["I1", "I1", "I2"],
                "gds_no": ["G1", "G1", "G2"],
                "gds_ctgry_3_lvel": ["C3", "C3b", "D3"],
                "gds_ctgry_4_lvel": ["C4", "C4b", "D4"],
                "gds_ctgry_5_lvel": ["C5", "C5b", "D5"],
                "price": [10.0, 11.0, 20.0],
                "total_qty": [3.0, 2.0, 0.0],
                "offline_qty": [3.0, 0.0, 0.0],
                "online_qty": [0.0, 2.0, 0.0],
                "unknown_channel_qty": [0.0, 0.0, 0.0],
                "total_tlp": [30.0, 20.0, 0.0],
                "total_tsp": [24.0, 18.0, 5.0],
                "sales_count": [1, 1, 1],
                "return_count": [0, 0, 0],
                "return_qty": [0.0, 0.0, 0.0],
            }
        )

        monthly = build_monthly_sales(daily)
        i1 = monthly[monthly["item_id"] == "I1"].iloc[0]
        i2 = monthly[monthly["item_id"] == "I2"].iloc[0]

        self.assertEqual(i1["month"], "2026-01")
        self.assertEqual(i1["sales_days"], 2)
        self.assertEqual(i1["total_qty"], 5.0)
        self.assertAlmostEqual(i1["avg_real_price"], 42.0 / 5.0)
        self.assertAlmostEqual(i1["discount_rate"], 42.0 / 50.0)
        self.assertTrue(np.isnan(i2["avg_real_price"]))
        self.assertTrue(np.isnan(i2["discount_rate"]))

    def test_model_dataset_uses_only_prior_months_for_features(self):
        monthly = pd.DataFrame(
            {
                "month": ["2023-01", "2023-03", "2023-04"],
                "site_no": ["S1", "S1", "S1"],
                "blt_site_no": ["B1", "B1", "B1"],
                "item_id": ["I1", "I1", "I1"],
                "isbn": ["I1", "I1", "I1"],
                "gds_no": ["G1", "G1", "G1"],
                "gds_ctgry_3_lvel": ["C3", "C3", "C3"],
                "gds_ctgry_4_lvel": ["C4", "C4", "C4"],
                "gds_ctgry_5_lvel": ["C5", "C5", "C5"],
                "price": [10.0, 10.0, 10.0],
                "total_qty": [2.0, 5.0, 7.0],
                "offline_qty": [2.0, 5.0, 7.0],
                "online_qty": [0.0, 0.0, 0.0],
                "unknown_channel_qty": [0.0, 0.0, 0.0],
                "total_tlp": [20.0, 50.0, 70.0],
                "total_tsp": [18.0, 45.0, 63.0],
                "avg_real_price": [9.0, 9.0, 9.0],
                "discount_rate": [0.9, 0.9, 0.9],
                "sales_days": [1, 1, 1],
                "sales_count": [1, 1, 1],
                "return_count": [0, 0, 0],
                "return_qty": [0.0, 0.0, 0.0],
            }
        )

        dataset, panel_rows = build_site_model_dataset(monthly, max_month_ord=monthly["month"].map(lambda m: pd.Period(m, freq="M").ordinal).max())

        self.assertEqual(panel_rows, 4)
        jan = dataset[dataset["month"] == "2023-01"].iloc[0]
        feb = dataset[dataset["month"] == "2023-02"].iloc[0]
        self.assertEqual(jan["qty_lag_1m"], 0)
        self.assertEqual(jan["qty_sum_last_3m"], 0)
        self.assertEqual(jan["future_qty_1m"], 0)
        self.assertEqual(jan["future_qty_2m"], 5)
        self.assertEqual(feb["qty_lag_1m"], 2)
        self.assertEqual(feb["qty_sum_last_3m"], 2)
        self.assertEqual(feb["future_qty_1m"], 5)
        self.assertEqual(feb["future_qty_2m"], 12)
        self.assertEqual(feb["zero_sales_months_last_3m"], 0)

    def test_model_dataset_returns_empty_when_future_two_month_target_unavailable(self):
        monthly = pd.DataFrame(
            {
                "month": ["2026-06"],
                "site_no": ["S1"],
                "blt_site_no": ["B1"],
                "item_id": ["I1"],
                "isbn": ["I1"],
                "gds_no": ["G1"],
                "gds_ctgry_3_lvel": ["C3"],
                "gds_ctgry_4_lvel": ["C4"],
                "gds_ctgry_5_lvel": ["C5"],
                "price": [10.0],
                "total_qty": [2.0],
                "offline_qty": [2.0],
                "online_qty": [0.0],
                "unknown_channel_qty": [0.0],
                "total_tlp": [20.0],
                "total_tsp": [18.0],
                "avg_real_price": [9.0],
                "discount_rate": [0.9],
                "sales_days": [1],
                "sales_count": [1],
                "return_count": [0],
                "return_qty": [0.0],
            }
        )

        dataset, panel_rows = build_site_model_dataset(
            monthly,
            max_month_ord=pd.Period("2026-06", freq="M").ordinal,
        )

        self.assertEqual(panel_rows, 1)
        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
