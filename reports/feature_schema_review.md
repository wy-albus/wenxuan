# Feature Schema Review

## Dataset Overview

- Dataset: `D:/codex_wenxuan/data/processed/model_dataset_monthly.parquet`
- Total rows: 89529779
- Field count: 66

## All Fields

| field                     | dtype   |   missing_count | missing_rate   |   unique_values |
|:--------------------------|:--------|----------------:|:---------------|----------------:|
| month                     | string  |               0 | 0.0000%        |              40 |
| site_no                   | string  |               0 | 0.0000%        |             230 |
| blt_site_no               | string  |               0 | 0.0000%        |               2 |
| item_id                   | string  |               0 | 0.0000%        |          351134 |
| isbn                      | string  |               0 | 0.0000%        |          351134 |
| gds_no                    | string  |               0 | 0.0000%        |          390496 |
| gds_ctgry_3_lvel          | string  |               0 | 0.0000%        |              17 |
| gds_ctgry_4_lvel          | string  |               0 | 0.0000%        |              18 |
| gds_ctgry_5_lvel          | string  |               0 | 0.0000%        |             553 |
| price                     | float   |               0 | 0.0000%        |            2435 |
| total_qty                 | float   |               0 | 0.0000%        |             760 |
| offline_qty               | float   |               0 | 0.0000%        |             715 |
| online_qty                | float   |               0 | 0.0000%        |             386 |
| unknown_channel_qty       | float   |               0 | 0.0000%        |               1 |
| total_tlp                 | float   |               0 | 0.0000%        |           15620 |
| total_tsp                 | float   |               0 | 0.0000%        |           88183 |
| avg_real_price            | float   |               0 | 0.0000%        |          129758 |
| discount_rate             | float   |               0 | 0.0000%        |          197568 |
| sales_days                | int32   |               0 | 0.0000%        |              32 |
| sales_count               | int32   |               0 | 0.0000%        |             433 |
| return_count              | int32   |               0 | 0.0000%        |              29 |
| return_qty                | float   |               0 | 0.0000%        |             129 |
| qty_lag_1m                | float   |               0 | 0.0000%        |             756 |
| qty_lag_2m                | float   |               0 | 0.0000%        |             751 |
| qty_lag_3m                | float   |               0 | 0.0000%        |             748 |
| qty_lag_6m                | float   |               0 | 0.0000%        |             731 |
| offline_qty_lag_1m        | float   |               0 | 0.0000%        |             714 |
| online_qty_lag_1m         | float   |               0 | 0.0000%        |             385 |
| qty_sum_last_3m           | float   |               0 | 0.0000%        |            1116 |
| qty_mean_last_3m          | float   |               0 | 0.0000%        |            1505 |
| qty_max_last_3m           | float   |               0 | 0.0000%        |             725 |
| qty_std_last_3m           | float   |               0 | 0.0000%        |            6387 |
| qty_sum_last_6m           | float   |               0 | 0.0000%        |            1389 |
| qty_mean_last_6m          | float   |               0 | 0.0000%        |            2551 |
| qty_max_last_6m           | float   |               0 | 0.0000%        |             722 |
| qty_std_last_6m           | float   |               0 | 0.0000%        |           31492 |
| offline_qty_sum_last_3m   | float   |               0 | 0.0000%        |            1049 |
| online_qty_sum_last_3m    | float   |               0 | 0.0000%        |             503 |
| offline_qty_sum_last_6m   | float   |               0 | 0.0000%        |            1338 |
| online_qty_sum_last_6m    | float   |               0 | 0.0000%        |             593 |
| offline_ratio_last_3m     | float   |               0 | 0.0000%        |            5010 |
| online_ratio_last_3m      | float   |               0 | 0.0000%        |            5010 |
| offline_ratio_last_6m     | float   |               0 | 0.0000%        |            7652 |
| online_ratio_last_6m      | float   |               0 | 0.0000%        |            7652 |
| sales_days_lag_1m         | float   |               0 | 0.0000%        |              32 |
| sales_days_sum_last_3m    | float   |               0 | 0.0000%        |              91 |
| sales_days_sum_last_6m    | float   |               0 | 0.0000%        |             173 |
| active_months_last_3m     | int32   |               0 | 0.0000%        |               4 |
| active_months_last_6m     | int32   |               0 | 0.0000%        |               7 |
| zero_sales_months_last_3m | int32   |               0 | 0.0000%        |               4 |
| zero_sales_months_last_6m | int32   |               0 | 0.0000%        |               7 |
| is_sold_last_1m           | int32   |               0 | 0.0000%        |               2 |
| is_sold_last_3m           | int32   |               0 | 0.0000%        |               2 |
| is_sold_last_6m           | int32   |               0 | 0.0000%        |               2 |
| months_since_last_sale    | int32   |               0 | 0.0000%        |              40 |
| tsp_sum_last_3m           | float   |               0 | 0.0000%        |          171967 |
| tlp_sum_last_3m           | float   |               0 | 0.0000%        |           25122 |
| discount_rate_last_3m     | float   |               0 | 0.0000%        |          470750 |
| tsp_sum_last_6m           | float   |               0 | 0.0000%        |          249601 |
| tlp_sum_last_6m           | float   |               0 | 0.0000%        |           34472 |
| discount_rate_last_6m     | float   |               0 | 0.0000%        |          761531 |
| future_qty_1m             | float   |               0 | 0.0000%        |             695 |
| future_qty_2m             | float   |               0 | 0.0000%        |             878 |
| future_has_sales_1m       | int32   |               0 | 0.0000%        |               2 |
| future_has_sales_2m       | int32   |               0 | 0.0000%        |               2 |
| split                     | string  |               0 | 0.0000%        |               3 |

## Numeric Field Statistics

| field                     |              min |         mean |            std |              max |
|:--------------------------|-----------------:|-------------:|---------------:|-----------------:|
| price                     |      0.01        | 48.6097      |   62.589       |  39000           |
| total_qty                 |  -2650           |  0.255446    |    2.92767     |   3770           |
| offline_qty               |  -2650           |  0.224591    |    2.69295     |   3770           |
| online_qty                |    -50           |  0.0308552   |    1.08551     |   2020           |
| unknown_channel_qty       |      0           |  0           |    0           |      0           |
| total_tlp                 | -53000           | 11.5033      |  355.199       |      1.6065e+06  |
| total_tsp                 | -32860           |  8.98591     |  137.836       | 438829           |
| avg_real_price            |    -64.1         |  4.64407     |   17.0679      |  25350           |
| discount_rate             |     -5.7         |  0.109953    |    0.298533    |      3.81        |
| sales_days                |      0           |  0.201598    |    0.757176    |     31           |
| sales_count               |      0           |  0.237446    |    1.58998     |   2324           |
| return_count              |      0           |  0.00225328  |    0.0550629   |     64           |
| return_qty                |      0           |  0.00319651  |    0.764198    |   2650           |
| qty_lag_1m                |  -2650           |  0.252126    |    2.91241     |   3770           |
| qty_lag_2m                |  -2650           |  0.246098    |    2.88756     |   3770           |
| qty_lag_3m                |  -2650           |  0.24052     |    2.87572     |   3770           |
| qty_lag_6m                |  -2650           |  0.226396    |    2.80483     |   3770           |
| offline_qty_lag_1m        |  -2650           |  0.221853    |    2.68378     |   3770           |
| online_qty_lag_1m         |    -50           |  0.0302726   |    1.06707     |   2020           |
| qty_sum_last_3m           |  -2649           |  0.738744    |    5.54824     |   3781           |
| qty_mean_last_3m          |   -883           |  0.303334    |    2.43815     |   3770           |
| qty_max_last_3m           |   -380           |  0.571649    |    4.83477     |   3770           |
| qty_std_last_3m           |      0           |  0.26474     |    2.64655     |   3747.67        |
| qty_sum_last_6m           |  -2649           |  1.42983     |    8.37626     |   3985           |
| qty_mean_last_6m          |   -441.5         |  0.348214    |    2.33832     |   3770           |
| qty_max_last_6m           |   -380           |  0.939608    |    6.66311     |   3770           |
| qty_std_last_6m           |      0           |  0.385715    |    2.96418     |   3747.67        |
| offline_qty_sum_last_3m   |  -2649           |  0.650633    |    5.12283     |   3781           |
| online_qty_sum_last_3m    |    -50           |  0.0881106   |    1.9647      |   2179           |
| offline_qty_sum_last_6m   |  -2649           |  1.26124     |    7.74076     |   3985           |
| online_qty_sum_last_6m    |    -50           |  0.168593    |    2.92528     |   2450           |
| offline_ratio_last_3m     |     -3.5         |  0.234383    |    0.422077    |      4.4         |
| online_ratio_last_3m      |     -3.4         |  0.0360841   |    0.182985    |      4.5         |
| offline_ratio_last_6m     |     -3.5         |  0.359574    |    0.477288    |     11           |
| online_ratio_last_6m      |    -10           |  0.063321    |    0.238401    |      4.5         |
| sales_days_lag_1m         |      0           |  0.198853    |    0.75388     |     31           |
| sales_days_sum_last_3m    |      0           |  0.582901    |    1.74787     |     90           |
| sales_days_sum_last_6m    |      0           |  1.12834     |    2.92683     |    174           |
| active_months_last_3m     |      0           |  0.362273    |    0.673657    |      3           |
| active_months_last_6m     |      0           |  0.700968    |    1.07814     |      6           |
| zero_sales_months_last_3m |      0           |  2.40811     |    1.0069      |      3           |
| zero_sales_months_last_6m |      0           |  4.5063      |    2.06314     |      6           |
| is_sold_last_1m           |      0           |  0.123448    |    0.328951    |      1           |
| is_sold_last_3m           |      0           |  0.270431    |    0.444182    |      1           |
| is_sold_last_6m           |      0           |  0.422927    |    0.494024    |      1           |
| months_since_last_sale    |     -1           | 10.1958      |    9.21664     |     39           |
| tsp_sum_last_3m           | -32847.6         | 26.0003      |  278.808       | 493429           |
| tlp_sum_last_3m           | -52980           | 33.2704      |  706.785       |      1.80642e+06 |
| discount_rate_last_3m     |     -5.43387e+15 |  1.21709e+08 |    8.43902e+11 |      2.6938e+15  |
| tsp_sum_last_6m           | -32847.6         | 50.3347      |  439.836       | 665139           |
| tlp_sum_last_6m           | -52980           | 64.3623      | 1178.31        |      2.43474e+06 |
| discount_rate_last_6m     |     -4.63721e+15 |  7.01409e+08 |    3.38817e+12 |      2.17355e+16 |
| future_qty_1m             |  -2650           |  0.192956    |    2.49286     |   3489           |
| future_qty_2m             |  -2650           |  0.368448    |    3.72772     |   3748           |
| future_has_sales_1m       |      0           |  0.088732    |    0.284357    |      1           |
| future_has_sales_2m       |      0           |  0.137308    |    0.344172    |      1           |

## Field Classification

### Identifier Fields
- `month`
- `site_no`
- `item_id`
- `isbn`
- `gds_no`
- `blt_site_no`

### Numeric Feature Candidates
- `price`
- `total_qty`
- `offline_qty`
- `online_qty`
- `unknown_channel_qty`
- `total_tlp`
- `total_tsp`
- `avg_real_price`
- `discount_rate`
- `sales_days`
- `sales_count`
- `return_count`
- `return_qty`
- `qty_lag_1m`
- `qty_lag_2m`
- `qty_lag_3m`
- `qty_lag_6m`
- `offline_qty_lag_1m`
- `online_qty_lag_1m`
- `qty_sum_last_3m`
- `qty_mean_last_3m`
- `qty_max_last_3m`
- `qty_std_last_3m`
- `qty_sum_last_6m`
- `qty_mean_last_6m`
- `qty_max_last_6m`
- `qty_std_last_6m`
- `offline_qty_sum_last_3m`
- `online_qty_sum_last_3m`
- `offline_qty_sum_last_6m`
- `online_qty_sum_last_6m`
- `offline_ratio_last_3m`
- `online_ratio_last_3m`
- `offline_ratio_last_6m`
- `online_ratio_last_6m`
- `sales_days_lag_1m`
- `sales_days_sum_last_3m`
- `sales_days_sum_last_6m`
- `active_months_last_3m`
- `active_months_last_6m`
- `zero_sales_months_last_3m`
- `zero_sales_months_last_6m`
- `is_sold_last_1m`
- `is_sold_last_3m`
- `is_sold_last_6m`
- `months_since_last_sale`
- `tsp_sum_last_3m`
- `tlp_sum_last_3m`
- `discount_rate_last_3m`
- `tsp_sum_last_6m`
- `tlp_sum_last_6m`
- `discount_rate_last_6m`

### Categorical Feature Candidates
- `site_no`
- `blt_site_no`
- `gds_ctgry_3_lvel`
- `gds_ctgry_4_lvel`
- `gds_ctgry_5_lvel`

### Target Fields
- `future_qty_1m`
- `future_qty_2m`
- `future_has_sales_1m`
- `future_has_sales_2m`

### Split Fields
- `split`

### Temporarily Excluded High-cardinality Fields
- `item_id`
- `isbn`
- `gds_no`

### Potential Leakage Fields
- `future_qty_1m`
- `future_qty_2m`
- `future_has_sales_1m`
- `future_has_sales_2m`

## Fields Not Recommended As Inputs

| field/group | reason |
|---|---|
| `future_* target fields` | 目标变量，包含未来销量或未来是否有销量，作为输入会直接数据泄露。 |
| `split` | 只能用于 train / valid / test 时间切分，不能作为模型输入。 |
| `item_id / isbn / gds_no` | 高基数商品 ID，第一版先不直接输入，避免模型记忆商品编号；后续可考虑频率编码、目标编码或 embedding。 |
| `month` | 样本时间标识，第一版只用于时间切分和回溯审计，避免模型记忆绝对月份。 |

## Recommended Feature Lists

### baseline_features (5)
- `qty_lag_1m`
- `qty_lag_2m`
- `qty_lag_3m`
- `qty_mean_last_3m`
- `qty_sum_last_3m`

### lightgbm_features (57)
- `price`
- `total_qty`
- `offline_qty`
- `online_qty`
- `unknown_channel_qty`
- `total_tlp`
- `total_tsp`
- `avg_real_price`
- `discount_rate`
- `sales_days`
- `sales_count`
- `return_count`
- `return_qty`
- `qty_lag_1m`
- `qty_lag_2m`
- `qty_lag_3m`
- `qty_lag_6m`
- `offline_qty_lag_1m`
- `online_qty_lag_1m`
- `qty_sum_last_3m`
- `qty_mean_last_3m`
- `qty_max_last_3m`
- `qty_std_last_3m`
- `qty_sum_last_6m`
- `qty_mean_last_6m`
- `qty_max_last_6m`
- `qty_std_last_6m`
- `offline_qty_sum_last_3m`
- `online_qty_sum_last_3m`
- `offline_qty_sum_last_6m`
- `online_qty_sum_last_6m`
- `offline_ratio_last_3m`
- `online_ratio_last_3m`
- `offline_ratio_last_6m`
- `online_ratio_last_6m`
- `sales_days_lag_1m`
- `sales_days_sum_last_3m`
- `sales_days_sum_last_6m`
- `active_months_last_3m`
- `active_months_last_6m`
- `zero_sales_months_last_3m`
- `zero_sales_months_last_6m`
- `is_sold_last_1m`
- `is_sold_last_3m`
- `is_sold_last_6m`
- `months_since_last_sale`
- `tsp_sum_last_3m`
- `tlp_sum_last_3m`
- `discount_rate_last_3m`
- `tsp_sum_last_6m`
- `tlp_sum_last_6m`
- `discount_rate_last_6m`
- `site_no`
- `blt_site_no`
- `gds_ctgry_3_lvel`
- `gds_ctgry_4_lvel`
- `gds_ctgry_5_lvel`

### mlp_features (52)
- `price`
- `total_qty`
- `offline_qty`
- `online_qty`
- `unknown_channel_qty`
- `total_tlp`
- `total_tsp`
- `avg_real_price`
- `discount_rate`
- `sales_days`
- `sales_count`
- `return_count`
- `return_qty`
- `qty_lag_1m`
- `qty_lag_2m`
- `qty_lag_3m`
- `qty_lag_6m`
- `offline_qty_lag_1m`
- `online_qty_lag_1m`
- `qty_sum_last_3m`
- `qty_mean_last_3m`
- `qty_max_last_3m`
- `qty_std_last_3m`
- `qty_sum_last_6m`
- `qty_mean_last_6m`
- `qty_max_last_6m`
- `qty_std_last_6m`
- `offline_qty_sum_last_3m`
- `online_qty_sum_last_3m`
- `offline_qty_sum_last_6m`
- `online_qty_sum_last_6m`
- `offline_ratio_last_3m`
- `online_ratio_last_3m`
- `offline_ratio_last_6m`
- `online_ratio_last_6m`
- `sales_days_lag_1m`
- `sales_days_sum_last_3m`
- `sales_days_sum_last_6m`
- `active_months_last_3m`
- `active_months_last_6m`
- `zero_sales_months_last_3m`
- `zero_sales_months_last_6m`
- `is_sold_last_1m`
- `is_sold_last_3m`
- `is_sold_last_6m`
- `months_since_last_sale`
- `tsp_sum_last_3m`
- `tlp_sum_last_3m`
- `discount_rate_last_3m`
- `tsp_sum_last_6m`
- `tlp_sum_last_6m`
- `discount_rate_last_6m`

### two_stage_features (57)
- `price`
- `total_qty`
- `offline_qty`
- `online_qty`
- `unknown_channel_qty`
- `total_tlp`
- `total_tsp`
- `avg_real_price`
- `discount_rate`
- `sales_days`
- `sales_count`
- `return_count`
- `return_qty`
- `qty_lag_1m`
- `qty_lag_2m`
- `qty_lag_3m`
- `qty_lag_6m`
- `offline_qty_lag_1m`
- `online_qty_lag_1m`
- `qty_sum_last_3m`
- `qty_mean_last_3m`
- `qty_max_last_3m`
- `qty_std_last_3m`
- `qty_sum_last_6m`
- `qty_mean_last_6m`
- `qty_max_last_6m`
- `qty_std_last_6m`
- `offline_qty_sum_last_3m`
- `online_qty_sum_last_3m`
- `offline_qty_sum_last_6m`
- `online_qty_sum_last_6m`
- `offline_ratio_last_3m`
- `online_ratio_last_3m`
- `offline_ratio_last_6m`
- `online_ratio_last_6m`
- `sales_days_lag_1m`
- `sales_days_sum_last_3m`
- `sales_days_sum_last_6m`
- `active_months_last_3m`
- `active_months_last_6m`
- `zero_sales_months_last_3m`
- `zero_sales_months_last_6m`
- `is_sold_last_1m`
- `is_sold_last_3m`
- `is_sold_last_6m`
- `months_since_last_sale`
- `tsp_sum_last_3m`
- `tlp_sum_last_3m`
- `discount_rate_last_3m`
- `tsp_sum_last_6m`
- `tlp_sum_last_6m`
- `discount_rate_last_6m`
- `site_no`
- `blt_site_no`
- `gds_ctgry_3_lvel`
- `gds_ctgry_4_lvel`
- `gds_ctgry_5_lvel`

## Two-stage Model Target Advice

- Stage 1 classification targets: `future_has_sales_1m`, `future_has_sales_2m`.
- Stage 2 regression targets should be non-negative demand targets.
- Current dataset does not contain `target_qty_1m` or `target_qty_2m`; create them during training as:
  - `target_qty_1m = max(future_qty_1m, 0)`
  - `target_qty_2m = max(future_qty_2m, 0)`
- Reason: negative net sales are useful history, but should not become negative replenishment demand.

## Leakage Review

- Detected leakage-risk fields by prefix: `future_qty_1m`, `future_qty_2m`, `future_has_sales_1m`, `future_has_sales_2m`
- These fields are all target fields and are excluded from every feature list.
- `split` is excluded from every feature list.
- `item_id`, `isbn`, and `gds_no` are excluded from first-version model features.

## Next-stage Modeling Suggestions

- First run historical mean and weighted moving average baselines using `baseline_features` and the target fields only for evaluation.
- For LightGBM / RandomForest, use `lightgbm_features`; treat category fields as categorical or encode them explicitly.
- For MLP, start with `mlp_features`; add category embeddings only after numeric-feature baseline is stable.
- Evaluate both raw target and clipped demand target for replenishment scenarios, especially because `future_qty_*` contains negative net-sales values from returns.
