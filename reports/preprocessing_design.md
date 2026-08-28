# Preprocessing Design

## One Public Parquet

The project will continue to use a single shared dataset:

```text
data/processed/model_dataset_monthly.parquet
```

There is no `model_dataset_v1.parquet`. The file has about 89.5 million rows, so copying it for every modeling variant would waste disk and create version drift. Parquet supports column projection, so each training script should call `pd.read_parquet(..., columns=required_columns)` and load only the feature, target, split, and month columns needed by that model.

## Feature Counts

| model | config key | feature count |
|---|---|---:|
| Historical mean | `historical_mean_features` | 5 |
| Weighted moving average | `weighted_moving_average_features` | 5 |
| Random forest | `random_forest_features` | 49 |
| LightGBM | `lightgbm_features` | 49 |
| MLP | `mlp_features` | 44 |
| Two-stage model | `two_stage_features` | 49 |

## Excluded Fields

The following fields are excluded from every model feature list because they are targets, split metadata, or high-cardinality product identifiers:

```text
future_qty_1m
future_qty_2m
future_has_sales_1m
future_has_sales_2m
split
item_id
isbn
gds_no
```

The first version also excludes these anomalous or currently low-value fields:

```text
unknown_channel_qty
discount_rate
discount_rate_last_3m
discount_rate_last_6m
offline_ratio_last_3m
online_ratio_last_3m
offline_ratio_last_6m
online_ratio_last_6m
```

These can be revisited after baseline training and diagnostics.

## Runtime Columns

`src/features/preprocessing.py` creates the following columns at runtime only:

```text
target_qty_1m = max(future_qty_1m, 0)
target_qty_2m = max(future_qty_2m, 0)
year
month_of_year
quarter
time_index
```

The runtime target columns are for replenishment-oriented training and evaluation. They are not written back to `model_dataset_monthly.parquet`.

## Model-specific Preprocessing

Historical mean and weighted moving average use only the small lag/rolling quantity feature lists. Training code can use these fields to compute rule-based predictions without loading category columns.

Random forest, LightGBM, and the two-stage model use numeric history, rolling, frequency, long-tail, amount, store, and category features. Their category fields are:

```text
site_no
blt_site_no
gds_ctgry_3_lvel
gds_ctgry_4_lvel
gds_ctgry_5_lvel
```

`preprocessing.py` fits integer encodings on the train split only and maps unseen validation/test categories to `-1`.

MLP uses numeric features only in v1. `preprocessing.py` fills missing numeric values with `0` and standardizes numeric features using train-split mean and standard deviation. Category fields are intentionally excluded until an embedding or one-hot strategy is designed.

All model paths replace positive and negative infinity with `NaN`, then apply consistent missing-value handling.

## Leakage Prevention

Feature lists are loaded from `config/model_features.yaml`, which explicitly excludes all `future_*` target fields, `split`, and high-cardinality product IDs. Runtime preprocessing reads target columns only to build `target_qty_1m` and `target_qty_2m`; it does not add those targets to feature matrices.

Scaling and category encodings are fit using the `train` split only, then applied to validation and test rows. This avoids learning category vocabularies or feature statistics from future periods.
