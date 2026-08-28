# Active-Store Model Dataset Report

- Input monthly rows: 11,619,660
- Active-store panel rows before target eligibility: 95,516,713
- Final unified rows (1M target available): 88,726,470
- Rows with complete 2M target: 88,512,677
- Fields: 72
- Output range: 2023-01 to 2026-04
- Output size: 1.412 GiB
- Build elapsed: 6.9 minutes

## Structural-Zero Correction

- Store activity boundaries and horizon eligibility are recomputed from `monthly_item_store_sales.parquet`; leakage-safe lag/rolling columns are streamed from the existing monthly-derived model table because retained pre-closure rows are mathematically unchanged.
- Old model rows: 89,529,779
- Old rows after store closure: 602,499
- Old zero-current-sales ratio: 87.46%
- Active-store zero-current-sales ratio: 87.35%
- Difference between old and active-store unified row count: 803,309

## Horizon Eligibility

- 1M rows require month + 1 <= store_last_active_month.
- 2M rows require month + 2 <= store_last_active_month.
- Rows with a valid 1M target but unavailable 2M target remain in the common Parquet and are excluded by the 2M trainer using target_available_2m.
- A store closing during the forecast horizon is never converted into a zero-sales target.

## Split Counts

| split | Old dataset | Active-store 1M eligible | Change vs old | Active-store 2M eligible |
|---|---:|---:|---:|---:|
| train | 57,178,758 | 57,163,300 | -15,458 | 57,161,638 |
| valid | 18,789,982 | 18,563,573 | -226,409 | 18,491,375 |
| test | 13,561,039 | 12,999,597 | -561,442 | 12,859,664 |
| total | 89,529,779 | 88,726,470 | -803,309 | 88,512,677 |

The 602,499 old rows strictly after a store's last active month are structural-zero removals. The larger 803,309 total reduction also includes rows removed to keep the complete 1M forecast horizon inside each store's active interval.

## Leakage and Quality Checks

- Lag and rolling features use shifted historical quantities only.
- MC targets are derived only from matching future quantity targets and are excluded from all input feature lists.
- Duplicate keys: 0
- Rows after store closure: 0
- Invalid 1M horizon rows: 0
- Invalid 2M horizon rows marked available: 0
- Existing model_dataset_monthly.parquet was not modified.
