# Store Activity Audit

## Scope And Policy

- Configured abnormal stores: 28
- Configuration: `config/active_store_closures.csv`
- `closure_month` is the first month in which the store is treated as closed.
- Only the configured 28 stores receive closure filtering; all other stores follow the original pipeline.
- Rows at or after `closure_month` are not valid store-item observations.
- A 1M/2M sample is invalid when its forecast horizon reaches or crosses `closure_month`; it is not relabeled as zero demand.
- No active-store count, lifecycle embedding, or cross-store aggregate is added as a model feature.

## Exact Row Reconciliation

| Deletion reason | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| After-closure structural zeros | 14,160 | 157,577 | 430,762 | 602,499 |
| 1M crosses closure boundary | 1,298 | 68,832 | 130,680 | 200,810 |
| 1M total excluded | 15,458 | 226,409 | 561,442 | 803,309 |
| Additional 2M boundary rows | 1,662 | 72,198 | 139,933 | 213,793 |
| 2M total excluded (separate sample set) | 17,120 | 298,607 | 701,375 | 1,017,102 |
| Other reasons in 1M reconciliation | 0 | 0 | 0 | 0 |

- 1M reconciliation: 89,529,779 - 803,309 = 88,726,470.
- 2M reconciliation: 89,529,779 - 1,017,102 = 88,512,677.
- The reported 803,309-row reduction is exactly 602,499 structural-zero rows plus 200,810 1M boundary-crossing rows.
- The 213,793 additional rows apply only to the 2M eligible set and are not forced into the 803,309 1M total.

## Split Comparison

| Split | Old | Active 1M | Change | Active 2M | 2M change |
|---|---:|---:|---:|---:|---:|
| train | 57,178,758 | 57,163,300 | -15,458 | 57,161,638 | -17,120 |
| valid | 18,789,982 | 18,563,573 | -226,409 | 18,491,375 | -298,607 |
| test | 13,561,039 | 12,999,597 | -561,442 | 12,859,664 | -701,375 |
| total | 89,529,779 | 88,726,470 | -803,309 | 88,512,677 | -1,017,102 |

## Zero-Sales Ratio

- Old dataset: 78,303,588 / 89,529,779 = 87.4609%.
- Active-store 1M dataset: 77,502,305 / 88,726,470 = 87.3497%.
- Change: -0.1112 percentage points.

## Closure Configuration

| site_no | last_active_month | closure_month |
|---|---|---|
| 8203 | 2023-03 | 2023-04 |
| 8009 | 2023-09 | 2023-10 |
| 6201 | 2024-06 | 2024-07 |
| 8202 | 2024-11 | 2024-12 |
| 6812 | 2025-07 | 2025-08 |
| 5001 | 2025-09 | 2025-10 |
| 6109 | 2025-09 | 2025-10 |
| 6826 | 2025-09 | 2025-10 |
| 8204 | 2025-10 | 2025-11 |
| 6308 | 2025-12 | 2026-01 |
| 6126 | 2026-01 | 2026-02 |
| 6157 | 2026-02 | 2026-03 |
| 6811 | 2026-02 | 2026-03 |
| 6181 | 2026-03 | 2026-04 |
| 6404 | 2026-03 | 2026-04 |
| 6801 | 2026-03 | 2026-04 |
| 6806 | 2026-03 | 2026-04 |
| 6808 | 2026-03 | 2026-04 |
| 6810 | 2026-03 | 2026-04 |
| 7015 | 2026-03 | 2026-04 |
| 8998 | 2026-03 | 2026-04 |
| 8999 | 2026-03 | 2026-04 |
| 6186 | 2026-04 | 2026-05 |
| 6800 | 2026-04 | 2026-05 |
| 6813 | 2026-04 | 2026-05 |
| 6803 | 2026-05 | 2026-06 |
| 7012 | 2026-05 | 2026-06 |
| 8200 | 2026-05 | 2026-06 |

## Quality Checks

- The 28-row configuration exactly matches stores whose final monthly flow precedes the global maximum month.
- 1M and 2M counts reconcile independently with the existing active-store Parquet.
- Existing active-store Parquet metadata was inspected; no Parquet was rebuilt or modified by this audit.
- Audit elapsed: 2.29 seconds.
