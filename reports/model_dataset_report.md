# Model Dataset Report

## Overview

- Input monthly rows: 11619660
- Rows after zero-month panel completion: 96534793
- Final model dataset rows: 89529779
- Field count: 66
- Time range: 2023-01 to 2026-04
- Output file: `D:/codex_wenxuan/data/processed/model_dataset_monthly.parquet`
- Build elapsed seconds: 294.9

## Split Counts

| value | count | ratio |
|---|---:|---:|
| train | 57178758 | 63.87% |
| valid | 18789982 | 20.99% |
| test | 13561039 | 15.15% |

## future_qty_1m Distribution

| metric | value |
|---|---:|
| count | 89529779 |
| mean | 0.1930 |
| std | 2.4929 |
| min | -2650.0000 |
| max | 3489.0000 |

## future_qty_2m Distribution

| metric | value |
|---|---:|
| count | 89529779 |
| mean | 0.3684 |
| std | 3.7277 |
| min | -2650.0000 |
| max | 3748.0000 |

## future_has_sales_1m Ratio

| value | count | ratio |
|---|---:|---:|
| 0 | 81585619 | 91.13% |
| 1 | 7944160 | 8.87% |

## future_has_sales_2m Ratio

| value | count | ratio |
|---|---:|---:|
| 0 | 77236618 | 86.27% |
| 1 | 12293161 | 13.73% |

## Current Month total_qty Buckets

| value | count | ratio |
|---|---:|---:|
| 0 | 78303588 | 87.46% |
| 1 | 7778569 | 8.69% |
| 2-5 | 2940909 | 3.28% |
| 5-20 | 443421 | 0.50% |
| 20+ | 63292 | 0.07% |

## Leakage Check

- Lag and rolling features are built from shifted historical values only.
- `future_qty_1m` and `future_qty_2m` are created after feature construction and are not used in input feature windows.
- Last two months per `site_no x item_id` are dropped because `future_qty_2m` cannot be fully observed.
- Splits are time-based: train `2023-01` to `2025-06`, valid `2025-07` to `2025-12`, test `2026-01` to `2026-04`.

## Next-stage Modeling Suggestions

- Start with historical mean and weighted moving average baselines using this dataset.
- Use `future_qty_1m` first, then extend to `future_qty_2m` once baseline evaluation is stable.
- For LightGBM, use `log1p(future_qty_1m)` or Poisson/Tweedie objectives because the target remains strongly long-tailed.
- For the later two-stage model, use `future_has_sales_1m` or `future_has_sales_2m` as the classification target and the positive-sales subset for regression.
