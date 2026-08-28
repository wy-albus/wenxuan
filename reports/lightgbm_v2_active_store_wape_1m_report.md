# LightGBM V2 Active-Store 1M WAPE Report

## Experiment

- Dataset: `data/processed/model_dataset_monthly_active_store.parquet`
- Target: `max(future_qty_1m, 0)`; training target is `log1p(target)`.
- Objective: `regression_l2`; prediction is `clip(expm1(log_prediction), 0, None)`.
- Features: 57; feature order and Train-only category maps are stored in the model metadata.
- Formal selection metric: complete Valid WAPE on the original quantity scale.
- WAPE is accumulated globally as total absolute error divided by total true quantity; row-group WAPEs are never averaged.
- Test is not used for iteration, parameter, threshold, or model selection.
- Model: `models/final/lightgbm_v2_logl2_active_store_wape_1m.txt`
- Prediction: `data/outputs/lightgbm_v2_active_store_wape_1m_test_predictions.parquet`
- Log: `logs/lightgbm/active_store_wape_1m/run_20260807_143118.log`

## Training And Search

- Base checkpoint: 700 trees; extended training limit: 2000 trees.
- Train sample: 4,335,628; sampled Valid: 546,370.
- Sampling: 306.8s; continuation training: 317.8s.
- Training peak RAM: 3.04 GiB.
- Selected iteration: 850; search status: `internal_minimum`.
- Selected Valid WAPE: 117.176105%.
- The best candidate is an internal minimum in the evaluated range.

## Candidate Valid WAPE

| Iteration | Valid rows | WAPE | MAE | RMSE | True total | Predicted total | Total bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | 18,563,573 | 117.302746% | 0.155987 | 2.154810 | 2468546.00 | 1781893.97 | -27.816052% |
| 650 | 18,563,573 | 117.295216% | 0.155977 | 2.153575 | 2468546.00 | 1784800.19 | -27.698322% |
| 700 | 18,563,573 | 117.209516% | 0.155863 | 2.150879 | 2468546.00 | 1788878.99 | -27.533091% |
| 750 | 18,563,573 | 117.233622% | 0.155895 | 2.149559 | 2468546.00 | 1794171.40 | -27.318697% |
| 800 | 18,563,573 | 117.186570% | 0.155832 | 2.148677 | 2468546.00 | 1795428.80 | -27.267760% |
| 850 | 18,563,573 | 117.176105% | 0.155818 | 2.147801 | 2468546.00 | 1801295.95 | -27.030084% |
| 900 | 18,563,573 | 117.192755% | 0.155841 | 2.147051 | 2468546.00 | 1806732.44 | -26.809853% |
| 1000 | 18,563,573 | 117.195034% | 0.155844 | 2.146363 | 2468546.00 | 1813101.18 | -26.551858% |
| 1200 | 18,563,573 | 117.241624% | 0.155906 | 2.145132 | 2468546.00 | 1822954.67 | -26.152696% |
| 1400 | 18,563,573 | 117.276809% | 0.155952 | 2.142624 | 2468546.00 | 1835248.57 | -25.654674% |
| 1600 | 18,563,573 | 117.343786% | 0.156041 | 2.141452 | 2468546.00 | 1846997.12 | -25.178744% |
| 1800 | 18,563,573 | 117.420860% | 0.156144 | 2.140200 | 2468546.00 | 1857044.69 | -24.771720% |
| 2000 | 18,563,573 | 117.490970% | 0.156237 | 2.140088 | 2468546.00 | 1865462.99 | -24.430698% |

Curve: `reports/lightgbm_v2_active_store_wape_curve_1m.png`

## Active-Store Test

| Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 12,999,597 | 0.125308 | 1.206316 | 117.1949% | 1389956.00 | 975992.27 | -29.7825% |
| 0 | 12,312,185 | 0.044853 | 0.148389 | N/A | 0.00 | 552233.39 | N/A |
| 1 | 464,196 | 0.812584 | 0.890296 | 81.2584% | 464196.00 | 136399.33 | -70.6160% |
| 2-5 | 179,928 | 1.880199 | 2.092555 | 75.9355% | 445510.00 | 149211.12 | -66.5078% |
| 5-20 | 39,307 | 5.384843 | 6.326169 | 69.0609% | 306486.00 | 113234.08 | -63.0541% |
| 20+ | 3,981 | 37.569421 | 63.231640 | 86.0730% | 173764.00 | 24914.36 | -85.6620% |
| nonzero | 687,412 | 1.566346 | 5.208142 | 77.4647% | 1389956.00 | 423758.88 | -69.5128% |
| ge_5 | 43,288 | 8.344712 | 20.100743 | 75.2162% | 480250.00 | 138148.43 | -71.2341% |
| ge_20 | 3,981 | 37.569421 | 63.231640 | 86.0730% | 173764.00 | 24914.36 | -85.6620% |

### Zero-Sales Diagnostics

- Mean prediction: 0.044853.
- Prediction > 0.5: 1.0031%.
- Prediction > 1: 0.2852%.

## Fair Old/New Comparison

Both models are evaluated only on the `month + site_no + item_id` Test intersection. Targets are required to match exactly.

| Model | Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active_store_wape | 0 | 12,312,185 | 0.044853 | 0.148389 | N/A | 0.00 | 552233.39 | N/A |
| active_store_wape | 20+ | 3,981 | 37.569421 | 63.231639 | 86.072988% | 173764.00 | 24914.36 | -85.661958% |
| active_store_wape | 5-20 | 39,307 | 5.384843 | 6.326169 | 69.060915% | 306486.00 | 113234.08 | -63.054079% |
| active_store_wape | nonzero | 687,412 | 1.566346 | 5.208142 | 77.464661% | 1389956.00 | 423758.88 | -69.512785% |
| active_store_wape | overall | 12,999,597 | 0.125308 | 1.206316 | 117.194940% | 1389956.00 | 975992.27 | -29.782506% |
| old_v2_logl2 | 0 | 12,312,185 | 0.042848 | 0.142193 | N/A | 0.00 | 527555.63 | N/A |
| old_v2_logl2 | 20+ | 3,981 | 37.745888 | 63.349141 | 86.477281% | 173764.00 | 24021.56 | -86.175755% |
| old_v2_logl2 | 5-20 | 39,307 | 5.437919 | 6.364181 | 69.741616% | 306486.00 | 109332.72 | -64.327010% |
| old_v2_logl2 | nonzero | 687,412 | 1.574551 | 5.218943 | 77.870475% | 1389956.00 | 408753.07 | -70.592373% |
| old_v2_logl2 | overall | 12,999,597 | 0.123844 | 1.208076 | 115.825319% | 1389956.00 | 936308.70 | -32.637530% |

## Scope Notes

- Existing LightGBM V2 models, predictions, and reports were not overwritten.
- MC remains a post-processing evaluation using the provisional five-level project rule; it is not treated as a confirmed Wenxuan business definition.
- No 2M, two-stage, independent MC, trend, random forest, MLP, or baseline training is performed by this script.
