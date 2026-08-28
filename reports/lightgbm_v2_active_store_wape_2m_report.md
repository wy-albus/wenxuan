# LightGBM V2 Active-Store 2M WAPE Report

## Experiment

- Dataset: `data/processed/model_dataset_monthly_active_store.parquet`
- Target: `max(future_qty_2m, 0)`; training target is `log1p(target)`.
- Objective: `regression_l2`; prediction is `clip(expm1(log_prediction), 0, None)`.
- Features: 59; feature order and Train-only category maps are stored in the model metadata.
- Formal selection metric: complete Valid WAPE on the original quantity scale.
- WAPE is accumulated globally as total absolute error divided by total true quantity; row-group WAPEs are never averaged.
- Test is not used for iteration, parameter, threshold, or model selection.
- Model: `models/final/lightgbm_v2_logl2_active_store_wape_2m.txt`
- Prediction: `data/outputs/lightgbm_v2_active_store_wape_2m_test_predictions.parquet`
- Log: `logs/lightgbm/active_store_wape_2m/run_20260807_174222.log`

## Training And Search

- Formal training starts from scratch; training/search limit: 2000 trees.
- Train sample: 5,399,458; sampled Valid: 698,560.
- Sampling: 495.2s; formal training: 453.3s.
- Training peak RAM: 3.68 GiB.
- Selected iteration: 1600; search status: `internal_minimum`.
- Selected Valid WAPE: 105.564179%.
- The best candidate is an internal minimum in the evaluated range.

## Candidate Valid WAPE

| Iteration | Valid rows | WAPE | MAE | RMSE | True total | Predicted total | Total bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 18,491,375 | 108.410153% | 0.282947 | 3.260442 | 4826189.00 | 2961176.58 | -38.643584% |
| 300 | 18,491,375 | 106.436573% | 0.277796 | 3.199706 | 4826189.00 | 3070759.23 | -36.373001% |
| 500 | 18,491,375 | 106.036818% | 0.276753 | 3.177001 | 4826189.00 | 3128229.01 | -35.182211% |
| 700 | 18,491,375 | 105.866069% | 0.276307 | 3.163295 | 4826189.00 | 3165941.42 | -34.400799% |
| 850 | 18,491,375 | 105.729299% | 0.275950 | 3.152242 | 4826189.00 | 3185417.40 | -33.997251% |
| 1000 | 18,491,375 | 105.691558% | 0.275852 | 3.148149 | 4826189.00 | 3206403.94 | -33.562404% |
| 1200 | 18,491,375 | 105.661258% | 0.275772 | 3.142296 | 4826189.00 | 3232558.81 | -33.020468% |
| 1400 | 18,491,375 | 105.625193% | 0.275678 | 3.137442 | 4826189.00 | 3259257.57 | -32.467262% |
| 1600 | 18,491,375 | 105.564179% | 0.275519 | 3.134306 | 4826189.00 | 3268664.47 | -32.272348% |
| 1800 | 18,491,375 | 105.564393% | 0.275520 | 3.131193 | 4826189.00 | 3290822.72 | -31.813223% |
| 2000 | 18,491,375 | 105.582961% | 0.275568 | 3.126874 | 4826189.00 | 3310105.79 | -31.413673% |

Curve: `reports/lightgbm_v2_active_store_wape_curve_2m.png`

## Active-Store Test

| Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 12,859,664 | 0.207498 | 1.765772 | 107.2476% | 2488035.00 | 1629505.51 | -34.5063% |
| 0 | 11,821,536 | 0.068474 | 0.198027 | N/A | 0.00 | 809469.73 | N/A |
| 1 | 633,395 | 0.771417 | 0.867991 | 77.1417% | 633395.00 | 226000.76 | -64.3191% |
| 2-5 | 310,992 | 1.855136 | 2.076735 | 73.4720% | 785241.00 | 273359.13 | -65.1879% |
| 5-20 | 84,333 | 5.482766 | 6.428531 | 69.3661% | 666576.00 | 244109.79 | -63.3786% |
| 20+ | 9,408 | 35.179316 | 60.405922 | 82.1619% | 402823.00 | 76566.10 | -80.9926% |
| nonzero | 1,038,128 | 1.790617 | 6.178724 | 74.7132% | 2488035.00 | 820035.78 | -67.0408% |
| ge_5 | 93,741 | 8.463160 | 20.084461 | 74.1861% | 1069399.00 | 320675.89 | -70.0134% |
| ge_20 | 9,408 | 35.179316 | 60.405922 | 82.1619% | 402823.00 | 76566.10 | -80.9926% |

### Zero-Sales Diagnostics

- Mean prediction: 0.068474.
- Prediction > 0.5: 2.0136%.
- Prediction > 1: 0.5439%.

## Fair Old/New Comparison

Both models are evaluated only on the `month + site_no + item_id` Test intersection. Targets are required to match exactly.

| Model | Segment | Count | MAE | RMSE | WAPE | True total | Predicted total | Total bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active_store_wape | 0 | 11,821,536 | 0.068474 | 0.198027 | N/A | 0.00 | 809469.73 | N/A |
| active_store_wape | 20+ | 9,408 | 35.179316 | 60.405922 | 82.161893% | 402823.00 | 76566.10 | -80.992619% |
| active_store_wape | 5-20 | 84,333 | 5.482766 | 6.428531 | 69.366143% | 666576.00 | 244109.79 | -63.378551% |
| active_store_wape | nonzero | 1,038,128 | 1.790617 | 6.178724 | 74.713150% | 2488035.00 | 820035.78 | -67.040826% |
| active_store_wape | overall | 12,859,664 | 0.207498 | 1.765772 | 107.247650% | 2488035.00 | 1629505.51 | -34.506327% |
| old_v2_logl2 | 0 | 11,821,536 | 0.068473 | 0.196368 | N/A | 0.00 | 809450.67 | N/A |
| old_v2_logl2 | 20+ | 9,408 | 35.232677 | 60.471395 | 82.286519% | 402823.00 | 75978.21 | -81.138562% |
| old_v2_logl2 | 5-20 | 84,333 | 5.485356 | 6.430813 | 69.398913% | 666576.00 | 243254.59 | -63.506848% |
| old_v2_logl2 | nonzero | 1,038,128 | 1.791608 | 6.184390 | 74.754509% | 2488035.00 | 816229.11 | -67.193825% |
| old_v2_logl2 | overall | 12,859,664 | 0.207577 | 1.767202 | 107.288242% | 2488035.00 | 1625679.78 | -34.660092% |

## Scope Notes

- Existing LightGBM V2 models, predictions, and reports were not overwritten.
- MC remains a post-processing evaluation using the provisional five-level project rule; it is not treated as a confirmed Wenxuan business definition.
- No 2M, two-stage, independent MC, trend, random forest, MLP, or baseline training is performed by this script.
