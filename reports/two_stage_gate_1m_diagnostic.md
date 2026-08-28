# Two-Stage 1M Zero-Sales Gate Diagnostic

## Scope

- No model was trained or adjusted. Existing 1M classifier and Tweedie regressor were loaded read-only.
- Classifier and regressor were each run once per active-store Valid row group; every threshold was computed from binned sufficient statistics.
- Threshold selection uses complete Valid original-scale WAPE only. Test is never used for threshold search.
- Valid rows: 18,563,573; runtime: not retained; peak RAM: not retained.
- Log: `logs/lightgbm/two_stage_gate_1m/run_20260815_173421.log`.

## Existing Test Schema

| Field | Type | Source | Meaning |
|---|---|---|---|
| month | string | original_test_key | 样本观察月份 |
| site_no | string | original_test_key | 门店标识 |
| item_id | string | original_test_key | 商品标识，仅用于定位 |
| target_qty_1m | float | target | 未来1个月非负真实销量 |
| target_qty_2m | float | target | 未来连续2个月非负真实累计销量 |
| p_sale_1m | float | classifier_output | 1M分类器预测的未来有销量概率 |
| conditional_qty_1m | float | tweedie_output | 1M Tweedie回归器预测的有销量条件数量 |
| two_stage_pred_1m | float | combined_prediction | 1M旧组合预测 p_sale × conditional_qty |
| p_sale_2m | float | classifier_output | 2M分类器预测的未来有销量概率 |
| conditional_qty_2m | float | tweedie_output | 2M Tweedie回归器预测的有销量条件数量 |
| two_stage_pred_2m | float | combined_prediction | 2M旧组合预测 p_sale × conditional_qty |

All 1M and 2M classifier probabilities, conditional quantities, and final products are present. No saved Two-stage Valid prediction file was found, so Valid inference was necessary.

## Classifier Probability Diagnostic

| True quantity | Count | Mean p | Median | P10 | P25 | P75 | P90 | p<0.2 | p<0.3 | p<0.5 | p>=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17,433,654 | 0.0531 | 0.0182 | 0.0045 | 0.0078 | 0.0574 | 0.1395 | 94.02% | 97.10% | 99.24% | 0.76% |
| 1 | 756,543 | 0.2265 | 0.1607 | 0.0261 | 0.0687 | 0.3283 | 0.5365 | 57.72% | 71.91% | 88.08% | 11.92% |
| 2-5 | 296,534 | 0.4090 | 0.3741 | 0.0689 | 0.1697 | 0.6278 | 0.8122 | 29.12% | 41.45% | 63.12% | 36.88% |
| 5-20 | 68,403 | 0.6141 | 0.7028 | 0.1207 | 0.3549 | 0.8913 | 0.9603 | 16.01% | 21.74% | 34.26% | 65.74% |
| 20+ | 8,439 | 0.6388 | 0.7813 | 0.0736 | 0.3251 | 0.9482 | 0.9856 | 19.08% | 23.82% | 33.64% | 66.36% |

For true 5-20 demand, p_sale median is 0.7028 and 34.26% fall below 0.50. For true 20+ demand, median is 0.7813 and 33.64% fall below 0.50.
The history-stratified version of the same diagnostic is in `reports/two_stage_gate_1m_classifier_diagnostic.csv`.

## Valid Gate Search

| Method | Threshold | Valid WAPE | MAE | Total bias | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | 5-20 pass | 20+ pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | N/A | 135.032534% | 0.179564 | 9.2669% | 72.4685% | 65.1186% | 75.1093% | 65.74% | 66.36% |
| gate_a | 0.65 | 92.472683% | 0.122968 | -65.6507% | 87.5086% | 71.3489% | 76.5508% | 54.75% | 59.15% |
| gate_b | 0.79 | 94.145970% | 0.125193 | -73.1773% | 91.7770% | 76.7799% | 77.4540% | 40.20% | 49.28% |

- Original WAPE: 135.032534%.
- Best Gate-A: threshold 0.65, WAPE 92.472683%.
- Best Gate-B: threshold 0.79, WAPE 94.145970%.
- Selected: `gate_a` threshold 0.02, WAPE 130.519870%, relative improvement 3.3419%.
- Zero prediction total changes from 1544420.74 to 1431934.72.
- 5-20 and 20+ gate pass rates are 98.12% and 96.75%.
- Test eligibility rule: relative WAPE improvement >= 0.50%, 5-20 pass >= 95%, 20+ pass >= 95%.
- Test executed: yes.

## Locked Test Comparison

The method and threshold below were locked using Valid only.

| Model | Segment | Count | WAPE | MAE | RMSE | True total | Predicted total | Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active_store_lightgbm_v2_wape | 20+ | 3,981 | 86.0730% | 37.569421 | 63.231639 | 173764.00 | 24914.36 | -85.6620% |
| active_store_lightgbm_v2_wape | 5-20 | 39,307 | 69.0609% | 5.384843 | 6.326169 | 306486.00 | 113234.08 | -63.0541% |
| active_store_lightgbm_v2_wape | nonzero | 687,412 | 77.4647% | 1.566346 | 5.208142 | 1389956.00 | 423758.88 | -69.5128% |
| active_store_lightgbm_v2_wape | overall | 12,999,597 | 117.1949% | 0.125308 | 1.206316 | 1389956.00 | 975992.27 | -29.7825% |
| active_store_lightgbm_v2_wape | zero | 12,312,185 | N/A | 0.044853 | 0.148389 | 0.00 | 552233.39 | N/A |
| best_gated_two_stage | 20+ | 3,981 | 82.6696% | 36.083891 | 62.319425 | 173764.00 | 32154.29 | -81.4954% |
| best_gated_two_stage | 5-20 | 39,307 | 65.3745% | 5.097402 | 6.177274 | 306486.00 | 138542.10 | -54.7966% |
| best_gated_two_stage | nonzero | 687,412 | 73.5225% | 1.486635 | 5.134487 | 1389956.00 | 552416.73 | -60.2565% |
| best_gated_two_stage | overall | 12,999,597 | 126.1098% | 0.134840 | 1.199564 | 1389956.00 | 1283357.38 | -7.6692% |
| best_gated_two_stage | zero | 12,312,185 | N/A | 0.059367 | 0.217715 | 0.00 | 730940.65 | N/A |
| old_lightgbm_v2 | 20+ | 3,981 | 86.4773% | 37.745888 | 63.349141 | 173764.00 | 24021.56 | -86.1758% |
| old_lightgbm_v2 | 5-20 | 39,307 | 69.7416% | 5.437919 | 6.364181 | 306486.00 | 109332.72 | -64.3270% |
| old_lightgbm_v2 | nonzero | 687,412 | 77.8705% | 1.574551 | 5.218943 | 1389956.00 | 408753.07 | -70.5924% |
| old_lightgbm_v2 | overall | 12,999,597 | 115.8253% | 0.123844 | 1.208076 | 1389956.00 | 936308.70 | -32.6375% |
| old_lightgbm_v2 | zero | 12,312,185 | N/A | 0.042848 | 0.142193 | 0.00 | 527555.63 | N/A |
| old_two_stage | 20+ | 3,981 | 82.6669% | 36.082711 | 62.317853 | 173764.00 | 32158.99 | -81.4927% |
| old_two_stage | 5-20 | 39,307 | 65.3697% | 5.097034 | 6.176785 | 306486.00 | 138556.59 | -54.7919% |
| old_two_stage | nonzero | 687,412 | 73.4718% | 1.485609 | 5.134117 | 1389956.00 | 553121.91 | -60.2058% |
| old_two_stage | overall | 12,999,597 | 131.7375% | 0.140858 | 1.199520 | 1389956.00 | 1362989.69 | -1.9401% |
| old_two_stage | zero | 12,312,185 | N/A | 0.065778 | 0.217947 | 0.00 | 809867.77 | N/A |

## Conclusion

The locked gate passed the predeclared Valid value/safety rule, so it was evaluated once on Test.
- On the locked common Test, Gate-A lowers old Two-stage WAPE from 131.737481% to 126.109825% (5.627656 percentage points; 4.27% relative).
- Zero-sample predicted quantity falls from 809867.77 to 730940.65 (9.75% reduction). This confirms that persistent small positive predictions on true zeros are a material part of old Two-stage WAPE.
- Nonzero WAPE changes from 73.471768% to 73.522502%; 5-20 from 65.369741% to 65.374469%; 20+ from 82.666876% to 82.669580%. The safe gate leaves head-demand accuracy essentially unchanged.
- Total bias moves from -1.9401% to -7.6692%: gating reduces false positives but increases aggregate underestimation.
- The gated result remains worse in overall WAPE than old LightGBM V2 (115.825319%) and active-store LightGBM V2 (117.194940%). It is a useful Two-stage correction, not a new overall winner.
This experiment diagnoses post-processing only and does not justify retraining or extending to 2M without a separate decision.
