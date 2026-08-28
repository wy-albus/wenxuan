# Active-Store 业务评价

本报告复用现有 Active-Store LightGBM 正式 Test 预测，不重新训练或推理模型。Test 只用于已冻结模型的业务效果汇报，不参与轮次或阈值选择。

## 口径

- MC 使用 `config/mc_sales_levels.yaml` 的五级规则：0、1、2-4、5-19、20+。预测先非负截断并执行 `floor(x + 0.5)`。
- 2M MC 使用未来两个月累计销量除以 2 后的月均值，真实值和预测值顺序完全一致。
- 项目已有正式趋势实现：比较当前月销量与未来销量；2M 比较未来两个月月均。严格大于为上升、严格小于为下降、相等为持平，不使用额外百分比阈值。
- MC 边界当前状态为项目暂定口径，后续业务确认后通过统一配置调整。

## LightGBM Active-Store 1M

| 指标 | 结果 |
|---|---:|
| Overall WAPE | 117.1949% |
| Nonzero WAPE | 77.4647% |
| 5-20 WAPE | 69.0609% |
| 20+ WAPE | 86.0730% |
| MAE | 0.125308 |
| RMSE | 1.206316 |
| True total | 1389956.00 |
| Predicted total | 975992.27 |
| Total Bias | -29.7825% |
| Zero sample count | 12,312,185 |
| Zero mean prediction | 0.044853 |
| Zero predicted total | 552233.39 |
| Zero > 0.5 | 1.0031% |
| Zero > 1 | 0.2852% |
| MC Accuracy | 94.4448% |
| MC Macro-F1 | 34.6924% |
| MC Weighted-F1 | 93.2791% |
| MC High-demand Recall | 6.1040% |
| Trend Accuracy | 23.0509% |
| Trend Macro-F1 | 40.9949% |
| Trend Up Recall | 83.4878% |
| Trend Down Recall | 98.9524% |
| Trend severe direction error | 0.6078% |

### MC 各等级

| 等级 | Precision | Recall | F1 |
|---|---:|---:|---:|
| no_sales | 96.0196% | 98.9969% | 97.4855% |
| low | 25.1931% | 12.2306% | 16.4669% |
| normal | 37.0827% | 13.1942% | 19.4632% |
| medium_high | 50.6064% | 20.3806% | 29.0585% |
| high | 54.9774% | 6.1040% | 10.9880% |

MC confusion matrix:

| True \ Pred | no_sales | low | normal | medium_high | high |
|---|---:|---:|---:|---:|---:|
| no_sales | 12188678 | 108256 | 14060 | 1184 | 7 |
| low | 393990 | 56774 | 12311 | 1120 | 1 |
| normal | 101623 | 50570 | 23740 | 3971 | 24 |
| medium_high | 8712 | 9248 | 13169 | 8011 | 167 |
| high | 948 | 507 | 739 | 1544 | 243 |

Trend confusion matrix (`down`, `flat`, `up`):

| True \ Pred | down | flat | up |
|---|---:|---:|---:|
| down | 709190 | 0 | 7508 |
| flat | 114456 | 1895019 | 9803510 |
| up | 71505 | 6088 | 392321 |

## LightGBM Active-Store 2M

| 指标 | 结果 |
|---|---:|
| Overall WAPE | 107.2476% |
| Nonzero WAPE | 74.7132% |
| 5-20 WAPE | 69.3661% |
| 20+ WAPE | 82.1619% |
| MAE | 0.207498 |
| RMSE | 1.765772 |
| True total | 2488035.00 |
| Predicted total | 1629505.51 |
| Total Bias | -34.5063% |
| Zero sample count | 11,821,536 |
| Zero mean prediction | 0.068474 |
| Zero predicted total | 809469.73 |
| Zero > 0.5 | 2.0136% |
| Zero > 1 | 0.5439% |
| MC Accuracy | 92.2485% |
| MC Macro-F1 | 32.8547% |
| MC Weighted-F1 | 89.8285% |
| MC High-demand Recall | 4.1005% |
| Trend Accuracy | 26.3424% |
| Trend Macro-F1 | 44.7707% |
| Trend Up Recall | 89.7419% |
| Trend Down Recall | 99.2090% |
| Trend severe direction error | 0.5255% |

### MC 各等级

| 等级 | Precision | Recall | F1 |
|---|---:|---:|---:|
| no_sales | 93.3217% | 99.4561% | 96.2913% |
| low | 37.9618% | 9.3639% | 15.0223% |
| normal | 47.3347% | 12.9435% | 20.3283% |
| medium_high | 59.1008% | 15.8364% | 24.9794% |
| high | 57.1429% | 4.1005% | 7.6520% |

MC confusion matrix:

| True \ Pred | no_sales | low | normal | medium_high | high |
|---|---:|---:|---:|---:|---:|
| no_sales | 11757235 | 59656 | 4397 | 245 | 3 |
| low | 735398 | 77002 | 9602 | 326 | 3 |
| normal | 96973 | 57781 | 23318 | 2077 | 3 |
| medium_high | 8080 | 7933 | 11358 | 5166 | 84 |
| high | 917 | 469 | 587 | 927 | 124 |

Trend confusion matrix (`down`, `flat`, `up`):

| True \ Pred | down | flat | up |
|---|---:|---:|---:|
| down | 779831 | 0 | 6218 |
| flat | 52910 | 1964484 | 9339462 |
| up | 61364 | 12162 | 643233 |

## 历史 Two-stage Gate 诊断参考

以下仅为 `Pre-Active-Store model / diagnostic only`，不是新版 Active-Store 正式 Two-stage：

| 方案 | tau | Overall WAPE | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | Total Bias |
|---|---:|---:|---:|---:|---:|---:|
| Original | N/A | 135.0325% | 72.4685% | 65.1186% | 75.1093% | 9.2669% |
| Gate-A low-cost reference | 0.020 | 130.5199% | 72.5127% | 65.1239% | 75.1114% | 4.6660% |
| Gate-A unconstrained minimum | 0.650 | 92.4727% | 87.5086% | 71.3489% | 76.5508% | -65.6507% |

无约束最低点会大量牺牲中高销量覆盖，只保留为历史诊断，不作为本轮阈值结论。

## 数据一致性

- 1M rows: 12,999,597; target max difference: 0.
- 2M rows: 12,859,664; target max difference: 0.
- 所有预测均为有限非负值，预测文件与统一 Active-Store 数据集目标一致。
