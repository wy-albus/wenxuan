# Active-Store Two-stage 1M 训练与 Valid Gate-A 报告

## 正式范围

- Dataset: `data/processed/model_dataset_monthly_active_store.parquet`.
- Components: binary classifier and positive-sales Tweedie regressor.
- Combination: `p_sale * conditional_qty`.
- 组件候选先在 Sampled Valid 筛选，Complete Valid 只评价 3 个组合。
- 正式组合仅按 Complete Valid 原始销量尺度 Overall WAPE 选择；接近时依次比较绝对总量偏差、趋势 Macro-F1 和轮次。
- Gate-A 在组合冻结后评价；本报告不自动选定最终 tau，也未访问留出评估集。

## 训练资源

- Features: 57.
- Classifier train/valid: 4,379,008 / 556,913.
- Regressor positive train/valid: 1,827,297 / 209,368.
- Sampling: 545.6s; classifier training: 386.0s; regressor training: 181.4s.
- Complete Valid inference: 5673.8s.
- Peak RAM: 2.567 GiB.

## 组件诊断

Classifier 记录 PR-AUC、ROC-AUC、Logloss 和 threshold=0.5 Recall；Regressor 仅在真实正销量样本记录 WAPE、MAE、RMSE。完整明细见 `reports/two_stage_active_store_1m_component_candidates.csv`。

## 组合选择

最终选择 classifier=750 轮，regressor=1054 轮。
Complete Valid Overall WAPE=134.598921%，Total Bias=8.932879%，Trend Macro-F1=32.1414%。
选择原因是该组合在受控的 3 个 Complete Valid 候选中 Overall WAPE 最低；未使用组件单独损失替代组合选模。

## Original Complete Valid 业务指标

| Metric | Value |
|---|---:|
| Overall WAPE | 134.598921% |
| Nonzero WAPE | 72.604150% |
| 5-20 WAPE | 64.464076% |
| 20+ WAPE | 75.148203% |
| MAE | 0.178987 |
| RMSE | 2.099695 |
| True total | 2468546.00 |
| Predicted total | 2689058.23 |
| Total Bias | 8.932879% |
| Zero sample count | 17,433,654 |
| Zero mean prediction | 0.087782 |
| Zero predicted total | 1530369.44 |
| Zero > 0.5 | 2.945229% |
| Zero > 1 | 0.968655% |

## Gate-A 值得关注的观察点

以下只是便于阅读的观察点，不是自动选择结果。完整 97 个阈值见 `reports/two_stage_active_store_1m_gate_tradeoff.csv`。

| tau | Overall WAPE | Nonzero WAPE | 5-20 WAPE | 20+ WAPE | Zero total | Zero >0.5 | Zero >1 | Total Bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 134.5989% | 72.6041% | 64.4641% | 75.1482% | 1530369.44 | 2.9452% | 0.9687% | 8.9329% |
| 0.010 | 132.8071% | 72.6139% | 64.4656% | 75.1485% | 1485897.74 | 2.9452% | 0.9687% | 7.1216% |
| 0.020 | 130.0212% | 72.6493% | 64.4695% | 75.1506% | 1416250.95 | 2.9444% | 0.9686% | 4.2648% |
| 0.050 | 124.0053% | 72.8420% | 64.4956% | 75.1579% | 1262988.68 | 2.9401% | 0.9679% | -2.1366% |
| 0.100 | 115.2983% | 73.4997% | 64.6035% | 75.1828% | 1031819.98 | 2.9173% | 0.9644% | -12.1599% |
| 0.200 | 104.4744% | 75.5479% | 65.0122% | 75.3116% | 714062.32 | 2.7607% | 0.9462% | -27.0930% |
| 0.300 | 98.4844% | 77.9741% | 65.5432% | 75.4167% | 506308.00 | 2.3254% | 0.8873% | -37.9855% |
| 0.500 | 93.1696% | 83.5545% | 67.7956% | 75.8976% | 237353.57 | 0.7308% | 0.5975% | -55.1312% |

## 正式产物

- `models/final/two_stage_classifier_active_store_1m.txt`
- `models/final/two_stage_regressor_active_store_1m.txt`
- `reports/two_stage_active_store_1m_complete_valid_pairs.csv`
- `reports/two_stage_active_store_1m_gate_tradeoff.csv`

流程在 Complete Valid Gate-A trade-off 后停止。
