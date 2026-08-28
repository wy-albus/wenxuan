# 存储清单

本清单记录模型阶段的主要正式产物。公共数据集 `data/processed/model_dataset_monthly.parquet` 未被修改或覆盖。

| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |
|---|---|---:|---|---|---|
| `reports/baseline_model_report.md` | Baseline 正式报告 | 0.01 MiB | 是 | 否，正式结果应保留 | `scripts/09_train_baselines.py` |
| `data/outputs/baseline_predictions.parquet` | Baseline 测试预测 | 27.59 MiB | 是 | 否，正式结果应保留 | `scripts/09_train_baselines.py` |
| `models/final/lightgbm_1m.txt` | LightGBM V1 1M 正式模型 | 0.70 MiB | 是 | 否，正式模型应保留 | `scripts/10_train_lightgbm.py` |
| `models/final/lightgbm_2m.txt` | LightGBM V1 2M 正式模型 | 1.85 MiB | 是 | 否，正式模型应保留 | `scripts/10_train_lightgbm.py` |
| `reports/lightgbm_model_report.md` | LightGBM V1 正式报告 | 0.01 MiB | 是 | 否，正式结果应保留 | `scripts/10_train_lightgbm.py` |
| `reports/lightgbm_feature_importance_1m.csv` | LightGBM V1 1M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/10_train_lightgbm.py` |
| `reports/lightgbm_feature_importance_2m.csv` | LightGBM V1 2M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/10_train_lightgbm.py` |
| `data/outputs/lightgbm_test_predictions.parquet` | LightGBM V1 测试预测 | 21.04 MiB | 是 | 否，正式结果应保留 | `scripts/10_train_lightgbm.py` |
| `models/final/lightgbm_v2_logl2_1m.txt` | LightGBM V2 1M 正式模型 | 5.38 MiB | 是 | 否，正式模型应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `models/final/lightgbm_v2_logl2_2m.txt` | LightGBM V2 2M 正式模型 | 11.24 MiB | 是 | 否，正式模型应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `reports/lightgbm_v2_logl2_model_report.md` | LightGBM V2 正式报告 | 0.02 MiB | 是 | 否，正式结果应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `reports/lightgbm_v2_logl2_feature_importance_1m.csv` | LightGBM V2 1M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `reports/lightgbm_v2_logl2_feature_importance_2m.csv` | LightGBM V2 2M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `data/outputs/lightgbm_v2_logl2_test_predictions.parquet` | LightGBM V2 测试预测 | 147.34 MiB | 是 | 否，正式结果应保留 | `scripts/12_train_lightgbm_v2_logl2.py` |
| `models/final/random_forest_1m.joblib` | Random Forest 1M 正式模型 | 79.10 MiB | 是 | 否，正式模型应保留 | `scripts/13_train_random_forest.py` |
| `models/final/random_forest_2m.joblib` | Random Forest 2M 正式模型 | 85.45 MiB | 是 | 否，正式模型应保留 | `scripts/13_train_random_forest.py` |
| `reports/random_forest_model_report.md` | Random Forest 正式报告 | 0.02 MiB | 是 | 否，正式结果应保留 | `scripts/13_train_random_forest.py` |
| `reports/random_forest_feature_importance_1m.csv` | Random Forest 1M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/13_train_random_forest.py` |
| `reports/random_forest_feature_importance_2m.csv` | Random Forest 2M 特征重要性 | <0.01 MiB | 是 | 否，正式结果应保留 | `scripts/13_train_random_forest.py` |
| `data/outputs/random_forest_test_predictions.parquet` | Random Forest 测试预测 | 149.65 MiB | 是 | 否，正式结果应保留 | `scripts/13_train_random_forest.py` |

## 可清理文件

以下内容可安全删除，需要时可由正式脚本重新生成：

- `models/checkpoints/random_forest/smoke_*.joblib`
- `models/checkpoints/random_forest/pilot_*.joblib`
- `models/checkpoints/random_forest/formal_*.joblib`
- `logs/random_forest/` 下的运行日志和阶段 JSON 摘要

不要删除公共 processed 数据、最终模型、正式测试预测和正式报告。

## MLP 正式产物

| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |
|---|---|---:|---|---|---|
| `src/models/mlp_model.py` | 代码、报告或训练历史 | 0.00 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `scripts/14_train_mlp.py` | 代码、报告或训练历史 | 0.03 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `models/final/mlp_1m.pt` | 正式模型 | 0.34 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `models/final/mlp_2m.pt` | 正式模型 | 0.34 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `reports/mlp_model_report.md` | 代码、报告或训练历史 | 0.02 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `reports/mlp_training_history_1m.csv` | 代码、报告或训练历史 | 0.00 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `reports/mlp_training_history_2m.csv` | 代码、报告或训练历史 | 0.00 MiB | 是 | 否 | `scripts/14_train_mlp.py` |
| `data/outputs/mlp_test_predictions.parquet` | 正式测试预测 | 91.23 MiB | 是 | 是（可由正式模型重建） | `scripts/14_train_mlp.py` |

## Two-Stage 正式产物

| 文件 | 用途 | 大小 | 可重新生成 | 可安全删除 | 生成脚本 |
|---|---|---:|---|---|---|
| `src/models/two_stage_model.py` | 代码、报告或特征重要性 | 0.00 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `scripts/15_train_two_stage.py` | 代码、报告或特征重要性 | 0.03 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `models/final/two_stage_classifier_1m.txt` | 正式模型 | 7.94 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `models/final/two_stage_regressor_1m.txt` | 正式模型 | 6.41 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `models/final/two_stage_classifier_2m.txt` | 正式模型 | 7.94 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `models/final/two_stage_regressor_2m.txt` | 正式模型 | 12.01 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `reports/two_stage_model_report.md` | 代码、报告或特征重要性 | 0.01 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `reports/two_stage_feature_importance.csv` | 代码、报告或特征重要性 | 0.01 MiB | 是 | 否 | `scripts/15_train_two_stage.py` |
| `data/outputs/two_stage_test_predictions.parquet` | 正式测试预测 | 239.96 MiB | 是 | 是（可由正式模型重建） | `scripts/15_train_two_stage.py` |

## 集中式预测阶段新增文件


| 文件 | 用途 | 大小(MiB) | 可重新生成 | 可安全删除 | 生成脚本 |
|---|---|---:|---|---|---|
| `reports\centralized_stage_summary.md` | 集中式阶段总结报告 | 0.01 | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |
| `reports\centralized_model_comparison.csv` | 六模型统一点预测指标 | 0.02 | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |
| `reports\centralized_sales_level_metrics.csv` | 暂定动销等级评价与混淆矩阵 | 0.06 | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |
| `reports\centralized_topk_metrics.csv` | 门店-月份 Top-K 重点图书识别指标 | 0.00 | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |
| `data\outputs\centralized_prediction_sample.csv` | 人工检查样本 | 8.00 | 是 | 是 | `scripts/16_finalize_centralized_stage.py` |

<!-- TWO_STAGE_ACTIVE_STORE_1M_V2 -->

## Active-Store Two-stage 1M V2

| File | Size MiB | Regenerable |
|---|---:|---|
| `scripts/25_train_two_stage_active_store_1m.py` | 0.03 | Yes |
| `models/final/two_stage_classifier_active_store_1m.txt` | 6.54 | No (formal model) |
| `models/final/two_stage_regressor_active_store_1m.txt` | 8.54 | No (formal model) |
| `reports/two_stage_active_store_1m_component_candidates.csv` | 0.00 | Yes |
| `reports/two_stage_active_store_1m_sampled_pair_candidates.csv` | 0.00 | Yes |
| `reports/two_stage_active_store_1m_complete_valid_pairs.csv` | 0.00 | Yes |
| `reports/two_stage_active_store_1m_gate_tradeoff.csv` | 0.03 | Yes |
| `reports/two_stage_active_store_1m_training_report.md` | 0.00 | Yes |
| `reports/two_stage_active_store_1m_runtime.json` | 0.00 | Yes |
