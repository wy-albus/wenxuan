# 文轩集团图书销量预测项目实验可追溯性审计

> 本审计为只读扫描结果。除本文件和 `reports/model_parameter_audit.csv` 外，未训练模型、未修改公共数据、配置、代码、模型或既有报告。

## 1. 扫描范围与总体结果

| 目录 | 是否存在 | 文件数 | 总大小(MiB) |
|---|---|---:|---:|
| `data/raw/` | 是 | 7 | 429.14 |
| `data/interim/` | 是 | 112 | 390.72 |
| `data/processed/` | 是 | 3 | 2059.01 |
| `data/outputs/` | 是 | 8 | 684.82 |
| `scripts/` | 是 | 32 | 0.71 |
| `src/data/` | 是 | 12 | 0.06 |
| `src/features/` | 是 | 2 | 0.02 |
| `src/models/` | 是 | 16 | 0.06 |
| `src/evaluation/` | 是 | 4 | 0.02 |
| `config/` | 是 | 3 | 0.01 |
| `models/final/` | 是 | 12 | 218.71 |
| `reports/` | 是 | 27 | 0.27 |

- 共识别阶段输入项：17 个；执行脚本项：15 个；阶段输出项：52 个。
- 参数审计表记录：18 行模型/诊断参数。
- 关键正式文件缺失：未发现。
- 参数字段中写入“未找到”的项目：0 项；详见 `reports/model_parameter_audit.csv`。

## 2. 项目阶段追溯

### 阶段 1：原始数据接收

- 目的：接收并固定销售流水 zip 原始输入。
- 输入文件：`data/raw/*.zip`
- 执行脚本：`无训练脚本；原始文件由人工/外部流程放入 data/raw。`
- 主要函数、算法或工具：ZIP/CSV 原始文件管理。
- 输出文件：`data/raw/*.zip`
- 输出数据记录：Zip文件: 7; CSV文件: 7; 原始行数: 21771594; 时间范围: 2023-01-01 to 2026-06-21; 大小: 429.14 MiB
- 质量检查：检查 data/raw 是否存在 7 个 zip，后续质量报告确认每个 zip 内 CSV 可读。
- 主要结论：原始数据覆盖 2023-01-01 至 2026-06-21，符合项目时间跨度。
- 相比上一阶段新增/改变：从外部原始数据进入项目受控 raw 区。
- 为下一阶段提供：为原始读取与字段质量检查提供输入。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 2：原始数据读取与质量检查

- 目的：逐个读取 zip 内 CSV，确认编码、字段、日期/数值可转换性和异常值。
- 输入文件：`data/raw/*.zip`
- 执行脚本：`scripts/01_read_and_check_raw.py`
- 主要函数、算法或工具：分块读取；编码尝试 utf-8-sig/gbk/gb18030；字段完整性和转换失败率统计。
- 输出文件：`reports/data_quality_report.md`
- 输出数据记录：原始行数: 21771594; 字段数: 22; 时间范围: 2023-01-01 to 2026-06-21; 报告大小: 0.00 MiB
- 质量检查：检查核心字段、period/qty/price/tlp/tsp 转换、isbn/gds_no 缺失、oln_or_ofln/rtn_flag 取值、qty<0。
- 主要结论：读取成功；qty<0 记录 205810；isbn/gds_no 缺失率为 0；rtn_flag 主要为空和 X。
- 相比上一阶段新增/改变：从文件存在性推进到可解析、可统计的数据质量事实。
- 为下一阶段提供：为清洗脚本确定字段保留、隐私字段删除和退货处理规则。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 3：数据清洗

- 目的：保留建模核心字段、去除隐私字段、转换日期/数值、构造 item_id/channel_type。
- 输入文件：`data/raw/*.zip`
- 执行脚本：`scripts/02_clean_sales.py`
- 主要函数、算法或工具：src/data/clean_sales.py；分块清洗；item_id=isbn 优先否则 gds_no；channel_type 映射。
- 输出文件：`data/interim/cleaned_sales_*.parquet; reports/cleaning_summary.md`
- 输出数据记录：文件数: 112; 行数: 21771594; 字段数: 23; 时间范围: 2023-01-01 00:00:00 to 2026-06-21 00:00:00; 大小: 390.72 MiB; cleaning_summary大小: 0.01 MiB
- 质量检查：删除缺失 period/item_id/site_no 记录；保留 rtn_flag；不使用客户隐私字段。
- 主要结论：生成分片 Parquet 中间层，后续聚合不再直接依赖原始 CSV。
- 相比上一阶段新增/改变：从原始流水变为标准字段、无隐私输入的清洗流水。
- 为下一阶段提供：为日度聚合提供 period/site/item 粒度流水。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 4：日度聚合

- 目的：按 period × site_no × item_id 聚合净销量和渠道销量。
- 输入文件：`data/interim/cleaned_sales_*.parquet`
- 执行脚本：`scripts/03_build_daily_sales.py`
- 主要函数、算法或工具：src/data/aggregate_sales.py；groupby 聚合；退货记录 qty<0 或 rtn_flag 非正常。
- 输出文件：`data/processed/daily_item_store_sales.parquet`
- 输出数据记录：文件数: 1; 行数: 18521277; 字段数: 19; 时间范围: 2023-01-01 00:00:00 to 2026-06-21 00:00:00; 大小: 375.88 MiB
- 质量检查：核对 total_qty、online/offline/unknown qty、sales_count/return_count/return_qty。
- 主要结论：得到日度门店-图书销售基础表。
- 相比上一阶段新增/改变：从流水粒度收敛到日度样本粒度。
- 为下一阶段提供：为月度聚合和后续时序特征提供日级基础。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 5：月度聚合

- 目的：按 month × site_no × item_id 聚合月度销量、金额、折扣和销售天数。
- 输入文件：`data/processed/daily_item_store_sales.parquet`
- 执行脚本：`scripts/04_build_monthly_sales.py`
- 主要函数、算法或工具：src/data/aggregate_sales.py；最近一次非空描述字段；除零保护。
- 输出文件：`data/processed/monthly_item_store_sales.parquet`
- 输出数据记录：文件数: 1; 行数: 11619660; 字段数: 22; 时间范围: 2023-01 to 2026-06; 大小: 337.06 MiB
- 质量检查：检查 sales_days、avg_real_price、discount_rate 除零处理和描述字段逻辑。
- 主要结论：得到实际有流水的月度门店-图书表。
- 相比上一阶段新增/改变：从日度表汇总到月度建模基础。
- 为下一阶段提供：为长尾分析和监督学习面板补齐提供输入。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 6：长尾分析

- 目的：分析月销量长尾、头部贡献和线上/线下差异。
- 输入文件：`data/processed/monthly_item_store_sales.parquet`
- 执行脚本：`scripts/05_long_tail_analysis.py`
- 主要函数、算法或工具：分位数、销量桶、Top 10/20% 贡献、类目/渠道分布统计。
- 输出文件：`reports/long_tail_analysis.md`
- 输出数据记录：输入行数: 11619660; 报告大小: 0.00 MiB
- 质量检查：检查销量桶、类目和渠道汇总。
- 主要结论：月度样本中低销量极多，Top 10% 图书贡献 88.19%，Top 20% 贡献 94.72%；LSTM 不宜作为优先模型。
- 相比上一阶段新增/改变：新增长尾业务认知，不改变数据文件。
- 为下一阶段提供：为补齐零销量和选择 log1p/树模型/两阶段方案提供依据。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 7：零销量月份补齐

- 目的：对每个 site_no × item_id 从 first_month 到 max_month 补齐合理范围内零销量月份。
- 输入文件：`data/processed/monthly_item_store_sales.parquet`
- 执行脚本：`scripts/06_build_model_dataset.py`
- 主要函数、算法或工具：src/data/model_dataset.py；按组合局部面板补齐，避免全量笛卡尔积。
- 输出文件：`内存中零销量补齐面板；最终写入 data/processed/model_dataset_monthly.parquet`
- 输出数据记录：补齐后行数: 96534793; 未单独保存补齐中间大表。
- 质量检查：确认不做全量笛卡尔积；缺失销量字段填 0；描述字段延续最近/众数。
- 主要结论：补齐后样本规模约 9653 万，为监督学习构造提供完整时间窗。
- 相比上一阶段新增/改变：从“有流水才有记录”变为“合理观察期内含零销量月份”。
- 为下一阶段提供：为 lag/rolling/target 构造提供连续月度面板。
- 缺失、已删除或无法确认：补齐后的完整中间面板未单独保存；这是按设计避免生成大型临时文件，不视为缺失。

### 阶段 8：模型数据集构造

- 目的：构造 lag、rolling、渠道、频次、金额折扣和未来目标，并按时间划分 split。
- 输入文件：`data/processed/monthly_item_store_sales.parquet`
- 执行脚本：`scripts/06_build_model_dataset.py`
- 主要函数、算法或工具：历史窗口 shift/rolling；future_qty_1m/2m 标签；时间切分 train/valid/test。
- 输出文件：`data/processed/model_dataset_monthly.parquet; reports/model_dataset_report.md`
- 输出数据记录：文件数: 1; 行数: 89529779; 字段数: 66; 时间范围: 2023-01 to 2026-04; 大小: 1346.08 MiB; split: train 57,178,758, valid 18,789,982, test 13,561,039
- 质量检查：检查未来信息只用于标签；最后两月无法构造 2M 标签样本删除；时间划分非随机。
- 主要结论：最终监督学习月度数据集 89,529,779 行、66 字段。
- 相比上一阶段新增/改变：新增训练特征、目标字段和 split 字段。
- 为下一阶段提供：为字段审查、预处理和模型训练提供唯一公共数据集。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 9：特征审查与模型配置

- 目的：审查 66 字段，排除泄露/高基数/异常比例字段，形成六类算法特征列表。
- 输入文件：`data/processed/model_dataset_monthly.parquet; reports/feature_schema_review.md`
- 执行脚本：`scripts/07_feature_schema_review.py`
- 主要函数、算法或工具：字段审计；config/model_features.yaml；src/features/preprocessing.py 按列读取和运行时时间特征。
- 输出文件：`reports/feature_schema_review.md; config/model_features.yaml; reports/preprocessing_design.md; src/features/preprocessing.py`
- 输出数据记录：model_features.yaml大小: 0.01 MiB; preprocessing.py大小: 0.01 MiB
- 质量检查：检查 future_*、split、item_id/isbn/gds_no 和异常比例字段未进入模型输入。
- 主要结论：确立唯一公共 Parquet + 按列读取策略。
- 相比上一阶段新增/改变：从数据集构造推进到模型可用特征边界。
- 为下一阶段提供：为所有模型训练阶段提供统一配置和防泄漏预处理。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 10：历史均值

- 目的：用过去 3 月平均销量建立公式 baseline。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml`
- 执行脚本：`scripts/09_train_baselines.py`
- 主要函数、算法或工具：src/models/historical_mean.py；pred_1m=qty_mean_last_3m，pred_2m=2×pred_1m，非负截断。
- 输出文件：`data/outputs/baseline_predictions.parquet; reports/baseline_model_report.md`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 10; 时间范围: 2026-01 to 2026-04; 大小: 27.59 MiB
- 质量检查：smoke test 检查公式、target 非负截断、分桶边界、流式指标一致性。
- 主要结论：得到最低基准；均值法在整体和长尾层级上误差较高。
- 相比上一阶段新增/改变：新增无需训练的第一类 baseline 预测。
- 为下一阶段提供：为所有复杂模型提供最低比较基准。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 11：加权移动平均

- 目的：用最近 3 月 lag 销量按 0.5/0.3/0.2 加权建立 baseline。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml`
- 执行脚本：`scripts/09_train_baselines.py`
- 主要函数、算法或工具：src/models/weighted_moving_average.py；流式计算 MAE/RMSE/SMAPE/WAPE。
- 输出文件：`data/outputs/baseline_predictions.parquet; reports/baseline_model_report.md`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 10; 时间范围: 2026-01 to 2026-04; 大小: 27.59 MiB
- 质量检查：同历史均值阶段；只保存 test 全量预测。
- 主要结论：加权移动平均优于历史均值，但仍明显高估总量。
- 相比上一阶段新增/改变：新增第二类公式 baseline，并与历史均值共用输出文件。
- 为下一阶段提供：为复杂模型比较提供传统基准。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 12：LightGBM V1

- 目的：训练 log1p + regression_l1 的第一版 LightGBM。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml; reports/baseline_model_report.md`
- 执行脚本：`scripts/10_train_lightgbm.py`
- 主要函数、算法或工具：src/models/lightgbm_model.py；确定性分层采样、逆概率权重、类别编码、early stopping。
- 输出文件：`models/final/lightgbm_1m.txt; models/final/lightgbm_2m.txt; data/outputs/lightgbm_test_predictions.parquet; reports/lightgbm_model_report.md; reports/lightgbm_feature_importance_*.csv`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 7; 时间范围: 2026-01 to 2026-04; 大小: 21.04 MiB
- 质量检查：smoke/pilot/formal 三阶段；内存阈值；完整 valid/test 流式评价；模型保存重载一致性。
- 主要结论：V1 总体 MAE 改善但严重低估总量，尤其非零和 20+ 图书。
- 相比上一阶段新增/改变：从公式 baseline 进入机器学习树模型。
- 为下一阶段提供：为目标函数修正实验提供对照。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 13：LightGBM目标函数Pilot

- 目的：验证低估是否由 log1p + L1 造成，比较 Tweedie 与 Log-L2。
- 输入文件：`data/processed/model_dataset_monthly.parquet; models/final/lightgbm_*.txt`
- 执行脚本：`scripts/11_lightgbm_objective_pilot.py`
- 主要函数、算法或工具：LightGBM Tweedie power=1.2/1.4/1.6；Log-L2；同采样/权重/编码；完整 valid 流式评价。
- 输出文件：`reports/lightgbm_v2_objective_pilot_report.md; models/checkpoints/lightgbm_v2_pilot/*/*.txt; logs/lightgbm/v2_pilot/objective_pilot_*.json`
- 输出数据记录：pilot候选日志: 1; checkpoint大小: 18.09 MiB
- 质量检查：只做 pilot，不正式训练全部候选；valid 指标、总量偏差、零销量误报和非零/头部层级检查。
- 主要结论：推荐 Log-L2 作为 V2 正式方案。
- 相比上一阶段新增/改变：新增目标函数实验，不覆盖 V1。
- 为下一阶段提供：为 LightGBM V2 正式训练确定 objective。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 14：LightGBM V2

- 目的：正式训练 log1p + regression_l2 LightGBM。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml; models/final/lightgbm_*.txt`
- 执行脚本：`scripts/12_train_lightgbm_v2_logl2.py`
- 主要函数、算法或工具：ObjectiveSpec log_l2；复用 V1 采样、权重、类别编码和时间特征；完整 valid/test 评价。
- 输出文件：`models/final/lightgbm_v2_logl2_1m.txt; models/final/lightgbm_v2_logl2_2m.txt; data/outputs/lightgbm_v2_logl2_test_predictions.parquet; reports/lightgbm_v2_logl2_model_report.md; reports/lightgbm_v2_logl2_feature_importance_*.csv`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 9; 时间范围: 2026-01 to 2026-04; 大小: 147.34 MiB
- 质量检查：保护 V1 文件哈希；模型重载一致；test 不参与早停；valid/test 全量流式评价。
- 主要结论：V2 相对 V1 明显缓解低估，但仍低估总量；整体 MAE 是六模型中最好。
- 相比上一阶段新增/改变：替换 LightGBM 主比较版本为 V2 Log-L2 原始预测。
- 为下一阶段提供：为后续模型和集中比较提供主树模型基线。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 15：校准诊断

- 目的：用完整 valid 的真实总量/预测总量计算 V2 乘法校准因子，仅作诊断。
- 输入文件：`LightGBM V2 valid 流式评价结果`
- 执行脚本：`scripts/12_train_lightgbm_v2_logl2.py`
- 主要函数、算法或工具：valid-only calibration_factor；test 仅应用因子诊断，不覆盖原始预测。
- 输出文件：`reports/lightgbm_v2_logl2_model_report.md; data/outputs/lightgbm_v2_logl2_test_predictions.parquet 中 calibrated_pred 列`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 9; 时间范围: 2026-01 to 2026-04; 大小: 147.34 MiB
- 质量检查：校准因子只来自 valid；禁止 test 反推因子；原始 Log-L2 保留为主结果。
- 主要结论：校准可改善总量诊断，但不作为默认点预测模型。
- 相比上一阶段新增/改变：新增对总量偏差的诊断，不改变模型文件。
- 为下一阶段提供：为集中阶段解释“原始预测 vs 总量修正”提供背景。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 16：随机森林

- 目的：训练 log1p 目标的随机森林回归，对照 LightGBM。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml; LightGBM V2 结果`
- 执行脚本：`scripts/13_train_random_forest.py`
- 主要函数、算法或工具：sklearn RandomForestRegressor；确定性分层采样容量缩放；逆概率权重；类别编码；逐步树数选择。
- 输出文件：`models/final/random_forest_1m.joblib; models/final/random_forest_2m.joblib; data/outputs/random_forest_test_predictions.parquet; reports/random_forest_model_report.md; reports/random_forest_feature_importance_*.csv`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 7; 时间范围: 2026-01 to 2026-04; 大小: 149.65 MiB
- 质量检查：smoke/pilot/formal；模型重载一致；完整 valid/test 流式评价；内存与模型大小审计。
- 主要结论：RF 优于加权移动平均，并缓解总量低估；整体和 20+ 仍不如 LightGBM V2。
- 相比上一阶段新增/改变：新增非 boosting 树模型对照。
- 为下一阶段提供：为 MLP 和集中比较提供传统集成模型结果。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 17：MLP

- 目的：训练 PyTorch 表格 MLP，验证神经网络结构对照。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml; RF/LGB V2 结果`
- 执行脚本：`scripts/14_train_mlp.py`
- 主要函数、算法或工具：TabularMLP；数值标准化；类别 embedding；weighted MSE on log_target；AdamW；early stopping。
- 输出文件：`models/final/mlp_1m.pt; models/final/mlp_2m.pt; data/outputs/mlp_test_predictions.parquet; reports/mlp_model_report.md; reports/mlp_training_history_*.csv`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 7; 时间范围: 2026-01 to 2026-04; 大小: 91.23 MiB
- 质量检查：smoke/pilot/formal；train-only 标准化/类别映射；模型重载一致；完整流式评价。
- 主要结论：MLP 优于移动平均，但整体未超过 LightGBM V2；存在轻微过拟合倾向。
- 相比上一阶段新增/改变：新增神经网络结构对照。
- 为下一阶段提供：为最终六模型比较提供 MLP 结果。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 18：两阶段模型

- 目的：训练是否有销量分类器 + 正销量 Tweedie 回归器，改善长尾与总量。
- 输入文件：`data/processed/model_dataset_monthly.parquet; config/model_features.yaml; LightGBM V2 结果`
- 执行脚本：`scripts/15_train_two_stage.py`
- 主要函数、算法或工具：LightGBM binary + LightGBM Tweedie(power=1.4)；p_sale × conditional_qty；valid-best-F1 仅诊断。
- 输出文件：`models/final/two_stage_classifier_1m.txt; models/final/two_stage_regressor_1m.txt; models/final/two_stage_classifier_2m.txt; models/final/two_stage_regressor_2m.txt; data/outputs/two_stage_test_predictions.parquet; reports/two_stage_model_report.md; reports/two_stage_feature_importance.csv`
- 输出数据记录：文件数: 1; 行数: 13561039; 字段数: 11; 时间范围: 2026-01 to 2026-04; 大小: 239.96 MiB
- 质量检查：四组件 smoke/formal；分类 PR-AUC/ROC-AUC/logloss；正样本回归和组合预测完整评价；模型重载一致。
- 主要结论：两阶段显著改善总量偏差和非零/头部 WAPE，但零销量误报与整体 MAE 劣于 LightGBM V2。
- 相比上一阶段新增/改变：新增结构化长尾友好模型。
- 为下一阶段提供：为集中比较提供辅助模型候选。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

### 阶段 19：六模型统一比较

- 目的：在同一 test 样本上比较六类算法，生成暂定动销等级和 Top-K 识别评价。
- 输入文件：`data/outputs/baseline_predictions.parquet; data/outputs/lightgbm_v2_logl2_test_predictions.parquet; data/outputs/random_forest_test_predictions.parquet; data/outputs/mlp_test_predictions.parquet; data/outputs/two_stage_test_predictions.parquet`
- 执行脚本：`scripts/16_finalize_centralized_stage.py`
- 主要函数、算法或工具：按 row group 同步读取；一致性检查；LongTailStreamingMetrics；动销等级 Macro-F1/混淆矩阵；Top-K Precision/Sales Capture。
- 输出文件：`reports/centralized_stage_summary.md; reports/centralized_model_comparison.csv; reports/centralized_sales_level_metrics.csv; reports/centralized_topk_metrics.csv; data/outputs/centralized_prediction_sample.csv`
- 输出数据记录：比较CSV: 0.09 MiB; 人工样本: 8.00 MiB; test样本: 13561039
- 质量检查：检查样本数一致、key 对齐、target 一致、无重复/缺失/NaN/Inf/负预测。
- 主要结论：整体点预测和零销量控制以 LightGBM V2 最优；非零/头部/动销等级/Top-K 和总量接近度以两阶段最优。
- 相比上一阶段新增/改变：从单模型报告收口到集中式阶段统一评价。
- 为下一阶段提供：为老师阶段实验报告和后续联邦学习设计提供可追溯结论。
- 缺失、已删除或无法确认：未发现关键缺失；若使用通配符，表示存在多个分片文件。

## 3. 文件命名说明

- `raw`：原始输入区，保存未清洗 zip/csv。
- `interim`：中间层，保存清洗后分片流水 Parquet。
- `processed`：建模基础层，保存日度、月度和监督学习公共数据集。
- `outputs`：模型预测、人工检查样本等可由模型/脚本重新生成的输出。
- `daily`：日度粒度，通常为 `period × site_no × item_id`。
- `monthly`：月度粒度，通常为 `month × site_no × item_id`。
- `future`：未来真实观测标签来源，例如 `future_qty_1m`；不能作为输入特征。
- `target`：训练或评价时由 future 字段加工出的目标，例如 `target_qty_1m=max(future_qty_1m,0)`。
- `pred`：模型预测值，例如 `lightgbm_v2_logl2_pred_1m`。
- `qty`：销量数量；负值来自退货或净退货。
- `lag`：滞后特征，例如 `qty_lag_1m` 表示当前观察月之前 1 个月销量。
- `last`：历史窗口聚合，例如 `qty_sum_last_3m` 表示当前月之前 3 个月累计销量。
- `1m`：预测未来 1 个月。
- `2m`：预测未来 2 个月合计；集中阶段另有 2M 月均动销等级解释口径。
- `train / valid / test`：时间切分，不随机划分；train 训练，valid 早停/诊断，test 最终评价。
- `smoke`：小样本流程验证。
- `pilot`：中等样本资源与效果试运行。
- `formal`：正式训练或正式评价产物。
- `checkpoint`：中间模型或试验模型，可按脚本重建。
- `final`：正式模型，阶段报告引用的保留产物。

## 4. 模型参数审计说明

- 详细参数见 `reports/model_parameter_audit.csv`。
- LightGBM 参数来自正式模型文本末尾内嵌 metadata、`config/model_features.yaml` 和结构化日志。
- 随机森林参数来自 joblib bundle metadata 和 `logs/random_forest/formal_run_summary.json`。
- MLP 参数来自 `.pt` bundle metadata、网络结构和 `logs/mlp/formal_run_summary.json`。
- 两阶段模型参数分别来自四个 LightGBM 组件模型 metadata 和 `logs/two_stage/formal_run_summary.json`。
- 公式 baseline 没有训练过程和模型文件，审计表明确标记为“不适用”。

## 5. 缺失与可确认性结论

- 未发现关键正式输入、模型、预测或报告文件缺失。
- 审计表中“未找到”参数项数量：0。这类字段主要用于区分不适用与无法从项目文件确认的值。
- `data/interim/cleaned_sales.parquet` 未出现单文件版，但存在 `data/interim/cleaned_sales_*.parquet` 分片输出，符合前期“大数据按分片保存”的设计。
- 零销量月份补齐后的完整中间面板没有单独保存为大型 Parquet，只有最终 `data/processed/model_dataset_monthly.parquet`；这符合“不生成新大型中间文件”的设计。
- 部分早期报告正文在 Windows 控制台中可能显示为乱码；本审计优先使用结构化 JSON、模型 metadata、Parquet metadata 和 CSV 指标文件核对。

## 6. 审计生成信息

- 生成时间：2026-07-30 20:25:11 Asia/Shanghai
- 审计输出：`reports/project_artifact_audit.md`；`reports/model_parameter_audit.csv`
