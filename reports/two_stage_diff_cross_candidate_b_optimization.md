# Two-stage 1M Diff、Cross-store 与 Candidate B-v2 优化报告

## 执行边界

- 仅使用 Active-Store 1M Train 与 Complete Valid；未运行 Test、2M 或联邦学习。
- 高需求分类器未使用 `q`，因此不存在 Train in-sample q 泄露。
- 所有采样 Valid 指标均使用逆采样概率权重恢复真实分布。
- 现有 750/1054 正式模型和 Active-Store 数据集均未覆盖。

## Cross-store 旁表

- Rows: 2,499,208；size: 10.49 MiB；reused: True。
- 旁表只保存逐月集团正销量和正销量门店数；目标样本按 t/t-1/t-2 逐月扣除自身后再滚动。

## 全 Train 统计审计

- 扫描 57,163,300 行，确定性均匀样本 1,143,368 行。
- Diff 进入 Pilot：False。
- Cross-store 进入 Pilot：True。

| Feature | Group | AP(5+) | AP(20+) | Top-decile lift(5+) | Top-decile lift(20+) | Monthly direction | Max existing | Kept |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `qty_diff_current_lag1` | Diff | 0.1253 | 0.0663 | 4.5536 | 5.2352 | 0.6667 | qty_lag_1m (0.5928) | True |
| `qty_diff_lag1_lag2` | Diff | 0.0534 | 0.0157 | 3.1877 | 3.1524 | 0.6000 | qty_lag_2m (0.5938) | True |
| `log_qty_diff_current_lag1` | Diff | 0.0778 | 0.0347 | 3.8216 | 4.7936 | 0.6667 | qty_lag_1m (0.5990) | True |
| `sales_days_diff_current_lag1` | Diff | 0.1365 | 0.0599 | 4.5153 | 5.1147 | 0.6667 | sales_days_lag_1m (0.5934) | True |
| `qty_diff_recent3_previous3` | Diff | 0.0371 | 0.0103 | 0.9014 | 0.8661 | 0.3667 | qty_sum_last_3m (0.3078) | True |
| `qty_diff_6m_history_available` | Diff | 0.0071 | 0.0007 | 0.8338 | 0.8895 | 0.4333 | zero_sales_months_last_6m (0.7719) | True |
| `xstore_qty_current` | Cross-store | 0.0987 | 0.0373 | 6.2923 | 7.0607 | 1.0000 | qty_sum_last_6m (0.3435) | True |
| `xstore_positive_sites_current` | Cross-store | 0.0886 | 0.0356 | 5.7233 | 6.3484 | 1.0000 | qty_sum_last_6m (0.3438) | True |
| `xstore_qty_per_positive_site_current` | Cross-store | 0.0628 | 0.0165 | 5.4996 | 6.5643 | 1.0000 | qty_sum_last_6m (0.3088) | True |
| `xstore_qty_mean_last_3m` | Cross-store | 0.0680 | 0.0198 | 5.5049 | 6.2017 | 1.0000 | qty_sum_last_6m (0.3940) | True |
| `xstore_qty_max_last_3m` | Cross-store | 0.0701 | 0.0212 | 5.8540 | 6.8152 | 1.0000 | qty_sum_last_6m (0.3874) | True |
| `xstore_qty_diff_1m` | Cross-store | 0.0754 | 0.0302 | 5.2410 | 6.4311 | 1.0000 | qty_mean_last_3m (0.1299) | True |

## Pilot 消融

- Passed: True；selected: `E2`。
- E2: passed=True, AP gate=True, q gate=True, mean 20+ q-WAPE gain=1.6234 pp。

## Forward-time OOF 与 Candidate B-v2

- Passed: False；feature set: `E2`。
- Aggregate OOF diagnostic rule (not promoted): tau5=0.0520, tau20=0.0923, c5=1.0667, c20=1.1878。
- Last two folds same direction: False。
- q4_2024: 20+ WAPE gain 0.0000 pp, 20+ Recall gain 0.0000 pp, Zero total change -0.0000%。
- q1_2025: 20+ WAPE gain -0.0000 pp, 20+ Recall gain 0.0000 pp, Zero total change 0.0000%。
- q2_2025: 20+ WAPE gain -0.0000 pp, 20+ Recall gain 0.0000 pp, Zero total change -0.0000%。
- q4_2024 rule fit: eligible 0/25; rejected by 5-19=25, nonzero=0, zero-total=0, absolute-bias=15.
- q1_2025 rule fit: eligible 0/25; rejected by 5-19=20, nonzero=0, zero-total=1, absolute-bias=22.
- q2_2025 rule fit: eligible 0/25; rejected by 5-19=20, nonzero=0, zero-total=0, absolute-bias=6.

### Candidate B-v2的25个候选是什么

- `medium`候选比例为预测期分数分布的Top 0.25%、0.5%、1%、2%、5%；`high`候选比例为Top 0.05%、0.1%、0.25%、0.5%、1%。两个集合各5档，逐一组合得到`5 x 5 = 25`个候选。
- 每个规则拟合期均使用逆采样概率权重恢复真实分布，再分别从`s5=P(Y>=5)`和`s20=min(P(Y>=20), s5)`求对应Top比例的加权分位点`tau5/tau20`。`s20>=tau20`为high，`s5>=tau5`且不属于high为medium，其余为ordinary。
- ordinary保持`c=1`。medium和high分别按本档聚合后的`sum(y)/sum(p_sale*q)`估计低估比例，并使用`c=clip(sqrt(max(ratio,1)),1,1.5)`得到`c5/c20`；不存在聚合低估时倍率自动退回1，禁止逐样本使用真实未来销量定倍率。
- forward流程严格为：2024-Q3拟合规则后评价Q4；Q3-Q4拟合后评价2025-Q1；Q3-Q1拟合后评价2025-Q2。下一期不参与上一期阈值或倍率拟合。
- 每个候选必须同时满足：5-19 WAPE恶化不超过0.25个百分点、Nonzero WAPE恶化不超过0.5个百分点、Zero Pred Total增加不超过1%、绝对Total Bias恶化不超过2个百分点。
- Q4规则拟合期的25个候选中：25个触发5-19门槛，0个触发Nonzero，0个触发Zero Pred Total，15个触发Total Bias。Q1对应为20、0、1、22；Q2对应为20、0、0、6。同一候选可以同时触发多个失败条件，因此各项失败计数不应相加为25。
- 三个拟合期均没有候选通过全部保护条件，实际评价规则因此回退为`tau=inf, c=1`，也就是保持原预测不变。报告中的20+ WAPE gain和20+ Recall gain约为0，含义是“没有补偿规则获准应用”，不是25种补偿公式都运行后恰好得到零收益。
- Cross-store显著提升5+/20+分类器PR-AUC，说明**高需求识别分数更有效**；Candidate B仍失败，说明当前“按分数分档后乘固定倍率”的**补偿机制无法保护5-19等业务指标**。识别能力和补偿决策是两个不同环节，两者结论可以同时成立。

### Diff为什么Kept=True但整组未进入Pilot

- `Kept=True`只表示单字段未被预删除：该字段不是99.99%以上同值的近常量，并且没有出现“与已有字段绝对Spearman相关性>=0.98且PR-AUC不优于已有字段”的冗余组合。它不表示该字段已经证明具有跨月份稳定增量。
- Diff组进入Pilot的组级条件更严格：至少一个Kept字段必须对5+或20+达到Top-decile Lift>1.10，并且该方向在至少75%的Train月份一致。
- `qty_diff_current_lag1`相对最好：5+/20+ Top-decile Lift为4.5536/5.2352；`sales_days_diff_current_lag1`为4.5153/5.1147；`log_qty_diff_current_lag1`为3.8216/4.7936。它们的Lift足够，但最高monthly direction仅66.67%，低于75%门槛。
- `qty_diff_recent3_previous3`和`qty_diff_6m_history_available`的Top-decile Lift仅约0.83-0.90，方向一致率最高43.33%，既没有稳定提升，也没有达到Lift门槛。
- 因此Diff不是因PR-AUC字段值为空、近常量或0.98高度重复而被删，而是因为**没有任何单个Diff信号同时满足Lift和跨月稳定性两个组级条件**。所有字段可以保持Kept=True，但E1仍按stop gate不进入Pilot。

## Candidate B-v2 Full Train 与 Complete Valid

Candidate B-v2分支因forward-time OOF业务门槛失败而未执行；该停止结论不等同于Cross-store E2特征分支失败。

## 结论

**Candidate B-v2分支在OOF门槛后停止。Cross-store E2特征分支随后按独立正式验证继续，结果见报告末节。**

- Cross-store通过全Train统计审计和两折Pilot，但Candidate B-v2未通过forward-time OOF业务门槛。
- 三个滚动规则拟合期的可用候选数为 `[0, 0, 0]`；主要停止原因是5–19 WAPE保护门槛，非模型训练失败。
- 因此Candidate B-v2分支未执行全Train重训、Complete Valid或Test，也未生成或提升任何Candidate B optimization_v2正式模型。

本流程中的 0.5pp、1pp、Zero Pred +1% 等均为项目工程筛选规则，不解释为理论显著性标准。

## 追溯

- Log: `logs\lightgbm\two_stage_optimization\run_20260819_203315.log`。
- Ablation CSV: `reports\two_stage_diff_cross_ablation.csv`。

## Cross-store E2独立正式验证

### 正式训练与选轮

- 本分支只重训positive-sales Tweedie regressor；原`P(Y>0)` classifier继续冻结为750轮。高需求5+/20+分类器未重训，也未接回Candidate B。
- 轮次只来自四个Train内部forward-time E2 regressor fold的验证量加权中位数：`657`轮；Complete Valid未参与选轮。
- 完整Train覆盖`2023-01`至`2025-06`共30个月；正式正销量分层样本1,827,396行。
- 特征数63；训练耗时0.73分钟；训练峰值RAM 1.62 GiB；模型大小5.36 MiB。
- Complete Valid共18,563,573行，流式评价耗时33.99分钟，评价峰值RAM 1.21 GiB；未运行Test。

四折选轮依据：

| Fold | Best iteration | Valid rows |
|---|---:|---:|
| q1_2025 | 1136 | 97,798 |
| q2_2025 | 1357 | 121,023 |
| q3_2024 | 382 | 122,836 |
| q4_2024 | 657 | 107,822 |

### q层结果

| Model | Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall | 20+ median(q/y) |
|---|---:|---:|---:|---:|---:|---:|
| E0 q1054 | 57.9745% | 52.4573% | 71.0175% | 42.4046% | 28.5697% | 0.301663 |
| E2 Cross-store q | 55.2883% | 52.3412% | 68.7368% | 41.3783% | 32.5749% | 0.350284 |

### 最终p_sale × q结果

| Model | Raw Nonzero WAPE | Integer Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall | MC Macro-F1 | Overall Raw WAPE | Integer WAPE | Total Bias | Zero Pred Total | Zero >0.5 | Zero >1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 750×1054 | 72.6041% | 75.5189% | 64.4641% | 75.1482% | 30.4241% | 24.5053% | 42.4214% | 134.5989% | 101.9824% | 8.9329% | 1530369.4350 | 2.9452% | 0.9687% |
| E2 750×Cross-store q | 71.5927% | 74.5370% | 64.2626% | 73.0038% | 29.6420% | 27.6692% | 43.5114% | 130.3257% | 97.8903% | 4.9883% | 1449851.9737 | 2.6756% | 0.8242% |

### 月度稳定性（最终p_sale × q）

| Month | Model | Nonzero WAPE | 5-19 WAPE | 5-19 Recall | 20+ WAPE | 20+ Recall | Total Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025-07 | E0 | 72.4260% | 58.8501% | 40.3522% | 69.4473% | 35.3798% | 19.7476% |
| 2025-07 | E2 | 71.1340% | 58.6826% | 38.6474% | 68.3182% | 35.3798% | 13.3230% |
| 2025-08 | E0 | 70.7091% | 64.1712% | 33.0325% | 69.5506% | 34.0764% | -9.7009% |
| 2025-08 | E2 | 68.6172% | 63.4520% | 32.0367% | 66.3904% | 39.3843% | -12.0329% |
| 2025-09 | E0 | 72.5075% | 63.9342% | 32.2873% | 80.1954% | 16.8919% | 8.1055% |
| 2025-09 | E2 | 72.6712% | 65.0132% | 31.5462% | 80.3268% | 17.4831% | 6.4685% |
| 2025-10 | E0 | 72.7014% | 61.3102% | 28.0775% | 88.0174% | 12.4424% | 14.3885% |
| 2025-10 | E2 | 72.3513% | 61.2920% | 27.3472% | 88.7160% | 10.5991% | 11.8038% |
| 2025-11 | E0 | 75.6453% | 68.0069% | 21.1433% | 89.2768% | 10.0503% | 14.7200% |
| 2025-11 | E2 | 75.5941% | 68.1919% | 19.8512% | 89.6859% | 7.5377% | 11.3182% |
| 2025-12 | E0 | 73.9726% | 69.5503% | 22.0509% | 84.4492% | 8.8548% | 21.9238% |
| 2025-12 | E2 | 72.7671% | 68.7220% | 22.3951% | 81.5077% | 13.4593% | 14.2749% |

### 阶段判断

- E2 Cross-store正式结果：**未通过正式验收，不晋级**。月度20+改善月份数为4/6。
- 业务门槛：`{"one_head_wape_improved": true, "other_head_wape_protected": true, "one_head_recall_improved": true, "other_head_recall_protected": false, "overall_protected": true, "zero_protected": true, "bias_protected": true, "mc_protected": true, "monthly_stable": true}`。
- q层20+ WAPE变化：2.2807个百分点；20+ Recall变化：4.0052个百分点；median(q/y)变化：0.048621。
- 最终5-19 WAPE变化：-0.2015个百分点；Nonzero WAPE变化：-1.0115个百分点。
- Zero Pred Total变化：-5.2613%；Total Bias变化：-3.9446个百分点。
- Train-only类别映射在Complete Valid的Unknown占比：`{"blt_site_no": 0.0, "gds_ctgry_3_lvel": 0.0, "gds_ctgry_4_lvel": 0.0, "gds_ctgry_5_lvel": 0.0, "site_no": 0.0}`。
- 晋级模型：`无`。原750/1054模型未覆盖。
- Candidate B-v2仍冻结；本轮结果不改变其当前失败结论，也未自动重新启动补偿实验。
- Cross-store对条件销量幅度具有明确增量：q层20+ WAPE改善2.2807个百分点、20+ Recall提高4.0052个百分点，`median(q/y)`由0.301663升至0.350284；导师提出的“同书跨门店相关性”可以认定为**有效预测信号**。
- 该信号传递到最终`p_sale × q`后，20+ WAPE改善2.1444个百分点、20+ Recall提高3.1639个百分点，Nonzero WAPE改善1.0115个百分点；同时Zero Pred Total下降5.2613%，Total Bias从+8.9329%收窄到+4.9883%，不是以放大零销量误报换取头部收益。
- 但5-19 Recall由30.4241%降至29.6420%，下降0.7821个百分点，超过项目允许下降0.25个百分点的保护线；此外10月、11月20+ WAPE和Recall均反向恶化。因而E2只能认定为**值得保留研究的特征方向**，不能认定为稳定的正式替代模型。
- 下一阶段如继续，应保留Cross-store特征分支并专门解决5-19掉级与10/11月漂移；不应复用Complete Valid重新调657轮，也不应在本轮恢复Candidate B。由于未通过既定正式门槛，本次checkpoint已删除，未写入`models/final`。

### 本次追溯

- Log: `logs\lightgbm\two_stage_optimization\run_20260819_220046.log`。
- Ablation CSV已追加E2正式行：`reports\two_stage_diff_cross_ablation.csv`。

## E2 5–19误差迁移与10/11月20+时间漂移诊断

### 重建边界与一致性

- 诊断恢复模型固定为63特征、657轮、Tweedie regressor；训练样本1,827,396，未重新选特征、轮次、参数或采样率。
- Complete Valid行数18,563,573；流式推理23.94分钟；峰值RAM 1.29 GiB；未运行Test。
- 重建checkpoint：`models\checkpoints\two_stage_optimization\cross_store_diagnostic\two_stage_regressor_cross_store_1m_rebuilt.txt`，本轮保留但不进入`models/final`。

| Metric | Previous E2 | Rebuilt E2 | Difference | Tolerance | Pass |
|---|---:|---:|---:|---:|---|
| q_20_plus_wape | 68.736826 | 68.819528 | 0.08270171 | 0.000100 | False |
| q_20_plus_recall | 32.574950 | 32.420903 | -0.15404669 | 0.000100 | False |
| final_20_plus_wape | 73.003756 | 73.054681 | 0.05092462 | 0.000100 | False |
| final_20_plus_recall | 27.669155 | 27.526958 | -0.14219694 | 0.000100 | False |
| final_nonzero_wape | 71.592691 | 71.454857 | -0.13783423 | 0.000100 | False |
| overall_wape | 130.325726 | 129.925144 | -0.40058216 | 0.000100 | False |
| total_bias | 4.988277 | 4.439726 | -0.54855095 | 0.000100 | False |
| zero_prediction_total | 1449851.973675 | 1443365.920202 | -6486.05347213 | 0.010000 | False |

- Strict aggregate reproduction: failed because the prior DuckDB cache scan did not impose a stable row order.
- The source data, deterministic sampled row set, 63-feature order, category maps, Tweedie parameters and 657 rounds are unchanged. A same-cache repeat produced identical predictions (maximum absolute difference 0), isolating the variance to row-order-sensitive LightGBM bagging.
- Diagnostic equivalence: passed. Maximum metric delta=0.548551 percentage points; Zero Pred Total relative delta=0.4474%. This does not claim bitwise reproduction.

一致性校验通过，以下诊断仅使用同一批Complete Valid样本。

### E2 5–19误差迁移诊断

真实5–19样本的预测MC边际分布：

| Predicted MC | E0 count | E0 share | E2 count | E2 share | Count change |
|---|---:|---:|---:|---:|---:|
| 0 | 11,780 | 17.2215% | 11,690 | 17.0899% | -90 |
| 1 | 12,428 | 18.1688% | 12,363 | 18.0738% | -65 |
| 2-4 | 22,201 | 32.4562% | 22,946 | 33.5453% | +745 |
| 5-19 | 20,811 | 30.4241% | 20,258 | 29.6157% | -553 |
| 20+ | 1,183 | 1.7295% | 1,146 | 1.6754% | -37 |

E0判对5–19但E2判错的拆分：

| E2 destination | Count | y mean/median | E0 pred mean/median | E2 pred mean/median | E0 MAE | E2 MAE | E0 WAPE | E2 WAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20+ | 345 | 12.0087/12.0000 | 15.3105/16.2107 | 23.9176/22.5573 | 5.0980 | 11.9089 | 42.4530% | 99.1691% |
| 2-4 | 2,481 | 7.6775/7.0000 | 5.5375/5.1236 | 3.7894/3.9441 | 2.6402 | 3.8882 | 34.3879% | 50.6434% |
| 1/0 | 3 | 11.3333/11.0000 | 5.7283/5.7318 | 1.3737/1.3787 | 5.6050 | 9.9596 | 49.4560% | 87.8789% |

完整E0→E2配对转移（只列非零格）：

| E0 MC | E2 MC | Count |
|---|---|---:|
| 0 | 0 | 10,929 |
| 0 | 1 | 842 |
| 0 | 2-4 | 9 |
| 1 | 0 | 761 |
| 1 | 1 | 10,013 |
| 1 | 2-4 | 1,638 |
| 1 | 5-19 | 16 |
| 2-4 | 1 | 1,505 |
| 2-4 | 2-4 | 18,817 |
| 2-4 | 5-19 | 1,875 |
| 2-4 | 20+ | 4 |
| 5-19 | 1 | 3 |
| 5-19 | 2-4 | 2,481 |
| 5-19 | 5-19 | 17,982 |
| 5-19 | 20+ | 345 |
| 20+ | 2-4 | 1 |
| 20+ | 5-19 | 385 |
| 20+ | 20+ | 797 |

Cross-store与本店历史信号（mean / median）：

| Feature | E2 correct 5-19 | E2 to 20+ | E2 to <=2-4 |
|---|---:|---:|---:|
| `xstore_qty_current` | 613.1250 / 208.0000 | 1854.6431 / 732.5000 | 165.8830 / 48.0000 |
| `xstore_positive_sites_current` | 71.4036 / 64.0000 | 106.5087 / 117.5000 | 37.3419 / 25.0000 |
| `xstore_qty_per_positive_site_current` | 5.6521 / 3.3733 | 12.9786 / 6.9616 | 2.7434 / 1.9000 |
| `xstore_qty_mean_last_3m` | 452.2920 / 186.3333 | 1101.9008 / 525.3333 | 157.2381 / 45.6667 |
| `xstore_qty_max_last_3m` | 832.9200 / 295.0000 | 2155.8019 / 907.5000 | 280.3739 / 75.0000 |
| `xstore_qty_diff_1m` | 168.1341 / 17.0000 | 817.3909 / 165.0000 | -3.9163 / 0.0000 |
| `total_qty` | 12.5579 / 9.0000 | 45.8368 / 20.0000 | 2.1238 / 1.0000 |
| `qty_lag_1m` | 9.8626 / 6.0000 | 23.0942 / 13.0000 | 2.4476 / 1.0000 |
| `qty_lag_2m` | 6.9749 / 2.0000 | 11.4852 / 3.0000 | 2.0798 / 0.0000 |
| `qty_lag_3m` | 5.1767 / 0.0000 | 7.2408 / 0.0000 | 1.5454 / 0.0000 |
| `qty_mean_last_3m` | 7.7595 / 4.6667 | 15.3245 / 8.1667 | 2.1698 / 0.6667 |
| `qty_max_last_3m` | 13.9323 / 8.0000 | 27.8752 / 16.0000 | 4.0239 / 2.0000 |
| `sales_days` | 7.1019 / 6.0000 | 11.8578 / 11.0000 | 1.6963 / 1.0000 |
| `active_months_last_6m` | 3.2268 / 3.0000 | 3.4721 / 3.0000 | 2.1833 / 2.0000 |
| `months_since_last_sale` | 1.5988 / 1.0000 | 1.2740 / 1.0000 | 3.6071 / 1.0000 |

### 20+月度时间漂移诊断

| Month | N | y total | y median/P90/P95 | E0 q/y med | E2 q/y med | E0/E2 q Recall | E0/E2 final Recall | E0/E2 WAPE | p_sale mean/median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07 | 961 | 45357.0000 | 29.0000/81.0000/129.0000 | 0.417143 | 0.397525 | 37.1488%/37.5650% | 35.3798%/35.0676% | 69.4473%/68.4741% | 0.7442/0.9304 |
| 2025-08 | 3,768 | 225213.0000 | 35.0000/107.0000/166.6500 | 0.378911 | 0.460190 | 38.3758%/44.5860% | 34.0764%/38.7473% | 69.5506%/66.4923% | 0.7391/0.8620 |
| 2025-09 | 1,184 | 52087.0000 | 29.0000/67.0000/102.8500 | 0.196948 | 0.214956 | 18.4966%/20.2703% | 16.8919%/18.3277% | 80.1954%/79.7882% | 0.5620/0.6493 |
| 2025-10 | 434 | 23467.0000 | 27.0000/67.5000/94.8000 | 0.138081 | 0.130505 | 13.3641%/12.9032% | 12.4424%/11.7512% | 88.0174%/88.9254% | 0.5119/0.5504 |
| 2025-11 | 398 | 21050.0000 | 35.0000/85.6000/121.1500 | 0.104044 | 0.099467 | 12.0603%/10.5528% | 10.0503%/8.2915% | 89.2768%/89.5332% | 0.4427/0.3408 |
| 2025-12 | 1,694 | 70629.0000 | 28.0000/67.0000/103.0000 | 0.255144 | 0.294520 | 16.7060%/21.0744% | 8.8548%/13.2822% | 84.4492%/81.7716% | 0.4561/0.4377 |

Cross-store字段月度分布；`Train pct`表示该月真实20+中位数位于Train真实20+分布的百分位：

| Month | Feature | Mean | P25 | Median | P75 | Train pct |
|---|---|---:|---:|---:|---:|---:|
| 2025-07 | `xstore_qty_current` | 1311.5411 | 177.0000 | 721.0000 | 1327.0000 | 80.7100% |
| 2025-07 | `xstore_positive_sites_current` | 101.2425 | 52.0000 | 118.0000 | 143.0000 | 82.1567% |
| 2025-07 | `xstore_qty_per_positive_site_current` | 9.0907 | 3.1667 | 6.3766 | 9.3427 | 74.6501% |
| 2025-07 | `xstore_qty_mean_last_3m` | 615.3313 | 88.6667 | 426.0000 | 654.0000 | 79.2726% |
| 2025-07 | `xstore_qty_max_last_3m` | 1328.5702 | 206.0000 | 729.0000 | 1331.0000 | 77.6377% |
| 2025-07 | `xstore_qty_diff_1m` | 912.6795 | 30.0000 | 316.0000 | 955.0000 | 77.3695% |
| 2025-08 | `xstore_qty_current` | 1027.3888 | 78.0000 | 311.0000 | 990.0000 | 64.0359% |
| 2025-08 | `xstore_positive_sites_current` | 84.7107 | 28.0000 | 80.0000 | 135.0000 | 66.8282% |
| 2025-08 | `xstore_qty_per_positive_site_current` | 7.8066 | 2.4425 | 4.1837 | 11.3604 | 58.8864% |
| 2025-08 | `xstore_qty_mean_last_3m` | 674.7338 | 55.3333 | 337.0000 | 802.2500 | 75.6804% |
| 2025-08 | `xstore_qty_max_last_3m` | 1249.6746 | 122.0000 | 515.0000 | 1883.2500 | 70.3592% |
| 2025-08 | `xstore_qty_diff_1m` | 277.3140 | -198.0000 | -10.0000 | 177.5000 | 21.4261% |
| 2025-09 | `xstore_qty_current` | 1373.6292 | 19.7500 | 129.0000 | 646.2500 | 45.2681% |
| 2025-09 | `xstore_positive_sites_current` | 60.1774 | 10.7500 | 42.0000 | 98.0000 | 46.5925% |
| 2025-09 | `xstore_qty_per_positive_site_current` | 10.5114 | 1.5000 | 2.8571 | 8.7678 | 41.4877% |
| 2025-09 | `xstore_qty_mean_last_3m` | 726.8840 | 22.6667 | 100.5000 | 361.0000 | 49.1590% |
| 2025-09 | `xstore_qty_max_last_3m` | 1441.6841 | 31.0000 | 161.0000 | 708.7500 | 45.0258% |
| 2025-09 | `xstore_qty_diff_1m` | 991.8117 | 0.0000 | 30.0000 | 549.0000 | 48.6556% |
| 2025-10 | `xstore_qty_current` | 288.7811 | 7.0000 | 73.0000 | 292.5000 | 34.6303% |
| 2025-10 | `xstore_positive_sites_current` | 46.7604 | 4.0000 | 25.5000 | 79.5000 | 33.2330% |
| 2025-10 | `xstore_qty_per_positive_site_current` | 4.2587 | 1.5000 | 2.6991 | 5.1020 | 38.7871% |
| 2025-10 | `xstore_qty_mean_last_3m` | 478.8095 | 7.7500 | 72.5000 | 306.4167 | 42.6263% |
| 2025-10 | `xstore_qty_max_last_3m` | 857.6429 | 13.0000 | 113.5000 | 444.2500 | 37.6203% |
| 2025-10 | `xstore_qty_diff_1m` | -534.9839 | -62.0000 | 0.0000 | 9.0000 | 30.3465% |
| 2025-11 | `xstore_qty_current` | 152.3065 | 4.0000 | 48.0000 | 176.0000 | 28.0035% |
| 2025-11 | `xstore_positive_sites_current` | 37.0000 | 3.0000 | 19.0000 | 55.0000 | 27.3824% |
| 2025-11 | `xstore_qty_per_positive_site_current` | 2.9795 | 1.1190 | 2.2087 | 4.1634 | 30.5747% |
| 2025-11 | `xstore_qty_mean_last_3m` | 224.0302 | 9.7500 | 62.1667 | 154.7500 | 39.4458% |
| 2025-11 | `xstore_qty_max_last_3m` | 405.7161 | 16.0000 | 93.5000 | 281.5000 | 34.0187% |
| 2025-11 | `xstore_qty_diff_1m` | -20.7563 | -38.7500 | -2.0000 | 7.0000 | 25.8039% |
| 2025-12 | `xstore_qty_current` | 74.8312 | 7.0000 | 29.0000 | 84.0000 | 21.3461% |
| 2025-12 | `xstore_positive_sites_current` | 25.8837 | 5.0000 | 16.0000 | 43.0000 | 24.4513% |
| 2025-12 | `xstore_qty_per_positive_site_current` | 2.0927 | 1.2222 | 1.6000 | 2.3800 | 18.5937% |
| 2025-12 | `xstore_qty_mean_last_3m` | 57.5933 | 3.3333 | 16.3333 | 59.6667 | 18.2220% |
| 2025-12 | `xstore_qty_max_last_3m` | 95.4793 | 8.0000 | 35.0000 | 108.7500 | 19.3277% |
| 2025-12 | `xstore_qty_diff_1m` | 29.9776 | 0.0000 | 13.0000 | 39.0000 | 40.9419% |

逐月`Cross-store热度 → 未来20+`关系（全部20+正例 + 2%确定性负例，逆概率加权）：

| Month | Feature | Weighted PR-AUC | Top-decile actual 20+ rate | Top-decile lift |
|---|---|---:|---:|---:|
| 2025-07 | `xstore_qty_current` | 0.047062 | 0.2653% | 8.2793 |
| 2025-07 | `xstore_positive_sites_current` | 0.058979 | 0.2530% | 7.8956 |
| 2025-07 | `xstore_qty_per_positive_site_current` | 0.018544 | 0.2287% | 7.1352 |
| 2025-07 | `xstore_qty_mean_last_3m` | 0.049072 | 0.2703% | 8.4348 |
| 2025-07 | `xstore_qty_max_last_3m` | 0.045388 | 0.2680% | 8.3623 |
| 2025-07 | `xstore_qty_diff_1m` | 0.044668 | 0.2472% | 7.7135 |
| 2025-08 | `xstore_qty_current` | 0.237837 | 1.0388% | 8.3459 |
| 2025-08 | `xstore_positive_sites_current` | 0.222280 | 1.0115% | 8.1268 |
| 2025-08 | `xstore_qty_per_positive_site_current` | 0.097654 | 1.0543% | 8.4704 |
| 2025-08 | `xstore_qty_mean_last_3m` | 0.151128 | 0.9935% | 7.9823 |
| 2025-08 | `xstore_qty_max_last_3m` | 0.144236 | 1.0443% | 8.3902 |
| 2025-08 | `xstore_qty_diff_1m` | 0.170725 | 0.3236% | 2.5995 |
| 2025-09 | `xstore_qty_current` | 0.015646 | 0.2569% | 6.6460 |
| 2025-09 | `xstore_positive_sites_current` | 0.013532 | 0.2384% | 6.1674 |
| 2025-09 | `xstore_qty_per_positive_site_current` | 0.009859 | 0.2298% | 5.9442 |
| 2025-09 | `xstore_qty_mean_last_3m` | 0.013475 | 0.2571% | 6.6515 |
| 2025-09 | `xstore_qty_max_last_3m` | 0.013748 | 0.2590% | 6.7005 |
| 2025-09 | `xstore_qty_diff_1m` | 0.016082 | 0.2226% | 5.7589 |
| 2025-10 | `xstore_qty_current` | 0.005306 | 0.0834% | 5.9648 |
| 2025-10 | `xstore_positive_sites_current` | 0.004187 | 0.0739% | 5.2846 |
| 2025-10 | `xstore_qty_per_positive_site_current` | 0.001571 | 0.0786% | 5.6210 |
| 2025-10 | `xstore_qty_mean_last_3m` | 0.003207 | 0.0835% | 5.9735 |
| 2025-10 | `xstore_qty_max_last_3m` | 0.002908 | 0.0829% | 5.9307 |
| 2025-10 | `xstore_qty_diff_1m` | 0.000742 | 0.0468% | 3.3501 |
| 2025-11 | `xstore_qty_current` | 0.003290 | 0.0756% | 5.9884 |
| 2025-11 | `xstore_positive_sites_current` | 0.002940 | 0.0650% | 5.1474 |
| 2025-11 | `xstore_qty_per_positive_site_current` | 0.001964 | 0.0769% | 6.0963 |
| 2025-11 | `xstore_qty_mean_last_3m` | 0.001513 | 0.0755% | 5.9828 |
| 2025-11 | `xstore_qty_max_last_3m` | 0.001376 | 0.0764% | 6.0541 |
| 2025-11 | `xstore_qty_diff_1m` | 0.001400 | 0.0375% | 2.9746 |
| 2025-12 | `xstore_qty_current` | 0.007414 | 0.3127% | 5.8945 |
| 2025-12 | `xstore_positive_sites_current` | 0.004321 | 0.2924% | 5.5116 |
| 2025-12 | `xstore_qty_per_positive_site_current` | 0.002638 | 0.2607% | 4.9139 |
| 2025-12 | `xstore_qty_mean_last_3m` | 0.002322 | 0.2180% | 4.1083 |
| 2025-12 | `xstore_qty_max_last_3m` | 0.002473 | 0.2573% | 4.8504 |
| 2025-12 | `xstore_qty_diff_1m` | 0.009874 | 0.3390% | 6.3893 |

极端销量对20+误差贡献：

| Month | Top1% y threshold | N | E0 abs-error share | E2 abs-error share | Share of E2-E0 deterioration |
|---|---:|---:|---:|---:|---:|
| 2025-07 | 254.0000 | 10 | 14.9579% | 15.4672% | N/A% |
| 2025-08 | 433.0000 | 39 | 16.3274% | 16.8477% | N/A% |
| 2025-09 | 278.0000 | 12 | 14.4175% | 13.9928% | N/A% |
| 2025-10 | 566.0000 | 5 | 35.6843% | 35.3199% | -0.0090% |
| 2025-11 | 353.0000 | 4 | 11.6838% | 11.6497% | -0.2137% |
| 2025-12 | 267.0000 | 17 | 9.2294% | 9.5294% | N/A% |

10/11月明显恶化样本集中性：

- 定义：`E2绝对误差-E0绝对误差 > max(2本, 10%真实销量)`；命中63/832条。

`site_no`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| 6013 | 10 | 15.8730% | 7.9327% | 2.0010 |

`category_3`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| R1300 | 30 | 47.6190% | 16.1058% | 2.9566 |
| R1200 | 18 | 28.5714% | 23.6779% | 1.2067 |

`category_5`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| MC130302 | 21 | 33.3333% | 3.3654% | 9.9048 |

`history_level`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| 20+ | 42 | 66.6667% | 22.4760% | 2.9661 |
| 5-19 | 19 | 30.1587% | 20.5529% | 1.4674 |
| 1-4 | 2 | 3.1746% | 56.9712% | 0.0557 |

`short_history`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| 3plus_active_months | 49 | 77.7778% | 40.2644% | 1.9317 |
| lt3_active_months | 14 | 22.2222% | 59.7356% | 0.3720 |

`xstore_heat`集中度最高项：

| Value | Worse count | Worse share | All share | Concentration ratio |
|---|---:|---:|---:|---:|
| very_high | 3 | 4.7619% | 0.8413% | 5.6599 |
| high | 33 | 52.3810% | 12.2596% | 4.2726 |
| mid | 15 | 23.8095% | 19.3510% | 1.2304 |
| low | 12 | 19.0476% | 67.5481% | 0.2820 |

### 本轮可读结论：5-19误差迁移与10/11月时间漂移

#### 重建一致性说明

- 本轮严格复用了完整 Active-Store Train、1,827,396 条确定性分层正销量样本、57 个基础特征、6 个 Cross-store 特征、Train-only 类别映射、原 Tweedie 参数和固定 657 轮。
- 严格逐值复现没有通过。原因不是数据、特征、参数或样本集合变化，而是上一轮 DuckDB 训练缓存没有固定行顺序；LightGBM 同时启用了 `bagging_fraction=0.8`、`bagging_freq=1`，因此相同样本在不同行序下会形成不同的行位置抽样。
- 证据：两次缓存最终样本数均为 1,827,396，但分批累计行数和 Parquet 大小不同；在本轮固定缓存上重复训练 657 轮，抽查预测最大绝对差为 0。
- 重建模型相对上一轮 E2 的最大指标差为 0.5486 个百分点，Zero Pred Total 相对差为 0.4474%；E0 到 E2 的主要改善方向均未反转。因此以下内容只作为“同规格随机实现的误差结构诊断”，不声称逐位恢复了已删除的上一轮 E2。
- 以后若要正式复现模型，训练缓存必须按稳定样本键排序，并记录样本集合哈希、顺序哈希和模型文件哈希。

#### 问题A：5-19 Recall下降去了哪里

本次重建实现中，真实5-19共68,403条。E0正确识别20,811条，E2正确识别20,258条，净减少553条，即0.8084个百分点；上一轮正式E2记录的降幅为0.7821个百分点，两者方向一致。

| 去向/来源 | 样本数 | 对Recall的含义 |
|---|---:|---|
| E0判对5-19，E2降到2-4 | 2,481 | 主要向下损失 |
| E0判对5-19，E2降到1/0 | 3 | 向下损失很少 |
| E0判对5-19，E2升到20+ | 345 | 向上越界损失 |
| E0判成1/2-4，E2改正为5-19 | 1,891 | 从下方补回 |
| E0判成20+，E2改正为5-19 | 385 | 从上方补回 |

净变化可以拆成：下边界净损失`2,484 - 1,891 = 593`条；上边界反而净补回`385 - 345 = 40`条；合计净减少553条。因此 **0.78至0.81个百分点的Recall下降主要流向2-4，不是主要流向20+**。

这些E0判对但E2判错的样本，连续数值误差也确实变差，而不只是MC边界效应：

| E2去向 | N | E0到E2 MAE | E0到E2 WAPE | 判断 |
|---|---:|---:|---:|---|
| 20+ | 345 | 5.0980到11.9089 | 42.4530%到99.1691% | Cross-store过热，明显推高过头 |
| 2-4 | 2,481 | 2.6402到3.8882 | 34.3879%到50.6434% | Cross-store/本店历史偏冷，进一步压低 |
| 1/0 | 3 | 5.6050到9.9596 | 49.4560%到87.8789% | 样本极少，但同样是真实恶化 |

5-19总体WAPE没有同步恶化，是因为WAPE按销量绝对误差加权，E2在其他大量5-19样本上的连续误差改善抵消了边界附近的Recall损失；Recall则只看整数化后是否仍落在5-19，一旦跨过5或20就整条计错。两者衡量的不是同一件事。

Cross-store信号对迁移方向非常清楚：

- 被推到20+的样本明显“跨店和本店都很热”：`xstore_qty_current`中位数732.5、活跃其他门店117.5、每店销量6.96、近3月跨店最大值907.5；本店当前销量中位数20、lag1为13。
- 被压到2-4及以下的样本明显偏冷：对应中位数分别为48、25、1.90、75；本店当前销量和lag1都只有1，近3月均值仅0.67。
- 保持正确5-19的样本位于两者之间。说明E2并非随机掉级，而是对Cross-store热度较强地进行幅度迁移：极热时容易推高，偏冷时容易压低。

#### 问题B：2025-10、11月20+为什么反向恶化

这是多因素共同作用，其中“分布漂移、关系漂移、原`p_sale`同期偏低”比极端爆款更重要。

1. **Cross-store分布明显偏冷。** 真实20+样本中，10月六个Cross-store字段的中位数大多只处于Train 20+分布的第30%至43%；11月约第26%至39%。例如`xstore_qty_current`中位数从7月的721降到10月73、11月48。
2. **相同信号的关系也变弱，尤其月度变化特征。** `xstore_qty_diff_1m`的未来20+ Top-decile Lift从7月7.71、9月5.76降到10月3.35、11月2.97。其他Cross-store水平特征在10/11月仍有约5至6倍Lift，说明信号没有失效，但强度显著低于7/8月。
3. **不能只归因于字段分布。** 12月Cross-store水平比10/11月更低，但E2的20+ WAPE和Recall反而改善；因此“信号关系在不同月份的含义变化”同样重要。
4. **原classifier同期也在压低最终预测。** 真实20+的`p_sale`中位数在8月为0.8620，10月降至0.5504，11月降至0.3408。E2的q/y中位数也从E0的0.1381/0.1040降至0.1305/0.0995，两个组件在同一方向上叠加低估。
5. **明显恶化主要是进一步低估。** 10/11月共832条真实20+，其中63条达到“E2绝对误差比E0增加超过max(2本, 10%真实销量)”；51条为低估，预测中位数从E0的20.21降到E2的13.39，真实销量中位数为27。
6. **存在群体集中，但不是单一门店问题。** 63条中三级类目`R1300`占47.62%（相对总体集中度2.96倍），五级类目`MC130302`占33.33%（9.90倍）；门店`6013`占15.87%（2.00倍）。66.67%的恶化样本本店当前历史销量已是20+，52.38%位于Cross-store high档，说明模型在“本店本来很强、跨店也较热”的部分样本上反而出现不稳定收缩。
7. **不是少量极端爆款主导。** 10月Top 1%极端销量贡献E2 20+绝对误差的35.32%，但相对E0略有改善；11月同样略有改善。两月反向恶化来自更广泛的普通20+样本，而非最顶端几本书。

#### 阶段判断与下一步

- **E2仍值得保留研究，但不应直接晋升替代E0。** Cross-store已被证明是有效增量信息，且总体改善并非靠放大零销量误报换取；问题在于作用强度随月份、类目和本店历史状态不稳定。
- **问题A最推荐的修复方向：** 在Train内部forward folds上研究E0/E2的受限收缩或路由，只在“Cross-store与本店历史方向一致且关系可靠”时充分采用E2；冲突或极端热/冷时向E0收缩。规则必须由Train内部OOF确定，不能再用Complete Valid调阈值。
- **问题B最推荐的修复方向：** 增加Cross-store可靠性/相对强度表达，并做月份、类目稳定性约束或分组消融；优先检查`xstore_qty_diff_1m`，因为它在10/11月关系衰减最明显。不要简单全局降低Cross-store权重，否则会牺牲8月和12月的明确收益。
- **暂不建议回到Diff或Candidate B。** Diff已在组级统计门槛停止；Candidate B的补偿机制也已冻结。下一轮若允许一个小实验，优先做“Cross-store E2的forward-OOF稳健收缩/路由”，而不是继续扩大补偿倍率。
- 本轮未运行Test、2M、Candidate B或任何参数搜索；原E0/E2正式文件均未修改。重建checkpoint保留在`models/checkpoints/two_stage_optimization/cross_store_diagnostic/two_stage_regressor_cross_store_1m_rebuilt.txt`。

### 诊断边界

- 本节没有训练新候选、没有重新选轮、没有运行Test/2M，也没有恢复Candidate B。
- 诊断表只用于解释E2误差迁移与时间漂移，不据此修改657轮模型。
- Log: `logs\lightgbm\two_stage_optimization\run_20260820_213841.log`。
## 三条优化路线阶段性收口（2026-08-21）

### 1. 本轮边界与可追溯性

- 本轮唯一新增训练是受控的E1 Diff regressor Pilot；没有运行Test、2M、Complete Valid、Candidate B或Cross-store微调。
- 两个fold均让E0控制和E1使用完全相同的样本、样本顺序、逆采样权重、类别映射和LightGBM参数；E1只多6个Diff字段。
- Q4 fold为训练至2024-09、验证2024-10至12；Q2 fold为训练至2025-03、验证2025-04至06。Train覆盖各截止日前全部历史月份。
- 总耗时14.69分钟；训练阶段观测峰值RSS为1.48 GiB。
- 日志：`logs\lightgbm\two_stage_optimization\run_20260821_010703.log`；逐折结果已追加到`reports/two_stage_diff_cross_ablation.csv`。

### 2. E1 Diff受控Pilot

E1仅加入现有6个字段：当前月与lag1差、lag1与lag2差、log1p当前与lag1差、销售天数差、真正的最近3月均值减此前3月均值、6月历史可用标记。它们只依赖观察月及更早历史，不含未来销量标签。

| Forward fold | Model | Train/positive-valid | Best iter | Nonzero WAPE | 5-19 WAPE | 20+ WAPE | 5-19 Recall | 20+ Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| q4_2024 | E0_DIFF_CONTROL | 1,311,125/108,038 | 1240 | 55.6728% | 55.6017% | 82.4207% | 31.3226% | 10.4043% |
| q4_2024 | E1_DIFF | 1,311,125/108,038 | 1320 | 55.6566% | 55.6671% | 81.8763% | 31.1489% | 10.3774% |
| q2_2025 | E0_DIFF_CONTROL | 1,640,499/120,644 | 1112 | 52.8319% | 56.8940% | 75.4825% | 26.3028% | 10.4646% |
| q2_2025 | E1_DIFF | 1,640,499/120,644 | 978 | 52.9305% | 56.4129% | 75.3653% | 27.0545% | 10.3778% |

相对同折E0控制的变化（正数WAPE gain表示改善）：

| Fold | Nonzero WAPE变化 | 5-19 WAPE gain | 20+ WAPE gain | 5-19 Recall变化 | 20+ Recall变化 | Pilot门槛 |
|---|---:|---:|---:|---:|---:|---|
| q4_2024 | -0.0161 pp | -0.0653 pp | +0.5444 pp | -0.1738 pp | -0.0270 pp | 通过 |
| q2_2025 | +0.0986 pp | +0.4811 pp | +0.1172 pp | +0.7517 pp | -0.0868 pp | 未通过 |

结论为`time_conditional`，因此`allow_complete_valid=false`。Q4依靠20+ WAPE改善0.5444pp通过；Q2的5-19 WAPE改善0.4811pp、Recall提高0.7517pp，但均略低于0.5pp/1pp工程门槛。两个fold的20+ WAPE都改善，说明Diff存在弱的幅度信息；但20+ Recall分别下降0.0270pp和0.0868pp，收益没有转化为高动销等级识别。按照预先约束，本轮没有运行Complete Valid。

Diff字段的LightGBM gain重要性（只说明模型使用程度，不等同于单字段因果贡献）：

| Diff feature | Q4 gain | Q2 gain | 两折均值 |
|---|---:|---:|---:|
| `qty_diff_recent3_previous3` | 32168.6 | 58756.6 | 45462.6 |
| `log_qty_diff_current_lag1` | 36720.3 | 37624.8 | 37172.5 |
| `qty_diff_current_lag1` | 29671.2 | 27914.4 | 28792.8 |
| `qty_diff_lag1_lag2` | 24364.0 | 25202.8 | 24783.4 |
| `sales_days_diff_current_lag1` | 12801.3 | 12302.6 | 12551.9 |
| `qty_diff_6m_history_available` | 9528.6 | 12349.4 | 10939.0 |

`qty_diff_recent3_previous3`和`log_qty_diff_current_lag1`是两折中最强的Diff输入，`qty_diff_current_lag1`次之。前者单字段统计Lift较弱，却在树模型中获得较高gain，说明它更可能作为条件分裂与其他历史特征联合使用。`qty_diff_6m_history_available`和`sales_days_diff_current_lag1`整体贡献较低。所有Diff与已有lag/rolling的相关性都远低于0.98删除线，因此不是完全重复，但多数是已有水平信息的变化表达，增量有限。

**Diff阶段判断：不是无效，而是有信息、模型层面增量偏弱且时间/指标条件性明显。** 它改善了两个时期的20+连续误差，却没有改善20+等级Recall，也没有跨折达到项目晋级线；当前不值得为它单独做全Train和Complete Valid。

### 3. Cross-store阶段性结论及与未来分布式研究的衔接

- Cross-store是三条路线中证据最强的一条：全Train统计、双折Pilot、正式全Train和Complete Valid均已完成。q层20+ WAPE改善2.2807pp、20+ Recall提高4.0052pp、median(q/y)由0.301663升至0.350284；最终20+ WAPE改善2.1444pp、Recall提高3.1639pp，Nonzero与Overall WAPE也改善，Zero Pred Total下降5.2613%，Total Bias由+8.9329%收窄到+4.9883%。
- 它并非简单把所有预测抬高；但5-19 Recall下降0.7821pp，且10/11月20+发生关系漂移。本店历史较强而跨店偏冷时，E2可能过度压低q。因此保留为有明确增量价值的优化候选和checkpoint，暂不替代E0，也不继续做路由、收缩或月份修正。
- 集中式结果已经证明：同一本书在其他门店的当月/近期销量、活跃门店数和热度变化具有预测增量。未来分布式研究可考虑门店不共享交易明细，而协同计算item级聚合销量、活跃门店数、滚动集团热度与变化信号，再研究安全聚合、差分隐私等机制。当前Cross-store是集中式明文聚合实验，**尚未实现或证明任何隐私保护**。
- 重建E2 checkpoint继续保留：`models/checkpoints/two_stage_optimization/cross_store_diagnostic/two_stage_regressor_cross_store_1m_rebuilt.txt`。

### 4. Candidate B完整阶段性结论

1. A2通过提高5-19/20+训练权重证明头部幅度可改善：最终5-19/20+ Recall分别提高2.60/3.05pp，WAPE改善0.52/1.42pp；但Zero Pred Total增加8.22%，Total Bias由+8.93%扩大到+17.92%，属于全局regressor一起变积极，不能正式采用。
2. Candidate B-v1改为只补偿有历史高需求证据的样本。时间外Holdout仅补偿28,031条（0.2960%），Zero Pred Total只增加0.0348%，Bias只增加0.1739pp，证明“选择性积极”能够隔离副作用；但命中样本中真正5+只有7,579条（约27.04%），20+ Recall仅提高1.3064pp，5-19 Recall反降0.2251pp，综合收益不足。
3. Candidate B-v2训练独立`P(Y>=5)`/`P(Y>=20)`分数，Cross-store使高需求分类PR-AUC明显提高；随后用5个medium Top比例乘5个high Top比例形成25组分档规则。Q4/Q1/Q2规则拟合期分别有25/20/20个候选触发5-19保护失败，Total Bias失败15/22/6个，另有Q1一个Zero失败；同一候选可同时失败。没有候选通过全部保护条件，所以正式规则回退为不补偿，报告中的收益为0表示“未采用规则”，不是25个公式碰巧都得到0。
4. 当前保留的OOF产物只有按门槛汇总的淘汰计数，没有保存25个候选逐项指标，因此无法在不重跑既有OOF的前提下可靠列出“最接近通过”的若干项。本轮禁止重跑，故明确记录证据缺口，不猜测。B-v1是现有资料中最接近可控方案的实例，但收益仍不足。

Candidate B的优点是直接针对`p_sale*q`的头部压缩并实现选择性补偿；困难同时存在于两个环节：未来高需求仍难稳定识别，而即使识别PR-AUC提升，固定分档乘固定倍率也会在5-19与20+边界制造trade-off。当前应保留机制证据，不继续扩大倍率或制造B-v3。

### 5. 三路线研究地图

| 路线 | 核心思想 | 已做到什么 | 优点 | 副作用/困难 | 当前状态 | 后续潜力 |
|---|---|---|---|---|---|---|
| Diff | 把历史销量变化而非仅销量水平作为regressor输入 | 全Train统计审计；本轮两折受控E1 Pilot | 两折20+ WAPE均小幅改善；成本低；符合导师差分建议 | 收益弱且未转化为20+ Recall；跨折未同时过门槛；部分信息与lag/rolling重叠 | 有信息但不够稳定，停止在Pilot | 可与Cross-store合并做一次受控消融，但不应单独正式化 |
| Cross-store | 用同书在其他门店的历史热度补充本店历史 | 旁表、审计、Pilot、全Train、Complete Valid、误差迁移和漂移诊断 | q幅度、20+、Nonzero、Overall、Bias、Zero均有实质改善；证据最完整 | 5-19 Recall下降；10/11月关系漂移；尚无隐私保护 | 保留E2 checkpoint和研究结论，暂不替代E0、不再微调 | 最适合作为跨店协同/未来分布式特征基线 |
| Candidate B | 只对高需求候选有限补偿，避免全局积极 | A2、B-v1、5+/20+分类、25规则、forward OOF stop gate | 能把头部积极性限制在少数样本，副作用远小于A2 | 候选Precision不足；固定倍率造成5-19/20+ trade-off；OOF无候选通过 | 机制有价值，当前实现失败，冻结 | 以后可研究连续、受约束的专家/损失机制，但必须先有稳定高需求识别 |

三条路线不互斥：Diff和Cross-store都是预测时可得的输入信息，未来可组合；Candidate B是预测后的决策/组合层，可使用前两者产生的高需求信号。但本轮证据排序明确：**Cross-store最强，Diff次之且条件性明显，Candidate B当前机制最不成熟。** Cross-store也最符合“门店保护本地明细、跨店参考聚合热度”的研究衔接，但隐私机制必须在后续另行实现和验证。

### 6. 已回答、仍未知与停止点

**已回答：** Diff进入模型后确有弱连续误差增量，但不足以晋级；Cross-store提供真实且较强的跨店增量；A2的全局积极不可接受；Candidate B可控制副作用，但当前分档倍率机制不能把识别收益稳定转成业务收益。

**仍未知：** Cross-store+Diff是否存在互补；Cross-store信号在隐私保护聚合后保留多少效用；连续受约束的高需求专家能否避免固定MC边界trade-off；这些都尚未使用Test验证。

**下一阶段值得讨论的方向（本轮不执行）：**
1. 优先讨论Cross-store如何形成可共享的item级聚合协议，并设计集中式等价基线，再进入隐私/分布式研究。
2. 可以后做一次严格受控的`Cross-store + Diff`消融，只在forward Pilot中验证Diff是否补足10/11月关系漂移；失败即停。
3. Candidate B暂不继续固定分档倍率；若未来重启，应转向带业务约束的连续专家/损失设计，并先证明forward高需求识别稳定。

本轮停止于Diff两折Pilot。未运行Test、2M、Complete Valid或分布式训练，未覆盖E0正式模型，未修改E2 checkpoint。

## 统一Train→Valid开发口径重评

### 协议审计

新默认协议为完整Active-Store Train训练，正式Complete Valid用于轮次、特征和规则开发；本节没有读取Test。历史forward结果仍保留在前文，只是不再作为当前候选的选轮依据。

| 实验 | 旧选择来源 | 本轮处理 |
|---|---|---|
| p_sale classifier (750) | Complete Valid在750/850/864与固定1054轮regressor组合中按Overall WAPE选择750 | 保留；不是四fold加权轮次，且本轮只统一重评positive regressor及其衍生候选 |
| E0 positive regressor (1054) | 采样Valid组件早停后，Complete Valid三组组合筛选得到1054 | 按完整Train训练、正式Complete Valid WAPE重新选轮 |
| E1 Diff | Q4-2024/Q2-2025两个forward Pilot分别选轮，未形成Complete Valid正式结论 | 按完整Train训练、正式Complete Valid WAPE重新选轮 |
| E2 Cross-store | 四个forward fold的382/657/1136/1357按验证量加权中位数取657 | 按完整Train训练、正式Complete Valid WAPE重新选轮 |
| Candidate B-v2 | 高需求分类器及25组规则由Q4/Q1/Q2 forward OOF筛选并stop | 仅按既有25组和既有倍率公式，在正式Valid重新形成开发集trade-off |

### 可复现性

- 统一稳定Train缓存共`3,699,878`行，覆盖2023-01至2025-06；各组件保持原确定性采样率与逆概率权重。
- 特征顺序hash：`7ca87a5aebae67a0b054741b60acfc5de32413f24dccf2d3a54d417ba578fb1e`；缓存hash：`12f7b23cf31c6f7b60cebd43793667cfd65fd26b55b6c67a7d5b6735fded357a`。
- 重复训练smoke：100,000行、25轮，两次模型及预测完全一致；hash=`d125c5f81048bd0be8d3db34e5ef70f9005fbb87af0c9859fb538c626e08580d`。
- 三个regressor共享完全相同的正销量样本集合与顺序；每个模型metadata另存sample-set、order和model hash。
- `p_sale`继续使用750轮正式classifier；其来源是旧版Complete Valid组合WAPE选择，并非四fold加权中位数。

### 统一Valid结果

| Model | 新增机制/特征 | Valid best iteration | Overall Raw WAPE | Nonzero WAPE | 5-19 WAPE | 5-19 Recall | 20+ WAPE | 20+ Recall | MC Macro-F1 | Total Bias | Zero Pred Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_VALID_SELECTED | 无（统一基准） | 1964 | 133.8868% | 72.5816% | 64.7151% | 29.7560% | 75.7708% | 24.0787% | 42.3127% | 7.4447% | 1513346.6082 |
| E1_DIFF_VALID_SELECTED | 6个Diff字段 | 4988 | 132.5758% | 72.6346% | 64.9817% | 28.6844% | 76.7027% | 22.1709% | 41.8326% | 4.4912% | 1479677.2626 |
| E2_CROSS_STORE_VALID_SELECTED | 6个Cross-store字段 | 2885 | 128.5494% | 71.6208% | 64.3078% | 29.3569% | 73.0472% | 27.1478% | 43.4879% | 2.3764% | 1405308.6222 |
| E3_DIFF_CROSS_STORE_VALID_SELECTED | 6个Diff + 6个Cross-store | 4562 | 127.8279% | 71.5332% | 64.4846% | 28.4403% | 73.3090% | 26.3420% | 43.2174% | 0.6450% | 1389660.2552 |
| Candidate B | 既有25组规则 | 不晋级 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

- `E1_DIFF_VALID_SELECTED`相对E0（WAPE正数表示降低/改善）：Overall WAPE +1.3110 pp，Nonzero WAPE -0.0529 pp，5–19 Recall -1.0716 pp，20+ Recall -1.9078 pp，Bias -2.9535 pp，Zero Pred -2.2248%。
- `E2_CROSS_STORE_VALID_SELECTED`相对E0（WAPE正数表示降低/改善）：Overall WAPE +5.3374 pp，Nonzero WAPE +0.9608 pp，5–19 Recall -0.3991 pp，20+ Recall +3.0691 pp，Bias -5.0683 pp，Zero Pred -7.1390%。
- `E3_DIFF_CROSS_STORE_VALID_SELECTED`相对E0（WAPE正数表示降低/改善）：Overall WAPE +6.0589 pp，Nonzero WAPE +1.0484 pp，5–19 Recall -1.3157 pp，20+ Recall +2.2633 pp，Bias -6.7997 pp，Zero Pred -8.1730%。

补充业务指标：

| Model | Integer Overall WAPE | Integer Nonzero WAPE | Zero >0.5 | Zero >1 |
|---|---:|---:|---:|---:|
| E0_VALID_SELECTED | 101.6995% | 75.4544% | 2.9408% | 0.9548% |
| E1_DIFF_VALID_SELECTED | 100.6645% | 75.5616% | 2.8355% | 0.9085% |
| E2_CROSS_STORE_VALID_SELECTED | 96.9031% | 74.5756% | 2.5717% | 0.7821% |
| E3_DIFF_CROSS_STORE_VALID_SELECTED | 96.3393% | 74.4734% | 2.5326% | 0.7621% |

q层诊断：

| Model | q Nonzero WAPE | q 5-19 WAPE | q 5-19 Recall | q 20+ WAPE | q 20+ Recall | 20+ median(q/y) |
|---|---:|---:|---:|---:|---:|---:|
| E0_VALID_SELECTED | 57.5221% | 52.7193% | 41.3491% | 71.4834% | 28.3801% | 0.298493 |
| E1_DIFF_VALID_SELECTED | 56.8042% | 53.0511% | 40.2205% | 72.3480% | 26.0931% | 0.290515 |
| E2_CROSS_STORE_VALID_SELECTED | 54.5366% | 52.6007% | 41.1269% | 68.5613% | 32.6697% | 0.351183 |
| E3_DIFF_CROSS_STORE_VALID_SELECTED | 54.0403% | 52.7781% | 40.1327% | 68.6847% | 31.6033% | 0.342458 |

Candidate B：25组中保护条件可接受7组，但没有方案同时达到既有20+ WAPE/Recall收益门槛，故不晋级。
最接近晋级的保护方案为medium Top 1.00%、high Top 0.50%：20+ WAPE改善0.4141 pp、Recall提高1.6827 pp，但20+ WAPE改善未达到0.5 pp门槛；同时5–19 WAPE恶化0.2045 pp、Nonzero WAPE恶化0.2482 pp、Zero Pred增加0.3086%、绝对Bias恶化1.2996 pp。

轮次选择记录：

- `E0_VALID_SELECTED`：粗选1950轮，在粗选点附近逐轮精扫后锁定1964轮；Valid Overall WAPE=133.8868%。
- `E1_DIFF_VALID_SELECTED`：粗选5000轮，在粗选点附近逐轮精扫后锁定4988轮；Valid Overall WAPE=132.5758%。
- `E2_CROSS_STORE_VALID_SELECTED`：粗选2875轮，在粗选点附近逐轮精扫后锁定2885轮；Valid Overall WAPE=128.5494%。
- `E3_DIFF_CROSS_STORE_VALID_SELECTED`：粗选4550轮，在粗选点附近逐轮精扫后锁定4562轮；Valid Overall WAPE=127.8279%。
- `E1_DIFF_VALID_SELECTED`在5000轮开发预算附近锁定4988轮；这是预算内Valid最佳点，但未形成常规early-stopping内部最低点，复杂度与稳定性需作为限制记录。

### 当前判断

- 统一开发口径下，当前Overall WAPE最低的1M候选是`E3_DIFF_CROSS_STORE_VALID_SELECTED`（127.8279%）。
- 但E3相对E2的5–19 Recall下降0.9166 pp、20+ WAPE恶化0.2617 pp、20+ Recall下降0.8058 pp、MC Macro-F1下降0.2705 pp。因此业务多指标首选仍为`E2_CROSS_STORE_VALID_SELECTED`；E3作为低Overall WAPE、低Bias和低零销量误报的第二候选保留。
- Diff是否保留、Cross-store是否仍有增量以及Candidate B是否晋级，均以本节同一批Complete Valid结果为准；旧forward结果仅作时间稳定性背景。
- 本节所有结果都是开发集结果。Test尚未运行，不能称为最终泛化结论。
- E0/E1/E2统一重评日志：`logs\lightgbm\two_stage_optimization\run_20260822_013504.log`；E3日志：`logs\lightgbm\two_stage_optimization\run_20260822_165842.log`。


### E3 Diff + Cross-store消融结论

变化表中WAPE正数表示误差降低/改善；Recall、Bias和Zero Pred为候选减基准的变化。

| 方案 | 相对E0 Overall WAPE | Nonzero WAPE | 5–19 Recall | 20+ WAPE | 20+ Recall | Bias | Zero Pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 Diff | +1.3110 pp | -0.0529 pp | -1.0716 pp | -0.9319 pp | -1.9078 pp | -2.9535 pp | -2.2248% |
| E2 Cross-store | +5.3374 pp | +0.9608 pp | -0.3991 pp | +2.7236 pp | +3.0691 pp | -5.0683 pp | -7.1390% |
| E3 Diff+Cross-store | +6.0589 pp | +1.0484 pp | -1.3157 pp | +2.4619 pp | +2.2633 pp | -6.7997 pp | -8.1730% |

E3相对E2：Overall WAPE改善+0.7215 pp，Nonzero WAPE改善+0.0876 pp，5–19 WAPE改善-0.1767 pp，5–19 Recall变化-0.9166 pp，20+ WAPE改善-0.2617 pp，20+ Recall变化-0.8058 pp，MC Macro-F1变化-0.2705 pp，Bias变化-1.7314 pp，Zero Pred变化-1.1135%。

**结论：E3降低了Overall WAPE、Bias和零销量误报，但5–19/20+ Recall、20+ WAPE及MC均弱于E2；Diff未证明与Cross-store互补，业务多指标下优先保留E2，E3作为低Overall误差候选保留。**

### 导师汇报摘要

1. **统一E0基准**：57个基础特征，Valid最佳1964轮；Overall WAPE=133.8868%，Nonzero WAPE=72.5816%。
2. **Diff**：加入6个历史销量变化字段，Overall WAPE降低1.3110 pp，但Nonzero和中高需求指标未改善。
3. **Cross-store**：加入6个同书其他门店热度字段，Overall WAPE降低5.3374 pp，20+ Recall提高3.0691 pp。
4. **Diff + Cross-store**：E3的Overall WAPE比E2再降低0.7215 pp，但5–19/20+ Recall分别下降0.9166/0.8058 pp，20+ WAPE恶化0.2617 pp，未证明两类信息互补。
5. **Candidate B**：原25组选择性补偿规则没有方案同时达到既定头部收益与副作用保护门槛，不进入最终候选。
6. **当前Valid结论**：纯Overall WAPE最低的是`E3_DIFF_CROSS_STORE_VALID_SELECTED`（127.8279%）；兼顾Nonzero、5–19、20+和MC后，业务首选仍为`E2_CROSS_STORE_VALID_SELECTED`。
7. **相对E0与Test候选**：E3 Overall WAPE降低6.0589 pp，但相对E2的5–19/20+ Recall分别下降0.9166/0.8058 pp；建议E2与E3共同进入最终Test。
8. **分布式衔接**：Cross-store验证了同书跨门店聚合热度的增量价值，可作为未来门店在不直接交换原始明细时协同计算item级统计信号的集中式依据；当前尚未实现隐私保护。
