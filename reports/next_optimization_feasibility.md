# 下一阶段模型优化可行性评估

## 1. 本轮范围与证据

本轮只读取现有配置、报告、模型元数据和 Parquet schema，没有训练模型、没有重新运行 Test、没有启动 Two-stage 2M，也没有执行正式校准或 Hybrid。

采用的正式口径来自 `reports/evaluation_protocol.md`：训练和轮次选择继续使用 Complete Valid 原始连续销量 Overall WAPE；最终业务评价重点看 Nonzero WAPE、5-19、20+、零销量风险、MC、趋势和总量。

当前可直接使用的 Active-Store Valid 证据如下：

| 模型 | Complete Valid 行级预测 | 已有 Valid 汇总 |
|---|---|---|
| LightGBM V2 1M | 未保存 | 轮次候选的 Overall WAPE、MAE、RMSE、总量偏差 |
| Two-stage 1M | 未保存 | Original 分层 WAPE、零销量风险、总量偏差、Gate-A trade-off |

LightGBM 和旧 Gate 文件中的行级预测都是 Test 预测，不能拿来拟合 Valid 校准系数。新 Active-Store Two-stage 1M 没有行级 Valid 预测缓存。因此，取整指标、按预测 MC 分组、月度稳定性和同一行上的模型互补性不能从现有聚合表精确反推。

Valid 1M 共 18,563,573 行，月份为 2025-07 至 2025-12。每月样本量约 299 万至 319 万。

## 2. 当前能够确认的偏差结构

### LightGBM V2 1M

Complete Valid 最佳轮次 850：

- True total：2,468,546
- Predicted total：1,801,295.95
- Total Bias：-27.03%
- Overall WAPE：117.1761%

因此 LightGBM 在 Valid 总体上存在明确低估。已有冻结 Test 报告中，1、2-4、5-19 和 20+ 也都表现为低估，但这些 Test 结果只作为历史背景，不能用于本轮校准、阈值或 Hybrid 选择。

### Two-stage 1M

Complete Valid 正式组合为 classifier 750 轮、regressor 1054 轮：

- True total：2,468,546
- Predicted total：2,689,058.23
- Total Bias：+8.93%
- 真实零样本预测总量：1,530,369.44
- 真实非零样本预测总量（由总量扣除零样本预测量）：1,158,688.80
- 真实非零样本总量偏差：-53.06%
- Nonzero WAPE：72.60%
- 5-19 WAPE：64.46%
- 20+ WAPE：75.15%

这说明 Two-stage 的总体高估主要来自真实零销量样本上的小额正预测；在真正发生销售的样本上，模型总体仍严重低估。一个全局乘法因子会把这两类相反问题混在一起。

现有 Valid 聚合结果没有保存 1、2-4、5-19、20+ 各组的预测总量，也没有按月份保存偏差。因此，目前不能严谨声称新 Two-stage 在每个 MC 等级上分别是高估还是低估，也不能确认偏差是否跨月份稳定。

## 3. 方案一：Raw 与 Integer 双指标

统一定义：

`pred_integer = floor(max(pred, 0) + 0.5)`

该方案统计和工程上合理，建议正式加入项目，但用途必须分开：

- Raw WAPE、Raw MAE：继续用于 Boosting 轮次、组件组合和内部模型比较。连续预测保留了模型置信程度，指标随树轮次变化较平滑。
- Integer WAPE、Integer MAE：用于最终业务交付评价和 MC 映射，反映实际整数本数。

固定同一取整规则并同时报告 Raw/Integer，不会造成不公平；模型排名发生变化本身就是业务离散化影响。不能只报告 Integer 指标，也不建议用 Integer WAPE 选 Boosting 轮次，因为 0.49/0.50 等边界会造成不连续跳变，使轮次选择不稳定。

现有 Active-Store Valid 行级预测未保存，因此本轮无法直接计算 Integer WAPE。若已有行级预测，取整汇总只需约 1-3 分钟；当前需要先对冻结模型做一次 Valid 推理。该推理一旦执行，应在同一遍历中同时完成后续 MC、月份和互补性统计，避免重复推理。

预期影响：零样本上的大量小于 0.5 的预测会归零，Overall/Zero Integer 误差可能明显下降；真实销量为 1 的样本也可能被归零，因此 Nonzero Integer WAPE、MC Recall 不一定改善，必须单独报告。

结论：**建议优先做，作为正式业务评价，不改变现有训练选轮口径。**

## 4. 方案二：真实 MC 与预测 MC 偏差诊断

两种分组必须严格区分：

### 按真实 MC 分组

用途仅限诊断模型在哪个真实需求区间失效。真实未来 MC 在部署时不可知，绝对不能用于选择校准系数或路由模型，否则构成目标泄露。

### 按预测 MC 分组

预测 MC 来自模型当时可用的预测值，可以用于研究部署可行的分层校准。分组必须由校准前预测固定一次；应用系数后可以重新计算最终 MC，但不能反复重新分组和迭代套用系数。

每个真实/预测 MC 等级和月份应输出：count、true total、predicted total、segment bias、mean/median error、under/over ratio、Raw WAPE、Integer WAPE。连续误差的中位数可用流式分位数草图或临时预测缓存计算。

当前能确认：

- LightGBM 总体低估，历史冻结结果也显示中高等级低估明显。
- Two-stage 在真实非零样本总体低估 53.06%，但在真实零样本产生 153 万预测量。
- 两模型都存在非零需求低估倾向；Two-stage 不是简单的“头部高估模型”，其优势主要体现为 Nonzero/中高销量绝对误差较低。
- 各 MC 等级的精确方向和跨月稳定性尚未由 Active-Store Complete Valid 行级证据确认。

结论：**非常值得先做诊断；它是校准和 Hybrid 的前置条件。**

## 5. 方案三：预测 MC 分层校准

`pred_calibrated = c_k * pred` 在工程上可行，且预测 MC 是部署可用信息，不构成泄露。但不能直接默认 `sum(y)/sum(pred)` 是最终方案。

风险包括：

1. 预测高动销等级样本可能很少，原始比值方差很大。
2. 等级边界两侧使用不同系数，会产生不连续跳变。
3. 预测 MC=0 组同时包含大量真实零样本和被严重低估的真实动销样本，一个系数很难兼顾两者。
4. 在完整 Valid 上拟合并在同一批 Valid 上报告提升，会产生校准选择偏乐观；Test 仍不能参与。

全局因子尤其不合适：LightGBM Valid 的总量因子约为 1.370；Two-stage 总体因子约为 0.918，但 Two-stage 真实非零部分对应因子约为 2.130。Two-stage 的总体缩小和非零放大方向相反，证明全局乘法会掩盖零误报与非零低估的结构冲突。

若下一轮诊断发现按预测 MC 的偏差跨月份稳定，推荐从以下低复杂度方案开始：

- 先合并为较稳健的三段：预测 0、预测 1-4、预测 5+，而不是直接拟合五个高方差系数。
- 系数只用较早 Valid 月份拟合，例如 2025-07 至 2025-09；用 2025-10 至 2025-12 做时间外校验。
- 对系数做向 1 收缩，并设置经 Valid 预先确认的安全范围；不要无约束使用极端比值。
- 保持“按校准前预测等级分组，一次乘法，之后重新计算最终 MC”的固定流程。
- 同时观察 Nonzero/5-19/20+、Integer WAPE、MC Macro-F1、高动销 Recall和零销量风险，不能只优化 Overall WAPE。

更简单的全局乘法不推荐。五级无约束独立系数也不推荐。若三段系数仍不稳定，再考虑单调的 log1p 预测校准，但不应直接升级为复杂模型。

结论：**值得在 Valid 做低成本诊断实验，但必须先证明预测等级偏差稳定；本轮不应直接实施正式校准。**

## 6. 方案四：LightGBM 与 Two-stage 融合

目前存在合理的结构性假设：LightGBM 的零销量控制更好；Two-stage 的 Nonzero 和中高销量 WAPE 更低。但现有证据还不足以证明稳定互补：

- 新 Two-stage 没有正式 Test 结果，本轮也不应运行 Test。
- Active-Store Complete Valid 没有两模型同一行的预测，无法计算残差相关性、逐行胜率和 Oracle 上限。
- LightGBM Valid 报告没有相同 MC 分层的预测总量；直接拿 LightGBM Test 分层指标与 Two-stage Valid 指标比较不公平。
- 两模型在真实非零需求上都可能低估。若两者在同一批样本上同时低估，普通凸组合不会改善头部需求。

在决定 Hybrid 前，至少需要在同一 Complete Valid 行上计算：

- 各销量/MC等级的 Raw 与 Integer WAPE、Bias、under/over ratio；
- 两模型残差相关系数和绝对误差胜率；
- `min(abs(error_lgb), abs(error_two_stage))` 对应的 Oracle WAPE，判断可融合上限；
- 按月份重复上述指标，确认互补不是单月偶然；
- `p_sale` 分段下两模型的相对优势。

若这些结果显示低 `p_sale` 区间 LightGBM 持续胜出、高 `p_sale` 和中高销量区间 Two-stage 持续胜出，`p_sale < tau` 路由才有数据依据。需要特别检查历史“突然爆发但 p_sale 很低”的样本，因为把它们路由给同样偏保守的 LightGBM 未必能解决头部低估。

结论：**理论上可行，但当前互补性证据不足；可以后做，暂不实施 Hybrid。**

## 7. 下一轮最小成本实验

不需要重新训练任何模型。推荐只进行一次冻结模型的 Active-Store Complete Valid 1M 联合推理：

1. 同一行同时得到 LightGBM、`p_sale`、`conditional_qty` 和 Original Two-stage 预测。
2. 同步累计 Raw/Integer、真实MC/预测MC、月份、销量层级和零风险指标。
3. 同步累计两模型残差相关性、胜率和 Oracle 指标。
4. 若要继续校准，使用 2025-07 至 2025-09 拟合，2025-10 至 2025-12 校验；不访问 Test。
5. 不保存永久大型预测集。需要精确中位数或多次后处理时，可保存约 0.2-0.4 GiB 的可删除临时压缩缓存，完成后清理。

基于历史正式运行估算：

| 工作 | 预计耗时 | 峰值 RAM | 新增规模 |
|---|---:|---:|---:|
| 仅已有聚合报告审计 | 小于 5 分钟 | 小于 0.5 GiB | 本报告约几十 KiB |
| 单模型 Integer/MC 汇总（已有行级预测时） | 1-3 分钟 | 小于 1 GiB | 小于 1 MiB 聚合表 |
| 冻结 LightGBM Complete Valid 推理 | 约 10-15 分钟 | 约 1-1.5 GiB | 可流式，不落大文件 |
| 冻结 Two-stage Complete Valid 单组合推理 | 约 25-40 分钟 | 约 0.5-1.5 GiB | 可流式，不落大文件 |
| 联合推理和全部诊断 | 约 40-60 分钟 | 约 1-2.5 GiB | 聚合结果小于 1 MiB；可选临时缓存 0.2-0.4 GiB |
| 基于已有预测缓存的分层校准/Hybrid阈值诊断 | 约 2-10 分钟 | 小于 1 GiB | 小于 1 MiB |

上述是基于现有日志的保守估算；不依赖重新训练，也不需要新的永久训练 Parquet。

## 8. 明确优先级

### 建议优先做

1. Raw + Integer 双指标正式化。
2. 在同一次 Complete Valid 推理中完成真实MC、预测MC、月份偏差和两模型同样本互补性诊断。
3. 只有预测MC偏差在月份外校验中稳定时，尝试三段、收缩、受限的乘法校准。

### 可以后做

1. 由 `p_sale` 路由的 LightGBM/Two-stage Hybrid。
2. 有限加权融合，但前提是残差相关性和 Oracle 增益证明两模型确实互补。
3. 五级更细校准，前提是各等级样本量与月度系数稳定。

### 不建议做

1. 使用真实未来 MC 选择校准系数或路由模型。
2. 使用 Integer WAPE 替代 Raw WAPE选择 Boosting 轮次。
3. 在全部 Valid 上拟合系数后仍用同一批数据宣称泛化提升。
4. 无约束五级系数、单一全局乘法，或在证据不足时直接实现 Hybrid。
5. 为本轮方案重新训练 LightGBM、Two-stage，或启动 Two-stage 2M。

## 9. 最终判断

- Integer / Business WAPE 合理且值得正式加入，但只作为业务评价，不替代 Raw 训练选轮指标。
- LightGBM 已确认总体系统性低估；Two-stage 已确认真实非零部分系统性低估，同时真实零样本误报严重。
- 具体 MC 等级偏差和跨月稳定性尚未被现有 Active-Store Valid 聚合结果证明。
- 预测 MC 分层校准有研究价值，但最推荐的是先做三段、收缩、时间外验证的低成本诊断，而不是直接采用五级原始比值。
- LightGBM 与 Two-stage 具有潜在互补机制，但现有同样本证据不足，暂不值得直接实现 Hybrid。
- 下一轮最值得投入的是一次冻结模型、Valid-only、单遍联合推理；它不需要重新训练任何模型，并能同时回答整数评价、MC偏差、月度稳定性、校准与融合可行性。
