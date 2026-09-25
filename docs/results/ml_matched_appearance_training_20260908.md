# 主体可见配对外观训练：六单元结果

六个固定终点单元全部完成，结论：没有家族通过全部开发门禁，不晋升。

K 为新位姿原始图，L 为同位姿材质／光照变体。共同基础、常规、困难负例成员及曝光位置一致，全图类别实例曝光一致；每单元 300 步、1800 次图像曝光，seed 7/17/27，v2.11 初始化、CPU 640、AdamW 恒定 0.001。只用末轮权重。

## 三 seed 平均结果

| 指标 | K-300 | L-300 | L−K |
| --- | ---: | ---: | ---: |
| 原始 计划命中 | 0.944444 | 0.833333 | -11.11 个百分点 |
| 原始 全图召回 | 0.744444 | 0.622222 | -12.22 个百分点 |
| 原始 匹配精度 | 0.824786 | 0.838230 | +1.34 个百分点 |
| 材质 计划命中 | 0.027778 | 0.166667 | +13.89 个百分点 |
| 材质 全图召回 | 0.011111 | 0.083333 | +7.22 个百分点 |
| 材质 匹配精度 | 0.750000（2 seed 有定义） | 0.660317 | 不计算完整三 seed 差值 |
| 背景 计划命中 | 0.750000 | 0.666667 | -8.33 个百分点 |
| 背景 全图召回 | 0.572222 | 0.550000 | -2.22 个百分点 |
| 背景 匹配精度 | 0.873678 | 0.892009 | +1.83 个百分点 |
| 光照 计划命中 | 0.416667 | 0.388889 | -2.78 个百分点 |
| 光照 全图召回 | 0.444444 | 0.377778 | -6.67 个百分点 |
| 光照 匹配精度 | 0.705291 | 0.847681 | +14.24 个百分点 |
| 无目标帧 FPR | 0.006944 | 0.048611 | +4.17 个百分点 |

## 各 seed 与波动

| 单元 | 原始计划命中 | 材质 | 背景 | 光照 | 无目标 FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| K-300-7 | 1.000000 | 0.083333 | 0.833333 | 0.333333 | 0.000000 |
| L-300-7 | 0.833333 | 0.083333 | 0.750000 | 0.333333 | 0.000000 |
| K-300-17 | 1.000000 | 0.000000 | 0.666667 | 0.416667 | 0.000000 |
| L-300-17 | 0.833333 | 0.333333 | 0.666667 | 0.416667 | 0.020833 |
| K-300-27 | 0.833333 | 0.000000 | 0.750000 | 0.500000 | 0.020833 |
| L-300-27 | 0.833333 | 0.083333 | 0.583333 | 0.416667 | 0.125000 |

无预测时匹配精度未定义，不填成 0 或 1；不足三个 seed 有定义时注明有效数量，不计算完整三 seed 精度差值。各条件全图／逐类召回、匹配精度、未匹配预测的均值、最差 seed、标准差及全部 seed 值保存在完成回执 aggregate；逐实例正式与低阈值预测保存在各 evaluation 文件。

## 未通过门禁

### K-300

- material.planned_instance_hit_rate.mean：0.027778，要求 >= 0.5。
- material.planned_instance_hit_rate.min_seed：0.000000，要求 >= 0.3333333333333333。
- lighting.planned_instance_hit_rate.mean：0.416667，要求 >= 0.6。
- original.all.instance_recall 相对 same_budget_R：-6.11 个百分点，要求不少于 -5.00 个百分点。
- original.switchgear.instance_recall 相对 same_budget_R：-15.05 个百分点，要求不少于 -5.00 个百分点。
- original.reactor.instance_recall 相对 same_budget_R：-13.33 个百分点，要求不少于 -5.00 个百分点。
- lighting.all.instance_recall 相对 same_budget_R：-17.78 个百分点，要求不少于 -5.00 个百分点。
- lighting.transformer.instance_recall 相对 same_budget_R：-12.96 个百分点，要求不少于 -5.00 个百分点。
- lighting.switchgear.instance_recall 相对 same_budget_R：-24.73 个百分点，要求不少于 -5.00 个百分点。
- lighting.reactor.instance_recall 相对 same_budget_R：-20.00 个百分点，要求不少于 -5.00 个百分点。
- original.all.instance_recall 相对 historical_A：-6.67 个百分点，要求不少于 -5.00 个百分点。
- original.switchgear.instance_recall 相对 historical_A：-12.90 个百分点，要求不少于 -5.00 个百分点。
- original.reactor.instance_recall 相对 historical_A：-20.00 个百分点，要求不少于 -5.00 个百分点。
- lighting.all.instance_recall 相对 historical_A：-22.22 个百分点，要求不少于 -5.00 个百分点。
- lighting.transformer.instance_recall 相对 historical_A：-20.37 个百分点，要求不少于 -5.00 个百分点。
- lighting.switchgear.instance_recall 相对 historical_A：-32.26 个百分点，要求不少于 -5.00 个百分点。

### L-300

- material.planned_instance_hit_rate.mean：0.166667，要求 >= 0.5。
- material.planned_instance_hit_rate.min_seed：0.083333，要求 >= 0.3333333333333333。
- lighting.planned_instance_hit_rate.mean：0.388889，要求 >= 0.6。
- original.all.instance_recall 相对 same_budget_R：-18.33 个百分点，要求不少于 -5.00 个百分点。
- original.transformer.instance_recall 相对 same_budget_R：-11.11 个百分点，要求不少于 -5.00 个百分点。
- original.switchgear.instance_recall 相对 same_budget_R：-19.35 个百分点，要求不少于 -5.00 个百分点。
- original.capacitor_bank.instance_recall 相对 same_budget_R：-11.11 个百分点，要求不少于 -5.00 个百分点。
- original.reactor.instance_recall 相对 same_budget_R：-46.67 个百分点，要求不少于 -5.00 个百分点。
- lighting.all.instance_recall 相对 same_budget_R：-24.44 个百分点，要求不少于 -5.00 个百分点。
- lighting.transformer.instance_recall 相对 same_budget_R：-12.96 个百分点，要求不少于 -5.00 个百分点。
- lighting.switchgear.instance_recall 相对 same_budget_R：-35.48 个百分点，要求不少于 -5.00 个百分点。
- lighting.reactor.instance_recall 相对 same_budget_R：-33.33 个百分点，要求不少于 -5.00 个百分点。
- original.all.instance_recall 相对 historical_A：-18.89 个百分点，要求不少于 -5.00 个百分点。
- original.transformer.instance_recall 相对 historical_A：-12.96 个百分点，要求不少于 -5.00 个百分点。
- original.switchgear.instance_recall 相对 historical_A：-17.20 个百分点，要求不少于 -5.00 个百分点。
- original.capacitor_bank.instance_recall 相对 historical_A：-16.67 个百分点，要求不少于 -5.00 个百分点。
- original.reactor.instance_recall 相对 historical_A：-53.33 个百分点，要求不少于 -5.00 个百分点。
- lighting.all.instance_recall 相对 historical_A：-28.89 个百分点，要求不少于 -5.00 个百分点。
- lighting.transformer.instance_recall 相对 historical_A：-20.37 个百分点，要求不少于 -5.00 个百分点。
- lighting.switchgear.instance_recall 相对 historical_A：-43.01 个百分点，要求不少于 -5.00 个百分点。
- lighting.reactor.instance_recall 相对 historical_A：-13.33 个百分点，要求不少于 -5.00 个百分点。
- original.all.instance_recall 相对 matched_K-300：-12.22 个百分点，要求不少于 -5.00 个百分点。
- original.transformer.instance_recall 相对 matched_K-300：-12.96 个百分点，要求不少于 -5.00 个百分点。
- original.capacitor_bank.instance_recall 相对 matched_K-300：-33.33 个百分点，要求不少于 -5.00 个百分点。
- original.reactor.instance_recall 相对 matched_K-300：-33.33 个百分点，要求不少于 -5.00 个百分点。
- lighting.all.instance_recall 相对 matched_K-300：-6.67 个百分点，要求不少于 -5.00 个百分点。
- lighting.switchgear.instance_recall 相对 matched_K-300：-10.75 个百分点，要求不少于 -5.00 个百分点。
- lighting.reactor.instance_recall 相对 matched_K-300：-13.33 个百分点，要求不少于 -5.00 个百分点。

## 配对与诊断解释

相同 pair_id 的 K→L 得失、低置信度／错类／定位不足等操作性漏检分类，以及训练损失端点，见独立 analysis.json。低阈值 0.001 诊断受 NMS 与 max_det 限制，不表示网络从未产生其他框；不替代 confidence 0.37 的正式结果。

L 的材质组共有 165 次漏检事件：111 次未见满足分类条件的保留预测、22 次同类低置信度、26 次错类、6 次定位不足。光照组 112 次漏检中，同类低置信度为 55 次。这里是三个 seed 的重复预测事件，不是新增独立实例。材质与光照的错误构成不同，不能统一解释为一个阈值问题。

### 无目标误检逐框观察

8 个正式误检事件涉及 7 张唯一图像，均已查看全图叠框与框内放大，记录 AI辅助审核、理由及图像／证据哈希。框内主要内容均为柜状结构：L-300-27 的 6 个框将蓝绿色柜状主体（有面板或无面板视角）预测为开关柜，其中 3 框置信度超过 0.84；另两次为灰色柜状结构分别被预测为变压器、电容器。不能依据场景中有杆而将这些误检归类为杆体。

同一张图在 K-300-27 与 L-300-27 中被框住的是不同柜体，不能将其当成同一对象跨 seed 一致误检。此证据支持优先核查普通柜体与目标设备在主体、面板、底座等可见特征上的可区分性；不证明二者几何完全相同，也不自动推导唯一根因。

本轮只回答固定曝光预算下，使用这批同位姿外观／光照变体相对原始图的变化。不能单独确定材质或光照的唯一病因，也不能凭 seed 平均改善宣称统计显著或跨站点泛化。

### 相对历史 I 的监督差异

K/L 每单元全图实例曝光为变压器 1278、开关柜 1080、电容器 714、电抗器 516，总计 3588；历史 I-300 分别为 1575、1575、660、660，总计 4470。图像曝光同为 1800，但本轮实例监督少 882 次（约 19.7%）。这些差异在三个 seed 中一致。K/L 彼此的实例曝光完全一致，因此其配对比较仍成立；但相对 I 的比较同时改变位姿、来源覆盖和监督次数，还可能改变实例尺度分布，不能归因于单一外观因素。

优先的后续开发方向是：冻结一个保留既有桥接覆盖与逐类全图实例曝光的对照，再有界替换为新补采图，并核对实例尺度与前面板覆盖。监督减少或覆盖变窄是可检验假设，不是本轮已证实的唯一病因。不要先扩大困难负例曝光、再搜索阈值或直接改模型结构。

原有能力是否保留以相对 R-300、历史 A 的原始／光照全图及逐类召回最多下降 5 个百分点为准，L 另需满足相对 K 的同样门禁。未达标不追加临时配比、不挑有利 seed。下一步应依据逐类失败项与已有前面板覆盖缺口设计独立开发修复，而不是直接晋升或解封测试。

## 完整性与边界

43 项启动前相关测试及另 15 项评估／历史训练回归通过，v2.11 固定 40 文件在完成时复核通过。实际曝光、300 个优化步、恒定学习率、关闭增强及 30 轮损失曲线逐单元验证。未声明全仓测试通过。训练成员末轮验证只作拟合诊断。

新场景未解封；training_admitted=false、promotable=false；后台定时任务未开启。仍为同一地图／资产，8 个位姿、7 个来源组；开关柜前面板覆盖不足仍存在。

- 完成回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/completion.json`）
- 逐 seed 诊断与配对变化（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/analysis.json`）
- 逐框 AI 辅助审核（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/negative-review.json`）
- 执行回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/execution.json`）
