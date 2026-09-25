# 困难负例组成对照：固定开发评估

六份预定终点权重均已完成正式 conf=0.37 和诊断 conf=0.001 推理。
状态：development_complete_no_candidate；开发候选：无。

## 协议与解释边界

CPU、640、类别相关 NMS IoU=0.7、max_det=300、同类一对一匹配 IoU≥0.5；未调阈值或挑选 seed。
旧48张配对开发图（12个位姿，每类3个）与48张无目标开发图重复评估，不是盲测；三个seed不是新增独立样本。
O/N均100步、600次图像曝光，正样本身份、顺序、位置一致，负例均108次。比较的是负例来源池扩展和组内曝光重新分配的组合效果，不是增加负例总曝光的纯因果效应。
本轮没有训练步数对照，不能据此判断延长训练的效果。新增96张训练图不作为泛化测试。
封存新场景未解封；training_admitted=false，promotable=false。

## 三seed均值

| 组 | 条件 | 计划命中 | 全图召回 | 匹配精度 | 未匹配预测均数 |
| --- | --- | ---: | ---: | ---: | ---: |
| O | original | 77.78% | 72.22% | 80.57% | 10.333 |
| O | material | 38.89% | 27.22% | 62.07% | 10.000 |
| O | background | 61.11% | 65.56% | 78.16% | 11.000 |
| O | lighting | 47.22% | 51.67% | 77.51% | 9.000 |
| N | original | 77.78% | 70.56% | 78.19% | 12.000 |
| N | material | 33.33% | 27.78% | 52.49% | 15.333 |
| N | background | 72.22% | 72.78% | 79.92% | 11.333 |
| N | lighting | 41.67% | 51.11% | 77.73% | 9.333 |

| 组 | 无目标FPR均值 | 最差seed | 三seed FPR（7/17/27） |
| --- | ---: | ---: | --- |
| O | 8.33% | 16.67% | 4.17% / 4.17% / 16.67% |
| N | 16.67% | 18.75% | 12.50% / 18.75% / 18.75% |

## 逐seed、逐条件及逐类

逐实例真值、预测、匹配分配、漏检归类及同位姿变体得失见 evaluation-v1 下六份单元JSON。以下逐类召回基于全图监督，计划命中只统计计划设备。

| 单元 | 条件 | 类别 | 计划命中数/帧数 | 全图召回 | 匹配精度 | 未匹配预测 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| O-100-7 | original | transformer | 1/3 | 66.67% | 92.31% | 1 |
| O-100-7 | original | switchgear | 2/3 | 61.29% | 79.17% | 5 |
| O-100-7 | original | capacitor_bank | 3/3 | 50.00% | 30.00% | 7 |
| O-100-7 | original | reactor | 2/3 | 80.00% | 100.00% | 0 |
| O-100-7 | material | transformer | 3/3 | 61.11% | 57.89% | 8 |
| O-100-7 | material | switchgear | 1/3 | 16.13% | 100.00% | 0 |
| O-100-7 | material | capacitor_bank | 0/3 | 0.00% | 0.00% | 3 |
| O-100-7 | material | reactor | 0/3 | 0.00% | 0.00% | 1 |
| O-100-7 | background | transformer | 2/3 | 66.67% | 100.00% | 0 |
| O-100-7 | background | switchgear | 2/3 | 58.06% | 90.00% | 2 |
| O-100-7 | background | capacitor_bank | 3/3 | 50.00% | 33.33% | 6 |
| O-100-7 | background | reactor | 1/3 | 40.00% | 50.00% | 2 |
| O-100-7 | lighting | transformer | 3/3 | 66.67% | 80.00% | 3 |
| O-100-7 | lighting | switchgear | 2/3 | 51.61% | 76.19% | 5 |
| O-100-7 | lighting | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| O-100-7 | lighting | reactor | 1/3 | 40.00% | 50.00% | 2 |
| O-100-17 | original | transformer | 3/3 | 83.33% | 75.00% | 5 |
| O-100-17 | original | switchgear | 3/3 | 83.87% | 96.30% | 1 |
| O-100-17 | original | capacitor_bank | 2/3 | 66.67% | 66.67% | 2 |
| O-100-17 | original | reactor | 2/3 | 60.00% | 100.00% | 0 |
| O-100-17 | material | transformer | 2/3 | 61.11% | 64.71% | 6 |
| O-100-17 | material | switchgear | 1/3 | 12.90% | 100.00% | 0 |
| O-100-17 | material | capacitor_bank | 1/3 | 16.67% | 100.00% | 0 |
| O-100-17 | material | reactor | 2/3 | 60.00% | 75.00% | 1 |
| O-100-17 | background | transformer | 3/3 | 83.33% | 71.43% | 6 |
| O-100-17 | background | switchgear | 3/3 | 77.42% | 80.00% | 6 |
| O-100-17 | background | capacitor_bank | 1/3 | 33.33% | 66.67% | 1 |
| O-100-17 | background | reactor | 0/3 | 0.00% | 0.00% | 0 |
| O-100-17 | lighting | transformer | 1/3 | 50.00% | 69.23% | 4 |
| O-100-17 | lighting | switchgear | 2/3 | 58.06% | 81.82% | 4 |
| O-100-17 | lighting | capacitor_bank | 1/3 | 16.67% | 100.00% | 0 |
| O-100-17 | lighting | reactor | 1/3 | 40.00% | 100.00% | 0 |
| O-100-27 | original | transformer | 3/3 | 83.33% | 65.22% | 8 |
| O-100-27 | original | switchgear | 3/3 | 70.97% | 91.67% | 2 |
| O-100-27 | original | capacitor_bank | 2/3 | 50.00% | 100.00% | 0 |
| O-100-27 | original | reactor | 2/3 | 80.00% | 100.00% | 0 |
| O-100-27 | material | transformer | 3/3 | 72.22% | 61.90% | 8 |
| O-100-27 | material | switchgear | 0/3 | 0.00% | 0.00% | 0 |
| O-100-27 | material | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| O-100-27 | material | reactor | 1/3 | 20.00% | 25.00% | 3 |
| O-100-27 | background | transformer | 3/3 | 83.33% | 75.00% | 5 |
| O-100-27 | background | switchgear | 3/3 | 77.42% | 82.76% | 5 |
| O-100-27 | background | capacitor_bank | 1/3 | 33.33% | 100.00% | 0 |
| O-100-27 | background | reactor | 0/3 | 20.00% | 100.00% | 0 |
| O-100-27 | lighting | transformer | 2/3 | 66.67% | 63.16% | 7 |
| O-100-27 | lighting | switchgear | 3/3 | 61.29% | 100.00% | 0 |
| O-100-27 | lighting | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| O-100-27 | lighting | reactor | 1/3 | 40.00% | 50.00% | 2 |
| N-100-7 | original | transformer | 3/3 | 94.44% | 73.91% | 6 |
| N-100-7 | original | switchgear | 3/3 | 74.19% | 76.67% | 7 |
| N-100-7 | original | capacitor_bank | 3/3 | 83.33% | 100.00% | 0 |
| N-100-7 | original | reactor | 2/3 | 80.00% | 57.14% | 3 |
| N-100-7 | material | transformer | 2/3 | 77.78% | 58.33% | 10 |
| N-100-7 | material | switchgear | 1/3 | 12.90% | 80.00% | 1 |
| N-100-7 | material | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| N-100-7 | material | reactor | 2/3 | 60.00% | 21.43% | 11 |
| N-100-7 | background | transformer | 3/3 | 94.44% | 73.91% | 6 |
| N-100-7 | background | switchgear | 3/3 | 77.42% | 72.73% | 9 |
| N-100-7 | background | capacitor_bank | 3/3 | 50.00% | 100.00% | 0 |
| N-100-7 | background | reactor | 0/3 | 40.00% | 100.00% | 0 |
| N-100-7 | lighting | transformer | 3/3 | 83.33% | 53.57% | 13 |
| N-100-7 | lighting | switchgear | 3/3 | 77.42% | 92.31% | 2 |
| N-100-7 | lighting | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| N-100-7 | lighting | reactor | 1/3 | 40.00% | 50.00% | 2 |
| N-100-17 | original | transformer | 3/3 | 83.33% | 57.69% | 11 |
| N-100-17 | original | switchgear | 2/3 | 41.94% | 100.00% | 0 |
| N-100-17 | original | capacitor_bank | 1/3 | 50.00% | 42.86% | 4 |
| N-100-17 | original | reactor | 1/3 | 60.00% | 100.00% | 0 |
| N-100-17 | material | transformer | 3/3 | 61.11% | 52.38% | 10 |
| N-100-17 | material | switchgear | 0/3 | 0.00% | 0.00% | 0 |
| N-100-17 | material | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| N-100-17 | material | reactor | 0/3 | 0.00% | 0.00% | 0 |
| N-100-17 | background | transformer | 3/3 | 83.33% | 71.43% | 6 |
| N-100-17 | background | switchgear | 3/3 | 70.97% | 84.62% | 4 |
| N-100-17 | background | capacitor_bank | 2/3 | 50.00% | 42.86% | 4 |
| N-100-17 | background | reactor | 0/3 | 20.00% | 100.00% | 0 |
| N-100-17 | lighting | transformer | 1/3 | 55.56% | 66.67% | 5 |
| N-100-17 | lighting | switchgear | 1/3 | 25.81% | 88.89% | 1 |
| N-100-17 | lighting | capacitor_bank | 1/3 | 16.67% | 100.00% | 0 |
| N-100-17 | lighting | reactor | 0/3 | 0.00% | 0.00% | 0 |
| N-100-27 | original | transformer | 3/3 | 83.33% | 93.75% | 1 |
| N-100-27 | original | switchgear | 3/3 | 70.97% | 84.62% | 4 |
| N-100-27 | original | capacitor_bank | 3/3 | 66.67% | 100.00% | 0 |
| N-100-27 | original | reactor | 1/3 | 60.00% | 100.00% | 0 |
| N-100-27 | material | transformer | 1/3 | 44.44% | 61.54% | 5 |
| N-100-27 | material | switchgear | 2/3 | 29.03% | 50.00% | 9 |
| N-100-27 | material | capacitor_bank | 1/3 | 16.67% | 100.00% | 0 |
| N-100-27 | material | reactor | 0/3 | 0.00% | 0.00% | 0 |
| N-100-27 | background | transformer | 3/3 | 83.33% | 88.24% | 2 |
| N-100-27 | background | switchgear | 3/3 | 74.19% | 95.83% | 1 |
| N-100-27 | background | capacitor_bank | 3/3 | 66.67% | 80.00% | 1 |
| N-100-27 | background | reactor | 0/3 | 40.00% | 66.67% | 1 |
| N-100-27 | lighting | transformer | 2/3 | 77.78% | 82.35% | 3 |
| N-100-27 | lighting | switchgear | 3/3 | 54.84% | 89.47% | 2 |
| N-100-27 | lighting | capacitor_bank | 0/3 | 0.00% | 0.00% | 0 |
| N-100-27 | lighting | reactor | 0/3 | 20.00% | 100.00% | 0 |

## 能力保留与数值门禁

下降容差为5个百分点；同时比较同预算R-100和历史A。所有失败项如下。

| 组 | 指标 | 参考 | 实际值或召回差（百分点） | 要求 |
| --- | --- | --- | ---: | --- |
| N | original.planned_instance_hit_rate.mean | 固定绝对门槛 | 77.78% | >= 83.33% |
| N | original.planned_instance_hit_rate.min_seed | 固定绝对门槛 | 58.33% | >= 75.00% |
| N | material.planned_instance_hit_rate.mean | 固定绝对门槛 | 33.33% | >= 50.00% |
| N | material.planned_instance_hit_rate.min_seed | 固定绝对门槛 | 25.00% | >= 33.33% |
| N | lighting.planned_instance_hit_rate.mean | 固定绝对门槛 | 41.67% | >= 60.00% |
| N | no_target.frame_false_positive_rate.mean | 固定绝对门槛 | 16.67% | <= 10.00% |
| N | original.all.instance_recall | same_budget_R | -7.78 | ≥ -5.00 pp |
| N | original.switchgear.instance_recall | same_budget_R | -11.83 | ≥ -5.00 pp |
| N | original.capacitor_bank.instance_recall | same_budget_R | -11.11 | ≥ -5.00 pp |
| N | original.reactor.instance_recall | same_budget_R | -20.00 | ≥ -5.00 pp |
| N | lighting.all.instance_recall | same_budget_R | -17.78 | ≥ -5.00 pp |
| N | lighting.transformer.instance_recall | same_budget_R | -7.41 | ≥ -5.00 pp |
| N | lighting.switchgear.instance_recall | same_budget_R | -25.81 | ≥ -5.00 pp |
| N | lighting.reactor.instance_recall | same_budget_R | -26.67 | ≥ -5.00 pp |
| N | original.all.instance_recall | historical_A | -10.56 | ≥ -5.00 pp |
| N | original.switchgear.instance_recall | historical_A | -16.13 | ≥ -5.00 pp |
| N | original.reactor.instance_recall | historical_A | -26.67 | ≥ -5.00 pp |
| N | lighting.all.instance_recall | historical_A | -15.56 | ≥ -5.00 pp |
| N | lighting.switchgear.instance_recall | historical_A | -26.88 | ≥ -5.00 pp |
| N | lighting.reactor.instance_recall | historical_A | -20.00 | ≥ -5.00 pp |
| O | original.planned_instance_hit_rate.mean | 固定绝对门槛 | 77.78% | >= 83.33% |
| O | original.planned_instance_hit_rate.min_seed | 固定绝对门槛 | 66.67% | >= 75.00% |
| O | material.planned_instance_hit_rate.mean | 固定绝对门槛 | 38.89% | >= 50.00% |
| O | lighting.planned_instance_hit_rate.mean | 固定绝对门槛 | 47.22% | >= 60.00% |
| O | original.all.instance_recall | same_budget_R | -6.11 | ≥ -5.00 pp |
| O | original.transformer.instance_recall | same_budget_R | -5.56 | ≥ -5.00 pp |
| O | original.capacitor_bank.instance_recall | same_budget_R | -22.22 | ≥ -5.00 pp |
| O | original.reactor.instance_recall | same_budget_R | -13.33 | ≥ -5.00 pp |
| O | lighting.all.instance_recall | same_budget_R | -17.22 | ≥ -5.00 pp |
| O | lighting.transformer.instance_recall | same_budget_R | -18.52 | ≥ -5.00 pp |
| O | lighting.switchgear.instance_recall | same_budget_R | -21.51 | ≥ -5.00 pp |
| O | lighting.reactor.instance_recall | same_budget_R | -6.67 | ≥ -5.00 pp |
| O | original.all.instance_recall | historical_A | -8.89 | ≥ -5.00 pp |
| O | original.transformer.instance_recall | historical_A | -11.11 | ≥ -5.00 pp |
| O | original.switchgear.instance_recall | historical_A | -6.45 | ≥ -5.00 pp |
| O | original.capacitor_bank.instance_recall | historical_A | -5.56 | ≥ -5.00 pp |
| O | original.reactor.instance_recall | historical_A | -20.00 | ≥ -5.00 pp |
| O | lighting.all.instance_recall | historical_A | -15.00 | ≥ -5.00 pp |
| O | lighting.transformer.instance_recall | historical_A | -12.96 | ≥ -5.00 pp |
| O | lighting.switchgear.instance_recall | historical_A | -22.58 | ≥ -5.00 pp |

## N相对O的同位姿得失

下表是三个seed的重复测量次数，不是独立图像数。

| 条件 | 计划命中新增 | 计划命中丢失 | 保持 |
| --- | ---: | ---: | ---: |
| original | 4 | 4 | 28 |
| material | 6 | 8 | 22 |
| background | 5 | 1 | 30 |
| lighting | 1 | 3 | 32 |

## 低阈值漏检诊断

分类有优先级：同类低置信度→错类→定位不足→未发现合格保留预测。最后一类受NMS和max_det限制，不能解释为网络从未产生候选。诊断输出不替代正式评估。

| 组 | 同类低置信度 | 错类 | 定位不足 | 未发现合格保留预测 |
| --- | ---: | ---: | ---: | ---: |
| O | 197 | 90 | 18 | 25 |
| N | 178 | 84 | 12 | 46 |

## 可复算性

O对历史Y-100的正式/低阈值全部逐框结果复现：{'17': True, '27': True, '7': True}。
六单元计划目标独立命中与全图匹配冲突总数：0。
完成回执递归绑定冻结协议、开发图/标签回执、审核、训练成员与曝光、六份权重、评估结果及测试。
相关回归通过，v2.11固定40文件完整性通过。未重跑全仓测试；历史12项失败和1项跳过仍单独保留，不声明全仓通过。

## 结论与下一步


本轮没有证明扩大负例来源池能降低误检：N的无目标FPR均值由O的8.33%升至16.67%，三个seed分别从4.17%/4.17%/16.67%变为12.50%/18.75%/18.75%，均未改善。

背景条件有方向性收益：计划命中由61.11%升至72.22%，全图召回由65.56%升至72.78%；但材质计划命中由38.89%降至33.33%，光照由47.22%降至41.67%。不能将背景收益解释为总体改进。

原始全图召回70.56%，较O下降1.67个百分点，但较R-100下降7.78个百分点、较历史A下降10.56个百分点；光照全图召回51.11%，较R下降17.78、较A下降15.56个百分点。能力保留门禁未通过，不能只与较弱O比较后宣称保留。

低阈值下仍有大量同类低置信度漏检（O 197、N 178次），同时N未发现合格保留预测由25增至46次。这些是三seed重复预测计数，不能单独认定置信阈值或网络结构是唯一根因。

下一步优先做旧负例锚定与覆盖分层的受控采样对照：保持正样本曝光和预算，避免扩展池后稀释原来有效的负例；配比应在下一轮推理前冻结，不在本轮追加试验。先逐框核对N新增与持续误检的实际内容，再决定是否补某类结构，不凭训练覆盖单元名称推断误检对象。

同时重点分析原始/光照下开关柜、电抗器召回退化与实例尺度、全图监督曝光及混淆类别的关系。当前证据不足以优先更换模型结构；先完成误检内容与采样诊断，再单独设计增强或结构对照。本轮未进行新的逐框人工/AI内容审核，不将自动漏检分类冒充视觉审核。

本阶段完整结束但无候选；不扩大采集、不追加临时配比、不挑选有利seed、不解封新场景。

