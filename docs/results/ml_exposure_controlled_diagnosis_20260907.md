# 曝光受控训练对照与错误诊断

本实验使用现有冻结数据，完成 R/X/Y × 100/300 步 × seed 7/17/27 共18个独立训练单元。正式推理和 confidence=0.001 诊断覆盖本轮18份权重及历史A/D各3份权重。

阶段状态：`development_complete_no_candidate`；开发候选家族：`None`。封存新场景未采集、未评估，未访问保护标签，全部产物 training_admitted=false、promotable=false。

## 冻结设计

| 组别 | 基础池 | 常规正样本 | 桥接正样本 | 困难负例 |
| --- | ---: | ---: | ---: | ---: |
| R | 330 | 270 | 0 | 0 |
| X | 216 | 156 | 156 | 72 |
| Y | 216 | 156 | 120 | 108 |

表为每600次图像曝光的配额，300步为三倍。X/Y每600次只有36次外观/负例交换，其他成员和位置一致。100/300步采样前缀一致，均从v2.11初始化，CPU/640/batch6/AdamW/恒定lr0=0.001，无预热和数据增强。采用终点权重；训练成员上的末轮验证只是拟合诊断。

## 正式开发评估

以下为三个seed均值；每个条件只有12个位姿，每类3个位姿，seed重复不增加独立样本数。固定confidence=0.37、同类NMS IoU=0.7、max_det=300、匹配IoU≥0.5。

| 家族 | 原始计划命中 | 材质计划命中 | 背景计划命中 | 光照计划命中 | 原始全图召回 | 光照全图召回 | 无目标FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R-100 | 0.889 | 0.167 | 0.694 | 0.611 | 0.783 | 0.689 | 0.569 |
| X-100 | 0.750 | 0.389 | 0.639 | 0.583 | 0.711 | 0.578 | 0.153 |
| Y-100 | 0.778 | 0.389 | 0.611 | 0.472 | 0.722 | 0.517 | 0.083 |
| R-300 | 0.944 | 0.111 | 0.667 | 0.528 | 0.806 | 0.622 | 0.368 |
| X-300 | 0.917 | 0.500 | 0.889 | 0.583 | 0.833 | 0.567 | 0.375 |
| Y-300 | 0.833 | 0.417 | 0.722 | 0.472 | 0.689 | 0.494 | 0.243 |

| 家族 | 材质计划命中seed范围 | 无目标FPR seed范围 |
| --- | ---: | ---: |
| R-100 | 0.083–0.333 | 0.354–0.896 |
| R-300 | 0.083–0.167 | 0.167–0.646 |
| X-100 | 0.250–0.500 | 0.062–0.208 |
| X-300 | 0.417–0.583 | 0.167–0.667 |
| Y-100 | 0.333–0.500 | 0.042–0.167 |
| Y-300 | 0.167–0.583 | 0.167–0.312 |

## 受控差异与门禁

X/Y差异是固定预算下交换外观与负例曝光的结果，不能解释成纯增加负例的因果效应。均值差异是本开发集方向性结果，不是统计显著性结论。

| 对照（后者减前者） | 材质命中差 | 原始全图召回差 | 光照全图召回差 | 无目标FPR差 |
| --- | ---: | ---: | ---: | ---: |
| R-300_minus_R-100 | -0.056 | +0.022 | -0.067 | -0.201 |
| X-300_minus_X-100 | +0.111 | +0.122 | -0.011 | +0.222 |
| Y-100_minus_X-100 | +0.000 | +0.011 | -0.061 | -0.069 |
| Y-300_minus_X-300 | -0.083 | -0.144 | -0.072 | -0.132 |
| Y-300_minus_Y-100 | +0.028 | -0.033 | -0.022 | +0.160 |

### X-100：未通过，共19项门槛失败

- original.planned_instance_hit_rate.mean：0.750，要求 >= 0.833。
- original.planned_instance_hit_rate.min_seed：0.667，要求 >= 0.750。
- material.planned_instance_hit_rate.mean：0.389，要求 >= 0.500。
- material.planned_instance_hit_rate.min_seed：0.250，要求 >= 0.333。
- lighting.planned_instance_hit_rate.mean：0.583，要求 >= 0.600。
- no_target.frame_false_positive_rate.mean：0.153，要求 <= 0.100。
- no_target.frame_false_positive_rate.max_seed：0.208，要求 <= 0.200。
- original.all.instance_recall 相对 same_budget_R：-0.072，允许最低−0.050。
- original.transformer.instance_recall 相对 same_budget_R：-0.148，允许最低−0.050。
- original.reactor.instance_recall 相对 same_budget_R：-0.133，允许最低−0.050。
- lighting.all.instance_recall 相对 same_budget_R：-0.111，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 same_budget_R：-0.204，允许最低−0.050。
- lighting.reactor.instance_recall 相对 same_budget_R：-0.067，允许最低−0.050。
- original.all.instance_recall 相对 historical_A：-0.100，允许最低−0.050。
- original.transformer.instance_recall 相对 historical_A：-0.204，允许最低−0.050。
- original.switchgear.instance_recall 相对 historical_A：-0.075，允许最低−0.050。
- original.reactor.instance_recall 相对 historical_A：-0.200，允许最低−0.050。
- lighting.all.instance_recall 相对 historical_A：-0.089，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 historical_A：-0.215，允许最低−0.050。

### X-300：未通过，共10项门槛失败

- lighting.planned_instance_hit_rate.mean：0.583，要求 >= 0.600。
- no_target.frame_false_positive_rate.mean：0.375，要求 <= 0.100。
- no_target.frame_false_positive_rate.max_seed：0.667，要求 <= 0.200。
- lighting.all.instance_recall 相对 same_budget_R：-0.056，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 same_budget_R：-0.108，允许最低−0.050。
- lighting.reactor.instance_recall 相对 same_budget_R：-0.067，允许最低−0.050。
- original.reactor.instance_recall 相对 historical_A：-0.067，允许最低−0.050。
- lighting.all.instance_recall 相对 historical_A：-0.100，允许最低−0.050。
- lighting.transformer.instance_recall 相对 historical_A：-0.093，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 historical_A：-0.183，允许最低−0.050。

### Y-100：未通过，共20项门槛失败

- original.planned_instance_hit_rate.mean：0.778，要求 >= 0.833。
- original.planned_instance_hit_rate.min_seed：0.667，要求 >= 0.750。
- material.planned_instance_hit_rate.mean：0.389，要求 >= 0.500。
- lighting.planned_instance_hit_rate.mean：0.472，要求 >= 0.600。
- original.all.instance_recall 相对 same_budget_R：-0.061，允许最低−0.050。
- original.transformer.instance_recall 相对 same_budget_R：-0.056，允许最低−0.050。
- original.capacitor_bank.instance_recall 相对 same_budget_R：-0.222，允许最低−0.050。
- original.reactor.instance_recall 相对 same_budget_R：-0.133，允许最低−0.050。
- lighting.all.instance_recall 相对 same_budget_R：-0.172，允许最低−0.050。
- lighting.transformer.instance_recall 相对 same_budget_R：-0.185，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 same_budget_R：-0.215，允许最低−0.050。
- lighting.reactor.instance_recall 相对 same_budget_R：-0.067，允许最低−0.050。
- original.all.instance_recall 相对 historical_A：-0.089，允许最低−0.050。
- original.transformer.instance_recall 相对 historical_A：-0.111，允许最低−0.050。
- original.switchgear.instance_recall 相对 historical_A：-0.065，允许最低−0.050。
- original.capacitor_bank.instance_recall 相对 historical_A：-0.056，允许最低−0.050。
- original.reactor.instance_recall 相对 historical_A：-0.200，允许最低−0.050。
- lighting.all.instance_recall 相对 historical_A：-0.150，允许最低−0.050。
- lighting.transformer.instance_recall 相对 historical_A：-0.130，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 historical_A：-0.226，允许最低−0.050。

### Y-300：未通过，共21项门槛失败

- original.planned_instance_hit_rate.min_seed：0.667，要求 >= 0.750。
- material.planned_instance_hit_rate.mean：0.417，要求 >= 0.500。
- material.planned_instance_hit_rate.min_seed：0.167，要求 >= 0.333。
- lighting.planned_instance_hit_rate.mean：0.472，要求 >= 0.600。
- no_target.frame_false_positive_rate.mean：0.243，要求 <= 0.100。
- no_target.frame_false_positive_rate.max_seed：0.312，要求 <= 0.200。
- original.all.instance_recall 相对 same_budget_R：-0.117，允许最低−0.050。
- original.switchgear.instance_recall 相对 same_budget_R：-0.172，允许最低−0.050。
- original.capacitor_bank.instance_recall 相对 same_budget_R：-0.167，允许最低−0.050。
- original.reactor.instance_recall 相对 same_budget_R：-0.200，允许最低−0.050。
- lighting.all.instance_recall 相对 same_budget_R：-0.128，允许最低−0.050。
- lighting.transformer.instance_recall 相对 same_budget_R：-0.093，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 same_budget_R：-0.183，允许最低−0.050。
- lighting.reactor.instance_recall 相对 same_budget_R：-0.133，允许最低−0.050。
- original.all.instance_recall 相对 historical_A：-0.122，允许最低−0.050。
- original.switchgear.instance_recall 相对 historical_A：-0.151，允许最低−0.050。
- original.capacitor_bank.instance_recall 相对 historical_A：-0.222，允许最低−0.050。
- original.reactor.instance_recall 相对 historical_A：-0.267，允许最低−0.050。
- lighting.all.instance_recall 相对 historical_A：-0.172，允许最低−0.050。
- lighting.transformer.instance_recall 相对 historical_A：-0.167，允许最低−0.050。
- lighting.switchgear.instance_recall 相对 historical_A：-0.258，允许最低−0.050。

## 低阈值漏检诊断

按每个未匹配真值实例依次判断低置信度同类、错类、定位不足、未发现满足条件的保留预测。NMS及max_det仍限制候选，第四类不能证明网络未生成候选；匹配竞争另外记录。

| 家族 | 条件 | 低置信度同类 | 错类 | 定位不足 | 无满足条件预测 |
| --- | --- | ---: | ---: | ---: | ---: |
| historical-D | material | 60 | 40 | 8 | 4 |
| historical-D | lighting | 49 | 22 | 4 | 2 |
| historical-D | original | 38 | 8 | 2 | 0 |
| R-100 | material | 20 | 43 | 9 | 91 |
| R-100 | lighting | 27 | 24 | 3 | 2 |
| R-100 | original | 25 | 10 | 2 | 2 |
| X-100 | material | 60 | 39 | 9 | 15 |
| X-100 | lighting | 45 | 21 | 5 | 5 |
| X-100 | original | 39 | 9 | 4 | 0 |
| Y-100 | material | 60 | 47 | 9 | 15 |
| Y-100 | lighting | 56 | 21 | 6 | 4 |
| Y-100 | original | 40 | 9 | 1 | 0 |
| R-300 | material | 16 | 22 | 4 | 129 |
| R-300 | lighting | 35 | 23 | 5 | 5 |
| R-300 | original | 16 | 12 | 3 | 4 |
| X-300 | material | 44 | 41 | 10 | 20 |
| X-300 | lighting | 41 | 23 | 6 | 8 |
| X-300 | original | 18 | 7 | 2 | 3 |
| Y-300 | material | 43 | 37 | 4 | 34 |
| Y-300 | lighting | 37 | 26 | 10 | 18 |
| Y-300 | original | 26 | 24 | 4 | 2 |

## 逐框AI辅助审核

历史D的67个误检框来自30张不同图像、20个位姿。已逐框查看全图上下文及裁剪，结果按实际框内内容登记，非按采集主题推断。图像、叠框证据、预测坐标、seed、置信度、理由及审核时间均已绑定。

| 实际内容 | 框数 |
| --- | ---: |
| building | 42 |
| ground_shadow | 12 |
| mixed_structure | 7 |
| cabinet | 4 |
| pole | 2 |

## 对三个问题的回答与下一步

- 100步加强负例曝光：FPR变化-0.069，材质计划命中变化+0.000，光照全图召回变化-0.061。这是曝光交换的组合效果，不能只用FPR一项判断整体改善。
- 300步加强负例曝光：FPR变化-0.132，材质计划命中变化-0.083，光照全图召回变化-0.072。这是曝光交换的组合效果，不能只用FPR一项判断整体改善。
- R延长至300步：原始全图召回变化+0.022，材质计划命中变化-0.056，光照全图召回变化-0.067，FPR变化-0.201。
- X延长至300步：原始全图召回变化+0.122，材质计划命中变化+0.111，光照全图召回变化-0.011，FPR变化+0.222。
- Y延长至300步：原始全图召回变化-0.033，材质计划命中变化+0.028，光照全图召回变化-0.022，FPR变化+0.160。

原有能力是否保留，以双参考全图/逐类门禁为准，不能用计划目标命中替代。上述结果也表明训练损失下降不等于鲁棒性同步提高。低阈值诊断将置信度不足、错类和定位不足分开，但它不是阈值调优，更不证明其中某一类是唯一根因。

下一步优先级：

1. 依据逐框证据补充新位姿下的灰色建筑/箱体、地面网格与投影、画面边缘截断结构负例；保留旧开发图的评估角色，不将其转入训练。先做小批来源匹配与审核。
2. 保持明确的共同基础曝光下限和负例配额，针对电容器组、电抗器较少的全图实例监督，单独冻结一次类别/实例曝光平衡对照；不把计划类别计数当作监督平衡。
3. 若采样和补缺后仍有置信度/错类问题，再分别对照温和光度增强、较低学习率或部分骨干冻结。每轮只增加一个变化，保留原始条件回归；现有结果不足以优先支持更大模型。
4. 新场景继续封存。通过开发门禁后再执行冻结的新场景复验，检验当前场景收益能否迁移。


## 验证与边界

候选必须通过旧计划命中/FPR门槛，以及原始和光照的全图、逐类召回相对同预算R和历史A均下降不超过5个百分点的门槛。所有计划目标与一对一匹配冲突必须清零才能完成。

基础与常规池沿用成员来源，成员ID不自动等同独立场景；新增桥接样本保留原始派生组。48张配对图和48张无目标图均为已查看开发集，不能用于跨站点或唯一根因结论。

协议身份：`6ce2eea8f76b1dae922779b7b2f74ab217a9f5e62a907e449e2e9dcc97d18d35`。
闭环身份：`fab1d8e197d3b4d3764c489487cf5dda3c75bf69fa1cd3ac0369073af4f8503e`。

数值明细、逐类结果、seed波动、配对得失、曝光与输入哈希位于独立实验目录。

验证记录：目标回归34项通过，v2.11固定40文件完整性通过，git diff --check通过。未重跑全仓测试；此前全仓876项中的12项失败和1项跳过仍单独保留，不声明全仓通过。

100/300步训练损失前缀完全一致的配对数：9/9。

来源位姿组数：基础池66、常规正样本48、桥接正样本16、困难负例12。这些是来源位姿身份，不能解释为142个独立场景。

审核工具勘误：首版仅按view_id索引，混合了同位姿的两种光照，收尾门禁因预测编号重复而拒绝。已改用view_id+variant，重新逐图审核全部30张图像和67个框。旧证据保留为superseded，有效审核为visual-review-v2.json，详细说明见review-index-erratum.json。训练和推理未因此改变。
