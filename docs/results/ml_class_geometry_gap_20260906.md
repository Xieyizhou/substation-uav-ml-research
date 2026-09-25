# 类别与几何覆盖核验结果

本轮把训练记忆诊断、19 个 held target、80 个 development control、25 个去除像素重叠的跨场景帧，以及一个真正不同 world 的 simple 场景放在同一套 IoU 0.5 匹配协议下比较。所有结果仍是开发诊断，未进入 protected validation，也没有改变 v2.11 基线。

训练记忆集只有 18 个 transformer、16 个 switchgear、7 个 capacitor_bank、3 个 reactor 对象。它能被模型拟合，但这个数量和外观变化不足以证明泛化，尤其 reactor 的有效几何覆盖最薄。

在同源开发复验中，加入 14 个经人工审核的背景 hard-negative 后，held target 从 44/46 提升到 46/46 个实例命中，预测框从 64 降到 47；80 个 control 的实例命中从 166/258 提升到 172/258，预测框从 334 降到 298。两组 control 的预期类别命中都为 38/40，说明这次改动主要改善了定位和背景抑制，尚未证明类别边界整体改善。

去除与既有诊断帧逐像素重叠后的 25 个跨场景帧中，仅正样本为 82/129 个实例命中、147 个预测框；加入 hard-negative 后为 83/129、132 个预测框。预期类别命中两组都是 23/25。按场景看，complex 为 47/88 对 47/88，medium 为 22/28 对 25/28，simple 为 13/13 对 11/13；收益集中在 medium，simple 出现回退，不能当作稳定泛化结论。

几何分桶（按图像框面积，小于 40,000 为 small，40,000–180,000 为 medium，大于 180,000 为 large）只用于定位后续采样方向，不能解释成物理距离结论。在开发聚合中，switchgear 的 complex 小框为 5/5、中框 10/11、大框 2/2；reactor 为 large 0/2、small 1/1。reactor 样本极少，large 结果只能标记为覆盖缺口候选，不能据此调阈值或裁剪策略。

真正不同 world 的 simple 扩展包含 3 个 cabinet、3 个 empty-ground、3 个 switchgear、3 个 transformer 视角。原始 `full_2d` truth 显示，3 个预期 switchgear 视角都没有 switchgear 实例，3 个预期 transformer 视角都有 transformer 实例。两组权重在可观测的 transformer 上都是 1/3；switchgear 的“0/3”是目标缺失/可见性或计划标注核验项，不是模型漏检率。presence audit 纳入修复采集后覆盖 193 个开发诊断帧，其中唯一的 3 个 `expected_target_absent` 都来自旧的 `label_mode=source` 且未声明 `hierarchy_mode` 的 legacy world；采用 `visual-instance + top-level-equipment` 的 129 个目标帧没有出现类别缺失。其余 80 个 switchgear、27 个 transformer、18 个 capacitor_bank、7 个 reactor 预期帧都有对应类别 truth。由此，source-plan 必须在采集前检查“预期类别确实存在且达到可见性门槛”，否则会把目标缺失帧误当成正样本失败或负样本。

下一步按以下顺序执行：

1. 给 source plan 增加 expected-class presence 与可见性门控；采集回执中同时保存 expected class、truth class 集合和 `expected_class_present`，目标缺失时标记 hold，不进入召回分母。采集器现已对 `full_2d` 强制要求 `label_mode=visual-instance` 与 `hierarchy_mode=top-level-equipment`；对必须有目标的诊断计划再设置 `diagnostic_require_expected_presence=true`，避免 `diagnostic_allow_expected_absence` 把计划错误静默放行。
2. 优先补 reactor 的完整设备实例，并覆盖 small/medium/large 三档；其次补 switchgear 的遮挡、相似背景和不同视角。每个新增视角先通过 target-presence、全设备语义和像素 hash 检查，再进入开发训练。
3. 保留 hard-negative A/B 作为背景抑制实验；只有在新的、类别可观测的跨场景集合上重复获得收益，才考虑调整模型或阈值。

证据文件：

- `data/research/ml_training_recovery_v1/class-geometry-gap-v1.json`
- `data/research/ml_training_recovery_v1/negative-ab-v1/independent-recheck.json`
- `data/research/ml_training_recovery_v1/cross-scene-recheck-v1/evaluation.json`
- `data/research/ml_training_recovery_v1/simple-independent-scene-v1/evaluation.json`
- `data/research/ml_training_recovery_v1/expected-class-presence-audit-v1/report.json`
- `docs/results/ml_annotation_repair_20260906.md`

本报告及上述结果均保留 `training_admitted=false`、`promotable=false`。
