# Simple 场景标注模式修复结果

本轮修复了 simple 独立场景中 3 张 switchgear truth 缺失的问题。根因是旧计划使用 `label_mode=source`，且没有声明 `hierarchy_mode`；同类可用 canonical 诊断计划使用 `label_mode=visual-instance` 与 `hierarchy_mode=top-level-equipment`。旧模式下图像中可见的 switchgear 没有稳定进入 `full_2d` truth，所以原来的“switchgear 0/3”不能作为模型漏检证据。

采集器现在对 `full_2d` 计划强制检查显式实例标签和顶层设备层级；必须有目标的诊断计划设置 `diagnostic_require_expected_presence=true`，目标类别缺失时直接拒绝采集。旧诊断数据没有覆盖或删除，仍保留为问题追踪证据。

修复计划基于 `simple-occlusion-filtered-plan-v1` 的已物化 world，选择 12 个未出现在历史回执中的视角：5 个 switchgear、7 个 transformer。采集结果：12/12 完成，12/12 预期类别存在，0 个 exact RGB overlap。全部帧经 contact-sheet 视觉检查和 `full_2d` truth 核验后接受，仍保持 `training_admitted=false`、`promotable=false`。

在同一推理设置下复测两组诊断权重：

| 权重 | 帧数 | 真值实例 | 实例命中 | 预测框 | 预期类别命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 仅正样本 | 12 | 29 | 25 | 54 | 11/12 |
| 加入 14 个难负样本 | 12 | 29 | 22 | 46 | 10/12 |

按类别看，switchgear 两组都是 5/5；transformer 为 6/7 对 5/7。难负样本减少 8 个预测框，但在这批修复后的 transformer 视角上少命中 1 个实例，说明背景抑制和目标几何覆盖存在权衡，不能把 hard-negative 版本直接设为默认模型。

错误拆分显示，hard-negative 版本在这 12 帧中漏掉 7 个 transformer truth，正样本版本漏掉 3 个；其余差异主要是背景候选和少量 wrong-class-on-target。该现象与训练内负样本 A/B 的“背景预测减少”一致，但说明当前 14 个裁剪可能过度压制了与 transformer 外观相近的区域，暂不应继续扩大这批负样本。

另外做了“18 帧原始正样本 + 12 帧修复正样本”的诊断训练。新权重在修复帧上拟合良好，但开发复验下降：held target 为 42/46，80 帧 control 为 135/258，25 帧跨场景为 72/129。这里的 held target 不能当作独立泛化证据：其中 18/19 帧与原始 18 帧训练图像按 exact RGB hash 重合，只有 control 和 cross-scene 两组与训练池完全分离。新增视角集中于同一个 simple world，导致场景和几何分布被过度加权；这证明直接追加正样本会造成分布漂移，不能作为训练修复方案。

下一步固定两项：

1. 所有新增 `full_2d` 计划必须使用修复后的标注模式，并在采集回执中保留 `expected_class_present`。
2. 继续补 transformer 的视角、尺度和遮挡变化；hard-negative 仅作为 A/B 变量，暂不纳入官方训练包。
3. 下一轮训练使用按类别、地图和尺度分层的采样配额；修复 simple 数据只作为受控比例的补充，并保留跨场景回归门槛，避免单一 world 过拟合。

证据：

- `data/research/ml_training_recovery_v1/simple-annotation-repair-v2/manifest.json`
- `data/research/ml_training_recovery_v1/simple-annotation-repair-v2/semantic-review.json`
- `data/research/ml_training_recovery_v1/simple-annotation-repair-v2/evaluation.json`
- `data/research/ml_training_recovery_v1/simple-annotation-repair-v2/error-audit.json`
- `data/research/ml_training_recovery_v1/repaired-positive-v1/independent-recheck.json`
- `scripts/vision/prepare_simple_annotation_repair.py`
- `scripts/vision/capture_simple_annotation_repair.py`
- `src/vision/canonical/collect.py`
