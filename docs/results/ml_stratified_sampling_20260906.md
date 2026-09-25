# 分层采样训练诊断结果

本轮在修复后的标注模式基础上，固定 30 张训练图像：18 张原始正样本 + 12 张 simple 修复正样本。比较两种采样臂：`uniform` 按原始行顺序重复曝光，`stratified` 按类别、地图和目标尺度配额曝光；两臂均使用种子 7、17、27，10 个 epoch、每 epoch 30 个样本、batch 6，严格得到 50 个优化步。协议在训练前冻结，24 个不存在的类别×地图×尺度 strata 被显式记录，没有伪造或补标签。

六次运行均成功完成，全部保持 `training_admitted=false`、`promotable=false`。评估只使用与 30 张训练池 exact RGB hash 完全分离的 80 帧控制集和 25 帧跨场景集；两组重叠数均为 0。指标使用既有设置：置信度 0.37、NMS IoU 0.7、实例匹配 IoU 0.5。

| 采样臂 | 控制集实例命中（均值±标准差） | 控制集预期类别命中 | 跨场景实例命中（均值±标准差） | 跨场景预期类别命中 |
| --- | ---: | ---: | ---: | ---: |
| uniform | 133.3 ± 17.5 / 258 | 35.7/40 | 69.0 ± 8.7 / 129 | 19.7/25 |
| stratified | 149.0 ± 6.2 / 258 | 36.3/40 | 75.3 ± 4.9 / 129 | 21.7/25 |

在冻结的配对规则下，`stratified` 相对 `uniform` 的控制集均值增加 15.7 个实例命中和 0.7 个预期类别命中；跨场景均值增加 6.3 个实例命中和 2.0 个预期类别命中。因此，分层配额作为“采样方法”通过了本轮 A/B 规则，而且种子间波动明显更小。

但它仍低于已有诊断基线：正样本基线在控制集和跨场景分别为 166/258、82/129，难负样本版本为 172/258、83/129。故本轮结论是“分层采样优于同数据的均匀重复曝光”，不是“30 张小训练池已经足够”。不能从这批结果选一个最佳 seed，也不能把任一权重放入正式训练包。

下一步应保持分层配额，扩大具有真实覆盖的类别×地图×尺度组合，优先补齐 transformer 的中/大尺度和多地图视角，以及 switchgear 的 complex 覆盖；每次扩充后继续使用 hash-disjoint 控制集和跨场景集做成对回归。

证据：

- `data/research/ml_training_recovery_v1/stratified-sampling-v1/protocol.json`
- `data/research/ml_training_recovery_v1/stratified-sampling-v1/recheck.json`
- `scripts/vision/run_recovery_stratified_experiment.py`
- `scripts/vision/evaluate_stratified_recheck.py`
