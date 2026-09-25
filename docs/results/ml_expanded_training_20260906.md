# 66 张扩展训练池结果

在 30 张诊断池的基础上，加入 36 张新采集的 full_2d 开发帧：simple 8 张、medium 12 张、complex 16 张。36 张全部通过人工语义审核，目标类别全部存在；与历史非保护采集的 exact RGB hash 重合为 0，最近 dHash 距离为 12，未触发近似重复隔离。

使用相同的 10 epoch、每 epoch 30 个样本、batch 6、50 个优化步和种子 7/17/27，比较扩展池的 `uniform` 与 `stratified` 两种采样。评估集包括 36 张扩展帧、80 张 hash-disjoint 控制帧和 25 张 hash-disjoint 跨场景帧。

| 训练池/采样 | 扩展帧实例命中（均值） | 控制集实例命中（均值） | 跨场景实例命中（均值） |
| --- | ---: | ---: | ---: |
| 30 张正样本基线 | 89/139 | 166/258 | 82/129 |
| 30 张负样本增强 | 100/139 | 172/258 | 83/129 |
| 66 张 uniform | 122.7/139 | 196/258 | 96.7/129 |
| 66 张 stratified | 115.7/139 | 175.3/258 | 93.3/129 |

扩展数据明显改善实例定位：`66 张 uniform` 相对 30 张负样本增强，在三组数据上分别增加 22.7、24 和 13.7 个实例命中。它仍有较大的 seed 方差，且预期类别命中均值略低于负样本基线：扩展帧 34/36 对 35/36，控制集 37.3/40 对 38/40，跨场景 22.3/25 对 23/25。因此这批权重仍是开发诊断结果，不能晋级为正式模型。

本轮分层采样在扩展池上没有复现先前 30 张小池的优势：相对 uniform，实例命中均值在扩展帧、控制集、跨场景分别下降 7.0、20.7、3.3；只有跨场景预期类别命中增加 1 个。结论是当前 66 张池应采用 uniform 作为临时采样基线，暂停继续增加分层权重，先补齐小目标和 reactor 的真实尺度覆盖，再重新设计配额。

下一步固定两件事：

1. 保留 66 张 uniform 结果作为开发参考，但不选择单一最佳 seed，也不改变正式 v2.11 基线。
2. 对 reactor 的小/中尺度、capacitor_bank 的小尺度和 switchgear 的复杂场景补采，随后用同一 hash-disjoint 门槛复测；只有预期类别命中和实例命中都不下降，才考虑进入更大规模训练池。

证据：

- `data/research/ml_training_recovery_v1/stratified-expansion-v1/recheck.json`
- `data/research/ml_training_recovery_v1/stratified-expansion-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/stratified-expansion-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/expanded-stratified-v1/protocol.json`
- `scripts/vision/run_recovery_expanded_experiment.py`
- `scripts/vision/evaluate_expanded_recheck.py`
