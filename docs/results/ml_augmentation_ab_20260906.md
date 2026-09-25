# 小目标训练 A/B 结果

本轮在相同的 66 张扩展训练池、10 epoch、每 epoch 30 个样本、batch 6、50 个优化步和种子 7/17/27 下，比较三种只改一个因素的方案：`scale_aug`（640 输入、scale=0.25）、`highres`（960 输入）和 `cls_weight`（分类损失权重 0.75）。对照是已有的 66 张 `uniform` 训练结果。所有权重仍保持 `training_admitted=false`、`promotable=false`。

## 三种子均值

| 方案 | 定向集实例命中 | 定向集预期类别 | 控制集实例命中 | 控制集预期类别 | 跨场景实例命中 | 跨场景预期类别 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 张 uniform | 98.3/127 | 19.7/24 | 196.0/258 | 37.3/40 | 96.7/129 | 22.3/25 |
| scale_aug | 73.3/127 | 22.7/24 | 154.0/258 | 30.7/40 | 75.7/129 | 19.7/25 |
| highres | 92.0/127 | 20.3/24 | 192.7/258 | 39.3/40 | 92.0/129 | 22.3/25 |
| cls_weight | 95.7/127 | 22.7/24 | 190.7/258 | 32.0/40 | 94.3/129 | 22.3/25 |

## 判断

三种方案都没有通过配对门槛。门槛要求在定向集、控制集和跨场景集上，实例命中与预期类别命中均不低于 `uniform` 均值。

- `scale_aug` 在定向集的预期类别命中增加 3 个，但实例命中下降 25 个；控制集和跨场景集也明显下降，说明这类尺度扰动破坏了当前小数据池的定位稳定性。
- `highres` 在控制集预期类别命中增加 2 个，定向集增加约 0.7 个，但三组实例命中都下降，跨场景预期类别没有增加。单纯把输入放大到 960 不能作为当前训练方案。
- `cls_weight` 在定向集预期类别命中增加 3 个，但控制集预期类别命中下降约 5.3 个，实例命中在三组都下降。它只改善了定向小样本上的分类，不具备泛化证据。

因此暂时保留 `66 张 uniform` 作为开发参考，不选择某个最佳 seed，也不改变正式 v2.11 基线。当前最可信的诊断是：数据覆盖和实例定位已经明显改善，剩余瓶颈是远距离小目标的分类置信度与跨场景稳定性；三种简单增广在当前数据量和训练预算下都不足以解决它。

下一轮应优先补充按类别和真实尺度分层的 hash-clean 帧，特别是 reactor、capacitor_bank 和远距离 switchgear，并把目标类别置信度、实例命中和控制集回归作为同一门槛。训练预算也应在扩充后再增加，而不是继续在 66 张池上叠加单一增广。

证据：

- `data/research/ml_training_recovery_v1/augmentation-ab-v1/recheck.json`
- `data/research/ml_training_recovery_v1/augmentation-ab-v1/protocol.json`
- `scripts/vision/run_recovery_augmentation_ab.py`
- `scripts/vision/evaluate_augmentation_ab.py`
- `data/research/ml_training_recovery_v1/targeted-coverage-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/targeted-coverage-v1/dedup-audit.json`
