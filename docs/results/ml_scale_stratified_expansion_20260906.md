# 尺度分层扩展结果

本轮新增 36 张 full_2d 开发帧：simple 6 张、medium 12 张、complex 18 张。采集计划按 near（5–9.99m）、mid（10–14.99m）、far（15m 以上）距离桶冻结，重点覆盖 complex reactor、capacitor_bank 和 switchgear。36/36 完成采集，36/36 通过人工语义审核，exact duplicate 为 0，最近历史 dHash 距离为 9。全部保持 `training_admitted=false`、`promotable=false`。

旧模型在这 36 张新帧上的基线为 136 个真值实例：`negative_augmented` 命中 93 个，已有 66 张 `uniform` 三个 seed 命中 100、105、111 个；类别命中分别为 31/36、31/36、32/36。reactor 的类别命中为 3–5/6，仍是主要不稳定来源。

随后将原 66 张与新 36 张合为 102 张开发训练池，统一采用 640 输入、10 epoch、每 epoch 60 个样本、batch 6、100 个优化步，使用 seed 7/17/27。三 seed 均完成，训练和权重 hash 已写入协议。

| 方案 | 新 36 张实例命中 | 新 36 张类别命中 | 定向覆盖实例命中 | 定向覆盖类别命中 | 控制集实例命中 | 控制集类别命中 | 跨场景实例命中 | 跨场景类别命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 张 uniform | 105.3/136 | 32.7/36 | 98.3/127 | 19.7/24 | 196.0/258 | 37.3/40 | 96.7/129 | 22.3/25 |
| 102 张 uniform，100 steps | 116.3/136 | 34.3/36 | 93.7/127 | 18.7/24 | 193.7/258 | 35.7/40 | 97.0/129 | 23.7/25 |

102 张池改善了新数据上的拟合，但没有通过配对门槛：定向覆盖集实例命中下降 4.7 个、类别命中下降 1 个；控制集实例命中下降 2.3 个、类别命中下降 1.7 个。跨场景集只有类别命中增加 1.3 个，不能抵消另外两组的回归。

结论是新增尺度数据本身有效，但当前一次性并入训练池并提高到 100 steps 会牺牲已有泛化。102 张权重不进入正式模型，也不作为新的统一基线。保留 66 张 uniform 作为当前开发参考，并继续把定向覆盖集、控制集和跨场景集作为门禁。

下一步应保留本轮 36 张数据作为候选，不直接扩大训练预算；重新采集一批与当前定向覆盖集同分布但 hash-clean 的新帧，用于训练和独立回归分离，再比较 50-step 与 100-step 预算，避免把训练步数和数据增量混在一次实验里。

证据：

- `data/research/ml_training_recovery_v1/scale-stratified-v1/manifest.json`
- `data/research/ml_training_recovery_v1/scale-stratified-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/scale-stratified-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/scale-stratified-v1/evaluation.json`
- `data/research/ml_training_recovery_v1/scale-expansion-v1/protocol.json`
- `data/research/ml_training_recovery_v1/scale-expansion-v1/recheck.json`
- `scripts/vision/prepare_scale_stratified_coverage.py`
- `scripts/vision/run_recovery_scale_expansion.py`
- `scripts/vision/evaluate_scale_expansion_recheck.py`
