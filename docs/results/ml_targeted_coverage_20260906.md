# 定向尺度与地图覆盖结果

为补齐上一轮发现的缺口，本轮冻结并采集 24 张 full_2d 开发帧：medium capacitor_bank 6 张，complex switchgear 6 张、capacitor_bank 6 张、reactor 6 张。视角优先选择距离 12 米以上的未使用位姿，以增加小/中尺度覆盖。

24/24 帧完成采集，24/24 通过人工语义审核，目标类别全部存在。与历史非保护采集的 exact RGB hash 重合为 0，最近 dHash 距离为 7，没有触发 dHash≤2 的隔离规则。全部保持 `training_admitted=false`、`promotable=false`。

现有诊断权重在该定向集上的结果：

| 权重 | 实例命中 | 预期类别命中 |
| --- | ---: | ---: |
| 仅正样本 | 75/127 | 19/24 |
| 难负样本增强 | 84/127 | 21/24 |
| 66 张 uniform，三种子均值 | 98.3/127 | 19.7/24 |

扩展 uniform 权重明显提升了远距离目标的定位命中，但预期类别命中没有超过难负样本基线，说明当前问题已经从“没有足够场景覆盖”转为“远距离小目标的分类置信度与定位稳定性不足”。因此这 24 张帧先作为下一轮训练候选和回归集，不直接并入正式训练包。

下一步应围绕远距离小目标做小规模 A/B：保留完整正样本框，比较轻量尺度增强、输入分辨率和分类损失权重；每个变量都要同时通过定向集、80 帧控制集和 25 帧跨场景集，避免只优化这 24 张新帧。

证据：

- `data/research/ml_training_recovery_v1/targeted-coverage-v1/manifest.json`
- `data/research/ml_training_recovery_v1/targeted-coverage-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/targeted-coverage-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/targeted-coverage-v1/evaluation.json`
- `scripts/vision/prepare_targeted_coverage_expansion.py`
- `scripts/vision/capture_targeted_coverage_expansion.py`
