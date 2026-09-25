# 新定向回归 holdout 结果

本轮新增 24 张完全独立的定向回归帧：medium capacitor_bank 6 张，complex switchgear、capacitor_bank、reactor 各 6 张。采集使用新的 canonical round 121，24/24 完成，24/24 通过人工语义审核，exact duplicate 为 0，最近历史 dHash 距离为 9。该批数据没有进入任何训练池，只作为新 holdout。

| 权重 | 实例命中 | 预期类别命中 |
| --- | ---: | ---: |
| 66 张 expanded uniform，三 seed 均值 | 93.3/127 | 18.3/24 |
| 102 张，100 steps，三 seed 均值 | 88.0/127 | 17.3/24 |
| 102 张，50 steps，三 seed 均值 | 92.7/127 | 19.3/24 |

100-step 权重在新 holdout 上同时下降实例命中 5.3 个、类别命中 1 个。50-step 权重类别命中增加 1 个，但实例命中仍低约 0.7 个，因此也没有通过严格门槛。50-step 的表现比 100-step 稳定，但不能证明已经改善泛化。

这批新 holdout 与上一轮控制集的方向一致：更大的训练池和更长的预算会提高部分训练分布上的拟合，却会让远距离目标的类别和定位结果随 seed 波动。当前最可靠的结论是，根因集中在模拟场景中小目标的类别可分性、遮挡/背景变化和训练池分布耦合，而不是单一的 epoch 或输入尺寸参数。

因此暂不晋级 50-step 或 100-step 权重，继续保留正式 v2.11 和 66 张 uniform 开发参考。下一步应针对 reactor 与 capacitor_bank 建立“同类别不同外观/遮挡”的新场景，而不是继续从同一 canonical 几何体生成更多相邻视角；每批新数据仍需独立 holdout、人工审核和去重后才可使用。

证据：

- `data/research/ml_training_recovery_v1/targeted-regression-v1/manifest.json`
- `data/research/ml_training_recovery_v1/targeted-regression-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/targeted-regression-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/targeted-regression-v1/evaluation.json`
- `scripts/vision/prepare_targeted_regression.py`
- `scripts/vision/evaluate_targeted_regression.py`
