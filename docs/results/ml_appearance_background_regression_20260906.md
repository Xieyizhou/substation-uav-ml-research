# 外观与背景分布变化回归结果

本轮在复杂场景中保持设备几何、目标名称、层级标签和相机采样逻辑不变，只建立了两类开发变体：

- `appearance`：改变 transformer、switchgear、capacitor_bank、reactor 主体材质颜色。
- `background`：改变天空背景、环境光、太阳光以及地面材质。

两组各 12 张，每个目标类别 3 张；24/24 采集成功，24/24 通过 contact sheet 人工语义审核，目标类别全部存在。历史非保护采集范围内 exact duplicate 为 0，最近历史 dHash 距离为 9。

| 开发权重 | 实例命中 | 预期类别命中 |
| --- | ---: | ---: |
| 66 张 expanded uniform，三 seed 均值 | 32.3/149 | 7.7/24 |
| 102 张，100 steps，三 seed 均值 | 27.7/149 | 6.7/24 |
| 102 张，50 steps，三 seed 均值 | 37.0/149 | 7.3/24 |

外观变体是主要失败来源：66 张参考权重在外观 12 张上的类别命中只有 3、1、2（不同 seed），而背景 12 张为 6、5、6。100-step 权重在外观变体上为 1、2、0，说明更长训练预算没有修复外观泛化；50-step 权重实例命中略高，但类别命中仍没有超过参考均值。

这次结果把问题进一步收窄到“类别语义依赖固定颜色/材质外观”。此前遮挡回归已经显示遮挡变化会导致明显下降，本轮说明即使目标无遮挡，只改变设备外观也会使模型大幅退化。因此下一步应把同类别多材质、多光照样本纳入训练候选，并用外观变体作为独立门槛；不能继续只从同一颜色和同一几何体生成相邻视角。

本轮所有产物均为开发用途，`training_admitted=false`、`promotable=false`，正式 v2.11 未修改。

证据：

- `data/research/ml_training_recovery_v1/appearance-background-v1/manifest.json`
- `data/research/ml_training_recovery_v1/appearance-background-v1/contact-sheet.png`
- `data/research/ml_training_recovery_v1/appearance-background-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/appearance-background-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/appearance-background-v1/evaluation.json`
- `scripts/vision/prepare_appearance_background_regression.py`
- `scripts/vision/capture_appearance_background_regression.py`
- `scripts/vision/finalize_appearance_background_review.py`
- `scripts/vision/audit_appearance_background_dedup.py`
- `scripts/vision/evaluate_appearance_background_regression.py`
