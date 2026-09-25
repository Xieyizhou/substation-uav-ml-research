# 遮挡分布变化回归结果

本轮针对前几轮“同一几何体的相邻视角”问题，冻结了一批真正带场景遮挡物的开发集。候选位姿来自 canonical 场景的原始连续采样，保留目标中心射线被 1–2 个其他设备、杆塔或建筑物穿过的视角；场景几何、类别契约和全 2D 真值生成方式保持不变。

采集阶段中等场景的第一个候选被遮挡物完全挡住，按“目标类别必须出现在真值中”的采集规则拒绝；该失败记录保留在 collection receipt 中，没有混入评估。复杂场景 24 张全部采集成功，人工查看 contact sheet 后 24/24 接受，目标类别均存在。历史非保护采集范围内 exact duplicate 为 0，最近历史 dHash 距离为 12，说明这批图像不是原有训练/回归帧的像素复制。

| 开发权重 | 实例命中 | 预期类别命中 |
| --- | ---: | ---: |
| 66 张 expanded uniform，三 seed 均值 | 77.7/120 | 14.0/24 |
| 102 张，100 steps，三 seed 均值 | 66.7/120 | 13.0/24 |
| 102 张，50 steps，三 seed 均值 | 71.3/120 | 13.3/24 |

与 66 张 expanded uniform 参考相比，100-step 权重在实例命中下降 11.0、类别命中下降 1.0；50-step 权重分别下降 6.3 和 0.7。单纯增加训练池或调整预算没有解决遮挡条件下的泛化问题。按遮挡数量拆分时，1 个遮挡物的 14 张帧为 68 个真值对象，2 个遮挡物的 10 张帧为 52 个真值对象；两组都没有出现某个权重稳定领先的证据。

当前结论是：模型在遮挡分布变化下明显退化，问题仍然集中在小目标可分性、遮挡后的可见外观和训练池与回归条件的耦合。下一步应补充同类别不同外观以及可控部分遮挡的训练样本，并继续使用独立遮挡 holdout 作为门槛；不能把这批 holdout 回灌训练，也不能据此晋级任何现有权重。

本轮所有产物均为开发用途，`training_admitted=false`、`promotable=false`，正式 v2.11 未修改。

证据：

- `data/research/ml_training_recovery_v1/occlusion-shift-v1/manifest.json`
- `data/research/ml_training_recovery_v1/occlusion-shift-v1/capture-progress.json`
- `data/research/ml_training_recovery_v1/occlusion-shift-v1/contact-sheet.png`
- `data/research/ml_training_recovery_v1/occlusion-shift-v1/semantic-review.json`
- `data/research/ml_training_recovery_v1/occlusion-shift-v1/dedup-audit.json`
- `data/research/ml_training_recovery_v1/occlusion-shift-v1/evaluation.json`
- `scripts/vision/prepare_occlusion_shifted_regression.py`
- `scripts/vision/capture_occlusion_shifted_regression.py`
- `scripts/vision/finalize_occlusion_shifted_review.py`
- `scripts/vision/audit_occlusion_shifted_dedup.py`
- `scripts/vision/evaluate_occlusion_shifted_regression.py`
