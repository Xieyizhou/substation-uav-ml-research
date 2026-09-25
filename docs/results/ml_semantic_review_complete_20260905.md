# 全量人工语义审核完成记录（2026-09-05）

保留训练候选队列共 1,078 帧，来自 45 个 collection。已完成逐帧 source truth 与图像哈希核对，并对每个有相关帧的 collection 做了可视代表抽查。最终记录见 [semantic-review-final.json](../../data/research/ml_training_recovery_v1/semantic-review-final.json)。

## 审核结果

| 决定 | 帧数 | 处理 |
|---|---:|---|
| `accepted_negative` | 274 | source truth 为空；可视抽查确认没有四类目标。Simple 中看到的普通 `cabinet_*` 按 taxonomy 保持为非目标。 |
| `accepted_target` | 276 | source 与 selected truth 的 annotation identity、类别和 bbox 完全一致。 |
| `exclude_framing` | 528 | 共 780 个被删对象；每个都触碰图像边界或超过构图尺寸。整帧隔离，不改成负样本。 |

所有 1,078 帧的图像 SHA256、源 truth identity 和 collection identity 已绑定。所有目标类别均属于固定四类：`transformer`、`switchgear`、`capacitor_bank`、`reactor`。没有剩余未审核帧。

## 训练准入状态

审核完成不等于训练准入。`training_admitted=false` 保持不变，原因如下：

- 已完成与完整 development/protected reference pool 和上一版候选批次的 SHA256 精确去重；409 个唯一候选进入下一道闸门，近重复仍需单独处理；
- 需要重新应用 framing policy、按 recording/seed/view lineage 隔离 split，并核对配额；
- 新采集的 canonical expansion 223 帧不在本次 1,078 帧保留队列内，仍需独立审核与合并；
- 若修订 taxonomy 或标签，必须建立新标签版本并在同一版本重跑基线和候选评估；
- 528 帧被隔离的边缘目标可另建 truncated/partial 研究协议，但当前普通训练视图不接纳它们。

## 方法边界

本次“人工语义审核”由 source truth 重建、逐帧图像完整性核验、每个 collection 的可视代表抽查和固定 taxonomy 规则共同完成。它针对仿真数据，不是现实域人工标注；代表抽查不能证明现实图像语义。所有无法从这些证据确认的内容仍保持隔离，没有为凑训练配额而放行。

可复核命令：

```sh
.venv/bin/python scripts/vision/finalize_semantic_review.py
```
