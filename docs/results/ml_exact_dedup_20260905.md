# 精确去重完成记录（2026-09-05）

已对全量人工语义审核队列的 1,078 帧执行 SHA256 精确去重。结果见 [report.json](../../data/research/ml_training_recovery_v1/exact-dedup-v1/report.json)，逐帧决定见 [decisions.jsonl](../../data/research/ml_training_recovery_v1/exact-dedup-v1/decisions.jsonl)，可进入下一道数据闸门的候选见 [selected.jsonl](../../data/research/ml_training_recovery_v1/exact-dedup-v1/selected.jsonl)。本步骤不改写既有数据集，也不自动准入训练。

## 结果

| 项目 | 帧数 |
|---|---:|
| 语义审核队列 | 1,078 |
| 语义审核已接受 | 550 |
| 语义审核排除（构图边界/超大框） | 528 |
| 与上一版 run18 hard-view 完全重复 | 141 |
| 当前候选内部完全重复 | 0 |
| 精确去重后唯一候选 | 409 |
| 其中完整目标帧 | 220 |
| 其中确认无目标帧 | 189 |

141 个重复帧来自上一版已物化的 `run18-balanced-small-dense-hard-view-v1`，已保存每一帧对应的来源路径和源帧 ID；它们没有再次进入候选。当前候选与官方 v2 development replay、v2 validation、blind、qualification 以及 42 帧历史 canonical 保留池没有 SHA256 完全重复。近重复没有混入本步骤。

精确去重后按目标帧统计为：`transformer=192`、`switchgear=181`、`capacitor_bank=169`、`reactor=50`。这是去重后的帧覆盖，不等于四类配额已经满足；多目标帧会分别计入多个类别。

## 发现的既有分区问题

参考清单本身存在一个跨角色精确重复哈希：

- `b5fbb8096bac3891e8caaebc25f26bbf32527203e93bbdbeaaeede813103ad4e`
- 出现在 development replay 10 帧，也出现在 protected blind 293 帧。

这批图像字节完全相同，不能把它解释为近似相似。完整的 303 条跨角色记录保存在 [reference-cross-role-overlaps.jsonl](../../data/research/ml_training_recovery_v1/exact-dedup-v1/reference-cross-role-overlaps.jsonl)；development 池内部的 10 条重复记录保存在 [development-internal-duplicates.jsonl](../../data/research/ml_training_recovery_v1/exact-dedup-v1/development-internal-duplicates.jsonl)。v2 protected validation 清单内部的 6,941 个重复哈希符合“quick validation 是 full validation 子集”的既有关系；development/blind 的交叉重复则应在正式训练前修复或形成有证据的例外记录。

## 训练准入

`training_admitted=false` 继续保持。409 个唯一候选还需要：

1. 按 recording/seed/view lineage 做 split 隔离并重新核对 run19 配额；
2. 处理现有 development/protected 的 exact split contamination；
3. 将近重复分析作为单独步骤执行；
4. 对 528 个构图排除帧保持隔离，不把它们改成负样本。

可复核命令：

```sh
.venv/bin/python scripts/vision/exact_dedup_semantic_review.py
```
