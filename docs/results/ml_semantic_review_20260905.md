# 首批人工语义审核记录（2026-09-05）

首批审核针对结构审计队列中的 20 帧：12 帧空 truth 负样本和 8 帧发生删框的多目标帧。审核依据是原图、源/选中 truth 的 bbox overlay、对应场景的 SDF 模型语义，以及四类产品 taxonomy。机器记录见 [semantic-review-batch1.json](../../data/research/ml_training_recovery_v1/semantic-review-batch1.json)。

## 决定

12 帧被人工确认在当前 taxonomy 下没有 transformer、switchgear、capacitor_bank 或 reactor。Simple 的两帧包含蓝色 `cabinet_1/2/3` 外观；SDF 将它们定义为普通 cabinet，产品契约明确不把普通 cabinet 当作目标，因此可以作为“无四类目标”的候选负样本。其余抽查帧只包含地面、围栏、墙面或空场景。

8 帧的源 truth 包含位于图像边界的目标，图像中仍可看到对应目标像素。它们的删框属于 framing/truncation 筛选结果，而不是已证实的类别错误。决定为 `exclude_framing`：保留原始 evidence，整帧隔离，不只删除这些框，也不把帧变成负样本。

## 约束

- 本批 20 帧仍然全部是 `training_admitted=false`。
- “accepted” 只表示完成了这一批的人工语义决定；它不代表完成 split isolation、exact/near dedup、配额或完整 reference-pool gate。
- 其余 1,058 帧未审核，仍为 `quarantined_pending_review`。
- 本批的负样本结论适用于现有四类 taxonomy。若产品决定把普通 cabinet 纳入 switchgear，必须新建 taxonomy 和标签版本，不能修改这份审核结论。
- 边界目标是否适合另建 truncated/partial 训练协议尚未决定；在当前 YOLO 训练契约下不进入普通训练视图。

## 可复核性

执行以下命令可重建并校验本批审核记录：

```sh
.venv/bin/python scripts/vision/create_semantic_review_batch.py
```

脚本检查所有 frame_id 存在于结构审计队列，重新绑定 image SHA256、source truth identity 和删框对象。它不会自动读取像素做语义判定，也不会改变训练 manifest。可视化复核图保存在 `outputs/research/ml_training_recovery_v1/semantic_review_batch1/`，其中 `removed_overlay.png` 使用红色 source bbox、绿色 selected bbox；`empty.png` 显示抽查的空 truth 原图。
