# 已审核样本近重复筛查

对全部 550 帧语义接受样本重新计算图像 SHA256 和 dHash64，与修正后的 15,575 帧 development replay、既有 validation/blind/qualification 指纹索引及本批已保留代表进行比较。采用现有 dHash64-v1 与 Hamming ≤2，直接与保留代表比较，不做传递聚类。

431 帧未命中相似项，119 帧进入相似性复核。431 是相对本次参考范围的保留量；上一轮 409 是相对旧 run18 批次的新增量，二者口径不同，不能直接相减评价去重效果。

| 类别 | 当前保留帧覆盖 | 距 600 帧缺口 |
|---|---:|---:|
| transformer | 214 | 386 |
| switchgear | 194 | 406 |
| capacitor_bank | 209 | 391 |
| reactor | 60 | 540 |
| no_target | 161 | 439 |

多目标帧会计入多个类别。保留样本地图分布为 Simple 19、Medium 296、Complex 116，来自 38 个 collection/seed 组。全部接受样本的 38 个 seed 与从保护索引显式 seed 或标准录制 ID 提取的 22 个已知 seed 无交集。这不是完整的视角谱系隔离证明。

开发回放池有 7,258 帧命中保护指纹近邻。dHash 在低纹理背景上可能发生大量相似命中，不能将这些命中等同于确认重复或数据泄漏；本次输出待复核清单，没有据此删除原数据或改写训练池。应优先做跨编码像素一致性检查，再对相似组判别。

输出位于 `data/research/ml_training_recovery_v1/near-dedup-v1`：`report.json` 记录输入输出哈希和覆盖；`decisions.jsonl` 保留全部 550 帧的匹配数量、最小距离与来源示例；`retained.jsonl` 为 431 帧；`replay-protected-near-hits.jsonl` 为回放侧待复核项。

后续状态：已完成本次提供参考范围内的[跨编码像素核验](ml_pixel_duplicate_verification_20260905.md)，未发现新增像素完全重复。119 帧候选和 7,258 帧回放的近似关系仍未解决，像素不同不等于无近重复。

范围限制：未穷尽其余历史 hard pool；qualification 清单全仓库完整性未证实；完整 recording/seed/view lineage 仍待核对。训练准入为 false。下一步优先核实相似命中，而不是按上述暂定缺口立即大规模补采。

已验证半径 0/1/2 命中、半径 3 排除、2,081 个唯一搜索掩码，以及报告身份和全部决定文件哈希。15,575 帧回放图像和 550 帧候选均重新核对文件哈希。

复核命令：`.venv/bin/python scripts/vision/audit_reviewed_near_duplicates.py`。
