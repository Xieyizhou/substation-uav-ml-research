# 场景扩展复验结果

本轮先检查备用 clean 计划是否提供新的 pilot 视角。结果发现其 pilot_views 与 calibration_views 重复：complex 12 帧、simple 6 帧逐像素重复已有跨场景采集；medium 批次在中止时没有有效帧。重复回执保留，但没有计入独立证据，也没有继续采集同一批重复视角。

随后使用一个真正不同的 simple world 快照（原始 world hash `58ac5dfed31de91f1c839ccc6bd0b167207fec7cc968f4ad1e1317de01051d1a`）采集 12 帧 full_2d：3 个 cabinet、3 个 empty_ground、3 个 switchgear、3 个 transformer。两组诊断权重使用原定 640/0.37/IoU 0.7 设置。

| 权重 | 目标实例命中 | 目标实例数 | 预测框 | 预期类别命中 |
| --- | ---: | ---: | ---: | ---: |
| 仅正样本 | 2 | 11 | 26 | 1/3（可观测预期类别） |
| 加入 14 个难负样本 | 2 | 11 | 23 | 1/3（可观测预期类别） |

原始 `full_2d` truth 复核后发现，3 个预期 switchgear 视角均没有 switchgear 实例，只有 transformer 或空 truth；因此它们属于目标缺失/可见性或计划标注核验项，不能计入 switchgear 漏检率。进一步核对发现，这批 simple world 使用旧的 `label_mode=source`、未声明 `hierarchy_mode`，而已有可用的 canonical 诊断批次使用 `label_mode=visual-instance`、`hierarchy_mode=top-level-equipment`；这解释了为什么同一类目标在旧 world 的 truth 中缺失，应该把它判为标注模式 hold。3 个预期 transformer 视角均有 transformer truth，且两组均命中 1/3。按可观测预期类别统计，两组都是 1/3；cabinet 与 empty-ground 也没有对应四类目标实例，不能按正样本召回统计。不同目标出现在同一画面时，实例真值数可能大于预期类别对象数，因此总目标命中与预期类别命中分开报告。无目标帧的预测存在情况由 evaluation.json 逐帧记录，不能解释为正式柜体误报率。

结论是：难负样本在 simple 独立场景只减少 3 个预测框，没有提高总目标命中；在唯一可观测的 transformer 预期类别上也没有改变命中。switchgear 在这 3 帧中没有可核验正实例，不能据此判断模型类别失败。结合前一轮跨场景结果，收益集中在 medium，simple 存在回退或不变，当前仍不具备稳定泛化证据。下一步应先修正 source-plan 的目标存在性/可见性门控，再继续扩大类别与几何覆盖；不能把这些目标缺失帧直接作为负样本或漏检证据。

数据目录：`data/research/ml_training_recovery_v1/simple-independent-scene-v1/`，包含计划、12 帧回执、逐帧 prediction 和 identity。重复扩展尝试位于 `cross-scene-additional-v1/`，状态已记录，未作为独立评估。所有新数据仍为开发诊断，training_admitted=false、promotable=false。
