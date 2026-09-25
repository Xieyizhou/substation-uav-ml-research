# ML 训练停滞诊断与解决方案（2026-09-05）

## 结论与范围

当前主要瓶颈是视觉候选模型达不到运行时验收要求，以及新训练数据迟迟不能完成准入；不是训练程序一直跑不起来。已有 LiDAR 验证回放通过，视觉 run18 也完成了 9 个 epoch 并保存了权重。近期 run19 尚未开始训练。

本报告检查本地代码、配置、训练 CSV、模型 provenance、数据 identity 和截至 9 月 5 日 08:24 的采集审计。没有启动训练、飞行或更换官方模型，也没有重新进行全量推理和人工标注审核。审计文件中的结论与计数属于已有证据；因果解释和实验参数建议另行标明。

近期工作明确以 v2.11 为官方四类模型；旧 PROJECT_STATUS.md 中的冻结 v2 基线是另一阶段证据。run15–18 的单类 equipment proposal 模型不能被当成四类识别模型；两者 mAP 不可直接排名。

## 已确认的问题

### 1. 学得出检测框，但达不到 proposal 验收要求

run18 的 results.csv 有 9 个 epoch。单类 equipment 验证 mAP50-95 从首轮 79.322% 到最佳第 6 轮 79.622%；训练 box loss 从 0.58234 降到最后一轮 0.53315。没有表现为 NaN 或完全不收敛。训练 provenance 记录 actual_device=mps，不能把这一轮失败归因于未使用加速设备。

相同运行时约束下，run18 proposal 审计结果如下。这里 recall 是不要求类别一致的候选框匹配召回，不是四类识别召回。

| 验证集 | 置信阈值 | 候选框总召回 | 无目标帧误检率 | 关键问题 |
|---|---:|---:|---:|---|
| v2，6,941 帧 | 0.01 | 97.070% | 8.100% | 总召回低于 98% |
| v2，6,941 帧 | 0.10 | 94.615% | 4.550% | 提阈值后召回继续下降 |
| v2，6,941 帧 | 0.25 | 93.302% | 3.400% | 阈值调节未消除取舍 |
| 多目标，60 帧 | 0.01 | 91.854% | 不适用 | transformer 83.333%，switchgear 93.500% |

两个审计各扫描 12 个阈值，均无通过项。当前代码门槛为总召回 ≥98%、每个有样本的类别召回 ≥95%、候选框数 P95 ≤16。无目标帧误检率虽然被记录，却没有直接纳入该函数的 passed 条件。因此，**这次 proposal gate 失败的直接原因是召回不足；背景误检是额外的质量问题，不能误称为现有 gate 的直接拒绝条件**。

已有失败归因记录：v2 验证的 259 个失败目标中，140 个定位 IoU 不足、66 个 detector miss、52 个被 proposal 数量上限截断、1 个匹配次序/共用框冲突；多目标的 29 个失败中，16 个定位不足、13 个截断。这些是特定审计配置下的诊断，不是跨所有阈值的统一统计。

推断：只调 confidence 或再跑一次相同微调，难以同时修复定位、重复框和候选框排序。

证据：
- `models/equipment/visual-yolo11n-proposal-v2.12-candidate-run18/results.csv`
- `models/equipment/visual-yolo11n-proposal-v2.12-candidate-run18/training_provenance.json`
- `data/research/visual_hard_examples_v2_12/proposal-audit-run18-v2-validation-v1.json`
- `data/research/visual_hard_examples_v2_12/proposal-audit-run18-multitarget-v1.json`
- `data/research/run19-product-scope-v1/existing-diagnosis-summary.json`
- `src/vision/evaluation/proposal_audit.py:36`

### 2. 历史标签存在完整性风险，自动真值不等于可直接训练的正确标签

历史标签保留审计检查 1,078 帧，其中 **528 帧**相对最近的有效源 truth 少了对象，550 帧对象数一致。528 是发生删框的帧数，不是删除框的总数；对象数一致也不证明语义正确。

当前 `apply_bbox_policy()` 在 `drop_invalid_objects` 模式下，只要还有合格框，就保留原图并删除不合格框。边界目标、过大目标或面积不符合要求的框可能被删掉，但目标像素仍在原图中。这种机制有给可见目标施加负向训练信号的风险；是否构成真实漏标，还需按统一的可见性定义复核，不能仅凭数量差认定所有 528 帧都错。

已有代码会拒绝“源图有目标但全部被过滤”的帧，不能声称当前实现会一律把这类帧变成空标签背景。更隐蔽的问题是多目标图片中只保留部分标签。

另一方面，普通 cabinet 在当前产品 taxonomy 中不是 switchgear。历史报告已纠正“看到柜体就说明漏标”的判断。无目标帧指没有四类目标，而非没有任何设备形状的物体。历史原始世界快照、原始 truth payload 绑定不足，也限制了事后追溯。

100 帧开发诊断中记录了 84 个重复目标框、77 个可能的设备部件框、91 个定位/部分重叠框、60 个待复核的背景候选框。它们是几何归类，不能全部当成已确认背景误检。

证据：
- `data/research/canonical_views_v1/historical-label-retention-audit-v1.json`
- `src/vision/training/hard_example_curator.py:114`
- `config/perception/visual_hard_examples_v2_12_run18_run19_full_strict_v1.json`
- `data/research/run19-product-scope-v1/truth-semantics-audit-v1.json`
- `data/research/run19-product-scope-v1/development-proposal-audit-v1/report.json`

### 3. 新数据有产出，但采集完整性、类别覆盖和人工审核尚未闭环

9 月 5 日最新一轮实际采到 223 帧：Simple 111、Medium 83、Complex 29。三份 collection receipt 的状态都为 blocked：

| 场景 | 成功采集帧 | 本轮停止原因 |
|---|---:|---|
| Simple | 111 | Pose service did not acknowledge |
| Medium | 83 | pose_or_pairing_timeout |
| Complex | 29 | pose_or_pairing_timeout |

这是不启动 PX4 的离线视角采集，不能把本轮问题描述成降落失败。旧飞行采集曾有落地确认与生命周期问题，需与当前 pose/配对错误分开处理。

本轮联合审计候选覆盖：transformer 174 帧、switchgear 134 帧、capacitor_bank 59 帧、reactor 16 帧，另有 34 帧空 truth 候选。多标签帧会重复计入多个类别。已计算与所提供历史/保护参考池的去重，duplicates 为空，但状态仍为 `pending_manual_object_review`、`training_admitted=false`。空 truth 不等于已经人工确认的负样本，提供了参考索引也不单独证明保护集清单完整。

run19 待定 recipe 要求四类各 600 帧、无目标 600 帧，并保留 development replay。旧覆盖账本有 reactor 缺 73、背景缺 288 的结论，但它引用的旧池仍有标签/语义待审项，不能直接与本轮候选相加后宣布达标。

推断：现在的实际耗时集中于“可用新样本的产出”，而非优化器运行时间。继续广泛采集会优先增加 transformer/switchgear，未必有效补到 reactor 和可信难负样本。

证据：
- `data/research/canonical_views_v1/expansion-round1-joint-audit-v1.json`
- `data/research/canonical_views_v1/{simple,medium,complex}-expansion-round1-pilot-v1/collection-receipt.json`
- `data/research/canonical_views_v1/run19-training-recipe-pending-v1.json`
- `data/research/run19-product-scope-v1/scope-decision.json`

### 4. 微调学习率配置存在已确认的不一致，性能影响需要对照验证

run18 显式采用 AdamW、lr0=3e-6、warmup_epochs=0.5，但实际 args.yaml 中 `warmup_bias_lr=0.1`，是主学习率的约 33,333 倍。首个 epoch CSV 的 `lr/pg2=0.000829421`，仍约为 lr0 的 276 倍。

本地 Ultralytics trainer 对 bias 参数组从 warmup_bias_lr 插值；只有 optimizer=auto 分支会把该参数自动设为 0。本项目显式使用 AdamW，且训练包装器允许字段未包含 warmup_bias_lr，因此仅在 recipe JSON 中添加该字段也不会传入 model.train。

这是可以直接定位的配置风险。它可能使原本意图为“小幅微调”的首轮更新过强，但现有证据不能证明它是精度退化的主要原因。应先固定数据、初始权重与随机种子，再比较默认 0.1 与显式 0 两组；记录逐 step 的学习率、更新幅度及验证指标。

另一个待验证因素是 lr0 很小、最多 10 epoch、patience=3、关闭 mosaic/mixup。这适合保守微调，却不一定足够学习新的多目标分布；不应在未排除标签问题前直接扩大训练时长或增强幅度。

证据：
- `models/equipment/visual-yolo11n-proposal-v2.12-candidate-run18/args.yaml`
- `models/equipment/visual-yolo11n-proposal-v2.12-candidate-run18/results.csv`
- `src/vision/training/yolo_training.py:17`
- `.venv/lib/python3.14/site-packages/ultralytics/engine/trainer.py:450`
- `.venv/lib/python3.14/site-packages/ultralytics/engine/trainer.py:1091`

### 5. 实验目标与评估契约没有充分对齐

run12–14 是四类训练，run15–18 改为单类 equipment proposal。run18 单类 mAP 高，并不能证明四类识别可用，也不能替代多目标 proposal 召回。run19 已计划从官方 v2.11 四类权重重新开始，这是合理的主线。

训练负样本也发生过变化：run12/14/15/16 的 training identity 中 train_no_target_count=0，run18 为 600/5,800；固定验证集有 4,000/6,941 个无目标帧。负样本并非始终缺失，但前几轮没有继续训练背景抑制，run18 加入的负样本是否覆盖 cabinet、部件和遮挡混淆仍需验证。比例不同本身不证明错误，关键是质量和条件覆盖。

proposal 审计的 P95≤16 在先截断到 16 后计算，无法衡量候选框饱和压力；它是报告盲点，不是这次召回不合格的解释。应另报截断前候选数、饱和帧比例和截断损失。匹配是顺序贪心匹配，已有一个顺序冲突记录，可用最大匹配做诊断上界，但不能静默替换原门槛或历史结果。

旧文档仍称 canonical 尚未 live capture，而最新 receipt 已有真实采集结果。当前状态应由带身份的最新完成/阻塞 receipt 汇总，避免围绕过期原因重复开发。

## 执行方案

以下为建议排期，按通过条件推进；不是对完成日期或精度的承诺。保持官方模型与原验收协议，诊断实验不具有部署准入资格。

| 阶段 | 工作 | 交付物与通过条件 |
|---|---|---|
| P0，约半天 | 冻结四类任务、v2.11 初始权重、开发/验证/保护分组和验收定义；整理最新状态 | 一份状态清单记录模型 hash、标签版本、采集状态、已审核帧数、下一阻塞项。单类 proposal 单独登记。 |
| P1，约 1–2 天 | 修 pose 服务确认及 RGB/truth/depth 配对超时；支持有界重试和幂等续采；完成取消/异常资源清理 | 三地图各一小批完整完成；针对 pose 不确认、配对超时、取消、退出增加必要回归；失败帧不得静默成为合格帧。已有部分采集是否可复用按协议重新审计。 |
| P1，约 1–2 天，可穿插 | 确定 visible/occluded/truncated 的标签规则；优先复核历史 528 帧删框情况、600 负样本的代表性及新 223 帧 | 对获准训练的每帧保留所有应标目标；未解决的样本隔离。当前 YOLO 管道未证明支持 ignore 区域时，采用整帧剔除或合规重标，禁止只删可见目标框。校准审核与训练标签审核分别留凭据。 |
| P2，约 1–3 天 | 从可信历史池恢复可用样本，然后只补缺失条件 | SHA256 精确去重已完成；下一步要修复既有 development/blind 交叉重复，再做近重复、按记录/视角组隔离并输出“原始→去重→审核→准入”计数；正式 run19 达到既定各类/负样本配额，且检查地图、距离、尺度、遮挡、视角覆盖。 |
| P3，约 1–2 天 | 使用下述小实验定位训练配置与数据增量的作用 | 生成同初始权重、同验证成员的实验表；首轮没有异常更新，改善来自可解释变量；不使用盲测挑选参数。 |
| P4，约 1–2 天 | 优胜者进入完整验证、导出等价、时序回放与原 qualification | 必须满足既有全部 gate；proposal 与四类分类分别报告；官方包仅在原流程通过后更新。 |

数据补采优先级：reactor 的新有效视角 → 经语义确认的 cabinet/部件等难负样本 → capacitor_bank 薄弱尺度 → transformer 定位差的遮挡/密集视角。每次先采 20–40 个计划视角，按实际“新增准入帧/采集时间”决定是否扩张。不同 seed 不自动意味着新的独立场景。

600 是当前实验配额，不是通用的学习充分性定理。正式 run19 不静默放宽它；若大规模补齐前需要验证学习率或数据方向，建立独立、仅供开发的诊断协议，使用已审核小集，明确不可晋升。

截至本报告后续的精确去重步骤，1,078 帧已完成 SHA256 精确匹配：语义接受的 550 帧中，141 帧与上一版 run18 hard-view 重复，409 帧保持唯一；官方 development replay、v2 validation、blind、qualification 和历史 canonical 保留池未发现候选交叉重复。去重同时发现既有参考清单中有 303 条 development/blind 完全相同的图像记录，必须在正式训练前处理，详见 [精确去重记录](ml_exact_dedup_20260905.md)。

### 最小对照实验

1. **零训练对照 A0**：官方 v2.11 在同一开发诊断集和固定验证集上测量四类指标、proposal 召回、误检、截断损失及延迟。先补上当前开发诊断中缺少的官方模型对照。
2. **学习率对照 A1/A2**：同一已审核数据子集、同一 v2.11、seed=7、batch 与步数相同，仅改变 warmup_bias_lr（历史 0.1 / 显式 0.0）。包装器必须显式传递并绑定该参数，记录逐 step LR；以短程诊断为目的，避免先重复长训练。
3. **数据对照 B1/B2**：采用上一阶段较稳定的配置，比较可信 development replay 与 replay+纠正标签的困难样本；再单独比较是否加入审核通过的难负样本。图像成员变化和标签变化分别记账，不能与换模型同时进行。
4. **必要时才扩展训练预算**：若干净小集仍学不动，先用 32–64 张全部审核的开发图做可记忆性检查；随后在开发协议内比较 lr0=3e-6 与 1e-5，或 10 与 30 epoch、patience=3 与 8，每次只改一个因素。这些是待试起点，不是保证最优的参数。
5. **选择与复验**：使用验证集选择阈值与候选；对优胜者换至少一个 seed 复验。按 recording/视角组计算不确定性，不能把相邻帧当作独立试验。已反复查看的 qualification 结果应视为诊断信息，不再声称是全新独立检验。

对每次实验记录：四类 mAP50-95、各类召回、背景帧误检率、IoU 分布、proposal recall@16、小目标召回、截断前 P95、16 框饱和率、重复框率、实际学习率、模型/数据身份。明确新增误检容忍值属于待制定的实验标准；现有门槛继续原样执行。

止损规则：两次只改变单一因素的实验均未改善目标错误类别，就返回数据/标注/后处理归因；不连续堆 run 编号。提高 max_proposals 到 32/64 只作诊断上界，不改变 16 框运行预算来宣布成功。只有干净数据、稳定优化和后处理诊断仍显示明显容量上限时，再评估更大模型或更高输入分辨率。

## LiDAR 与真实域补充

LiDAR 的 `outputs/research/lidar_replay_gate_v2/replay_gate.json` 已通过：validation 3,600 样本，danger recall=1.0，risk macro-F1≈0.99972，traversability IoU≈0.71954。它不是目前“没有训练出来”的主因，但验证回放不能代替正式动态闭环证据。

PROJECT_STATUS.md 记录的旧 120-run LiDAR 结果含多份研究身份且缺少 truth-danger，不能作为正式动态 ML 优势证据；本次未重新审计全树。后续应另立包含危险场景的隔离动态测试，保留几何安全基线。

真实图像迁移同样是独立任务。既有项目记录显示真实域压力测试效果很差且四类/独立分区覆盖不齐；本次没有重新推理。仿真修复完成后，仍需单独完成真实域数据、标签和跨站点评估，不能用仿真指标推断实景可用。

## 外部方法核对

Ultralytics 官方训练文档分别定义了 lr0、warmup_bias_lr、patience 等参数；这里的具体风险判断以本机实际 args、trainer 实现和 CSV 为准：[训练参数](https://docs.ultralytics.com/modes/train/)。

采集与标注应围绕任务类别和代表性条件设计；本文据此优先修正标签契约并按缺口补采，而不承诺通用样本数量：[数据采集与标注](https://docs.ultralytics.com/guides/data-collection-and-annotation/)。
