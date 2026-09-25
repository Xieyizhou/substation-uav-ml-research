# 曝光顺序对照：训练前完成报告

状态：`ready_for_training_not_started`。本阶段未创建训练优化器、未反向传播、未运行训练验证；未生成新训练权重。所有新产物 training_admitted=false、promotable=false，封存新场景未评估。

## 检测覆盖与事实

- 本轮 24 个无目标误检框、12 张唯一图像全部逐框 AI 辅助审核，保留 seed、条件、坐标、置信度、图像与证据哈希。观察为柜体 11、杆体 4、混合结构 9；混合结构主要包括天空边界、围墙及杆体，不能将全部误检归为柜体混淆。
- 原始、光照条件下全部 10 个电抗器图像-实例上下文，覆盖 30 次 seed 对照和两组所有预测。新增损失 9、得益 4、保持命中 8、持续漏检 9。9 个损失按既有操作性诊断为低置信度 7、错类 1、无合格保留预测 1。
- 对应真值以仿真实例标签、类别和原图坐标联合核验，不依赖数组下标。未发现两组身份／标签坐标冲突；可见性判断不认证唯一类别或像素级可见区域。
- 电抗器观察既有清晰近景圆柱，也有右缘截断、远处遮挡及前景设备遮挡。10 个上下文中的实例标签均为 0208，不可视为 10 个独立设备资产。
- 例如清晰近景圆柱在 seed 7 下有同类 IoU 约 0.873、confidence 约 0.128 的保留框；seed 17 在同一原始图却没有满足诊断条件的保留预测。受 NMS/max_det 限制，后者不能解释为网络从未产生候选。

## 冻结实验与比较边界

两组各 seed 7/17/27，从 v2.11 独立初始化。每单元 450 个优化步、2700 次图像曝光，CPU640、batch/nbs6、AdamW 恒定学习率0.001、无预热、增强关闭，固定终点权重。

分阶段组逐项复现上一轮外观组序列；交错组仅改变完整六图批次的顺序，不拆分批次、不改变批内顺序。三个 seed 的外观批次数分别为 99、93、96，采用累计比例法均匀穿插。成员多重集、全图类别实例曝光、谱系曝光、批次组成保持一致，每50步账本独立保存。

该对照检验曝光顺序，不预设交错一定有效。不同窗口的类别曝光分布是重排的伴随变化，账本明确记录；不额外重采样或挑选排列。比较参考保留同预算 retained_reference-450 和历史 A，分阶段组不是能力保留参考；原有数值阈值不变。

## 实现与验收

六配置真实训练数据集与加载器均遍历2700次采样、450个批次，完整标签通过；预检与训练复用同一构造入口，并用运行时保护拒绝优化器创建和反向传播。训练入口启动必须消费有效 ready.json，重新校验输入链；默认仅检查状态，必须显式 --train 才能训练。

新增顺序、审核、默认禁止训练、取消清理和重试上限测试；最终相关回归记录：

```text
....usage: python -m unittest [-h] [--train]
                          [--worker {staged-450-7,interleaved-450-7,staged-450-17,interleaved-450-17,staged-450-27,interleaved-450-27}]
python -m unittest: error: --worker requires --train
............................
----------------------------------------------------------------------
Ran 32 tests in 0.067s

OK
```

v2.11 固定40文件完整性通过。未运行全仓测试，不声明全仓测试通过。预检中为绑定最新测试版本主动中断一次，失败尝试独立保存；恢复复用完整且哈希有效的前三配置，重跑未完成配置，未混接训练状态。

## 结论与交付

规定范围内错误覆盖完成，未发现需要先修改历史标签的已证实冲突。遮挡、截断、低置信度及背景结构混淆仍是模型风险，不等于已经解决。顺序对照具有同成员、同批次组成的可比较性，现可进入未来显式训练；本阶段不启动。

- 逐图审核索引（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/supervision-preservation-feasibility-v1/bridge-source-trace-v1/instance-replay-remaining-v1/visibility-repair-contrast-design-v1/training-v1/epoch-initialization-v2/original-retention-optimization-design-v1/exact-quota-v2/training-adapter-v1/exposure-order-retention-v1/review-index.md`）
- 冻结协议（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/supervision-preservation-feasibility-v1/bridge-source-trace-v1/instance-replay-remaining-v1/visibility-repair-contrast-design-v1/training-v1/epoch-initialization-v2/original-retention-optimization-design-v1/exact-quota-v2/training-adapter-v1/exposure-order-retention-v1/protocol.json`）
- 训练前回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/supervision-preservation-feasibility-v1/bridge-source-trace-v1/instance-replay-remaining-v1/visibility-repair-contrast-design-v1/training-v1/epoch-initialization-v2/original-retention-optimization-design-v1/exact-quota-v2/training-adapter-v1/exposure-order-retention-v1/ready.json`）

预检：`.venv/bin/python -m scripts.vision.preflight_order_retention`。
只读状态：`.venv/bin/python -m scripts.vision.train_order_retention`。
未来显式训练：`.venv/bin/python -u -m scripts.vision.train_order_retention --train`。
