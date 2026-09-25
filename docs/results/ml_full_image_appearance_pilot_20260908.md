# 主体可见性预筛后的外观补采先导

本轮补齐原始 8 张的逐图审核与指纹检查，新增钢灰材质常规光 8 张、钢灰材质暖暗光 8 张。新增 16 张均首次采集成功；连同原始图共 24 张、33 次全图实例框观察。所有图均逐张查看，审核记录明确为 AI辅助审核，不是实例掩码像素认证，也不自动形成训练准入。

## 检查结果

- 实际模式、相机合法性、实例映射和恢复检查复用 canonical 门禁；两个世界经过允许字段结构比较。
- 新增图与对应原始图实例集合一致，框坐标最大差异为 0 像素。
- 文件、原尺寸 RGB 像素及近相似指纹排除未发现需暂缓命中。保护参考只读取既有成员／指纹缓存，没有打开保护图像或标签。设计内变体相似性保留为共同谱系，不删除或计作独立场景。
- 最初去重入口未接入既有像素缓存，留下明确缺口；随后独立 v2 入口复用 rgb8-row-major-v1 缓存口径完成检查。原入口、回执与历史实验保持不变。
- 25 项相关 unittest 通过；v2.11 固定 40 文件完整性通过。未声明全仓测试通过。

## 仍存在的边界

8 个位姿只对应 7 个登记来源组，全部来自同一个 complex 地图及原设备资产。F03 与 F08 是同源近邻开关柜视角；二者前面板均不可见，不能声称已经补齐面板或设备几何多样性。

钢灰暖暗图主体明显变暗，但本次逐图审核仍能观察连续主体、顶面／圆柱轮廓及基座。部分电容器和非计划变压器仍有前景遮挡，理由逐框保留，不将可见性简化为面积阈值。该结果证明这批图可继续走开发数据流程，不证明模型性能已经改善。

## 下一步与未完成项

这是一批 24 张的先导闭环，不是全部补采矩阵完成。还需冻结赭色、灰绿两种材质与常规／暖暗光组合的 32 张，采集后逐张审核和去重，再冻结完整开发数据与三 seed 训练对照。本轮未启动训练、未生成新模型、未解封新场景。所有产物 training_admitted=false、promotable=false。30 分钟自动任务保持关闭。

## 产物

- 冻结协议（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-expansion-v1/protocol.json`）
- 逐图逐框审核（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-expansion-v1/review/decisions.json`）
- 变体指纹检查（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-expansion-v1/review/intake-audit.json`）
- 先导交接回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-expansion-v1/pilot-handoff.json`）
