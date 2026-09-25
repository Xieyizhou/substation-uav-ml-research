# 主体可见性预筛后的外观补采矩阵结果

## 本轮结论

剩余赭色、灰绿两种材质 × 常规／暖暗两种光照 × 8 个冻结位姿，共 32 张均首次采集成功。连同原始 8 张和先前钢灰 16 张，完整矩阵为 56 张、77 次实例框观察，已完成逐图逐框 AI辅助审核与指纹检查。此处的 77 是跨变体重复观察次数，不是 77 个独立设备实例。

这是补采与审核闭环，不是训练数据版本冻结或模型性能改善的结论。本轮未训练、未生成新权重、未解封新场景，也未恢复已关闭的自动任务。

## 证据与检查

- 新增 32 张对应 44 次实例框观察，均逐张查看全图叠框并分别记录每框理由。图中连续主体可辨识；部分前景遮挡仍在记录中保留，未将遮挡等同于不可见。
- 四个新世界仅改变允许的主体／前面板材质及指定光照字段。赭色 ambient/diffuse 为 0.44 0.38 0.28 1，灰绿为 0.30 0.42 0.36 1；几何、背景、设备实例、相机和原有传感器保持一致。
- canonical 采集检查、实际模式、相机合法性与实例映射检查通过；新增图与对应原始图实例集合一致，框坐标最大差异为 0 像素。
- 对 156640 条参考记录进行文件、原尺寸像素和近相似指纹检查；32 张新增图无需要暂缓的外部命中或跨来源近相似。保护参考仅使用既有成员与指纹缓存，不读取保护标签。
- 设计内材质／光照变体保留共同谱系，不删除，也不当作独立场景扩充数量。
- 28 项相关 unittest 通过；v2.11 固定 40 文件完整性通过。未运行或宣称全仓测试通过。
- 完成交接前重新验证输入输出身份与递归哈希；相关采集、审核汇总进程已结束，未发现本次 Gazebo 采集进程残留。

## 审核边界与未解决问题

所有审核均明确标为 AI辅助审核，不是人工审核，也不是实例掩码像素认证。暖暗条件主体变暗，但审核中仍可见连续主体；这只能支持继续开发数据流程，不能保证训练收益。

8 个位姿对应 7 个登记来源组，仍为同一个 complex 地图及原设备资产。F03、F08 为同源近邻开关柜视角，前面板均不可见。不能把本批完成解释为前面板覆盖、几何多样性或跨场景泛化问题已经解决。

历史待定先导批和历史实验未修改；不得把其未通过图混入本次已审核矩阵。

## 下一判断点

下一步先导出与已审核全图实例一一对应的训练格式标签，冻结 56 张成员、标签哈希、角色及共同谱系，再冻结三 seed 的受控训练对照协议。训练前继续验证与固定开发回归集的隔离；已查看的开发图不变更为盲测。

不在导出时静默删掉非计划目标，不根据模型成绩挑图；本批尚未自动获得训练准入。所有新产物保持 training_admitted=false、promotable=false，新场景保持封存。

## 可核验产物

- 剩余 32 张冻结协议（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/protocol.json`）
- 逐图逐框审核决定（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/review/decisions.json`）
- 指纹检查回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/review/intake-audit.json`）
- 完整 56 张矩阵交接回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matrix-handoff.json`）

交接回执身份：6c925f5db56f0a36c619ad8be6dbb817df69734c9016d8338b7dca48d6a20107。

