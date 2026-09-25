# 扩大对照实采结果

80 个计划视角全部采集完成，9 个批次均为 complete_pending_review，无失败批次。complex 30 帧、medium 38 帧、simple 12 帧；按几何预测分层为 clear 57 帧、partial 16 帧、blocked 7 帧。这些分层尚非像素遮挡审核结论。

已从绑定传感器快照读取相机外参，将矩阵相机位姿转换为采集器位姿，并核验水平 FOV 为 1.466。各批次沿用 full_2d 快照，独立运行 Gazebo 校准采集，不启动 PX4 或训练。

新增显式诊断缺席规则：仅 calibration 模式、diagnostic_only=true、training_admitted=false 且 diagnostic_allow_expected_absence=true 时，允许预期类别缺席，保存原始真值。普通采集和 pilot 不适用。目标类别保持真实类别，没有通过改成背景类别绕过检查。类别存在仍不代表预期实例存在，后续需按实例核验。

80 帧的图像与深度 SHA256、实际位姿容差、计划和采集回执 identity 全部核验通过。12 项诊断隔离、拒绝帧、恢复及投影测试通过，git diff --check 无错误，结束后未发现遗留 gz sim 进程。verification.json 记录本次核验输入及 identity。

数据目录：data/research/ml_training_recovery_v1/control-matrix-capture-v1/。progress.json 汇总 9 批回执；每批包含 plan 和 capture；verification.json 汇总核验。执行脚本为 scripts/vision/capture_recovery_control_matrix.py，核验脚本为 scripts/vision/verify_recovery_control_capture.py。

本步完成实采和完整性核验，尚未完成 80 帧的像素遮挡、诊断区域视觉审核和新一轮推理。下一步按实例叠加区域核对实图，再冻结可见目标分母并运行原有 v2.11/640/0.37 设置。此次数据均不入训，426 帧候选和既有暂缓清单保持不变。
