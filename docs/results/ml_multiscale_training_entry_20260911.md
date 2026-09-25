# 多尺度显式训练入口接入

## 状态

接入和测试完成：`training_entry_verified_not_started`。本次仅接入，**没有启动训练**，没有创建训练授权回执或训练终点权重。

## 入口约束

`scripts/vision/train_frozen_multiscale.py` 默认只验证就绪状态；必须显式 `--train` 才进入训练。`--worker` 单独使用会被拒绝；工作进程还必须读取有效的训练授权回执。

启动时重新核验加载完成回执、完整依赖哈希、三个 seed 的唯一预检记录、初始化权重、配置与环境。训练固定/多尺度两组各 seed 7/17/27；不会静默改用加速器或复用历史终点代替新固定组。

每个真实训练批次通过 `BatchGate`：

- 成员和顺序必须与预检一致。
- 亮度处理后的原 640 图像张量及完整 `cls/bboxes/batch_idx` 必须与预检哈希一致。
- 调用已验证的共享 `frozen_multiscale_runtime.preprocess`；禁止第二次随机 multi_scale。
- 实际尺寸、缩放后模型输入张量及完整监督再次与预检逐批比较。不一致时在该批反向传播前拒绝。
- 结束时核验全部 2,700 次曝光、450 个优化步、完整亮度账本、配置及损失曲线；只保存并绑定预定终点 `last.pt` 的回执，不按验证成绩选 checkpoint。

各次尝试独立保存，最多三次。语义失败保留阻断，完整有效单元才复用；中断/超时后清理本次启动的进程组。训练成员末轮验证若由框架执行，只能称训练拟合诊断。

## 验证范围

此前六单元全量加载预检的 16,200 次加载继续有效。本次另用真正的训练入口批次钩子，每单元跑前 10 个真实 batch，共 60 个 batch、360 次加载，多尺度每个 seed 均覆盖 320/640/960。

此接入测试显式禁止优化器、反向传播、YOLO 训练与验证。输出输入张量与预检逐批一致。它不是完成一次优化训练，也不验证整轮优化稳定性或反向传播资源峰值。

46 项相关回归通过；v2.11 固定 40 文件完整性通过；真实默认命令验证为 `ENTRY_READY_NO_TRAINING_STARTED`。测试输出中的 `--worker requires --train` 是拒绝非法启动参数的预期测试。不声明全仓测试通过。

## 交付

- `scripts/vision/train_frozen_multiscale.py`：启动门禁、训练钩子、独立进程、恢复与终点回执。
- `scripts/vision/verify_multiscale_training_entry.py`：禁止训练的真实钩子测试与入口就绪签发。
- `tests/test_multiscale_training_entry.py`：默认不训练、非法 worker 参数、输入漂移拒绝、未完成拒绝及进程清理。
- `data/research/ml_training_recovery_v1/material-multiscale-loader-v1/entry-ready.json`：接入测试、依赖、基线与回归结果。

本次没有训练授权文件、训练完成回执、新模型或部署。全部新研究产物保持 `training_admitted=false`、`promotable=false`。后续需用户明确要求训练，才执行显式训练入口。
