# 本轮训练集交付与运行状态

## 已完成的判断

429 张原池成员中，376 张形成可用于本轮有界开发训练的冻结数据集，53 张整图暂缓。未修改历史图像、标签或审核；未删除单个标签，未将开发图转入训练。所有新产物继续保持 `training_admitted=false`、`promotable=false`，不构成正式训练准入或模型晋升。

376 张包括基础池 49、常规正例 9、桥接正例 198、困难负例 120。完整标签实例合计为变压器 201、开关柜 259、电容器组 81、电抗器 72。这些是标签数，不是独立场景数。

## 可复算依据

工作目录位于 `data/research/ml_training_recovery_v1/source-isolated-material-retention-v1/closed-exterior-coverage-v1/source-retention-control-v1/repeated-budget-control-v1/cool-light-coverage-v1/neutral-gray-calibration-v1/small-scale-material-control-v1/interleaved-tail-control-v1/reviewed-dataset-fit-v1`。

- `reviewed-dataset-v1/manifest.json`：成员、完整标签、来源、逐图质量及全图证据、53 张排除项。其“采样尚未就绪”状态描述该文件冻结时刻，不回写历史快照。
- `reviewed-dataset-v1/exposure-control-v1/design.json`：更新后的六个精确序列及按位置冻结的亮度数组、每 50 步曝光账本。
- 同目录六份计数解：两个优化阶段均达到最优；保持总量、子集预算、全类别实例曝光、负例成员及位置，原零曝光成员不恢复。
- `actual-preflight/*.json`：六个配置各完整遍历 6,600 次实际采样，核对图像及完整监督张量；未创建优化器、反向传播或训练验证。
- `entry-ready.json`：39,600 次预检、32 项针对性回归及 v2.11 固定 40 文件完整性核验通过。未声明全仓测试通过。

交付时再次核验全部 376 对图像和标签的文件哈希，并核验其全图证据记录。实际训练日志已进入第 2 个 epoch，说明首批两个单元不只是创建进程，而已执行训练批次。

## 运行安排

独立入口为 `scripts.vision.train_order_fit_reviewed`，默认仅核验，显式 `--train` 才启动训练。已授权并启动 ISR1100/ISM1100 两家族、seed 7/17/27，共六单元；每单元 1,100 优化步，采用 2 并发 × 4 CPU 线程。完成有效单元复用，不混接不完整训练状态。仅使用终点权重；训练结果及开发评估目前尚未完成，不能据此声称模型改进或门禁通过。

## 保留的边界

审核为 AI 辅助审核，不代表所有图像均取得实例掩码像素级可见性认证。共享布局、资产、同源变体及有限常规来源仍存在；历史来源补核不追认原采集通过新门禁。

本轮比较的是“整图风险暂缓及曝光补偿策略”，成员分布、部分批次组成和成员与亮度组合均发生变化，并非纯类别曝光量的因果实验。最小最大增量为 81–82 次并不证明来源充分或曝光量科学上最优。

下一判断应来自所有六个终点的固定开发评估和错误审核，不因单个 seed 的损失曲线或成绩另改训练配比。数据集可加载、证据可追溯，与模型是否达标是两个不同结论。
