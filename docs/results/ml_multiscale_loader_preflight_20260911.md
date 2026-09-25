# 固定/多尺度真实加载预检

## 结果

两组 × seed 7/17/27 的真实加载预检全部通过。共遍历 **16,200 次图像加载、2,700 个批次**；每个单元 2,700 次加载、450 个批次。全部首个尝试成功，无加载失败或重试。

回执状态：`loader_preflight_passed_training_not_started`。

没有创建优化器、反向传播、运行训练验证或启动训练。这里只验证输入链路，不证明模型已训练或改进，也不证明反向传播显存/内存和运行时间可接受。

## 已核验

- 两组使用独立真实 Dataset/DataLoader，每 seed 全量遍历 45 个逻辑 epoch；每个 epoch 10 个 batch，与冻结的 450 批计划一致。
- 固定组每批 `[6,3,640,640]`；多尺度组每 seed 为 320、640、960 各 150 批，严格执行冻结顺序。
- 每次成员和批内顺序与同 seed R-clean 实际历史序列一致；因此完整类别实例曝光、子集预算、负例成员与位置、暂缓零曝光均未改变。回执另列实际类别/谱系/子集计数。
- 两组逐次亮度系数及亮度前后像素哈希一致；固定组的全部 uint8 图像张量、完整 `cls/bboxes/batch_idx` 与历史预检哈希一致。
- 所有尺度下完整归一化监督保持不变；每批保存实际模型输入张量形状与哈希、原 640 张量哈希及完整监督哈希。
- 多尺度组处于 640 的批次，与固定组归一化输入字节一致；未启用框架第二次随机 multi_scale。
- 固定环境为 torch 2.13.0、ultralytics 8.4.107；初始化及依赖身份经回执核验。

## 适配的明确边界

复用既有 640 Dataset、亮度增强、固定采样器，并调用本机真实 `DetectionTrainer.preprocess_batch` 做 float/255 处理；关闭其随机 multi_scale，再由共享适配器进行冻结尺寸的 bilinear、`align_corners=False` 插值。

因此 960 张量由 640 张量插值得到，**不是从原图直接加载 960，也不增加原始细节**。这是标准输入尺度扰动的受控实现；与前阶段直接原图 960 的压力推理并非像素等价，不能把两者等同。归一化框在方形整图等比缩放下不变，没有裁剪、拼接或标签重写。

Ultralytics 自动重建了历史导出目录中的 `labels.cache`；该可再生缓存不是科学证据，原图、标签、协议、审核、权重未修改。

## 回归与交付

50 项相关测试通过，v2.11 固定 40 文件核验通过；不声明全仓测试通过。初次测试命令引用了不存在的测试模块名，定位后以实际 `tests.test_brightness_transfer` 重跑完整套件通过；不是加载门禁被放宽。

新增代码：

- `scripts/vision/frozen_multiscale_runtime.py`：未来训练与预检共用的确定性预处理。
- `scripts/vision/preflight_frozen_multiscale.py`：两组真实加载、历史张量比对、独立尝试与逐批账本。默认仅静态预检，`--check-loaders` 显式遍历，无训练入口。
- `scripts/vision/finalize_multiscale_loader.py`：读取完整记录，核验两组形状、标签和 640 一致性，再签发加载回执。
- `tests/test_frozen_multiscale_runtime.py`：尺度/数据类型、归一化、插值、重复缩放拒绝及记录一致性测试。

产物目录 `data/research/ml_training_recovery_v1/material-multiscale-loader-v1`，包含冻结协议、三个 seed 的两组逐批记录、亮度账本、加载汇总及 `completion.json`。

下一步需将同一适配器接入显式训练入口，在启动与每批运行时消费并重验这些回执；不得绕过预检另走一条缩放路径。当前尚未完成该训练入口，因此保留 `training_ready=false`、`training_started=false`。启动训练需后续授权。

所有新产物保持 `training_admitted=false`、`promotable=false`。
