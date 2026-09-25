# 主体可见补采数据冻结与配对训练设计

## 当前结果

56 张已审核图已导出为无损 RGB PNG 和 YOLO 格式全图标签，共保留 77 次实例框观察，包括非计划目标。原始图像、历史标签和实验未修改。导出后逐一核对原尺寸像素、标签坐标往返转换、原始框输出与仿真实例映射；任何越界框或审核缺失均拒绝，不静默裁剪或删除。

完整数据注册表为 56 张新图；训练池注册表另复用已审核共同基础 62 张、常规 39 张、困难负例 120 张，共 277 条。历史桥接正样本不进入这两个新对照。注册表成员数不等于每个训练单元实际使用数；K 使用 217 个独特图像成员，L 使用 257 个，精确成员及曝光均在协议中保存。

本阶段只完成数据和实验设计冻结，未训练、未产生新权重。training_admitted=false、promotable=false；新场景仍封存，后台自动任务未开启。

## 已冻结的对照

| 组 | 配对补充槽位使用内容 | 回答的问题 |
| --- | --- | --- |
| K-300 | 8 个新位姿原始图 | 新位姿、可见主体监督下的同预算参考 |
| L-300 | 相同位姿的 6 种材质／光照变体 | 同预算替换为外观变化是否改善敏感性 |

每组 seed 7、17、27，共 6 个独立单元，均为 300 优化步、batch 6、1800 次图像曝光。采用 v2.11 初始化、CPU、640、AdamW、恒定学习率 0.001、无预热、增强关闭、只使用末轮权重；恢复 I 组既有设置，不延续 J 的低学习率，也不再搜索学习率。初始化仍是 v2.11，不从 I/J 终点继续训练。

每 600 次曝光固定为基础 216、常规 156、配对补充 120、困难负例 108；300 步分别为 648、468、360、324。共同成员及其曝光位置直接沿用同 seed 的 I-300。仅在原桥接正样本槽位放入新配对图，K/L 的位姿顺序相同，每位姿曝光 45 次。L 的每位姿 6 种变体循环洗牌，各曝光 7 或 8 次。

两个新组的类别实例曝光逐位置一致：

| 全图类别 | 每单元实例曝光 |
| --- | ---: |
| 变压器 | 1278 |
| 开关柜 | 1080 |
| 电容器组 | 714 |
| 电抗器 | 516 |

因此 K/L 比较没有改变全图类别实例曝光量，但不能声称四类监督均衡。此次不另外引入类别重采样。L 是材质与光照组合干预，不能把总体差异解释成单独材质或单独光照的因果效应。相对历史 I 的比较还改变位姿及监督分布，不是仅颜色改变。

## 评估与启动门禁

沿用已查看的 48 张配对开发图和 48 张无目标开发图，CPU、640、正式 confidence 0.37、类别相关 NMS IoU 0.7、max_det 300、同类一对一匹配 IoU≥0.5；另以 confidence 0.001 做操作性错误归类，不替代正式评估。

保留原有所有数值门槛及相对 R-300、历史 A 的原始／光照全图及逐类召回最多下降 5 个百分点要求；对 L 另增加相对匹配 K-300 的相同能力保留检查。不选择有利 seed，候选家族只能保留全部三个 seed。训练成员的末轮验证仅是训练拟合诊断。

启动前仍需独立实现并验证执行适配器，将代码与测试哈希绑定到新执行回执，确保实际曝光逐项等于冻结序列、优化器控制一致、6 单元全部运行、失败尝试保留、恢复时重新校验全部身份。当前协议明确为 frozen_design_execution_preflight_required，不可将其当作训练已完成或正式准入回执。

## 检查与边界

36 项相关 unittest 通过；独立读回验证通过；v2.11 固定 40 文件完整性通过。未运行或宣称全仓测试通过。

56 张仍只对应同一地图和设备资产、8 个位姿及 7 个登记来源组。F03/F08 为同源近邻，所有同源变体同属开发训练候选角色，不拆分到盲测。开关柜前面板覆盖仍未解决。原有基础池部分谱系只解析至成员身份，不能宣称已建立独立场景隔离。前阶段文件／像素／指纹排除结果保留在输入哈希链中。

审核性质仍为 AI辅助审核，不是实例掩码像素认证。冻结数据不证明训练有效；下一重要判断点是 6 个单元完成统一评估后，外观收益与原有能力、误检约束能否同时满足。

## 产物与身份

- 56 张数据与全图标签注册表（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/attempt-001/dataset.json`）
- 6 单元协议、精确采样及曝光账本（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/protocol.json`）
- 独立读回及回归验证回执（本地研究资产：`data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/cleanup-validated-replay-v1/visibility-quality-training-v1/full-instance-exposure-balance-v1/lower-rate-control-v1/appearance-recovery-pilot-v1/instance-replay-v1/full-image-pose-screen-supplement-v1/local-supplement-v1/original-pilot-v1/appearance-remaining-v1/matched-appearance-training-v1/freeze-validation.json`）
- [标签导出与配对设计入口](../../scripts/vision/freeze_full_image_training.py)
- [独立验证入口](../../scripts/vision/verify_full_image_training_freeze.py)

数据身份：3820f295e420ea514e82c187c2b8b51f58427c15703f21778717e14def97d33f。
协议身份：92e9887a92c2be1f1324649a4c6573da2e5f5a55d3fb445dda55bf11758f8fea。
验证身份：8c08d1a3d8eddb696057fb312f14d9cd3a83022359623fc3cb357fbf286e8b38。

