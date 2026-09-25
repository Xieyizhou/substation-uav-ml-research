# 沙盒成熟度评估与改进记录（2026-09-24）

更新至 2026-09-25：本轮已交付训练、采集、存储与飞行视觉接入四方面的改进，并完成从桌面启动的真实 PX4 SITL 闭环验收。地图未扩展。当前是能力边界明确的桌面研究沙盒；通用场景、反馈审核与自动注册等产品流程仍有后续工作。

## 对照类似程序

| 参考 | 可借鉴的能力 | 对本项目的具体改进 |
| --- | --- | --- |
| [Project AirSim 数据采集配置](https://iamaisim.github.io/ProjectAirSim/datacollection/config.html)、[数据生成 API](https://iamaisim.github.io/ProjectAirSim/datacollection/api.html) | 环境和采集任务参数化、通过 API 执行采集 | 把现有散落脚本收束成带预算、恢复状态、采样规则、统一回执的采集任务；复用现有场景，不重做地图 |
| [FiftyOne Brain](https://docs.voxel51.com/brain/index.html) | 相似度检索、重复数据和样本问题分析 | 采集后按来源隔离、内容去重、类别/距离/困难条件统计，选择有价值的增量；不要把相邻重复帧当作新增覆盖 |
| [Ultralytics Train](https://docs.ultralytics.com/modes/train/) | 预训练微调、恢复训练、缓存及冻结层等训练选项 | 历史模型可直接成为新实验起点；以同一验证集比较时间、内存与指标，再决定精度或冻结策略，而不是盲目开启加速选项 |
| [PX4 Gazebo 仿真](https://docs.px4.io/main/en/sim_gazebo_gz/) | 现有 SITL 链路、受 CPU 和 I/O 约束的仿真速度 | 优先降低图像落盘和重复推理开销，保持飞控/传感器时钟与验证边界；不以任意提高仿真倍速代替端到端验收 |
| [MAVSDK Offboard](https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html) | 速度、航向设定点控制接口 | 推理与飞控发送循环分离；视觉只提供通过有效性检查的规划输入，保留停车、规划拒绝、降落和清理路径 |

以上是架构借鉴，不是这些项目与本项目的同机性能对比，也没有引入它们作为新运行依赖。

## 四项要求的当前状态

| 要求 | 本轮验证到的进展 | 尚需完成 |
| --- | --- | --- |
| 下一代训练更省空间、更省时间 | 历史权重微调完整链路通过；单次回放；消除重复 ONNX；冻结层小规模对照耗时减少 11.50%、恢复检查点减少 31.96% | 扩大增量与旧能力保留验证；单次小样本对照不能作为默认策略推广依据 |
| 先完善训练和数据收集，不做地图 | 历史模型来源可选；训练前检查验证类别覆盖；新增有容量上限、去重、无损 PNG、显式恢复的采集器，真实回放与实时 Gazebo 探针通过 | 将飞行困难样本筛选、审核、来源隔离、训练集注册接成受控任务 |
| 整体更小且旧模型仍可训练 | 实际归档一份旧旁路记录；既有模型保持可用；新模型由旧权重训练；新采集不落盘全部 raw 帧 | 扩大存储治理覆盖；不能把一份归档的节省率推算到全部数据 |
| 接入飞行中 YOLO + 改路线系统 | 同步 RGB-D 与姿态、连续三帧确认、停车、目标观察路线替换、恢复飞行与降落已完成；直接运行与桌面启动各一次正向闭环，中途运动状态下停止降落通过；原 6 正向 + 1 拒绝门禁仍有效 | 当前限定既有固定 SITL 场景、一次设备观察改线；不代表任意场景、动态目标或实机飞行能力 |

四项要求的本轮改进与验证已完成。表中的后续工作是成熟度扩展方向；小规模训练收益与固定场景飞行结果均保留其适用范围。

## 已落地的训练与存储改进

### 历史模型成为新实验起点

Training Studio 新增 `Start from`，可选择有完整验证回执的历史实验。新实验微调与恢复中断训练分别使用新配方与旧 `last.pt`，不会混用。

权重按 SHA-256 存入 `outputs/sandbox/workbench/weights/`。同一权重多个实验共用一个受管快照；导入时采用独立复制，避免修改原始文件影响后代实验。配方绑定权重、父实验和父回执。旧配方不增加空字段，因此原身份哈希保持不变。

外部历史 PyTorch 权重也能通过 CLI 指定；模型结构兼容性在实际 YOLO 加载/训练时检查，并非任意 `.pt` 都能训练：

```sh
.venv/bin/python main.py sandbox --profile development workbench-recipe-create \
  --experiment-id my-next-model --dataset-id imported-smoke-20260817 \
  --preset smoke --parent-experiment-id workbench-full-v2-20260817
# 外部历史权重使用 --initial-weights /path/to/model.pt，替换 --parent-experiment-id。
```

本机真实实验 `incremental-smoke-20260924` 使用历史完整模型，成功迁移 499/499 项权重，执行 256 图训练、64 图验证、ONNX 一致性检查和回放，最终回执通过。它是一轮兼容性冒烟试验，不是新模型推广或泛化精度提升证据。

### 回放只推理一次

原流程先推理所有图测耗时，再加载模型推理同一批图算指标。现在同一预测流同时输出耗时和预测，保留原来用于计算指标的 `rect=False`、置信度和 NMS 设置。新回放的耗时口径明确记录为读取下一帧预测结果所需的墙钟时间，首帧包含流初始化；不能直接混用旧回放的 P95 作回归门槛。

真实 CPU ABBA 顺序、每方式 2 次、64 张同一验证图的结果：旧流程中位数 3.398 秒，新流程 1.458 秒，约 2.33 倍；四次的检测指标完全一致。新导出的 10,480,516 字节 ONNX 只有模型目录中的一份。此结果不代表全部训练提速 2.33 倍。

可复现命令：

```sh
.venv/bin/python -m scripts.vision.benchmark_workbench_replay \
  --run outputs/sandbox/workbench/runs/incremental-smoke-20260924 \
  --output outputs/sandbox/efficiency/20260924/replay-benchmark.json
```

此外修正训练 ETA 的逐轮耗时计算，恢复训练时不再用已完成总轮次稀释本次耗时；存储统计也按 inode 去除同目录内硬链接的重复计数。

### 已完成摄像头旁路记录的无损冷归档

初次 `du` 盘点：`data` 约 457 GiB，其中 `data/research` 约 445 GiB；`outputs` 约 9.4 GiB，`.venv` 约 1.7 GiB。主要体积是采集与实验数据，不是页面或应用源代码。详细分组保存在本轮 storage-inventory.json；`du` 不提供 APFS 克隆共享区块的精确归属。

新增 `camera-archive`、`camera-archive-inspect`、`camera-archive-restore`。范围限定为完成回执可校验、未进入训练、没有飞行控制权的相机旁路记录，只归档哈希命名的 `.raw`。默认保留原文件，显式 `--compact` 才在完整解压哈希验证后移除原始帧。归档操作带进程锁，可重复执行；拒绝不完整输入、损坏归档、路径跳转和还原时覆盖冲突文件。

首次真实对象是 `live-replan-turn-hold-002/runtime/vision`：1,405 帧，原始 8,740,224,000 字节，ZIP 147,633,381 字节，载荷净减少 8,592,590,619 字节（约 8.00 GiB，98.31%，不含小型索引开销）。不改权重、训练集、检测日志与旧回执。

旧审计/回放工具仍可能要求原路径下的散装 `.raw`；运行这些工具前先还原。归档不是伪装成普通文件的透明替换：

```sh
.venv/bin/python main.py sandbox --profile development camera-archive-restore \
  --recording data/research/material-shadow-v1/autonomy-avoidance-v1/live-replan-turn-hold-002/runtime/vision
```

还原会逐项核对哈希，恢复文件路径和修改时间；恢复后归档仍保留，可再次执行 `camera-archive --compact` 回到冷存储状态。当前启动门禁使用的重复飞行证据未归档。

## 本轮执行顺序与验收依据（后文记录结果）

1. **采集体积从源头控制。** 将内存识别与有限预算证据写入分开，比较无损编码、采样间隔和难例保留；检查保存后的像素身份、标签同步、数据源分组、丢帧与磁盘耗尽行为。不要在推理/飞控关键路径里无界压缩或写盘。
2. **增量训练性能对照。** 固定新增样本、旧样本回放配额和验证来源，以相同质量门槛比较总训练耗时、内存峰值、磁盘新增量与各类召回，验证旧模型微调确实省训练工作，而不只是能加载。
3. **视觉接入现有飞行控制。** 复用当前稳定的停车—重规划—恢复—降落结构。YOLO 输出需明确模型、图像、采集时间、姿态配对与空间不确定性；只有通过时效、连续确认和几何检查的观测才能触发规划。没有空间证据时不能把 RGB 框直接当作地图障碍。
4. **闭环验收。** 在已有固定场景验证视觉影响路线的正向证据，以及失联、旧帧、错误类别/框、不可达路线、用户停止的拒绝或降级路径；保留原 LiDAR 停车保护。通过后再开放为桌面可选任务。

## 验证证据位置

- 真实训练：`outputs/sandbox/workbench/runs/incremental-smoke-20260924/receipt.json`。
- 回放对照：`outputs/sandbox/efficiency/20260924/replay-benchmark.json`。
- 盘点：`outputs/sandbox/efficiency/20260924/storage-inventory.json`。
- 归档索引：旧旁路记录目录下的 `raw-evidence-archive.json`，含每个成员原始哈希、大小和修改时间。
- 完整归档→还原→再次归档证据：`outputs/sandbox/efficiency/20260924/archive-roundtrip.json`。真实还原 1,405/1,405 帧；最终散装 raw 数量为 0；旧 4 个模型与新模型均校验通过，归档后重复飞行门禁仍通过。
- 相关 93 项回归测试同进程通过，覆盖模型来源兼容、原权重修改隔离、单次回放、归档损坏/还原冲突/缺失文件、存储计数、相机解码与现有飞行门禁。合并测试时暴露的原生 torch 重载崩溃已定位到单元测试的模块模拟清理，并通过隔离设备探测修复；真实训练另行完成，未用模拟测试替代。
- UI：桌面 Chrome、1440×1000，验证模型/数据集选择在轮询后保留，以及确认框来源正确；未提交额外 UI 训练任务。390×844 检查仍显示旧版最小 1024px 布局，此版本只验证桌面可用，未宣称手机适配。

## 第二轮实测：冻结训练与有上限的反馈采集

### 训练时间、检查点与内存对照

配方新增可选 `freeze=0/5/10`，CLI 为 `--freeze`。默认仍为不冻结，旧配方身份不变。训练回执绑定 `training_efficiency.json`，记录实际参与训练的参数量、逐轮时间、中途可恢复检查点最大值与进程内存峰值；检查点最终剥离优化器状态后的大小另列，避免混淆。

两组均从 `workbench-full-v2-20260817` 历史权重开始，使用 `visual_yolo_v2_12_run18` 中完全相同的 256 张训练图、64 张验证图，CPU、320px、batch=4、workers=0、seed=7、5 epochs。验证实例数为 transformer 18、switchgear 21、capacitor_bank 14、reactor 18。

| 指标 | 全量微调 | 冻结前 10 层 | 变化 |
| --- | ---: | ---: | ---: |
| 训练阶段墙钟时间 | 113.429 s | 100.381 s | 减少 11.50% |
| 中途恢复检查点最大值 | 21,201,123 B | 14,425,897 B | 减少 31.96% |
| 进程内存峰值 | 1,218,576,384 B | 1,028,734,976 B | 减少 15.58% |
| 最终 best.pt | 5,423,322 B | 5,423,322 B | 不变 |
| 可训练参数量 | 2,590,604 | 1,474,860 | 减少 43.07% |
| 验证 macro F1 | 0.94764 | 0.96211 | +0.01447 |
| 小目标召回（28 个） | 0.89286 | 0.92857 | +0.03571 |

两组均完成训练、验证、ONNX 一致性及回放门禁。阈值分别由验证集选择为 0.20 和 0.26。这是每种策略一次的小规模开发集试验，并非独立保留集、统计显著性或飞行推广证据；没有自动替换默认模型。内存为进程生命周期峰值（含导入、不含子进程），计时只覆盖训练阶段。

首次使用 `visual_yolo_v2_1_png` 的尝试训练后失败，原因是整个验证集没有 reactor 类，无法产出四类验证指标。失败实验 `efficiency-fullft-20260924` 保留；新增训练前类别覆盖检查，缺类会在调用训练之前拒绝。未降低验证要求，也未把失败试验纳入上述对照。

可复核两组产物的比较命令：

```sh
.venv/bin/python -m scripts.vision.benchmark_workbench_training \
  --baseline outputs/sandbox/workbench/runs/efficiency-fullft-v2-20260924 \
  --candidate outputs/sandbox/workbench/runs/efficiency-freeze10-v2-20260924 \
  --output outputs/sandbox/efficiency/20260924/training-comparison.json
```

### 有容量上限的在线反馈

`scripts.vision.collect_feedback` 提供 live/replay 两种输入。live 使用已验证的 ONNX 模型，接收端只保留最新一帧，后台 PNG 写入队列最多两帧；过期、重复像素、采样过密和超容量帧会被计数拒绝。保留帧携带原载荷哈希、无损像素哈希、模型或原记录来源。预测始终为 `unreviewed`，不能直接作为训练标签。输出显式标记 `control_authority=none` 和 `planning_eligible=false`。

样本数和 PNG+样本 JSON 字节数受配额约束；小型配置、摘要、latest/status 文件属于固定管理开销，不计入样本字节数。恢复要求显式 `--resume`、相同来源/策略、图片和元数据哈希完整；单写入者锁防止并发覆盖，未完成的孤立文件仍计入磁盘预算。

- **历史回放**：从已归档的 `live-replan-turn-hold-002` 直接读取前 600 条检测记录，无需展开所有 raw。上限 64 帧/32 MiB，实际保存 64 张 PNG 与元数据 8,872,222 B。逐张解码核对 64/64 像素哈希；显式恢复后重新读取 600 条记录，样本成员与占用不变。
- **实时静态场景探针**：12 秒内收到 355 帧、推理 165 帧，解码前替换 189 帧，非法/过期计数为 0；去重后保留 2 帧，样本共 377,524 B（上限 16 帧/8 MiB），无 raw 文件。启动的进程全部退出。此探针没有启动 PX4，也没有验证飞行改路线。
- **失败路径**：相机断开或清理失败均落下 failed 状态；单测覆盖容量、恢复冲突、图片篡改、队列替换和中断文件预算。

使用例：

```sh
.venv/bin/python -m scripts.vision.collect_feedback --mode live \
  --model-run outputs/sandbox/workbench/runs/workbench-full-v2-20260817 \
  --output outputs/sandbox/feedback/my-new-collection \
  --seconds 30 --max-frames 64 --max-mib 32
```

需要已有相机 topic；输出必须为新目录。当前为 CLI 功能，尚未接入桌面的审核与训练集注册流程。

第二轮证据：`training-comparison.json`、`feedback-replay-verification.json`、`outputs/sandbox/feedback/live-probe-20260924/probe.json`。105 项相关回归测试通过，结果为 `outputs/sandbox/efficiency/20260924/regression-tests-002.txt`。

补充最终核验：修正相机清理失败的状态传播后，再运行 `live-probe-20260924-final`，收到 353 帧、推理 165 帧、保存 2 帧/377,525 B，进程清理通过且证明文件绑定当前源码哈希。历史父模型及三份本轮后代模型回执再次通过；既有 6 正向 + 1 拒绝飞行门禁仍通过。结果为 `outputs/sandbox/efficiency/20260924/final-verification-002.json`。

第二轮结束时视觉接入尚未完成：旧 `ActiveRgbdRuntime._pose()` 读取最新姿态/位置，没有按图像采集时刻配对。第三轮使用独立的新链路解决该问题；没有直接启用旧开关，也没有宣称旧链路已全面修复。

## 第三轮实测：飞行中 YOLO 参与改路线

### 同步输入与受限控制

新链路由 `sim_pose_sync.py`、`gazebo_depth_memory.py`、`aligned_sim_rgbd.py`、`visual_replan_service.py` 和 `visual_target_route.py` 构成。只在已确认共享仿真时钟的 PX4/Gazebo SITL 中，将 RGB、深度与 MAVLink `time_boot_ms` 对齐。姿态/位置使用消息发布时刻，不声称是传感器原始采样时刻；时钟回退、来源改变、旧帧、过大时间差和无效深度均拒绝。推理结束后再次检查时效。

视觉目标需要置信度至少 0.85、连续三个不同采集时刻确认、有效深度与稳定空间位置。任务先在运动中发现目标并停车，再用新鲜观测生成约 5 米距离的观察路线；局部坐标通过已验证注册转换，几何规划与 LiDAR 停车保护继续生效。一次任务最多替换一次路线，不使用仿真物体真值生成视觉目标。

RGB/深度使用有界内存队列。飞行反馈最多 64 帧、16 MiB，另保存三次接受观测的 PNG/压缩深度与请求证据；不连续落盘全部 raw 帧。预测仍为未审核数据，不自动进入训练。

### 真实试验与失败记录

第一次试飞使用 Workbench 历史完整模型，空中视角未得到足够高置信度的目标，任务没有改线，随后降落和清理，按失败保留。离线诊断确认该视角模型适配不足。改用既有固定历史模型 `material-routed-480-7`（权重 SHA-256 `615db2cd151d6cc0015afa31f03470065e7e1f34d7b02be2957d24ecf1fe3923`），保留 0.85 接受门槛，没有将本轮小样本训练模型自动推广。

| 验收 | 结果 |
| --- | --- |
| 直接运行正向闭环 `…e208b6496d5440259e24d8a3b8a09240` | 0.137 m/s 运动中识别开关柜，停车后目标从 (3.0, 1.5) 改为 (4.3, 1.6)，恢复并到达，误差 2.10 cm；确认降落、解除武装、清理进程 |
| 改线恢复后的中途停止 `…e209b6496d5440259e24d8a3b8a09240` | 实测速度 0.108 m/s 时提交受管停止请求；按预期退出任务并完成停车降落、解除武装和进程清理 |
| 桌面启动完整闭环 `…733a32b9ea18422d977bf63365ce953d` | 目标从 (3.0, 1.5) 改为 (4.3, 1.7)，恢复运动样本 246 条，到达误差 2.57 cm，最大横向偏差 3.52 cm，最小 LiDAR 距离 6.75 m；离线真值碰撞球净距 2.77 m；确认降落、解除武装和清理 |

上述运行目录均位于 `data/research/material-shadow-v1/autonomy-avoidance-v1/`，前缀为 `sandbox-replan-v1-`。完成回执验证三帧深度定位与路线重算、事件顺序、原始 LiDAR 回放及离线真值净距；没有重新运行 YOLO 作为独立检测审计，不将回执等同于实机安全认证。

第一次桌面启动还暴露了模型冷初始化问题：字体缓存初始化发生在 MAVSDK/gRPC 线程启动之后，视觉启动超时，未解锁起飞。新增 `src.sandbox.visual_replan_entry`，先准备模型依赖和推理，再启动飞控连接。冷缓存预备检查及上述第二次桌面启动均成功。失败任务 `20260925T003110Z-visual-replan-flight-1fc03e89` 保留。

最终桌面任务 `20260925T004454Z-visual-replan-flight-495d4c50` 的 `workflow_receipt.json` 位于 `outputs/sandbox/operator/jobs/` 对应目录，状态 complete、退出码 0，三个预期输出的哈希均核对通过。该飞行保存 50 张反馈图及元数据，共 1,631,442 B。中途停止证据为 `outputs/sandbox/vision-control/stop-probe-001/proof.json`。启动时重新验证正向、停止和既有 6+1 门禁依赖，不只依赖界面显示状态。

### 桌面入口与最终检查

- Flight Console → Run → **识别设备后改为观察路线**：真实点击启动并运行完成；空闲后启动恢复可用、停止禁用。
- Training Studio → **Start from** 与 **训练范围**：可选历史模型和冻结 0/5/10 层；轮询保留模型、数据集、冻结选项，确认框正确显示参数。最后一次 UI 检查取消训练，避免重复创建实验。
- Chrome / Playwright，`http://127.0.0.1:8876/`，1440×1000：页面身份、非空内容、无错误遮罩、无相关页面运行错误、截图与交互检查通过。Browser 插件未安装，采用现有 Playwright 1.62.1。390×844 下沿用旧最小 1024px 布局，会横向溢出；本次交付仅验收桌面使用。
- 视觉接入阶段 130 项相关回归测试通过；冷启动入口与最终界面修改后再次运行 94 项相关测试通过。真实训练、实时采集、两次正向飞行与中途停止分别提供独立运行证据。
- 最终重新校验历史父模型、三份本轮后代模型、固定飞行模型，以及飞行/停止/原门禁回执；无模型删除。详细临时核验、UI 结果与截图保存在 `/private/tmp/sandbox-final-*`。

下一阶段优先完善反馈审核与训练集注册、扩大独立验证与旧能力保留检查，再考虑更多场景和动态目标；地图扩展继续暂缓。
