# 材质定向模型仿真旁路接入

仅研究用途，不晋升模型，不改变默认设备模型。独立进程订阅 Gazebo RGB，不连接 MAVSDK、不启动 PX4、不解锁、不发送航点、不将检测写入规划队列。无框不代表无障碍。运行失败只停止旁路。

默认 seed 7 是固定接线配置，不是最佳 seed；可选 17、27。运行前核验原材质定向组权重和审核身份。CPU 4线程，640，显式 rect=False 方形填充，confidence 0.37，类别相关 NMS IoU 0.7，max_det 300。默认命令只核验：

```sh
.venv/bin/python -m scripts.vision.material_shadow
```

现有已审核训练图只用于真实推理冒烟（不是效果评估）：

```sh
.venv/bin/python -m scripts.vision.material_shadow --mode smoke --output data/research/material-shadow-v1/smoke-001
```

在已运行的 Gazebo RGB 仿真旁运行，不启用 active_semantic_inspection：

```sh
.venv/bin/python -m scripts.vision.material_shadow --mode live --seconds 30 --output data/research/material-shadow-v1/live-001
```

输出目录必须不存在，历史结果不覆盖。最长120秒，最新帧队列容量1，跳过接收后超过1秒的帧；记录队列丢帧、原始图像、RGB哈希、预测框和推理调用耗时。仿真时钟与单调时钟不能直接相减，接收新鲜不等于传输无积压；尚不认证端到端实时性。原图保存在独立输出目录，短程运行仍需预留磁盘。进程隔离不等于CPU/内存资源隔离，尚未通过与飞控并发负载验证。

检测记录供离线查看，不生成通过判定。后续须验证飞行并发延迟、持续漏检、误检触发、断流和故障保护。本入口不提供实机模式，不能作为实机飞行许可。

实时模式在订阅之前使用三次640×640黑色合成图进行推理预热，结果丢弃，不写入检测帧或训练数据。`warmup.json` 单独记录预热时长及结束时刻；首次真实帧在预热完成后才开始接收。预热不省掉启动时间，也不保证传输端不存在历史积压。

修正复测：96张开发图的一致性核对及三次15秒隔离静态Gazebo测试已完成，见 `data/research/material-shadow-v1/rect-fixed-retest-v1/report-zh.md`。日志新增接收后编码投递、排队、解码、推理和接收到结果的分项延迟。它们不包括接收前的传输积压，不能替代端到端实时性认证。

## 内存优化入口

使用最新原始消息队列，在选中需要推理的消息后才执行JSON/base64解析及RGB转换；避免给随后丢弃的帧做解析。图像直接进入推理，已推理帧保存内容寻址的无损raw RGB证据，不再先PNG编码落盘再读回。原PNG入口继续保留作对照。

```sh
.venv/bin/python -m scripts.vision.material_shadow_latest --mode live --seconds 30 --output data/research/material-shadow-v1/latest-run-001
```

未加 `--mode live` 时仅预检，不订阅或飞行。队列仍只有一个待处理消息，丢弃帧仅计数、未解析也未保存，不可视作通过质量检查。动态图像的raw证据可能占较大空间，本入口仍限制最长120秒；无需长期录制时不要反复无界运行。证据保存耗时单列并计入 `receive_to_logged_result_ms`，实际日志写入flush时间不在该指标内。

仅去除PNG但仍逐帧解析的中间方案 `material_shadow_memory` 在本机对照中未加速，不作为推荐入口。最新消息优先方案的受控结果见 `data/research/material-shadow-v1/latest-benchmark-v1/report-zh.md`。没有接入规划或飞控指令。
