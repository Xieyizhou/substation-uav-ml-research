# 核心验收与历史研究复现

## 核心检查

完整核心集合需要 Python 3.12+：其中保留的几何适配器测试会导入使用新版语法的冻结研究辅助脚本。
Demo 仍支持 Python 3.11；CI 分别在 3.11 / 3.13 检查 Demo、3.12 / 3.13 检查完整核心。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/run_tests.py --suite core --report outputs/core-tests.json
```

核心检查覆盖桌面 HTTP 服务、主 CLI、采集、规划、飞行控制契约、任务管理与 Workbench。
使用临时数据和模拟接口，不会启动真实飞行或要求本机历史训练数据。
不需要 Torch、ONNX、SciPy、PX4 或 Gazebo；NumPy、Pillow、OpenCV 和 psutil 是明确的测试依赖。
飞行 API 使用 MAVSDK 的 gRPC 接口，因此固定为实测的 `mavsdk==3.17.2`。
不固定时，新环境会装到不兼容的 MAVSDK 4 原生接口。

CI 使用相同的 `scripts/run_tests.py --suite core` 入口。新测试默认进入核心集合。
`config/sandbox/test_suites.json` 显式列出历史研究例外，清单重复或引用不存在模块会失败。
`--list` 只列出选中的模块，不执行测试。报告包含失败、错误和跳过原因。

源码 ZIP 应使用支持 Unix 文件权限的解压工具（例如 `unzip`）；某些程序化解压 API
不还原可执行位。导出包含 `.gitignore` 和逐文件清单，不包含 `.git`、环境或实验数据。
仓库规范测试也支持在这种源码副本中执行。

## 历史研究复现

```sh
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python scripts/run_tests.py --suite research --report outputs/research-tests.json
```

研究集合包含已冻结的实验适配器、真实小型训练/导出以及历史审核回执检查。
需要原始 `data/research/`、相关模型和回执声明的依赖；仅下载源码不能复现这些历史结果。
缺失依赖或数据应报告失败，不会以跳过代替通过。不要从历史目录中抽取零散文件并修改哈希。

每个研究测试模块在独立进程中运行。原因是历史实验适配器会配置共享模块的全局目录；
这与它们原来的独立 CLI 执行方式一致，并保留冻结源码字节。
普通 `python -m unittest discover -s tests` 仍可用于已配置研究环境的同进程全量回归。

## 可搬迁历史证据

```sh
.venv/bin/python main.py sandbox evidence-freeze \
  --records named-records.json --output outputs/sandbox/evidence/snapshot.zip
.venv/bin/python main.py sandbox evidence-inspect \
  --input outputs/sandbox/evidence/snapshot.zip --identity <冻结时返回的清单哈希>
.venv/bin/python main.py sandbox evidence-flight-verify \
  --output outputs/sandbox/evidence/checks/my-verification.json
```

`named-records.json` 是“记录名称 → 文件路径”的 JSON 对象。包内保存原始记录字节和这些记录
`inputs` 声明的所有直接依赖，以 SHA-256 命名去重；原绝对路径仅作为来源标识保存，读取时不访问它们。
嵌套 JSON 输入不会自动变成独立审核记录；需要审核的记录必须显式列入清单。
清单哈希须在包外保管，不能从待验证包内读取一个哈希再用它证明同一个包可信。

当前注册包由 `config/sandbox/flight_evidence_archive.json` 绑定，覆盖原飞行门禁实际审核的
50 份记录及全部直接依赖。专用验证同时检查 6 次 LiDAR 正向、1 次无路拒绝、1 次视觉改线、
1 次受控停止，以及降落、解除武装和进程清理结果。桌面 Flight Console 可发起受管核验并查看日志。

包约 404 MB，不随源码导出；对应历史文件约 10.06 GB，原文件保留。
这份核验只证明历史运行。源码或模型变化后，必须产生相应的新运行验收，旧结果不能自动授权新实现。

## 桌面反馈迭代

Development profile 中依次使用：

1. Dataset Manager → 飞行并采集反馈（现有固定 SITL 场景）。
2. 逐张查看图片并纠正框，显式确认完整标注或无目标；无法可靠标注的图片可以拒绝。
   预测只作为草稿提示。审核会记录审核者、原始图片、上一版身份和时间，过期修改被拒绝。
3. 选择基础数据集并注册新 ID。整次采集必须全部完成接纳或拒绝，且至少接纳一张。
   新反馈仅进入训练，保留基础数据的固定验证子集；相邻反馈不会随机拆入验证。
4. Training Studio 选择注册数据和历史父模型。训练预算不足以保留全部接纳反馈时会拒绝，
   不会静默丢弃这些反馈。训练前重查受管反馈数据的图片、标签、成员及来源哈希。
5. Managed model runs 查看实际父模型比较。新模型阈值在开发验证集选择，父模型使用冻结阈值；
   差值和各自阈值写入 comparison.json，并由新完成回执绑定。负向差值不会隐藏。

注册数据保存当时的全部审核快照；后续纠正需创建新的数据集版本。
导入基础数据的原有划分会保留，不能仅因注册成功便声称不同物理场景或盲测独立。
完成训练和 ONNX 校验代表技术产物可读与一致，不自动代表质量提升或所选模型已经通过飞行验收。

6. Flight Console 选择已验证模型，先“验收完整任务”，再“验收中途停止”。
   停止验收必须在开始水平飞行后点击“停车并降落”，并确认已降落、解除武装与进程清理。
   两项都针对当前输入通过后，“执行已验收任务”才可用。每次任务仍保存有上限的反馈，可返回审核。

当前入口使用 `src/flight/semantic_mission.py` 与 `sitl_lifecycle.py`，输出到
`outputs/sandbox/semantic_runs/`。从受信任证据包仅提取固定场景资源到 `runtime_assets/`；
原实验目录中的绝对路径仅作为来源文字，不用于当前启动。
当前运行身份绑定核心源码、模型回执与 ONNX、场景、PX4 二进制/启动文件/模型资源/插件、
Python 及关键依赖版本与 Gazebo 版本。身份使用逻辑相对名称，因此位置改变不改变身份。
启动前与飞行结束后重新检查输入；任何身份变化都需要重新验收，历史通过不能替代。
这只适用于固定本机 SITL 场景，不代表真机或未验收场景的资格。

桌面验收范围为桌面浏览器；按用户要求，不包含手机布局。
