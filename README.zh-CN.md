# 无人机视觉研究工具箱

面向变电站设备视觉研究的离线工作流，以及 PX4/Gazebo 集成中的工程实验案例。示例沿用变压器、开关柜、电容器组和电抗器四类标签。

[English](README.md) · [快速开始](docs/SANDBOX_QUICKSTART.md) · [代码阅读指南](docs/CODE_GUIDE.md) · [工程案例](docs/case-studies/README.md)

## 运行三个离线样例

使用 macOS 或 Linux、**Python 3.11+**，在本快照的源码根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python examples/dataset_audit/run.py --output outputs/examples/dataset-audit
python examples/prediction_comparison/run.py --output outputs/examples/prediction-comparison
python examples/receipt_check/run.py --output outputs/examples/receipt-check
```

输出目录必须尚不存在。只需 Pillow 和 PyYAML，不需要权重、PX4、Gazebo 或额外研究资产。

| 样例 | 查看结果 | 用途 |
| --- | --- | --- |
| 数据审核 | `outputs/examples/dataset-audit/audit.json` | 经已有 YOLO 导入器审核 8 张训练图与 4 张验证图，检查标签结构、框与重复图像。 |
| 预测比较 | `outputs/examples/prediction-comparison/comparison.html` | 本机打开 HTML，查看相同输入上的标框图、各类计数与指标变化。 |
| 回执核验 | `outputs/examples/receipt-check/tamper_check.json` | 原结果通过；被改动的副本拒绝。此样例只用 Python 标准库。 |

图像是自生成几何图形，预测为**手工构造的缓存**，用于说明接口、匹配和结果追溯，不代表设备识别精度。结构有效不等于标注正确或站点独立；哈希回执验证记录文件的一致性，不等于数字签名或飞行资格。详见[样例说明](examples/README.md)。

## 值得阅读和复用的部分

| 能力 | 实现入口 |
| --- | --- |
| 数据审核与注册 | [workbench_datasets.py](src/sandbox/workbench_datasets.py) |
| 固定配方与实验回执 | [workbench_recipe.py](src/sandbox/workbench_recipe.py)、[workbench_receipt.py](src/sandbox/workbench_receipt.py) |
| 同组预测比较与指标 | [detection_metrics.py](src/vision/evaluation/detection_metrics.py)、[workbench_comparison.py](src/sandbox/workbench_comparison.py) |
| 三个轻量样例 | [offline_examples.py](src/sandbox/offline_examples.py) |
| CLI、浏览器工作台与可选仿真 | [代码指南](docs/CODE_GUIDE.md) |
| 导出与安装共用源码清单 | [source_bundle.py](src/sandbox/source_bundle.py) |

[精选案例](docs/case-studies/README.md)包含遥测事件循环被来源核验阻塞、清理诊断拖延退出与日志队列溢出、共享地磁模型的地面姿态比较，以及 E21 的三种子开发与 56 帧合成测试摘要。保留失败、负面结果和未解问题。

```sh
python3 scripts/verify_case_studies.py
```

该命令核验公开摘录的身份和内部计算。原始 ULog、权重及完整数据未打包，因此不等于独立复现原飞行或训练。

## 可选浏览器 Demo

已有桌面工作区提供配方、受管任务和回执的确定性分类器演示，仅需 **Python 3.11+，无需安装依赖**：

```sh
./scripts/run_sandbox_app.sh
```

打开 [localhost:8765](http://127.0.0.1:8765/#results/acceptance)，选择 **Acceptance → Demo classifier → Create recipe and run**。终端 `Ctrl-C` 停止服务。这个距离与密度分类器和视觉样例相互独立。[完整步骤](docs/SANDBOX_QUICKSTART.md)。

## 环境和项目边界

| 工作 | 所需条件 |
| --- | --- |
| Demo、回执样例 | Python 3.11+ 标准库 |
| 数据审核、缓存比较 | `requirements-offline.txt` |
| 核心测试 | Python 3.12–3.14、`requirements-test.txt`（含 pytest）和 PATH 中的 Node.js |
| 真实训练与推理 | 核心及 `requirements-ml.txt`、自备审核数据和权重；现有图片推理经过 Ultralytics |
| 原生集成与飞行 | 外部 PX4/Gazebo、平台插件、输入资产和重新验收 |

本 **research-toolbox-20261102 源码快照**以 9 月 25 日公开版本为基线，加入精选发布改动及 10 月案例摘录，没有整体迁入作者后续飞控研究工作区。完整自主飞行扩展已暂停，公开方向为离线研究工作流与工程案例。没有真机或真实变电站可用性结论。

旧 **v0.6.4 macOS 预发布安装包**对应其原提交，本快照不包含新安装包。详见[公开范围](docs/PUBLICATION_SCOPE.md)、[版本说明](docs/RELEASE_SNAPSHOT_20261102.md)、[历史索引](docs/HISTORY_INDEX.md)及[高级环境说明](docs/RESEARCH_PREVIEW.md)。

## 检查与分发

```sh
python -m pip install -r requirements-test.txt
python scripts/run_tests.py --suite core --report outputs/core-tests.json
python examples/generate_fixtures.py --check
python scripts/export_source.py --output dist/research-toolbox-source.zip
```

核心集合还需 Git、Bash/POSIX Shell 和 `/usr/bin/python3`（保留的打包拒绝检查使用），并需 Node.js 执行三项 JavaScript 契约检查，本机验证及 CI 配置使用 26.4.0。测试入口收集 unittest、pytest 函数及参数化测试；请求模块收集零项或必需检查被跳过时，不能算通过。研究、原生与 SITL 集合显式选择，详见[验证说明](docs/VALIDATION.md)与[贡献指南](CONTRIBUTING.md)。导出与安装验收共用已跟踪源码清单，源码包提供逐文件哈希和提交来源，排除本地数据、权重、环境和运行输出。

[GitHub main CI](https://github.com/Xieyizhou/substation-uav-ml-research/actions/workflows/ci.yml)只表示线上主分支；本地候选的验证与尚未触发的 CI 在版本说明中分开记录。

项目代码与生成格式样例使用 [MIT](LICENSE)。外部包与资产保留原条款，包括可选 Ultralytics 的 AGPL-3.0。参见[来源](PROVENANCE.md)与[第三方说明](THIRD_PARTY_NOTICES.md)。
