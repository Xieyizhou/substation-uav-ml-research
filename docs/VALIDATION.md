# 验证入口与证据边界

源码候选的验证分为无第三方依赖 Demo、轻量离线样例、便携核心检查，以及显式选择的研究检查。通过前一层不代表后续各层已经通过。历史飞行和模型结果也不会自动成为当前源码的资格。

## 无第三方依赖检查：Python 3.11+

在仓库根目录执行，输出目录必须尚不存在；重新运行时使用新的目录名。

```sh
python main.py sandbox --profile demo bootstrap
python examples/generate_fixtures.py --check
python examples/receipt_check/run.py --output outputs/examples/receipt-check
python main.py sandbox --profile demo release-gate --output outputs/sandbox/demo/release-gate/local
python main.py sandbox --profile demo release-gate-inspect --input outputs/sandbox/demo/release-gate/local/release_gate.json
python main.py sandbox --profile demo beta-install-gate --output outputs/sandbox/demo/beta-install/local
python main.py sandbox --profile demo beta-install-gate-inspect --input outputs/sandbox/demo/beta-install/local/beta_install_gate.json
```

Demo 使用构造输入。回执样例同时检查原始输出有效和修改后的副本被拒绝，不会修改历史证据。HTTP 服务和安装门禁中的服务检查需要允许绑定本机回环端口；权限拒绝必须保留为实际失败，不能当作通过。

## 轻量视觉离线样例：Python 3.11+

```sh
python -m venv .venv-offline
.venv-offline/bin/python -m pip install -r requirements-offline.txt
.venv-offline/bin/python examples/dataset_audit/run.py --output outputs/examples/dataset-audit
.venv-offline/bin/python examples/prediction_comparison/run.py --output outputs/examples/prediction-comparison
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/dataset-audit/receipt.json
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/prediction-comparison/receipt.json
```

这层只安装 Pillow 和 PyYAML。输入是项目自生成的格式图像、标签及手工构造的缓存预测，没有下载模型或重新推理。数据审核检验结构、类别和重复内容；比较结果演示指标计算；回执检验记录和输出字节的一致性。它们均不证明真实设备识别精度、标注正确、独立站点泛化或飞行安全。输出与限制见[样例说明](../examples/README.md)。

## 便携核心检查：Python 3.12–3.14

```sh
python -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python scripts/run_tests.py --suite core --report outputs/core-tests.json
```

即使从源码 ZIP 运行，核心集合也需要 Git CLI 和 Bash/POSIX Shell 工具。一项保留的 macOS 打包拒绝契约会先调用 `/usr/bin/python3` 读取版本，因此该路径也必须可执行；它与选择的测试虚拟环境 Python 分开。GitHub Ubuntu/macOS 宿主提供这些工具，精简容器需自行准备。

核心集合还需 PATH 中的 Node.js 执行三项 JavaScript 契约检查；本机使用 26.4.0，CI 通过 [setup-node](https://github.com/actions/setup-node) 显式安装同版本，不需 npm 依赖。缺失 Node 导致必需测试跳过，不能算通过。Demo 和三个轻量样例不需要 Node。

核心集合需要 Python 3.12+：保留的几何适配器测试会导入使用新版语法的研究辅助脚本。测试使用临时数据和模拟接口，不会启动真实飞行，也不需要作者的历史训练数据、模型权重、Torch、ONNX、SciPy、PX4 或 Gazebo。安装后的测试环境包含 NumPy、Pandas、Matplotlib、Pillow、PyYAML、OpenCV、psutil 和 MAVSDK 等依赖；直接依赖及约束以依赖文件为准。MAVSDK 固定为 `3.17.2`，对应此源码的 gRPC 接口。

执行器使用显式依赖 `pytest>=9,<10`，同时收集 `unittest.TestCase`、顶层测试函数和参数化测试。JSON 报告记录实际收集、执行、失败、错误、跳过及逐模块数量。pytest 的子测试数量可能单独显示，不应与顶层收集项混加。任何选中模块收集零项、必要测试跳过或空集合均不能通过。`--list` 仅列出模块，不执行测试。

`config/sandbox/test_suites.json` 将例外分别放在 `research_modules`、`native_modules`、`sitl_modules`，新测试默认进入 core，重复、交叉或不存在的模块会导致选择失败。当前 native 和 sitl 清单为空，表示此发布快照未选择独立原生或 SITL 测试；空集合不代表验收完成。core 内保留的飞行契约和模拟接口测试也不等于执行原生飞行。

## 历史研究复现：需要额外资产

```sh
.venv-test/bin/python -m pip install -r requirements-research.txt
.venv-test/bin/python scripts/run_tests.py --suite research --report outputs/research-tests.json
```

research 集合包含冻结的实验适配器、真实小型训练/导出和历史审核回执检查，需要原始数据、模型及回执声明的环境。它逐模块启动独立进程，避免历史适配器的全局目录配置互相污染。缺失依赖或数据应报告失败，不以跳过替代通过。`--suite all` 同样保留模块隔离，并包含所有显式组；仅下载源码通常不具备它所需的资产。

不要使用 `python -m unittest discover` 的结果声称已经执行完整发布集合：它不能收集顶层 pytest 函数和参数化测试。也不要为恢复历史结果改写冻结文件的哈希。

## 源码分发与 CI

```sh
python scripts/export_source.py --output outputs/research-toolbox-source.zip
unzip outputs/research-toolbox-source.zip -d outputs/research-toolbox-source
```

导出默认只包含共享清单允许的已跟踪文件，Beta 安装复制也使用同一清单。`--include-untracked` 只适合明确选择的开发快照。源码 ZIP 带有逐文件大小和 SHA-256 的 `SOURCE_MANIFEST.json`，以及绑定提交和清单身份的 `SOURCE_PROVENANCE.json`；不包含 `.git`、虚拟环境、模型权重或原始实验数据。清单校验可以发现字节变化，不能替代可信来源或数字签名。使用支持 Unix 文件权限的工具解压；某些程序化解压接口不恢复可执行位。

[Offline validation 工作流](../.github/workflows/ci.yml)配置了以下任务：

| 任务 | 配置的平台与 Python | 检查内容 |
| --- | --- | --- |
| 无第三方依赖 Demo | Ubuntu；3.11、3.13 | Demo、格式图像重生成对照、回执修改拒绝、发布门禁与独立安装 |
| 便携核心 | Ubuntu；3.12、3.13、3.14 | 实际 pytest 核心集合、编译、Shell 语法与 A-star 预览 |
| 源码包和轻量样例 | Ubuntu、macOS 15；3.11、3.13 | 导出与解压清单一致、仅轻量依赖、从解压源码运行三个样例 |
| 原生 macOS 应用 | macOS 15 | 既有 Swift 测试、构建、元数据、签名与预览打包检查 |

表格描述工作流配置，不是已通过的结果。未上传的本地候选不会触发 GitHub Actions；公开 main 分支或旧安装包的绿色检查不能证明新候选通过。准备源码时应另外保存实际本机平台、Python 版本、命令、报告和源码身份；Linux 与应用构建只有取得相应实际运行结果后才能列为通过。

## 历史证据、模型与飞行资格

精选[工程案例与模型摘要](case-studies/README.md)记录过去实验的观测和限制，其中小型片段可离线复查，完整原始数据、模型、ULog 和本机工具链不随源码提供。摘要检查不等于重新运行实验。

原有 `sandbox evidence-freeze`、`evidence-inspect` 和 `evidence-flight-verify` 支持绑定直接依赖的历史证据包。清单身份应在包外可信保存，不能从待验证包内取出哈希便据此证明来源可信。历史包及固定 SITL 所需资产不属于默认下载；更改源码、模型、场景、原生二进制或关键环境后，必须重新执行对应验收。

Workbench 的导入、反馈审核、开发集阈值选择和结果回执可阅读并按自身条件使用；数据注册成功不证明物理场景独立，ONNX 可读或数值一致不证明质量提升，历史任务完成不授予新源码、真机或未验收场景飞行资格。当前默认入口是离线工具与样例；完整训练和飞行复现的额外条件见[发布范围](PUBLICATION_SCOPE.md)。
