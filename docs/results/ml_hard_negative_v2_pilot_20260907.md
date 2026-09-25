# 困难负样本隔离矩阵 v2 pilot（2026-09-07）

已生成 `hard-negative-isolated-v2`，仅替代 `visual-augmentation-240-v1` 的困难负样本子集；原有 144 张材质/光照正样本和 48 张常规正样本计划不变。

新矩阵包含 24 个冻结位姿及 normal、cool-low 两种光照，共 48 张候选。负样本世界从 complex canonical 世界派生，在 XML 层移除全部 transformer、switchgear、capacitor_bank 和 reactor 模型，保留普通柜体、控制建筑、杆塔、围墙、地面和网格。每对光照共享同一 `derivation_group`，因此只计为 24 个独立位姿组。

先行 pilot 选择 4 个共享位姿并渲染两种光照。8/8 张通过新版采集检查；逐帧 full_2d truth 的实例数和仿真实例标签数均为 0。AI 辅助审核确认画面质量可用，包含普通柜体、控制建筑、杆塔及围墙/场地背景，8 张均通过 pilot 审核。

该结论只证明隔离世界、双光照谱系、零目标门禁和审核链路可用，不自动准入余下 40 张，也不授权训练。整批采集后仍须逐图审核、文件与原尺寸像素去重、来源隔离及账本冻结。全部产物保持 `training_admitted=false`、`promotable=false`。

## 整批后续结果

通过 pilot 后，以 continuation 计划采集剩余 20 个位姿的双光照版本，40/40 张通过实际配置、相机和全图零目标 truth 检查。与已审核 pilot 合并后共 48 张、24 个完整派生组。

Codex 已查看剩余 5 页双光照联系表。48/48 张均通过 AI 辅助图像审核；文件哈希与解码后原尺寸像素联合去重未发现意外重复。每组双光照相似性登记为设计谱系，不作为独立场景膨胀。冻结账本记录 48 个已审核候选成员，但数据集级 `training_admitted` 和 `promotable` 仍为 false；A/B/C/D 训练继续等待正样本子集完成统一准入。
