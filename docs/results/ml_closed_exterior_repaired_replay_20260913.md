# 修复后原帧重放核验

## 已完成

用户确认新增每帧最多三次技术重放预算后，在独立 `original-replays-repaired-v1` 目录执行。旧48次失败记录保留。16个原始帧全部在第1次新尝试通过，未消耗第2、3次预算。

共核对48个连续稳定帧：原帧RGB不同像素数为0，原有框最大坐标差为0像素，最大同步差约33毫秒，位姿和相机合法性检查通过。实例映射核验未发现有目标实例像素但没有对应标签框的情况。每次启动的进程均已清理。

这只认证16个原始帧；不能将原帧像素认证直接传播给48个材质变体，也不证明所有内容足够用于训练。现有逐图AI辅助内容审核继续有效，未自动生成新的通过决定。

18项针对性回归通过；v2.11固定40文件完整性通过。不声明全仓测试通过。

## 仍需处理的具名缺口

以下四个位姿及其全部变体继续暂缓，不修改或删除历史标签：

- layout-A:closed:reactor:3：非计划开关柜下半部遮挡。
- layout-A:closed:capacitor_bank:2：计划电容器组下部严重遮挡。
- layout-A:closed:switchgear:2：非计划电抗器较大下部遮挡。
- layout-B:closed:transformer:1：非计划电容器组被变压器遮挡，仅上部较清楚。

剩余12个位姿不能冒称原定16个全部合格。尤其B布局失去唯一的计划变压器位姿，尚不能冻结保持两布局四类覆盖的训练对照。B布局电容器位姿中的非计划变压器不能不作说明就替代计划来源。

## 下一步边界

建议仅为B布局增加一个固定变压器位姿及原始/暖/冷/中性四张候选，保持布局、资产、材质参数和训练预算不变。新增4张超出本次已冻结64张矩阵，因此需确认这个有界扩展；不是再次申请已授权的重放预算。新位姿必须采集前冻结并通过全图审核及实例核验，不依据模型成绩筛选。

该缺口解决后再执行来源隔离、精确30个可替换位置冻结、六个真实加载预检和训练启动。当前未训练，未签发训练就绪；所有新产物保持 training_admitted=false、promotable=false。

主要可核验输入位于 `data/research/ml_training_recovery_v1/source-isolated-material-retention-v1/closed-exterior-coverage-v1/`：`visual-review.json`、`original-replays-repaired-v1/retry-authorization.json`、`original-replays-repaired-v1/completion.json` 及各单元重放回执。
