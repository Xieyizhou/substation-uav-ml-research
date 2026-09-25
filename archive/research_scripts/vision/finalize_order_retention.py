"""Bind training-free readiness, complete coverage and a Chinese handoff report."""
from collections import Counter,defaultdict
from pathlib import Path
from scripts.vision.exposure_order_retention import *
from scripts.vision.preflight_order_retention import validate_ready
from scripts.vision.review_exposure_order_retention import validate_decisions

def reconstruct(batches,ids):
    return [member for index in ids for member in batches[index]]

def main():
    dest=OUT/'completion.json'
    if dest.exists():verify_tree(dest);print('COMPLETION_REVALIDATED');return
    p=validate_ready();ready=read(OUT/'ready.json');e=read(OUT/'evidence.json');review=read(OUT/'review.json')
    validate_decisions(e,review['decisions']);lookup={r['member_id']:r for r in p['pool_rows']}
    for row in lookup.values():
        actual=Counter(NAMES[int(line.split()[0])] for line in Path(row['label_path']).read_text().splitlines() if line.strip())
        if actual!=Counter(row['class_instances']):raise ValueError('Label-based exposure metadata mismatch')
    details=[]
    for seed in (7,17,27):
        prior=read(PRIOR/'protocol.json')['schedules'][f'retained_appearance-450-{seed}']
        b,ids=permutation(prior,p['variant_by_member'],seed)
        a=p['schedules'][f'staged-450-{seed}'];i=p['schedules'][f'interleaved-450-{seed}']
        if a!=prior or i!=reconstruct(b,ids) or Counter(a)!=Counter(i):raise ValueError('Sequence not reproducible')
        if p['batch_orders'][f'interleaved-450-{seed}']!=ids:raise ValueError('Batch indices changed')
        validate_permutation(b,ids)
        n=sum(any(p['variant_by_member'][m] in ('neutral_bridge','background_bridge') for m in batch) for batch in b)
        details.append(dict(seed=seed,appearance_batches=n,other_batches=450-n,unique_members=len(set(a))))
    if (OUT/'training').exists():raise ValueError('Training unexpectedly started before stage completion')
    grouped=defaultdict(list)
    for event in e['negative']:grouped[event['image_sha256']].append(event)
    index=OUT/'review-index.md'
    lines=['# 本轮唯一图像审核索引','', '以下每个图像组仅计一次；预测框、跨 seed 重复及光照派生不增加独立场景数。','']
    for digest,events in grouped.items():
        lines += [f'## 图像 {digest[:16]}','',f'原图：[打开]({events[0]["image_path"]})','']
        for ev in events:lines.append(f'- [{ev["event_id"]}]({ev["evidence_path"]})：{ev["cell"]}，{ev["variant"]}，{ev["prediction"]["class_name"]}，{ev["prediction"]["confidence"]:.4f}')
        lines.append('')
    lines += ['## 全部电抗器实例上下文','']
    for ev in e['reactors']:lines.append(f'- [{ev["event_id"]}]({ev["evidence_path"]})：{ev["variant"]}，实例标签 {ev["instance_label"]}，640 输入短边约 {ev["short_side_at_640"]:.1f} 像素，含三个 seed 两组全部预测明细。')
    if index.exists():raise ValueError('Preserve incomplete prior report attempt')
    index.write_text('\n'.join(lines)+'\n')
    report=ROOT/'docs/results/ml_exposure_order_retention_preflight_20260909.md'
    if report.exists():raise ValueError('Do not overwrite report')
    report.write_text(f'''# 曝光顺序对照：训练前完成报告

状态：`ready_for_training_not_started`。本阶段未创建训练优化器、未反向传播、未运行训练验证；未生成新训练权重。所有新产物 training_admitted=false、promotable=false，封存新场景未评估。

## 检测覆盖与事实

- 本轮 24 个无目标误检框、12 张唯一图像全部逐框 AI 辅助审核，保留 seed、条件、坐标、置信度、图像与证据哈希。观察为柜体 11、杆体 4、混合结构 9；混合结构主要包括天空边界、围墙及杆体，不能将全部误检归为柜体混淆。
- 原始、光照条件下全部 10 个电抗器图像-实例上下文，覆盖 30 次 seed 对照和两组所有预测。新增损失 9、得益 4、保持命中 8、持续漏检 9。9 个损失按既有操作性诊断为低置信度 7、错类 1、无合格保留预测 1。
- 对应真值以仿真实例标签、类别和原图坐标联合核验，不依赖数组下标。未发现两组身份／标签坐标冲突；可见性判断不认证唯一类别或像素级可见区域。
- 电抗器观察既有清晰近景圆柱，也有右缘截断、远处遮挡及前景设备遮挡。10 个上下文中的实例标签均为 0208，不可视为 10 个独立设备资产。
- 例如清晰近景圆柱在 seed 7 下有同类 IoU 约 0.873、confidence 约 0.128 的保留框；seed 17 在同一原始图却没有满足诊断条件的保留预测。受 NMS/max_det 限制，后者不能解释为网络从未产生候选。

## 冻结实验与比较边界

两组各 seed 7/17/27，从 v2.11 独立初始化。每单元 450 个优化步、2700 次图像曝光，CPU640、batch/nbs6、AdamW 恒定学习率0.001、无预热、增强关闭，固定终点权重。

分阶段组逐项复现上一轮外观组序列；交错组仅改变完整六图批次的顺序，不拆分批次、不改变批内顺序。三个 seed 的外观批次数分别为 99、93、96，采用累计比例法均匀穿插。成员多重集、全图类别实例曝光、谱系曝光、批次组成保持一致，每50步账本独立保存。

该对照检验曝光顺序，不预设交错一定有效。不同窗口的类别曝光分布是重排的伴随变化，账本明确记录；不额外重采样或挑选排列。比较参考保留同预算 retained_reference-450 和历史 A，分阶段组不是能力保留参考；原有数值阈值不变。

## 实现与验收

六配置真实训练数据集与加载器均遍历2700次采样、450个批次，完整标签通过；预检与训练复用同一构造入口，并用运行时保护拒绝优化器创建和反向传播。训练入口启动必须消费有效 ready.json，重新校验输入链；默认仅检查状态，必须显式 --train 才能训练。

新增顺序、审核、默认禁止训练、取消清理和重试上限测试；最终相关回归记录：

```text
{ready['regression_output'].strip()}
```

v2.11 固定40文件完整性通过。未运行全仓测试，不声明全仓测试通过。预检中为绑定最新测试版本主动中断一次，失败尝试独立保存；恢复复用完整且哈希有效的前三配置，重跑未完成配置，未混接训练状态。

## 结论与交付

规定范围内错误覆盖完成，未发现需要先修改历史标签的已证实冲突。遮挡、截断、低置信度及背景结构混淆仍是模型风险，不等于已经解决。顺序对照具有同成员、同批次组成的可比较性，现可进入未来显式训练；本阶段不启动。

- [逐图审核索引]({index})
- [冻结协议]({OUT/'protocol.json'})
- [训练前回执]({OUT/'ready.json'})

预检：`.venv/bin/python -m scripts.vision.preflight_order_retention`。
只读状态：`.venv/bin/python -m scripts.vision.train_order_retention`。
未来显式训练：`.venv/bin/python -u -m scripts.vision.train_order_retention --train`。
''')
    import torch,ultralytics
    paths=[OUT/'ready.json',OUT/'protocol.json',OUT/'review.json',OUT/'evidence.json',index,report,Path(__file__)]
    frozen(dest,dict(status='ready_for_training_not_started',training_started=False,coverage=dict(negative_events=24,negative_unique_images=12,
        reactor_contexts=10,reactor_seed_comparisons=30),batch_design=details,software=dict(torch=torch.__version__,ultralytics=ultralytics.__version__),
        report_path=str(report),inputs={str(path):file_sha256(path) for path in paths}))
    print('STAGE_COMPLETE_TRAINING_NOT_STARTED',report,flush=True)

if __name__=='__main__':main()
