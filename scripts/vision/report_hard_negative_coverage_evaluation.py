"""Render the verified development evaluation, without changing any decisions."""
import sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.evaluate_hard_negative_coverage import EVAL, KEYS, VARIANTS, NAMES, SEEDS, read, save, file_sha256, verify_tree


def build_report():
    path = EVAL / 'completion.json'
    verify_tree(path)
    c = read(path)
    records = {k: read(EVAL/f'{k}.json') for k in KEYS}
    p = lambda value: '未定义' if value is None else f'{100*value:.2f}%'
    lines = ['# 困难负例组成对照：固定开发评估', '',
        '六份预定终点权重均已完成正式 conf=0.37 和诊断 conf=0.001 推理。',
        f'状态：{c["status"]}；开发候选：{c["selected_candidate"] or "无"}。', '',
        '## 协议与解释边界', '',
        'CPU、640、类别相关 NMS IoU=0.7、max_det=300、同类一对一匹配 IoU≥0.5；未调阈值或挑选 seed。',
        '旧48张配对开发图（12个位姿，每类3个）与48张无目标开发图重复评估，不是盲测；三个seed不是新增独立样本。',
        'O/N均100步、600次图像曝光，正样本身份、顺序、位置一致，负例均108次。比较的是负例来源池扩展和组内曝光重新分配的组合效果，不是增加负例总曝光的纯因果效应。',
        '本轮没有训练步数对照，不能据此判断延长训练的效果。新增96张训练图不作为泛化测试。',
        '封存新场景未解封；training_admitted=false，promotable=false。', '',
        '## 三seed均值', '',
        '| 组 | 条件 | 计划命中 | 全图召回 | 匹配精度 | 未匹配预测均数 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for arm in ('O', 'N'):
        for v in VARIANTS:
            r = c['groups'][arm][v]
            lines.append(f'| {arm} | {v} | {p(r["planned_instance_hit_rate"]["mean"])} | {p(r["instance_recall"]["mean"])} | {p(r["matched_precision"]["mean"])} | {r["unmatched_predictions"]["mean"]:.3f} |')
    lines += ['', '| 组 | 无目标FPR均值 | 最差seed | 三seed FPR（7/17/27） |', '| --- | ---: | ---: | --- |']
    for arm in ('O', 'N'):
        r = c['groups'][arm]['no_target']['frame_false_positive_rate']
        lines.append(f'| {arm} | {p(r["mean"])} | {p(r["max_seed"])} | {" / ".join(p(x) for x in r["values"])} |')
    lines += ['', '## 逐seed、逐条件及逐类', '',
        '逐实例真值、预测、匹配分配、漏检归类及同位姿变体得失见 evaluation-v1 下六份单元JSON。以下逐类召回基于全图监督，计划命中只统计计划设备。', '',
        '| 单元 | 条件 | 类别 | 计划命中数/帧数 | 全图召回 | 匹配精度 | 未匹配预测 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for k in KEYS:
        for v in VARIANTS:
            for name in NAMES:
                r = records[k]['summary'][v]['per_class'][name]
                lines.append(f'| {k} | {v} | {name} | {r["planned_hits"]}/{r["planned_frames"]} | {p(r["instance_recall"])} | {p(r["matched_precision"])} | {r["unmatched_predictions"]} |')
    lines += ['', '## 能力保留与数值门禁', '', '下降容差为5个百分点；同时比较同预算R-100和历史A。所有失败项如下。', '',
        '| 组 | 指标 | 参考 | 实际值或召回差（百分点） | 要求 |', '| --- | --- | --- | ---: | --- |']
    for arm, gate in c['gates'].items():
        for check in gate['checks']:
            if check['passed']: continue
            if 'reference' in check:
                actual, requirement = f'{check["actual_delta"]*100:.2f}', '≥ -5.00 pp'
            else:
                actual, requirement = p(check['actual']), f'{check["op"]} {p(check["value"])}'
            lines.append(f'| {arm} | {check["metric"]} | {check.get("reference", "固定绝对门槛")} | {actual} | {requirement} |')
    lines += ['', '## N相对O的同位姿得失', '',
        '下表是三个seed的重复测量次数，不是独立图像数。', '',
        '| 条件 | 计划命中新增 | 计划命中丢失 | 保持 |', '| --- | ---: | ---: | ---: |']
    for v in VARIANTS:
        values = [r['planned_hit_delta'] for r in c['between_family_paired_deltas'] if r['variant'] == v]
        lines.append(f'| {v} | {values.count(1)} | {values.count(-1)} | {values.count(0)} |')
    lines += ['', '## 低阈值漏检诊断', '',
        '分类有优先级：同类低置信度→错类→定位不足→未发现合格保留预测。最后一类受NMS和max_det限制，不能解释为网络从未产生候选。诊断输出不替代正式评估。', '',
        '| 组 | 同类低置信度 | 错类 | 定位不足 | 未发现合格保留预测 |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('O', 'N'):
        counts = Counter(m['reason'] for k, r in records.items() if k.startswith(arm) for row in r['rows'] for m in row['misses'])
        lines.append('| '+arm+' | '+' | '.join(str(counts[x]) for x in ('low_confidence_same_class', 'wrong_class', 'localization', 'no_qualifying_retained_prediction'))+' |')
    lines += ['', '## 可复算性', '',
        f'O对历史Y-100的正式/低阈值全部逐框结果复现：{c["old_reference_prediction_reproduced"]}。',
        f'六单元计划目标独立命中与全图匹配冲突总数：{sum(r["matching_conflicts"] for r in records.values())}。',
        '完成回执递归绑定冻结协议、开发图/标签回执、审核、训练成员与曝光、六份权重、评估结果及测试。',
        '相关回归通过，v2.11固定40文件完整性通过。未重跑全仓测试；历史12项失败和1项跳过仍单独保留，不声明全仓通过。',
        '', '## 结论与下一步', '']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    print(build_report())
