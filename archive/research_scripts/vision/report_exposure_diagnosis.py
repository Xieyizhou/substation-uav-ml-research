"""Write the Chinese research report only from verified completion evidence."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, NAMES, SEEDS, read, save, verify, file_sha256


def main():
    completion_path = OUT/'completion.json'
    result = read(completion_path)
    verify(result)
    review = read(OUT/'visual-review-v2.json')
    counts = Counter(r['content_category'] for r in review['decisions'])
    report = ROOT/'docs/results/ml_exposure_controlled_diagnosis_20260907.md'
    lines = ['# 曝光受控训练对照与错误诊断', '',
        '本实验使用现有冻结数据，完成 R/X/Y × 100/300 步 × seed 7/17/27 共18个独立训练单元。'
        '正式推理和 confidence=0.001 诊断覆盖本轮18份权重及历史A/D各3份权重。', '',
        f'阶段状态：`{result["status"]}`；开发候选家族：`{result["selected_family"]}`。'
        '封存新场景未采集、未评估，未访问保护标签，全部产物 training_admitted=false、promotable=false。', '',
        '## 冻结设计', '',
        '| 组别 | 基础池 | 常规正样本 | 桥接正样本 | 困难负例 |',
        '| --- | ---: | ---: | ---: | ---: |',
        '| R | 330 | 270 | 0 | 0 |', '| X | 216 | 156 | 156 | 72 |', '| Y | 216 | 156 | 120 | 108 |', '',
        '表为每600次图像曝光的配额，300步为三倍。X/Y每600次只有36次外观/负例交换，其他成员和位置一致。'
        '100/300步采样前缀一致，均从v2.11初始化，CPU/640/batch6/AdamW/恒定lr0=0.001，无预热和数据增强。'
        '采用终点权重；训练成员上的末轮验证只是拟合诊断。', '',
        '## 正式开发评估', '',
        '以下为三个seed均值；每个条件只有12个位姿，每类3个位姿，seed重复不增加独立样本数。'
        '固定confidence=0.37、同类NMS IoU=0.7、max_det=300、匹配IoU≥0.5。', '',
        '| 家族 | 原始计划命中 | 材质计划命中 | 背景计划命中 | 光照计划命中 | 原始全图召回 | 光照全图召回 | 无目标FPR |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for family in ('R-100', 'X-100', 'Y-100', 'R-300', 'X-300', 'Y-300'):
        a = result['aggregate'][family]
        values = [a[v]['planned_instance_hit_rate']['mean'] for v in ('original', 'material', 'background', 'lighting')]
        values += [a[v]['instance_recall']['mean'] for v in ('original', 'lighting')]
        values += [a['no_target']['frame_false_positive_rate']['mean']]
        lines.append('| ' + family + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', '| 家族 | 材质计划命中seed范围 | 无目标FPR seed范围 |',
              '| --- | ---: | ---: |']
    for family, a in result['aggregate'].items():
        material = a['material']['planned_instance_hit_rate']
        negative = a['no_target']['frame_false_positive_rate']
        lines.append(f'| {family} | {material["min_seed"]:.3f}–{material["max_seed"]:.3f} | {negative["min_seed"]:.3f}–{negative["max_seed"]:.3f} |')
    lines += ['', '## 受控差异与门禁', '',
        'X/Y差异是固定预算下交换外观与负例曝光的结果，不能解释成纯增加负例的因果效应。'
        '均值差异是本开发集方向性结果，不是统计显著性结论。', '',
        '| 对照（后者减前者） | 材质命中差 | 原始全图召回差 | 光照全图召回差 | 无目标FPR差 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for contrast, delta in result['contrasts'].items():
        values = [delta['material']['planned_instance_hit_rate'], delta['original']['instance_recall'],
                  delta['lighting']['instance_recall'], delta['no_target_fpr']]
        lines.append('| ' + contrast + ' | ' + ' | '.join(f'{v:+.3f}' for v in values) + ' |')
    for family, policy in result['policy_results'].items():
        failed = [c for c in policy['checks'] if not c['passed']]
        lines += ['', f'### {family}：' + ('通过' if policy['passed'] else f'未通过，共{len(failed)}项门槛失败'), '']
        for check in failed:
            if 'reference' in check:
                lines.append(f'- {check["metric"]} 相对 {check["reference"]}：{check["actual_delta"]:+.3f}，允许最低−0.050。')
            else:
                lines.append(f'- {check["metric"]}：{check["actual"]:.3f}，要求 {check["op"]} {check["value"]:.3f}。')
    lines += ['', '## 低阈值漏检诊断', '',
        '按每个未匹配真值实例依次判断低置信度同类、错类、定位不足、未发现满足条件的保留预测。'
        'NMS及max_det仍限制候选，第四类不能证明网络未生成候选；匹配竞争另外记录。', '',
        '| 家族 | 条件 | 低置信度同类 | 错类 | 定位不足 | 无满足条件预测 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for family in ('historical-D', 'R-100', 'X-100', 'Y-100', 'R-300', 'X-300', 'Y-300'):
        for variant in ('material', 'lighting', 'original'):
            reasons = Counter()
            for seed in SEEDS:
                evaluation = read(OUT/'evaluation'/f'{family}-{seed}.json')
                reasons.update(evaluation['summary'][variant]['miss_reasons'])
            values = [reasons[k] for k in ('low_confidence_same_class', 'wrong_class', 'localization', 'no_qualifying_retained_prediction')]
            lines.append('| ' + family + ' | ' + variant + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## 逐框AI辅助审核', '',
        '历史D的67个误检框来自30张不同图像、20个位姿。已逐框查看全图上下文及裁剪，结果按实际框内内容登记，'
        '非按采集主题推断。图像、叠框证据、预测坐标、seed、置信度、理由及审核时间均已绑定。', '',
        '| 实际内容 | 框数 |', '| --- | ---: |']
    for category, count in counts.most_common():
        lines.append(f'| {category} | {count} |')
    lines += ['', '## 对三个问题的回答与下一步', '']
    for steps in (100, 300):
        d = result['contrasts'][f'Y-{steps}_minus_X-{steps}']
        lines.append(f'- {steps}步加强负例曝光：FPR变化{d["no_target_fpr"]:+.3f}，'
                     f'材质计划命中变化{d["material"]["planned_instance_hit_rate"]:+.3f}，'
                     f'光照全图召回变化{d["lighting"]["instance_recall"]:+.3f}。'
                     '这是曝光交换的组合效果，不能只用FPR一项判断整体改善。')
    for arm in 'RXY':
        d = result['contrasts'][f'{arm}-300_minus_{arm}-100']
        lines.append(f'- {arm}延长至300步：原始全图召回变化{d["original"]["instance_recall"]:+.3f}，'
                     f'材质计划命中变化{d["material"]["planned_instance_hit_rate"]:+.3f}，'
                     f'光照全图召回变化{d["lighting"]["instance_recall"]:+.3f}，FPR变化{d["no_target_fpr"]:+.3f}。')
    lines += ['', '原有能力是否保留，以双参考全图/逐类门禁为准，不能用计划目标命中替代。'
              '上述结果也表明训练损失下降不等于鲁棒性同步提高。低阈值诊断将置信度不足、错类和定位不足分开，'
              '但它不是阈值调优，更不证明其中某一类是唯一根因。', '']
    if result['selected_family'] is None:
        lines += ['下一步优先级：', '',
                  '1. 依据逐框证据补充新位姿下的灰色建筑/箱体、地面网格与投影、画面边缘截断结构负例；'
                  '保留旧开发图的评估角色，不将其转入训练。先做小批来源匹配与审核。',
                  '2. 保持明确的共同基础曝光下限和负例配额，针对电容器组、电抗器较少的全图实例监督，'
                  '单独冻结一次类别/实例曝光平衡对照；不把计划类别计数当作监督平衡。',
                  '3. 若采样和补缺后仍有置信度/错类问题，再分别对照温和光度增强、较低学习率或部分骨干冻结。'
                  '每轮只增加一个变化，保留原始条件回归；现有结果不足以优先支持更大模型。',
                  '4. 新场景继续封存。通过开发门禁后再执行冻结的新场景复验，检验当前场景收益能否迁移。', '']
    else:
        lines += [f'下一阶段优先使用已冻结新场景检验 `{result["selected_family"]}` 的三个seed，'
                  '确认方向性收益能否迁移；本阶段仍保持封存，尚无跨场景结论。', '']
    lines += ['', '## 验证与边界', '',
        '候选必须通过旧计划命中/FPR门槛，以及原始和光照的全图、逐类召回相对同预算R和历史A'
        '均下降不超过5个百分点的门槛。所有计划目标与一对一匹配冲突必须清零才能完成。', '',
        '基础与常规池沿用成员来源，成员ID不自动等同独立场景；新增桥接样本保留原始派生组。'
        '48张配对图和48张无目标图均为已查看开发集，不能用于跨站点或唯一根因结论。', '',
        f'协议身份：`{result["protocol_identity"]}`。',
        f'闭环身份：`{result["identity"]}`。', '',
        '数值明细、逐类结果、seed波动、配对得失、曝光与输入哈希位于独立实验目录。', '']
    lines += ['验证记录：目标回归34项通过，v2.11固定40文件完整性通过，git diff --check通过。'
              '未重跑全仓测试；此前全仓876项中的12项失败和1项跳过仍单独保留，不声明全仓通过。', '',
              f'100/300步训练损失前缀完全一致的配对数：{sum(r["first_100_step_losses_identical"] for r in result["training_loss_comparison"].values())}/9。', '',
              '来源位姿组数：基础池66、常规正样本48、桥接正样本16、困难负例12。'
              '这些是来源位姿身份，不能解释为142个独立场景。', '']
    lines += ['审核工具勘误：首版仅按view_id索引，混合了同位姿的两种光照，收尾门禁因预测编号重复而拒绝。'
              '已改用view_id+variant，重新逐图审核全部30张图像和67个框。旧证据保留为superseded，'
              '有效审核为visual-review-v2.json，详细说明见review-index-erratum.json。训练和推理未因此改变。', '']
    report.write_text('\n'.join(lines))
    save(OUT/'report-receipt.json', {'status': 'report_written', 'inputs': {str(completion_path): file_sha256(completion_path),
         str(report): file_sha256(report), str(Path(__file__)): file_sha256(Path(__file__))}})
    print(report)


if __name__ == '__main__':
    main()
