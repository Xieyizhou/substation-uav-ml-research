"""Summarize completed fixed-endpoint experiments without selecting favorable seeds."""
from collections import Counter
from pathlib import Path
from scripts.vision.run_matched_appearance_training import OUT, ROOT, KEYS, read, save, file_sha256, verify_tree
from scripts.vision.freeze_full_image_training import REFERENCE

def main():
    dest=OUT/'analysis.json';completion=OUT/'completion.json'
    receipt=OUT/'report-receipt.json'
    if receipt.exists():verify_tree(receipt);print('VERIFIED_EXISTING',receipt);return
    if dest.exists():verify_tree(dest)
    verify_tree(completion);c=read(completion);p=read(OUT/'protocol.json')
    if set(c['completed_cells'])!=set(KEYS):raise ValueError('Missing training cells')
    verify_tree(OUT/'evaluation-regression.json')
    verify_tree(OUT/'negative-review.json');negative_review=read(OUT/'negative-review.json')
    if negative_review['status']!='all_formal_negative_predictions_explicitly_reviewed':raise ValueError('Unresolved negative review')
    inputs={str(x):file_sha256(x) for x in (completion,OUT/'protocol.json',OUT/'evaluation-regression.json',OUT/'negative-review.json',Path(__file__))}
    old=read(REFERENCE/'protocol.json')
    inputs[str(REFERENCE/'protocol.json')]=file_sha256(REFERENCE/'protocol.json')
    supervision={str(s):dict(I=old['exposures'][f'I-300-{s}']['class_instance_exposure'],
        K=p['exposures'][f'K-300-{s}']['class_instance_exposure'],L=p['exposures'][f'L-300-{s}']['class_instance_exposure']) for s in (7,17,27)}
    diagnostics={};perseed={};losses={};miss_summary={a:{v:Counter() for v in ('original','material','background','lighting')} for a in ('K','L')}
    import csv
    for key in KEYS:
        ep=OUT/f'evaluation-{key}.json';r=read(ep);inputs[str(ep)]=file_sha256(ep)
        diagnostics[key]=dict(Counter(':'.join((x['variant'],m['class_name'],m['reason'])) for x in r['rows'] for m in x['misses']))
        for row in r['rows']:miss_summary[key[0]][row['variant']].update(m['reason'] for m in row['misses'])
        perseed[key]=dict(summary=r['summary'],negative_summary=r['negative_summary'])
        cp=OUT/key/'completion.json';cell=read(cp);lp=Path(cell['exposure_path']).parent/'results.csv'
        inputs[str(lp)]=file_sha256(lp)
        with lp.open() as stream:rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
        losses[key]=dict(first={k:v for k,v in rows[0].items() if k.startswith('train/')},
                        last={k:v for k,v in rows[-1].items() if k.startswith('train/')},role='training_fit_only')
    groups=c['aggregate'];delta={}
    for v in ('original','material','background','lighting'):
        delta[v]={m:(groups['L-300'][v][m]['mean']-groups['K-300'][v][m]['mean']
                    if groups['L-300'][v][m]['defined_seed_count']==3 and groups['K-300'][v][m]['defined_seed_count']==3 else None)
                  for m in ('planned_instance_hit_rate','instance_recall','matched_precision','unmatched_predictions')}
    delta['no_target']={m:groups['L-300']['no_target'][m]['mean']-groups['K-300']['no_target'][m]['mean'] for m in ('frame_false_positive_rate','unmatched_predictions')}
    analysis=read(dest) if dest.exists() else save(dest,dict(status='quantitative_development_diagnosis_complete',result_status=c['status'],
        selected_family=c['selected_family'],L_minus_K=delta,per_seed=perseed,miss_events=diagnostics,miss_summary=miss_summary,
        loss_endpoints=losses,paired_changes=c['paired_K_to_L'],historical_supervision_comparison=supervision,
        negative_review_summary=dict(prediction_events=negative_review['prediction_events'],unique_images=negative_review['unique_images'],
            content_categories=dict(Counter(x['content_category'] for x in negative_review['decisions']))),inputs=inputs,
        limits=['Single seen scene; correlated poses and variants; seed events are not new independent samples.',
          'L jointly changes material/light exposure; no isolated-factor causal identification.',
          'Prediction-box content descriptions are AI-assisted visual observations, not instance-mask certification.']))
    report=ROOT/'docs/results/ml_matched_appearance_training_20260908.md'
    lines=['# 主体可见配对外观训练：六单元结果','',
      f"六个固定终点单元全部完成，结论：{'形成开发候选 L-300（保留三个 seed），不晋升。' if c['selected_family'] else '没有家族通过全部开发门禁，不晋升。'}",'',
      'K 为新位姿原始图，L 为同位姿材质／光照变体。共同基础、常规、困难负例成员及曝光位置一致，全图类别实例曝光一致；每单元 300 步、1800 次图像曝光，seed 7/17/27，v2.11 初始化、CPU 640、AdamW 恒定 0.001。只用末轮权重。','',
      '## 三 seed 平均结果','',
      '| 指标 | K-300 | L-300 | L−K |','| --- | ---: | ---: | ---: |']
    labels={'original':'原始','material':'材质','background':'背景','lighting':'光照'}
    for v,label in labels.items():
        for m,name in [('planned_instance_hit_rate','计划命中'),('instance_recall','全图召回'),('matched_precision','匹配精度')]:
            sa=groups['K-300'][v][m];sb=groups['L-300'][v][m]
            a=sa['mean'];b=sb['mean']
            fmt=lambda value,n:('未定义' if value is None else f'{value:.6f}')+(f'（{n} seed 有定义）' if n!=3 else '')
            change=delta[v][m]
            text='不计算完整三 seed 差值' if change is None else f'{change*100:+.2f} 个百分点'
            lines.append(f'| {label} {name} | {fmt(a,sa["defined_seed_count"])} | {fmt(b,sb["defined_seed_count"])} | {text} |')
    m='frame_false_positive_rate';a=groups['K-300']['no_target'][m]['mean'];b=groups['L-300']['no_target'][m]['mean']
    lines.extend([f'| 无目标帧 FPR | {a:.6f} | {b:.6f} | {(b-a)*100:+.2f} 个百分点 |','',
      '## 各 seed 与波动','', '| 单元 | 原始计划命中 | 材质 | 背景 | 光照 | 无目标 FPR |','| --- | ---: | ---: | ---: | ---: | ---: |'])
    for key in KEYS:
        r=perseed[key];values=[r['summary'][v]['planned_instance_hit_rate'] for v in labels]+[r['negative_summary']['frame_false_positive_rate']]
        lines.append('| '+key+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines.extend(['','无预测时匹配精度未定义，不填成 0 或 1；不足三个 seed 有定义时注明有效数量，不计算完整三 seed 精度差值。各条件全图／逐类召回、匹配精度、未匹配预测的均值、最差 seed、标准差及全部 seed 值保存在完成回执 aggregate；逐实例正式与低阈值预测保存在各 evaluation 文件。','',
       '## 未通过门禁',''])
    for family,result in c['policy_results'].items():
        lines.append(f'### {family}');lines.append('')
        failed=[x for x in result['checks'] if not x['passed']]
        if not failed:lines.extend(['全部该家族门禁通过。','']);continue
        for rule in failed:
            if 'actual_delta' in rule:
                lines.append(f"- {rule['metric']} 相对 {rule['reference']}：{rule['actual_delta']*100:+.2f} 个百分点，要求不少于 {rule['minimum_delta']*100:.2f} 个百分点。")
            else:lines.append(f"- {rule['metric']}：{rule['actual']:.6f}，要求 {rule['op']} {rule['value']}。")
        lines.append('')
    lines.extend(['## 配对与诊断解释','',
      '相同 pair_id 的 K→L 得失、低置信度／错类／定位不足等操作性漏检分类，以及训练损失端点，见独立 analysis.json。低阈值 0.001 诊断受 NMS 与 max_det 限制，不表示网络从未产生其他框；不替代 confidence 0.37 的正式结果。','',
      'L 的材质组共有 165 次漏检事件：111 次未见满足分类条件的保留预测、22 次同类低置信度、26 次错类、6 次定位不足。光照组 112 次漏检中，同类低置信度为 55 次。这里是三个 seed 的重复预测事件，不是新增独立实例。材质与光照的错误构成不同，不能统一解释为一个阈值问题。','',
      '### 无目标误检逐框观察','',
      '8 个正式误检事件涉及 7 张唯一图像，均已查看全图叠框与框内放大，记录 AI辅助审核、理由及图像／证据哈希。框内主要内容均为柜状结构：L-300-27 的 6 个框将蓝绿色柜状主体（有面板或无面板视角）预测为开关柜，其中 3 框置信度超过 0.84；另两次为灰色柜状结构分别被预测为变压器、电容器。不能依据场景中有杆而将这些误检归类为杆体。','',
      '同一张图在 K-300-27 与 L-300-27 中被框住的是不同柜体，不能将其当成同一对象跨 seed 一致误检。此证据支持优先核查普通柜体与目标设备在主体、面板、底座等可见特征上的可区分性；不证明二者几何完全相同，也不自动推导唯一根因。','',
      '本轮只回答固定曝光预算下，使用这批同位姿外观／光照变体相对原始图的变化。不能单独确定材质或光照的唯一病因，也不能凭 seed 平均改善宣称统计显著或跨站点泛化。','',
      '### 相对历史 I 的监督差异','',
      'K/L 每单元全图实例曝光为变压器 1278、开关柜 1080、电容器 714、电抗器 516，总计 3588；历史 I-300 分别为 1575、1575、660、660，总计 4470。图像曝光同为 1800，但本轮实例监督少 882 次（约 19.7%）。这些差异在三个 seed 中一致。K/L 彼此的实例曝光完全一致，因此其配对比较仍成立；但相对 I 的比较同时改变位姿、来源覆盖和监督次数，还可能改变实例尺度分布，不能归因于单一外观因素。','',
      '优先的后续开发方向是：冻结一个保留既有桥接覆盖与逐类全图实例曝光的对照，再有界替换为新补采图，并核对实例尺度与前面板覆盖。监督减少或覆盖变窄是可检验假设，不是本轮已证实的唯一病因。不要先扩大困难负例曝光、再搜索阈值或直接改模型结构。','',
      '原有能力是否保留以相对 R-300、历史 A 的原始／光照全图及逐类召回最多下降 5 个百分点为准，L 另需满足相对 K 的同样门禁。未达标不追加临时配比、不挑有利 seed。下一步应依据逐类失败项与已有前面板覆盖缺口设计独立开发修复，而不是直接晋升或解封测试。','',
      '## 完整性与边界','',
      '43 项启动前相关测试及另 15 项评估／历史训练回归通过，v2.11 固定 40 文件在完成时复核通过。实际曝光、300 个优化步、恒定学习率、关闭增强及 30 轮损失曲线逐单元验证。未声明全仓测试通过。训练成员末轮验证只作拟合诊断。','',
      '新场景未解封；training_admitted=false、promotable=false；后台定时任务未开启。仍为同一地图／资产，8 个位姿、7 个来源组；开关柜前面板覆盖不足仍存在。','',
      f'- [完成回执]({completion})',f'- [逐 seed 诊断与配对变化]({dest})',f'- [逐框 AI 辅助审核]({OUT/"negative-review.json"})',f'- [执行回执]({OUT/"execution.json"})'])
    payload='\n'.join(lines)+'\n'
    if report.exists() and report.read_text()!=payload:raise ValueError('Refuse to overwrite changed report')
    if not report.exists():report.write_text(payload)
    save(OUT/'report-receipt.json',dict(status='report_complete',inputs={str(x):file_sha256(x) for x in (dest,report)},report_path=str(report)))
    print('REPORT',report,analysis['result_status'],flush=True)

if __name__=='__main__':main()
