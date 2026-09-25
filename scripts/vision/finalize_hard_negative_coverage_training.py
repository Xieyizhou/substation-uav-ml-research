"""Close training only after receipts, source chains, tests and pinned files verify."""
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.train_hard_negative_coverage import TRAIN,OUT,BASE,prepare,checked_cell,verify_tree
from scripts.vision.hard_negative_coverage import read,save,verify,file_sha256,COUNTS
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record
from scripts.vision.verify_pixel_duplicates import rgb_digest


def losses(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    rows=[{k.strip():v.strip() for k,v in r.items()} for r in rows]
    if len(rows)!=10 or [int(r['epoch']) for r in rows]!=list(range(1,11)):
        raise ValueError('Training loss curve incomplete')
    for r in rows:
        for name in ('train/box_loss','train/cls_loss','train/dfl_loss'):
            if not math.isfinite(float(r[name])):raise ValueError('Nonfinite training loss')
    return rows


def main():
    protocol=prepare();progress=read(TRAIN/'progress.json');verify(progress)
    if progress['status']!='training_complete' or set(progress['completed_cells'])!=set(protocol['schedules']):
        raise ValueError('Six completed training cells required')
    records={k:checked_cell(TRAIN/k/'completion.json',protocol) for k in protocol['schedules']}
    admission_path=OUT/'final-admission.json';verify_tree(admission_path);admission=read(admission_path)
    decisions={r['view_id']:r for r in admission['decisions']}
    rows_by_id={r['member_id']:r for r in protocol['pool_rows']}
    new_by_source={r.get('source_image_path'):r for r in protocol['pool_rows'] if 'source_image_path' in r}
    ledger=[]
    for r in admission['frames']:
        plan_path=Path(r['receipt_path']).parent.parent/'plan/plan.json';plan=read_record(plan_path)
        receipt=read_record(r['receipt_path']);capture=next(v for v in receipt['views'] if v['view_id']==r['view_id'])
        member=new_by_source[r['image_path']]
        if Path(member['label_path']).read_bytes()!=b'':raise ValueError('Negative label not empty')
        if rgb_digest(Path(member['image_path']).read_bytes())[0]!=r['pixel_sha256']:
            raise ValueError('Converted training image pixels changed')
        tree=ET.parse(plan_path.parent/'world.sdf')
        sensors=[ET.tostring(s,encoding='unicode') for s in tree.iter('sensor')]
        ledger.append(dict(frame_id=r['view_id'],pair_id=r['pair_id'],pose_id=r['pose_id'],variant=r['variant'],
            coverage_unit=r['coverage_unit'],correlation_group_id=r['correlation_group_id'],
            image_path=member['image_path'],image_sha256=member['image_sha256'],label_path=member['label_path'],label_sha256=member['label_sha256'],
            source_image_path=r['image_path'],source_image_sha256=r['image_sha256'],pixel_sha256=r['pixel_sha256'],
            source_layout_id=r['source_layout_id'],derived_layout_id=r['derived_layout_id'],
            configured_subject_instance=r['object_id'],projected_asset_candidates=r['asset_ids'],
            actual_visible_content_rois=decisions[r['view_id']]['rois'],
            world_sha256=plan['files']['world.sdf'],sensor_nodes_sha256=object_sha256(sensors),
            plan_identity=plan['identity'],collection_identity=receipt['identity'],recording_id=str(Path(r['receipt_path']).parent),
            actual_annotation_mode='full_2d',label_mode='visual-instance',hierarchy_mode='top-level-equipment',
            target_instance_presence=False,visibility_review='no_four_target_classes_visible',review=decisions[r['view_id']],
            actual_pose=capture['actual_pose'],check_version=capture['check_version'],
            data_role='development_training_candidate',training_admitted=False,promotable=False))
    if Counter(r['coverage_unit'] for r in ledger)!={k:2*n for k,n in COUNTS.items()}:raise ValueError('Coverage quota mismatch')
    input_paths=[TRAIN/'protocol.json',TRAIN/'progress.json',admission_path,Path(__file__)]
    input_paths.extend(TRAIN/k/'completion.json' for k in records)
    ledger_path=OUT/'frozen-intake-ledger.json'
    save(ledger_path,dict(status='frozen_development_only',frames=ledger,frame_count=96,pose_pairs=48,correlation_clusters=47,
        actual_capture_frames=100,held_capture_frames=4,
        asset_visibility_rule='Projected asset candidates are not visibility evidence; explicit content ROIs and reasons are authoritative.',
        inputs={str(p):file_sha256(p) for p in (admission_path,TRAIN/'protocol.json')}))
    suites=['tests.test_hard_negative_coverage','tests.test_canonical_gates','tests.test_canonical_diagnostic_absence',
            'tests.test_canonical_recovery','tests.test_canonical_shutdown','tests.test_exposure_diagnosis',
            'tests.test_visual_bridge_training','tests.test_visual_bridge_supplement','tests.test_paired_visual_factors']
    tests=subprocess.run([sys.executable,'-m','unittest',*suites],cwd=ROOT,capture_output=True,text=True)
    baseline=subprocess.run([sys.executable,'scripts/vision/verify_experiment_baseline.py'],cwd=ROOT,capture_output=True,text=True)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    baseline_data=json.loads(baseline.stdout) if baseline.returncode==0 else {}
    passed=tests.returncode==baseline.returncode==diff.returncode==0 and baseline_data.get('integrity_passed') and baseline_data.get('pinned_files_verified')==40
    check_paths=[ROOT/(suite.replace('.','/')+'.py') for suite in suites]
    check_paths += [ROOT/'scripts/vision'/n for n in ('hard_negative_coverage.py','hard_negative_coverage_admission.py',
        'repair_hard_negative_coverage.py','train_hard_negative_coverage.py','admit_hard_negative_coverage_final.py')]
    verification_path=OUT/'verification.json'
    save(verification_path,dict(status='passed' if passed else 'failed',tests_output=tests.stdout+tests.stderr,
        baseline=baseline_data,diff_output=diff.stdout+diff.stderr,inputs={str(p):file_sha256(p) for p in check_paths},
        scope='Targeted tests only. Whole repository not rerun; prior 12 failures and 1 skip remain separately noted.'))
    if not passed:raise ValueError('Verification failed')
    fits={};old_reference_reproduction={};correlation_exposures={}
    for k,record in records.items():
        curve=losses(Path(record['exposure_path']).parent/'results.csv')
        fits[k]={'first':curve[0],'last':curve[-1],'scope':'training-member fitting diagnosis, not development evaluation'}
        correlation_exposures[k]=dict(Counter(rows_by_id[mid].get('correlation_group_id',rows_by_id[mid]['lineage_id']) for mid in protocol['schedules'][k]))
        if k.startswith('O-'):
            historical=read(BASE/'exposure-controlled-diagnosis-v1'/k.replace('O-','Y-')/'completion.json');verify(historical)
            previous=losses(Path(historical['exposure_path']).parent/'results.csv')
            keys=('train/box_loss','train/cls_loss','train/dfl_loss')
            old_reference_reproduction[k]=all(a[n]==b[n] for a,b in zip(curve,previous) for n in keys)
    input_paths.extend((ledger_path,verification_path))
    completion_path=OUT/'training-completion.json'
    completion=save(completion_path,dict(status='training_complete_evaluation_pending',completed_cells=list(records),
        optimizer_steps=600,image_exposures=3600,accepted_frames=96,pose_pairs=48,correlation_clusters=47,
        held_frames=4,training_fits=fits,correlation_exposures=correlation_exposures,
        old_reference_loss_reproduced=old_reference_reproduction,selected_candidate=None,
        inputs={str(p):file_sha256(p) for p in input_paths}))
    report=ROOT/'docs/results/ml_hard_negative_coverage_training_20260907.md'
    lines=['# 困难负例覆盖补采与同预算训练完成报告','',
        '状态：训练完成，正式开发评估待执行；不选择候选，不声称模型改善。',
        '', '## 数据闭环','',
        '共采集100张：初版96张中4张因实际构图不满足覆盖要求暂缓，新增4张替换；最终96张、48个位姿对通过逐图AI辅助审核。全部技术采集在首次尝试完成。',
        'M1原位姿34实际只有建筑可见，C2原位姿41柜体实际未截断；失败帧、决定与替换谱系均保留。',
        '替换后的柜体位姿41与42低冷光图dHash距离为2，经单独查看确认为近相似同源构图；两对整体关联。因此48个位姿对只有47个相关来源簇，且不等于47个独立场景。',
        '无文件或原尺寸像素重复，无保护指纹阻断；保护参考仅用于成员/指纹排除，未访问保护标签。既有simple/medium/complex目标隔离世界复用，建筑仍只有一个既有资产。',
        '', '| 覆盖单元 | 收录图数 |','| --- | ---: |']
    lines += [f'| {k} | {2*v} |' for k,v in COUNTS.items()]
    lines += ['', '## 训练闭环','',
        'O：旧24张负例；N：旧24张加新增96张。每组100步，seed7/17/27，共6个独立单元；实际600优化步、3600图像曝光均与冻结序列一致。',
        '每单元基础216、常规156、桥接120、负例108次曝光。O/N同seed的正样本身份、顺序、位置以及类别实例监督次数完全相同。',
        'N每seed预定抽取60个位姿对中的54对、每光照一次；未曝光的6对逐seed登记。负例总曝光不增加，结论只能解释负例组成与组内曝光重分配效果。',
        '统一v2.11初始化、CPU/640/batch6/nbs6/AdamW/lr0.001，无预热、恒定学习率、增强关闭；只采用预定终点last.pt，框架自动保存的best.pt不用于选择。',
        '', '| 单元 | 实际曝光 | 独特图像 | 电容器组实例曝光 | 电抗器 | 开关柜 | 变压器 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in sorted(records):
        e=protocol['exposures'][k];c=e['class_instance_exposure']
        lines.append(f'| {k} | {e["draws"]} | {e["unique_frames"]} | {c["capacitor_bank"]} | {c["reactor"]} | {c["switchgear"]} | {c["transformer"]} |')
    lines += ['',f'旧参考O与历史Y-100的三条训练损失曲线逐轮完全复现：{sum(old_reference_reproduction.values())}/3。',
        '训练成员末轮验证数值只作拟合诊断，不作开发性能或候选依据。',
        '', '## 验证与下一步','',
        tests.stderr.strip(),'',
        'v2.11固定40文件完整性通过，git diff --check通过。未重跑全仓测试；此前12项失败和1项跳过仍单独保留，不声明全仓通过。',
        '下一步在旧48张配对开发图与48张无目标开发图上执行冻结正式协议评估，检查FPR、计划命中、原始/光照全图及逐类双参考5个百分点能力保留门禁。当前没有正式评估结果，不把补采成员拟合当作泛化证据。',
        '旧开发集不进入训练；封存新场景未解封。所有产物training_admitted=false、promotable=false。',
        '',f'训练协议身份：`{protocol["identity"]}`。',f'完成回执身份：`{completion["identity"]}`。','']
    report.write_text('\n'.join(lines))
    save(OUT/'report-receipt.json',dict(status='training_report_complete',report_path=str(report),
        inputs={str(report):file_sha256(report),str(completion_path):file_sha256(completion_path),str(Path(__file__)):file_sha256(Path(__file__))}))
    print(completion['status'],completion['identity'])

if __name__=='__main__':main()
