"""Explicit AI-assisted paired observations; separate geometry from class errors."""
import sys
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.inspect_stratified_material_errors import OUT,RUN,read,save,file_sha256,verify_tree,iou
OBS={
'P01':'圆柱形电抗器主体及基座；灰色替代原蓝灰色，形状未变。',
'P02':'开关柜大侧面、顶部与底座；原蓝色侧面变灰，不是电容器阵列。',
'P03':'开关柜正面、侧面及基座；原深色前面板与蓝色外壳的对比显著减弱，前景有遮挡。',
'L1':'后方电抗器圆柱仅上部可见，下部受前景设备遮挡；原图同一区域仍有圆柱结构。',
'P05':'变压器箱体及顶部套管可见，下部受前景圆柱遮挡。',
'P06':'变压器大侧面、顶部及套管，没有电容器阵列。',
'P07':'变压器箱体和套管，框内还含前景圆柱与遮挡结构，范围较宽。',
'P08':'开关柜前面板与侧面；原蓝外壳和深面板变为近似灰色，右侧有前景圆柱。',
'P09':'开关柜箱体正侧面及基座；灰色面板和外壳对比小。',
'P10':'近景变压器局部侧面及右侧立杆，框未完整包住变压器，属于局部框。',
'P11':'后排开关柜箱体，右下有前景柜体遮挡；原侧面为蓝色。',
'P12':'后排开关柜顶部和侧面，前景柜体遮住下部，灰色区域彼此对比减弱。',
'P13':'开关柜背侧大平面、顶部与基座，没有电容器阵列。',
'P14':'变压器主体和顶部套管，中央立杆遮挡；不是背景建筑。',
'P15':'变压器箱体、顶部套管及前景遮挡柜体。',
'P16':'电抗器圆柱主体与基座，背景包含变压器套管，框宽于圆柱主体。',
'P17':'开关柜顶部、无面板侧面与基座，原图为蓝色箱体。',
'P18':'远处变压器箱体与顶部套管，右侧立杆遮挡。',
'P19':'变压器箱体和顶部三根套管，材质变灰但套管保留。',
'L2':'同P19变压器，顶部套管清楚、主体可见，不是目标消失。',
'P21':'变压器箱体、顶部套管及基座；原蓝色外壳在材质图中变灰。',
'P22':'开关柜侧面、顶部和基座；未见电容器结构。',
'L3':'同P21变压器，主体和套管可见，原图外壳颜色不同。',
}

def validate(m,decisions):
    expected={x['item_id']:x for x in m['items']}
    if len(decisions)!=len(expected) or {d['item_id'] for d in decisions}!=set(expected):raise ValueError('Missing/duplicate decisions')
    for d in decisions:
        item=expected[d['item_id']]
        if d['decision']!='reviewed' or d['review_nature']!='AI-assisted' or not d['reason']:raise ValueError('Unresolved review')
        if d['bbox_xyxy']!=item['bbox_xyxy'] or d['evidence_sha256']!=item['evidence_sha256']:raise ValueError('Changed evidence identity')
        if file_sha256(item['evidence_path'])!=d['evidence_sha256']:raise ValueError('Stale evidence')
        for v,src in item['sources'].items():
            if d['image_hashes'][v]!=src['image_sha256'] or file_sha256(src['image_path'])!=d['image_hashes'][v]:raise ValueError('Stale image')

def main():
    mp=OUT/'manifest.json';verify_tree(mp);m=read(mp);decisions=[];geometric=[]
    evaluation=read(RUN/'evaluation.json');original={r['pair_id']:r for r in evaluation['rows'] if r['variant']=='original'}
    for item in m['items']:
        decisions.append(dict(item_id=item['item_id'],bbox_xyxy=item['bbox_xyxy'],reason=OBS[item['item_id']],review_nature='AI-assisted',decision='reviewed',reviewed_at=datetime.now(timezone.utc).isoformat(),
            evidence_sha256=item['evidence_sha256'],image_hashes={v:s['image_sha256'] for v,s in item['sources'].items()}))
        if item['kind']=='unmatched_capacitor_prediction':
            top=item['truth_overlaps'][0];row=original[item['pair_id']]
            material=next(r for r in evaluation['rows'] if r['variant']=='material' and r['pair_id']==item['pair_id'])
            t=material['truth'][top['truth_index']]
            choices=[(i,iou(t['bbox_xyxy'],q['bbox_xyxy'])) for i,q in enumerate(row['truth']) if q['class_name']==t['class_name']]
            idx,overlap=max(choices,key=lambda x:x[1])
            if overlap<.99:raise ValueError('Counterpart truth mismatch')
            geometric.append(dict(item_id=item['item_id'],nearest_truth=top,geometric_type='wrong_class_iou_ge_0.5' if top['iou']>=.5 else 'partial_box_wrong_class_iou_lt_0.5',
                original_truth_matched=any(x['truth_index']==idx for x in row['matches']),original_counterpart_iou=overlap))
    validate(m,decisions)
    save(OUT/'review.json',dict(status='AI_assisted_review_complete',decisions=decisions,geometric_checks=geometric,
        nearest_target_counts=dict(Counter(x['nearest_truth']['class_name'] for x in geometric)),geometric_counts=dict(Counter(x['geometric_type'] for x in geometric)),
        original_counterpart_matched_count=sum(x['original_truth_matched'] for x in geometric),
        scope='20 prediction events plus 3 target-loss events, not 23 independent objects. Counterpart matches are event counts, not independent samples.',
        inputs={str(q):file_sha256(q) for q in (mp,Path(__file__),RUN/'evaluation.json')}))
    print('SUMMARY',Counter(x['geometric_type'] for x in geometric),'original matched',sum(x['original_truth_matched'] for x in geometric))

if __name__=='__main__':main()
