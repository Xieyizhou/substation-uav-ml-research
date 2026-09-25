"""Explicit AI variant review and small candidate manifest; never training admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
import numpy as np
from PIL import Image
from scripts.vision.capture_supplemental_material_triplets import OUT,SOURCES,prior
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision.evaluate_scale_endpoints import baseline_verify

NOTES={
  "S01-warm-00": [
    "clear",
    "蓝灰变压器宽主体、顶面、三套管和基座清楚，未见前景遮挡。"
  ],
  "S01-warm-01": [
    "clear",
    "暖棕电容器宽顶面、正侧面与基座可辨，封闭主体完整。"
  ],
  "S01-cool-00": [
    "clear",
    "变压器主体、套管、顶面及基座清楚，保持蓝灰外观。"
  ],
  "S01-cool-01": [
    "clear",
    "冷灰电容器顶面亮于侧面，主体边界与基座可分。"
  ],
  "S02-warm-00": [
    "clear",
    "右侧蓝灰变压器主体、三套管和基座完整可辨。"
  ],
  "S02-warm-01": [
    "clear",
    "左侧蓝灰变压器宽主体、顶面和套管可辨，附近杆体未遮住主体。"
  ],
  "S02-warm-02": [
    "clear",
    "远处暖棕电容器正面、顶面、侧面与基座可辨，未见截断。"
  ],
  "S02-cool-00": [
    "clear",
    "右侧变压器宽主体、顶面、套管和基座可辨，未见遮挡。"
  ],
  "S02-cool-01": [
    "clear",
    "左侧变压器宽主体、三根套管和基座可辨。"
  ],
  "S02-cool-02": [
    "clear",
    "远处冷灰电容器宽正面、顶面及基座清楚，与地面可分。"
  ],
  "S03-warm-00": [
    "clear",
    "近处蓝柜宽顶面、正面与基座可辨；高俯视，未认证面板细节覆盖。"
  ],
  "S03-warm-01": [
    "clear",
    "暖棕计划柜体宽顶面、正面和基座可辨，四边在图内。"
  ],
  "S03-warm-02": [
    "clear",
    "蓝灰变压器宽主体、顶面、三套管与基座清楚。"
  ],
  "S03-warm-03": [
    "clear",
    "入口蓝柜顶面、宽正侧面和基座完整可辨。"
  ],
  "S03-cool-00": [
    "clear",
    "近处蓝柜顶面占比较高，宽正面及基座仍可辨。"
  ],
  "S03-cool-01": [
    "clear",
    "冷灰计划柜体顶面、较暗正面与基座可辨，轮廓清楚。"
  ],
  "S03-cool-02": [
    "clear",
    "蓝灰变压器宽主体、顶面和三套管完整可辨。"
  ],
  "S03-cool-03": [
    "clear",
    "入口蓝柜宽正侧面、顶面和基座可辨，没有图缘截断。"
  ],
  "S04-warm-00": [
    "partial",
    "右后蓝柜面板、顶面和宽侧面可辨，下方部分基座与主体被前景柜遮挡。"
  ],
  "S04-warm-01": [
    "partial",
    "蓝柜宽面板、顶面和基座可辨，右下邻柜轻微重叠。"
  ],
  "S04-warm-02": [
    "clear",
    "后排蓝柜宽面板、顶面和基座清楚。"
  ],
  "S04-warm-03": [
    "clear",
    "暖棕计划柜体宽背面、顶面和基座清楚，不把背面称为面板。"
  ],
  "S04-warm-04": [
    "clear",
    "变压器宽主体、顶面、三套管与基座可辨。"
  ],
  "S04-warm-05": [
    "clear",
    "前景蓝柜宽侧面、面板侧边、顶面及基座可辨。"
  ],
  "S04-warm-06": [
    "clear",
    "左侧蓝色入口柜宽背侧面、顶面和基座完整可辨。"
  ],
  "S04-cool-00": [
    "partial",
    "右后蓝柜顶面、面板及宽侧面可辨，下部受前景柜体遮挡。"
  ],
  "S04-cool-01": [
    "partial",
    "后排蓝柜宽面板和顶面可辨，右下边缘有邻柜侵入。"
  ],
  "S04-cool-02": [
    "clear",
    "后排蓝柜面板、顶面、侧边与基座清楚。"
  ],
  "S04-cool-03": [
    "clear",
    "冷灰计划柜体宽背面、顶面、侧边和基座可辨。"
  ],
  "S04-cool-04": [
    "clear",
    "变压器宽主体、顶面、三套管和基座完整可辨。"
  ],
  "S04-cool-05": [
    "clear",
    "前景蓝柜宽侧面、面板侧边及顶面可辨，主体完整。"
  ],
  "S04-cool-06": [
    "clear",
    "左侧入口蓝柜背侧面、顶面和基座清楚。"
  ],
  "S05-warm-00": [
    "clear",
    "暖棕电抗器宽圆柱、顶面与方形基座完整；曲面明暗可辨。"
  ],
  "S05-cool-00": [
    "clear",
    "冷灰电抗器圆柱、顶面与基座完整，暗侧曲面仍与背景可分。"
  ],
  "S06-warm-00": [
    "clear",
    "俯视暖棕电抗器宽曲面、顶面和基座完整清楚。"
  ],
  "S06-cool-00": [
    "clear",
    "俯视冷灰电抗器主体、顶面、基座可辨，没有前景遮挡。"
  ],
  "S07-warm-00": [
    "clear",
    "暖棕变压器宽主体、顶面、三根套管和基座完整，边界在图内。"
  ],
  "S07-warm-01": [
    "partial",
    "后方灰色电抗器宽圆柱及顶面清楚，基座右侧被暖色变压器局部遮挡。"
  ],
  "S07-cool-00": [
    "clear",
    "冷灰变压器宽正面、顶面、套管和基座可辨；较暗正面轮廓清楚。"
  ],
  "S07-cool-01": [
    "partial",
    "后方电抗器宽曲面和顶面可辨，基座右侧受冷灰变压器遮挡。"
  ],
  "S08-warm-00": [
    "clear",
    "暖棕变压器宽正面、顶面、套管及基座可辨，左侧杆体不遮住主体。"
  ],
  "S08-warm-01": [
    "clear",
    "后方蓝灰封闭电容器宽正面、顶面、侧面和基座可辨。"
  ],
  "S08-cool-00": [
    "clear",
    "冷灰变压器较暗正面、顶面、套管与基座可分，无截断。"
  ],
  "S08-cool-01": [
    "clear",
    "蓝灰电容器宽正面、顶面、侧边和基座清楚。"
  ]
}


def main():
    pp=OUT/'protocol.json';cp=OUT/'completion.json';ep=OUT/'review/evidence.json';sp=SOURCES/'reviewed-completion.json'
    p=prior.read(pp);c=prior.read(cp);e=prior.read(ep);source=prior.read(sp)
    for record in (p,c,e,source):prior.verify(record)
    if c['status']!='sixteen_variants_captured_review_pending':raise ValueError('Incomplete triplets')
    ids=[r['event_id'] for r in e['events']]
    if len(ids)!=44 or len(set(ids))!=44 or set(ids)!=set(NOTES):raise ValueError('Missing/duplicate visual review')
    now=datetime.now(timezone.utc).isoformat();decisions=[];members=[];paths=[pp,cp,ep,sp,Path(__file__)]
    for event in e['events']:
        condition,reason=NOTES[event['event_id']]
        decisions.append(dict(event,condition=condition,reason=reason,review_type='AI辅助审核',reviewed_at=now,
            decision='variant_content_review_passed',training_admitted=False,promotable=False))
    for s in source['sources']:
        members.append(dict(member_id=s['probe_id']+'-original',pair_id=s['lineage_id'],variant='original',image_path=s['image_path'],image_sha256=s['image_sha256'],
            full_truth=s['full_truth'],instance_mapping=s['instance_mapping'],actual_pose=s['actual_pose'],role='development_training_candidate',training_admitted=False,promotable=False))
    planned={r['key']:r for r in p['units']}
    for unit in c['units']:
        row=planned[unit['variant']];f=row['frame'];rp=Path(unit['receipt']);r=prior.read(rp);prior.verify(r)
        if r['status']!='candidate_rendered_review_pending' or not r['process_cleanup_complete']:raise ValueError('Unsuccessful unit')
        root=rp.parent/'first-stable-window';ip=root/'frame-1-rgb.png'
        if np.array_equal(np.asarray(Image.open(ip).convert('RGB')),np.asarray(Image.open(f['source_image']).convert('RGB'))):raise ValueError('No actual appearance change')
        for n in range(1,4):
            mp=root/f'frame-{n}-mask.bin'
            if prior.file_sha256(mp)!=prior.file_sha256(row['reference_mask']):raise ValueError('Mask identity changed')
            mask=np.fromfile(mp,dtype='u1').reshape(1080,1920,3)
            checked=coverage(mask,[x['runtime_label'] for x in f['events']],f['instance_mapping'])
            if checked['missing_targets']:raise ValueError('Unboxed target')
        if r['stable_frames']!=3 or any(x['maximum_box_delta_px']>1 for x in r['records']):raise ValueError('Alignment gate')
        full=prior.read(f['source_receipt'])['truth']
        members.append(dict(member_id=row['probe_id']+'-'+row['variant'],pair_id=f['lineage_id'],variant=row['variant'],
            image_path=str(ip),image_sha256=prior.file_sha256(ip),full_truth=full,instance_mapping=f['instance_mapping'],actual_pose=f['actual_pose'],
            world_path=row['world'],target_object_id=row['target_object_id'],max_box_delta_px=max(x['maximum_box_delta_px'] for x in r['records']),
            max_skew_ms=max(x['skew_ms'] for x in r['records']),role='development_training_candidate',training_admitted=False,promotable=False))
        paths += [rp,ip]
    groups={m['pair_id'] for m in members}
    if len(groups)!=8 or any({m['variant'] for m in members if m['pair_id']==g}!={'original','warm','cool'} for g in groups):raise ValueError('Missing triplet variant')
    tests=['tests.test_designed_material_triplets','tests.test_full_scene_pose_design','tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]+[prior.ROOT/'docs/results/ml_designed_material_triplets_20260911.md']
    prior.frozen(OUT/'reviewed-completion.json',dict(status='eight_triplets_24_candidates_reviewed_not_training_ready',decisions=decisions,members=members,
        complete_triplets=8,candidate_images=24,new_variant_images=16,training_ready=False,training_started=False,
        dataset_dedup_and_role_isolation_complete=False,historical_admission_chain_valid=False,exposure_feasibility_checked=False,
        limitations=['Eight supplemental poses; combine with original four only after whole-batch validation','Shared complex layout/assets','Only planned body material changes; nonplanned same-class equipment retains original material','No model improvement claim or training admission'],
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('VARIANT16_PASS; REVIEW44; COMPLETE_TRIPLETS8; CANDIDATES24; REGRESSIONS_PASS; PINNED40_PASS')

if __name__=='__main__':main()
