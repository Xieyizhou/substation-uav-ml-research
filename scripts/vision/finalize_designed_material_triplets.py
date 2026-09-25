"""Explicit AI variant review and small candidate manifest; never training admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
import numpy as np
from PIL import Image
from scripts.vision.capture_designed_material_triplets import OUT,SOURCES,prior
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision.evaluate_scale_endpoints import baseline_verify

NOTES={
 'G01-warm-00':('clear','俯视蓝灰色变压器宽主体、顶面、三根套管和基座清楚，无明显前景遮挡。'),
 'G01-warm-01':('clear','电容器变为暖灰棕色，完整封闭主体、顶面和基座可辨。'),
 'G01-cool-00':('clear','俯视变压器主体、套管及基座清楚，未见图缘截断。'),
 'G01-cool-01':('clear','冷灰色电容器宽顶面、侧面与基座可辨；侧面较暗但轮廓清楚。'),
 'G02-warm-00':('partial','远处蓝色柜体前面板、顶面及侧面可辨，左下邻柜局部遮挡。'),
 'G02-warm-01':('partial','后排蓝色柜体前面板、宽侧面和顶面清晰，边缘邻物重叠。'),
 'G02-warm-02':('partial','蓝色柜体宽主体和顶面清楚，下部被前景变压器及套管局部遮挡。'),
 'G02-warm-03':('partial','暖灰棕色计划柜体宽背面与顶面可辨，下部有前景变压器侵入，但并非窄片。'),
 'G02-warm-04':('clear','前景变压器完整宽主体、三根套管和基座清楚。'),
 'G02-warm-05':('clear','右侧蓝色柜体面板、顶面和宽侧面清晰，与地面可分。'),
 'G02-warm-06':('clear','左前蓝色柜体宽背面、顶面和基座完整。'),
 'G02-cool-00':('partial','远处蓝色柜体面板、顶面和侧面可见，局部受邻柜遮挡。'),
 'G02-cool-01':('partial','后排蓝色柜体宽侧面、面板、顶面可辨，邻物侵入边缘。'),
 'G02-cool-02':('partial','蓝色柜体宽主体及顶面可辨，下部被前景变压器局部遮挡。'),
 'G02-cool-03':('partial','计划柜体为冷灰色，背面、顶面和侧边可辨；下方遮挡仍保留。'),
 'G02-cool-04':('clear','变压器宽主体、顶面、三套管及基座完整可辨。'),
 'G02-cool-05':('clear','右侧柜体前面板、顶面和侧面清楚。'),
 'G02-cool-06':('clear','左前柜体背面、顶面与基座完整，无明显图缘截断。'),
 'G03-warm-00':('clear','暖灰棕色电抗器圆柱、顶面和基座完整，曲面明暗可辨。'),
 'G03-cool-00':('clear','冷灰色电抗器圆柱、顶面及基座完整，较暗曲面仍与背景可分。'),
 'G04-warm-00':('clear','暖灰棕色变压器完整宽主体、顶面、三套管和基座清楚。'),
 'G04-warm-01':('partial','后方灰色电抗器宽圆柱曲面、顶面和部分基座可见，右下被暖色变压器遮挡。'),
 'G04-cool-00':('clear','冷灰色变压器完整宽主体、顶面、套管和基座可辨。'),
 'G04-cool-01':('partial','后方电抗器较宽曲面和顶面可辨，右下受冷灰色变压器遮挡，未把两者当同一实例。'),
}


def main():
    pp=OUT/'protocol.json';cp=OUT/'completion.json';ep=OUT/'review/evidence.json';sp=SOURCES/'reviewed-completion.json'
    p=prior.read(pp);c=prior.read(cp);e=prior.read(ep);source=prior.read(sp)
    for record in (p,c,e,source):prior.verify(record)
    if c['status']!='eight_variants_captured_review_pending':raise ValueError('Incomplete triplets')
    ids=[r['event_id'] for r in e['events']]
    if len(ids)!=24 or len(set(ids))!=24 or set(ids)!=set(NOTES):raise ValueError('Missing/duplicate visual review')
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
    if len(groups)!=4 or any({m['variant'] for m in members if m['pair_id']==g}!={'original','warm','cool'} for g in groups):raise ValueError('Missing triplet variant')
    tests=['tests.test_designed_material_triplets','tests.test_full_scene_pose_design','tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]+[prior.ROOT/'docs/results/ml_designed_material_triplets_20260911.md']
    prior.frozen(OUT/'reviewed-completion.json',dict(status='four_triplets_12_candidates_reviewed_not_training_ready',decisions=decisions,members=members,
        complete_triplets=4,candidate_images=12,new_variant_images=8,training_ready=False,training_started=False,
        dataset_dedup_and_role_isolation_complete=False,historical_admission_chain_valid=False,exposure_feasibility_checked=False,
        limitations=['Four poses, one per class; not the planned twelve poses','Shared complex layout/assets','Only planned body material changes; nonplanned same-class equipment retains original material','No model improvement claim or training admission'],
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('VARIANT8_PASS; REVIEW24; COMPLETE_TRIPLETS4; CANDIDATES12; TEST35_PASS; PINNED40_PASS')

if __name__=='__main__':main()
