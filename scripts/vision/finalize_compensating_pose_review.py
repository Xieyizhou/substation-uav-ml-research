"""Explicit full-scene source review; source suitability is not training admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
import numpy as np
from scripts.vision.verify_compensating_pose_pilot import OUT,DESIGN,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify

OBSERVATIONS={
  "C01-00": [
    "clear",
    "右后柜体面板、顶面、宽侧面及基座可辨，没有图缘截断。"
  ],
  "C01-01": [
    "partial",
    "后排柜体宽面板、顶面和侧面可辨，左下被前景柜体局部遮挡。"
  ],
  "C01-02": [
    "clear",
    "后排柜体宽面板、顶面、侧面和基座可辨。"
  ],
  "C01-03": [
    "clear",
    "左后柜体宽背面、顶面和基座可辨，不能称为前面板覆盖。"
  ],
  "C01-04": [
    "partial",
    "远处变压器宽主体、三套管及顶面可辨，左侧有杆体和横件局部遮挡。"
  ],
  "C01-05": [
    "clear",
    "左侧变压器宽主体、三套管、顶面和基座清楚，附近杆体未遮住主体。"
  ],
  "C01-06": [
    "clear",
    "中央变压器完整宽主体、顶面、三套管和基座可辨。"
  ],
  "C01-07": [
    "clear",
    "前排柜体宽侧面、顶面、面板侧边与基座可辨。"
  ],
  "C01-08": [
    "clear",
    "右侧电抗器宽圆柱、顶面及方形基座完整可辨。"
  ],
  "C01-09": [
    "partial",
    "远处入口柜体宽背侧面、顶面可辨，下右部被变压器及杆体局部遮挡，并非窄条。"
  ],
  "C01-10": [
    "clear",
    "左侧封闭电容器宽主体、顶面和基座完整，未见图缘截断。"
  ]
}


def main():
    ep=OUT/'review/evidence.json';cp=OUT/'completion.json';p=prior.read(ep);c=prior.read(cp)
    for r in (p,c):prior.verify(r)
    ids=[e['review_id'] for e in p['events']]
    if len(ids)!=11 or len(set(ids))!=11 or set(ids)!=set(OBSERVATIONS):raise ValueError('Review coverage changed')
    now=datetime.now(timezone.utc).isoformat();decisions=[];sources=[];paths=[ep,cp,DESIGN/'protocol.json',Path(__file__)]
    for event in p['events']:
        condition,reason=OBSERVATIONS[event['review_id']]
        decisions.append(dict(event,condition=condition,reason=reason,review_type='AI辅助审核',reviewed_at=now,
            decision='source_content_review_passed',training_admitted=False,promotable=False))
    for u in c['units']:
        if u['status']!='original_pixel_evidence_certified':raise ValueError('Unqualified exact replay')
        rp=Path(u['replay_receipt']);r=prior.read(rp);prior.verify(r)
        up=OUT/u['probe_id']/'protocol.json';unit=prior.read(up);prior.verify(unit);f=unit['frame']
        if len(r['records'])!=3 or any(not x['rgb_exact'] for x in r['records']):raise ValueError('Incomplete exact frames')
        if any(x['missing_targets'] for x in r['full_mask_coverage']):raise ValueError('Unboxed target')
        # Verify visible pixels also fit each complete box within original 1px tolerance.
        for n in range(1,4):
            mask=np.fromfile(rp.parent/f'first-stable-window/frame-{n}-mask.bin',dtype='u1').reshape(1080,1920,3)
            for event in f['events']:
                yy,xx=np.where(mask[:,:,2]==event['runtime_label'])
                if not len(xx):raise ValueError('Label lacks visible pixels; cannot approve source')
                x1,y1,x2,y2=event['bbox_xyxy']
                if xx.min()<x1-1 or yy.min()<y1-1 or xx.max()+1>x2+1 or yy.max()+1>y2+1:raise ValueError('Mask outside full box tolerance')
        source=prior.read(f['source_receipt']);capture=prior.read(u['capture_receipt'])
        sources.append(dict(probe_id=u['probe_id'],member_id=f['member_id'],lineage_id=f['lineage_id'],
            status='source_ready_for_bounded_material_variants',image_path=f['source_image'],image_sha256=prior.file_sha256(f['source_image']),
            full_truth=source['truth'],instance_mapping=f['instance_mapping'],actual_pose=f['actual_pose'],
            exact_replay=True,unboxed_target_count=0,mask_inside_full_boxes=True,
            max_skew_ms=max(x['skew_ms'] for x in r['records']),max_box_delta_px=max(x['maximum_box_delta_px'] for x in r['records']),
            acquisition_attempts=len(capture['views'][0]['attempts']),training_admitted=False,promotable=False))
        paths += [rp,up,Path(f['source_receipt']),Path(u['capture_receipt'])]
    tests=['tests.test_full_scene_pose_design','tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]+[prior.ROOT/'docs/results/ml_designed_full_scene_poses_20260911.md']
    prior.frozen(OUT/'reviewed-completion.json',dict(status='one_compensating_source_verified_not_training_ready',sources=sources,decisions=decisions,
        new_original_images=1,source_pose_groups=1,truth_labels=11,training_ready=False,training_started=False,
        dataset_identity_isolation_complete=False,historical_admission_chain_valid=False,
        limitations=['Shared complex layout/assets; not independent new scenes','One additional compensation source, not a replacement for the original twelve groups','View distance and height distribution changed; does not alone validate failure-condition coverage','Remaining dataset dedup/source-role and exposure feasibility gates not executed'],
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SOURCE1_PASS; LABEL11_REVIEWED; REGRESSIONS_PASS; PINNED40_PASS; NO_TRAINING')

if __name__=='__main__':main()
