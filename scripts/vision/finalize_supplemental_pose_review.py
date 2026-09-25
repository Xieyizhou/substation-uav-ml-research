"""Explicit full-scene source review; source suitability is not training admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
import numpy as np
from scripts.vision.verify_supplemental_full_scene_poses import OUT,DESIGN,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify

OBSERVATIONS={
  "S01-00": [
    "clear",
    "俯视变压器宽主体、顶面、三根套管和基座可辨，未见图缘截断。"
  ],
  "S01-01": [
    "clear",
    "封闭电容器宽顶面、前侧主体及基座可辨；不声称内部组件可见。"
  ],
  "S02-00": [
    "clear",
    "右侧变压器主体、顶面、三根套管和基座完整可辨。"
  ],
  "S02-01": [
    "clear",
    "左侧变压器宽主体、套管与基座可辨，附近杆体未遮住主体。"
  ],
  "S02-02": [
    "clear",
    "远处封闭电容器顶面、正侧面及基座清楚，无明显前景遮挡。"
  ],
  "S03-00": [
    "clear",
    "俯视柜体顶面占比较高，但宽正面和基座仍可辨；非面板正视覆盖。"
  ],
  "S03-01": [
    "clear",
    "近处柜体宽顶面、正面和基座均可辨，边界在图内。"
  ],
  "S03-02": [
    "clear",
    "变压器宽主体、顶面、三根套管及基座完整可辨。"
  ],
  "S03-03": [
    "clear",
    "入口柜体顶面、宽正侧面和基座可辨；高俯视不等于前面板覆盖。"
  ],
  "S04-00": [
    "partial",
    "右后柜体前面板、顶面、宽侧面可辨；左下基座及少量主体被前景柜体遮挡。"
  ],
  "S04-01": [
    "partial",
    "后排柜体宽前面板、顶面和基座可辨，右下邻柜轻微重叠。"
  ],
  "S04-02": [
    "clear",
    "后排柜体前面板、顶面、宽侧面和基座可辨。"
  ],
  "S04-03": [
    "clear",
    "左后柜体宽背面、顶面与基座可辨，不将背面当成面板。"
  ],
  "S04-04": [
    "clear",
    "变压器完整主体、顶面、三根套管与基座可辨。"
  ],
  "S04-05": [
    "clear",
    "前景柜体宽侧面、面板侧边、顶面和基座可辨。"
  ],
  "S04-06": [
    "clear",
    "左侧入口柜体宽背侧面、顶面和基座完整可辨。"
  ],
  "S05-00": [
    "clear",
    "电抗器宽圆柱主体、顶面及方形基座完整清晰，附近杆体不遮挡。"
  ],
  "S06-00": [
    "clear",
    "俯视电抗器圆柱曲面、顶面和基座完整可辨，无明显前景遮挡。"
  ],
  "S07-00": [
    "clear",
    "近处变压器宽主体、顶面、三根套管与基座完整，边缘在图内。"
  ],
  "S07-01": [
    "partial",
    "后方电抗器宽圆柱主体和顶面可辨，基座右侧被变压器局部遮挡。"
  ],
  "S08-00": [
    "clear",
    "变压器宽正面、顶面、套管和基座清楚，前景杆体位于目标左侧。"
  ],
  "S08-01": [
    "clear",
    "封闭电容器宽正面、顶面、侧面与基座可辨，无图缘截断。"
  ]
}


def main():
    ep=OUT/'review/evidence.json';cp=OUT/'completion.json';p=prior.read(ep);c=prior.read(cp)
    for r in (p,c):prior.verify(r)
    ids=[e['review_id'] for e in p['events']]
    if len(ids)!=22 or len(set(ids))!=22 or set(ids)!=set(OBSERVATIONS):raise ValueError('Review coverage changed')
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
    prior.frozen(OUT/'reviewed-completion.json',dict(status='eight_new_sources_verified_material_variants_not_started',sources=sources,decisions=decisions,
        new_original_images=8,source_pose_groups=8,truth_labels=22,training_ready=False,training_started=False,
        dataset_identity_isolation_complete=False,historical_admission_chain_valid=False,
        limitations=['Shared complex layout/assets; not independent new scenes','Two supplemental poses per class; combined sources require final twelve-pose validation','View distance and height distribution changed; does not alone validate failure-condition coverage','Remaining dataset dedup/source-role and exposure feasibility gates not executed'],
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SOURCE8_PASS; LABEL22_REVIEWED; REGRESSIONS_PASS; PINNED40_PASS; NO_TRAINING')

if __name__=='__main__':main()
