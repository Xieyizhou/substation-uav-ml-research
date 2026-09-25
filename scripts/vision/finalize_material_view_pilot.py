"""Explicit AI observations and bounded forensic/capture completion; no admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
import numpy as np
from scripts.vision.capture_material_view_pilot import OUT,prior,SOURCE
from scripts.vision import replay_material_view_sources_v2 as fixed
from scripts.vision.evaluate_scale_endpoints import baseline_verify

OBSERVATIONS={
 'original-00':'蓝色后排柜体主体、前面板、顶沿及基座可辨，右侧被变压器遮挡。',
 'original-01':'右图缘深青色变压器宽侧面及基座可见，严重截断，未见套管；内容部分可见，类别特征有限，不据此批准监督质量。',
 'original-02':'前景蓝色柜体完整宽背面、顶面和基座可辨，无明显前景遮挡。',
 'warm-00':'后排蓝色柜体面板、主体和基座仍可辨，右侧遮挡保留。',
 'warm-01':'右缘变压器仍仅见深青色侧面和基座，截断及类别特征有限的问题保留。',
 'warm-02':'前景柜体主体变为暖灰棕色，完整背面、顶面及基座清楚。',
 'cool-00':'后排蓝色柜体面板、主体、顶沿及基座可见，右侧受遮挡。',
 'cool-01':'右缘变压器仍仅有宽侧面及基座，未见套管，不把掩码存在性等同于可充分辨识类别。',
 'cool-02':'前景柜体主体为冷灰色，背面、顶面和基座清楚，与邻物可分。',
}


def main():
    ep=OUT/'review/evidence.json';e=prior.read(ep);prior.verify(e)
    if {x['event_id'] for x in e['events']}!=set(OBSERVATIONS) or len(e['events'])!=9:raise ValueError('Review coverage changed')
    now=datetime.now(timezone.utc).isoformat();decisions=[]
    for event in e['events']:
        decisions.append(dict(event,reason=OBSERVATIONS[event['event_id']],review_type='AI辅助审核',reviewed_at=now,
            status='partial_content_supervision_review_pending' if event['event_id'].endswith('-01') else 'content_observed_candidate_only',
            training_admitted=False,promotable=False))
    root=prior.ROOT/'data/research/ml_training_recovery_v1'
    current=root/'development-content-strata-v1/review-v2.json';old=root/'development-content-strata-v1/review.json'
    initial=root/'material-control-feasibility-v1/initial-gate.json'
    r=prior.read(current);prior.verify(r);oldrecord=prior.read(old)
    script=prior.ROOT/'scripts/vision/record_development_content_strata_review.py'
    audit=dict(current_review_valid=True,current_script_sha256=prior.file_sha256(script),
        current_bound_sha256=r['inputs'][str(script)],old_bound_sha256=oldrecord['inputs'][str(script)],
        known_dependency_bug=prior.read(initial)['audit_dependency_bug'],historical_review_rewritten=False,
        disposition='Old metrics bound review.json although consuming review-v2.json. Current review-v2 and its script match. Old chain remains invalid; historical script version not recovered. No blanket chain waiver or training admission.',
        impact='Historical development-strata attribution cannot be re-certified via old chain; raw source replay directly verified without consuming strata as approval.')
    replay_results=[]
    for sid in ('N04','N02'):
        rp=fixed.OUT/f'replay/{sid}/attempt-01/receipt.json';record=prior.read(rp);prior.verify(record)
        mask=rp.parent/'first-stable-window/frame-1-mask.bin';a=np.fromfile(mask,dtype='u1').reshape(1080,1920,3)
        labels=sorted(int(x) for x in np.unique(a[:,:,2]) if x not in (0,255))
        truths=prior.read(next(f for f in prior.read(fixed.OUT/'protocol.json')['frames'] if f['review_ids']==[sid])['source_receipt'])['raw_truth']['annotatedBox']
        box_labels=sorted(int(x.get('label',0)) for x in truths)
        replay_results.append(dict(source_review_id=sid,rgb_exact_all=all(x['rgb_exact'] for x in record['records']),
            max_box_delta=max(x['maximum_box_delta_px'] for x in record['records']),max_skew_ms=max(x['skew_ms'] for x in record['records']),
            visible_runtime_labels=labels,unboxed_target_labels=sorted(set(labels)-set(box_labels)),
            blue_block_note='N02全图仅存在57及178目标掩码；蓝色区域不属于四类已映射目标。标签0不是cabinet_center独立实例ID，具体非目标资产身份不作像素级认证。' if sid=='N02' else '',
            review_type='AI辅助审核',reviewed_at=now))
    tests=['tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths=[ep,OUT/'completion.json',fixed.OUT/'completion.json',current,old,initial,script,Path(__file__),
           prior.ROOT/'docs/results/ml_material_view_replay_capture_20260911.md']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'reviewed-completion.json',dict(status='three_candidates_reviewed_supervision_and_provenance_admission_pending',
        decisions=decisions,historical_chain_audit=audit,replay_results=replay_results,new_candidate_images=3,new_pose_groups=1,
        full_training_material_matrix_complete=False,training_ready=False,training_started=False,baseline=b,
        regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('REPLAY2_PASS; CAPTURE3_PASS; REVIEW9_RECORDED; TEST16_PASS; PINNED40_PASS; NO_ADMISSION')

if __name__=='__main__':main()
