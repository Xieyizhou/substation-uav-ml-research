"""Explicit risk-review record after inspecting the two instance-overlay pages."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
from scripts.vision.replay_revision_siblings import OUT,base
from scripts.vision.test_body_material_applicability import baseline_verify


def main():
    ep=OUT/'evidence.json';base.verify(base.read(ep));e=base.read(ep)
    if [r['review_id'] for r in e['members']]!=['T027-original','T027-background_bridge']:raise ValueError('Reinspection required')
    paths=[ep,Path(__file__),base.ROOT/'docs/results/ml_revision_sibling_replay_20260910.md']
    for r in e['members']:
        base.verify(base.read(r['receipt']));paths.append(Path(r['receipt']))
        if r['pixels']!=3534 or r['bbox_half_open']!=[59,1052,306,1080] or not r['original_pixel_instance_certified']:raise ValueError('Changed evidence requires reinspection')
    tests=['tests.test_revision_sibling_replay','tests.test_revision_source_review','tests.test_annotation_revision_design','tests.test_depth_clip_acceptance','tests.test_dual_box_diagnosis']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    paths += [base.ROOT/(t.replace('.','/')+'.py') for t in tests]
    baseline=baseline_verify(base.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    base.guard();dest=OUT/'completion.json'
    if dest.exists():base.verify(base.read(dest));print('VALID_COMPLETION_REUSED');return
    decisions=[]
    for r in e['members']:
        page=Path(r['page']);paths.append(page)
        decisions.append(dict(review_id=r['review_id'],evidence_identity=e['identity'],page=str(page),page_sha256=base.file_sha256(page),
            reviewed_at=datetime.now(timezone.utc).isoformat(),review_nature='AI-assisted',
            status='instance_visible_content_insufficient',whole_frame_disposition='hold',
            reason='本轮逐一查看原始局部与实例叠图：仅底图缘极窄斜面片段，不能可靠辨识完整设备内容；实例存在不等于训练适用。',
            training_admitted=False,promotable=False))
    base.frozen(dest,dict(status='two_sibling_replays_complete_quality_held',decisions=decisions,baseline=baseline,
        regression=dict(returncode=run.returncode,output=run.stderr,whole_repository_tested=False),
        training_ready=False,training_started=False,historical_labels_changed=False,production_ready=False,
        inputs={str(p):base.file_sha256(p) for p in paths}))
    print(run.stderr);print('TWO_INDEPENDENT_REPLAYS_COMPLETE; QUALITY_HELD; BASELINE_40_PASSED')


if __name__=='__main__':main()
