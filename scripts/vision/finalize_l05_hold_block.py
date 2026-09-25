"""Validate count witnesses and explicit blocker; cannot issue training readiness."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
from scripts.vision.prepare_l05_compensation_review import OUT,prior
from scripts.vision.prepare_physical_lighting_control import SOURCE
from scripts.vision import optimize_revision_compensation as solver
from scripts.vision.replay_l05_increase_queue import OUT as REPLAY,capture
from scripts.vision.test_body_material_applicability import baseline_verify

def validate_counts(rows,old,new,held,allow,groups):
    if set(new)!={r['member_id'] for r in rows} or any(type(v)!=int or v<0 for v in new.values()):raise ValueError('Invalid count members')
    for r in rows:
        m=r['member_id'];v=old.get(m,0)
        if (m in held or v==0) and new[m]!=0:raise ValueError('Held or zero member restored')
        if r['subset']=='hard_negative' and new[m]!=v:raise ValueError('Negative count changed')
        if m not in allow and new[m]>v:raise ValueError('Unapproved increase')
        if m not in held and v>0 and new[m]<1:raise ValueError('Positive removed')
    for field,keys in [('subset',{r['subset'] for r in rows}),('class',{k for r in rows for k in r['class_instances']})]:
        for k in keys:
            coeff=lambda r:int(r['subset']==k) if field=='subset' else r['class_instances'].get(k,0)
            if sum(coeff(r)*old.get(r['member_id'],0) for r in rows)!=sum(coeff(r)*new[r['member_id']] for r in rows):raise ValueError('Exposure quota changed')
    for g in groups.values():
        if sum(new[m] for m in g)>sum(old.get(m,0) for m in g):raise ValueError('Risk cap increased')

def require_quality_ready(status):
    if status!='all_replayed_review_pending':raise ValueError('Blocked replay cannot enter review-complete/readiness')
    # Even technical completion is not an approval.
    return False

def main():
    cp=OUT/'counts.json';ep=OUT/'new-risk/evidence.json';rp=REPLAY/'receipt.json';op=OUT/'original-review.json'
    for x in [cp,ep,rp,op]:prior.verify(prior.read(x))
    c,e,r=map(prior.read,(cp,ep,rp));p=prior.read(SOURCE/'protocol.json');prior.verify(p)
    policy=prior.read(solver.prior.POLICY/'protocol.json');prior.verify(policy)
    for seed,result in c['results'].items():
        validate_counts(p['pool_rows'],Counter(p['schedules'][f'brightness-450-{seed}']),result['counts'],set(c['held_members']),set(c['allowlist']),policy['risk_lineage_groups'])
    if c['solve_round']!=1 or r['status']!='blocked':raise ValueError('Unexpected stopping condition')
    for record in e['records']:
        if not record['rgb_exact'] or not record['pose_pass'] or record['max_rgb_skew_ms']>33.334:raise ValueError('Alignment evidence changed')
    report=prior.ROOT/'docs/results/ml_l05_hold_compensation_block_20260910.md'
    review=OUT/'new-risk/review.json'
    if not review.exists():
        prior.frozen(review,dict(status='实例有可见证据，但内容不足',review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            member_id=e['member_id'],instance='transformer_se',runtime_label='57',component_identity='unknown',
            reason='已查看全图、原尺寸ROI及掩码叠图。左下边缘可见严重截断的箱状侧面局部；实例映射与像素对齐证据确认归属，单凭局部外形不能可靠辨类。历史标签未覆盖该实例。新增几何框横跨全宽，不能作为可见区域使用或直接补写标签。',
            inputs={str(x):prior.file_sha256(x) for x in [ep,report,*sorted(ep.parent.glob('*.png'))]}))
    prior.verify(prior.read(review))
    tests=['tests.test_l05_hold_gate','tests.test_l05_risk_scope','tests.test_physical_lighting_preflight','tests.test_physical_lighting_capture','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    for item in r['results']:
        receipt=prior.read(item['receipt']);prior.verify(receipt)
        if not receipt['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
    capture.base.guard()
    paths=[cp,ep,rp,op,review,report,SOURCE/'protocol.json',solver.prior.POLICY/'protocol.json',Path(__file__).resolve()]+[prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'completion.json',dict(status='stopped_new_risk_after_one_solve',training_ready=False,training_started=False,
        sequence_frozen=False,loader_preflight_run=False,second_solve_run=False,process_cleanup_complete=True,
        baseline=baseline,regression=dict(output=t.stderr,whole_repository_tested=False),inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('NEW RISK STOP VERIFIED; NO SECOND SOLVE; NO TRAINING');print(t.stderr)

if __name__=='__main__':main()
