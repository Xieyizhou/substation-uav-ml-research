"""Close scale diagnostics from explicit reviews, never generate approval."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.inspect_routed_scale_results import OUT,arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify,ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/'evidence.json',OUT/'other-errors-review.json',OUT/'material-evidence.json',
           OUT/'material-review.json',arm.OUT/'evaluation-v1/error-review-v1/evidence.json',
           arm.OUT/'evaluation-v1/summary.json',OUT/'evidence-build-receipt.json']
    e,r,m,mr,n,s,_=map(arm.checked,paths)
    validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions'])
    expected={(f['frame_id'],x['seed'],x['truth_index']):(f,x) for f in m['frames'] for x in f['events']}
    decisions=mr['decisions']
    if len(decisions)!=len(expected) or {(d['frame_id'],d['seed'],d['truth_index']) for d in decisions}!=set(expected):raise ValueError('Missing or duplicate material decision')
    for d in decisions:
        f,x=expected[d['frame_id'],d['seed'],d['truth_index']]
        if d['truth']!=x['truth'] or d['diagnosis']!=x['diagnosis']:raise ValueError('Changed prediction or truth')
        for name in ('image','evidence'):
            if d[name+'_sha256']!=f[name+'_sha256'] or file_sha256(f[name+'_path'])!=d[name+'_sha256']:raise ValueError('Stale reviewed content')
        if not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Missing explicit observation')
    if any(s['matching_conflicts'].values()):raise ValueError('Matching conflict')
    fit=arm.OUT.parent/'routed-scale-fit-diagnosis-v1'
    fitpath=fit/'summary.json';f=arm.checked(fitpath);paths.append(fitpath)
    counts={}
    for k in arm.KEYS:
        arm.verified_unit(k)
        q=fit/(k+'-verified.json');arm.checked(q);paths.append(q)
        c=f['groups'][k]['all']['classes'];counts[k]=dict(hits=sum(x['hits'] for x in c.values()),truth=sum(x['truth'] for x in c.values()))
    modules=('tests.test_contrast_cpu4_review','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit','tests.test_routed_scale_transform','tests.test_routed_scale_gate')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths.extend([contract,Path(__file__)])
    dest=OUT/'completion.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='scale_error_review_and_fit_diagnosis_complete_candidate_failed',
        counts=dict(original_lighting_events=len(r['positive_decisions']),material_events=len(decisions),negative_events=len(r['negative_decisions'])),
        fitting=counts,gates=s['gates'],integrity=integrity,test_output=t.stdout+t.stderr,
        conclusion='Original material training-member fitting remains near saturation, but development declines across all three seeds. Clear bodies as well as occluded fragments lose hits. Supports condition-transfer limitation, not proof of a unique cause or insufficient steps.',
        limits='Unaugmented member fitting is not augmented-tensor fitting. Visual shape is not asset identity. No pixel visibility certification. Repeated seeds and variants are not independent scenes.',
        selected_candidate=None,training_admitted=False,promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
