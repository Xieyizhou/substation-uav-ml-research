"""Record explicit FP observations, then sign a limited numerical audit report."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
from scripts.vision.audit_scale_endpoint_results import DEST,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify

NOTES={
 'F01':'青色柜状块体的顶面、正面深色面板、侧面及基座；未主要覆盖旁边杆体。',
 'F02':'灰色块体正面面板和底部，左下含前景青色块体遮挡。',
 'F03':'灰色块体正面深色面板及底部基座，背景杆体在框外。',
 'F04':'冷暗条件下灰色块体正面面板、顶部与基座。',
 'F05':'灰色块体宽侧面、右侧面板、顶面与基座完整可辨。',
 'F06':'右缘灰色块体的宽背侧面、顶面与基座，右侧截断。',
 'F07':'冷暗条件下右缘灰色块体背侧面、顶面与基座，右侧出画。',
 'F08':'混合结构：后方灰块被前景粗杆遮挡，右下还有青色块体；不能归为纯杆体或电抗器。',
 'F09':'混合结构：灰色块体侧面、面板局部，前景粗杆与青色块体共同入框。',
}

def main():
    ap=DEST/'audit.json';a=prior.read(ap);prior.verify(a)
    if {x['event_id'] for x in a['false_positives']}!=set(NOTES):raise ValueError('Explicit review coverage mismatch')
    ds=[]
    for x in a['false_positives']:
        if prior.file_sha256(x['page_path'])!=x['page_sha256']:raise ValueError('Changed evidence')
        ds.append(dict(event_id=x['event_id'],cell=x['cell'],prediction=x['prediction'],status='reviewed',reason=NOTES[x['event_id']],
            review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
            image_sha256=x['frame']['image_sha256'],evidence_sha256=x['page_sha256'],asset_identity_inferred_from_appearance=False))
    rp=DEST/'false-positive-review.json'
    prior.frozen(rp,dict(status='nine_false_positive_boxes_reviewed',decisions=ds,inputs={str(ap):prior.file_sha256(ap),str(Path(__file__).resolve()):prior.file_sha256(Path(__file__))}))
    suites=['tests.test_scale_endpoint_control','tests.test_multiscale_error_review','tests.test_frozen_multiscale_evaluation','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths=[ap,rp,Path(__file__).resolve(),prior.ROOT/'docs/results/ml_scale_endpoint_judgment_20260911.md']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in suites]
    prior.frozen(DEST/'completion.json',dict(status='bounded_judgment_complete_not_full_miss_visual_review',selected_candidate=None,
        numeric_transitions_verified=1440,false_positive_boxes_reviewed=9,
        scope_limits=['New positive-image losses have identity/numerical verification, not new exhaustive visual certification.','Existing neighboring-cabinet ambiguity retained; full metrics not filtered.'],
        regression_output=t.stdout+t.stderr,baseline=baseline,whole_repository_tests_claimed=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('BOUNDED_AUDIT_COMPLETE_NO_CANDIDATE')

if __name__=='__main__':main()
