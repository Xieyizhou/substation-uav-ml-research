"""Finalize bounded screening with explicit blockers; never issue readiness."""
import subprocess,sys
from pathlib import Path
from scripts.vision.review_redistribution_members import OUT,ROOT,read,verify,frozen,file_sha256,validate,RISK

def main():
    e=read(OUT/'evidence.json');r=read(OUT/'review.json');c=read(OUT/'risk-crops.json')
    for x in (e,r,c):verify(x)
    validate(e,r['decisions'])
    if set(r['pending_events'])!=set(RISK) or len(e['events'])!=56:raise ValueError('Scope mismatch')
    for d in r['decisions']:
        if d['event_id'] in RISK and (d['status']!='pending_target_evidence' or not d['risk_reason']):raise ValueError('Risk silently passed')
    if any((OUT/name).exists() for name in ['ready.json','protocol.json','training']):raise ValueError('Unexpected gate bypass')
    tests=['tests.test_redistribution_member_review','tests.test_positive_redistribution','tests.test_hold_compensation_feasibility']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    paths=[OUT/'evidence.json',OUT/'review.json',OUT/'risk-crops.json',Path(__file__),ROOT/'docs/results/ml_redistribution_quality_gate_20260909.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    frozen(OUT/'completion.json',dict(status='quality_gate_blocked_no_sequence',screened_members=56,pending_target_events=sorted(RISK),
        readiness_issued=False,sequence_generated=False,loader_preflight_run=False,training_started=False,labels_modified=False,
        regression_output=run.stderr,whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('QUALITY_GATE_BLOCKED_12_TARGETS; NO_SEQUENCE_NO_TRAINING')
if __name__=='__main__':main()
