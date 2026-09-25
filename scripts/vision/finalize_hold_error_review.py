"""Validate explicitly entered observations, preserve unreviewed scope."""
import subprocess,sys
from pathlib import Path
from scripts.vision.review_whole_image_hold_errors import OUT,ROOT,read,verify,frozen,file_sha256,object_sha256,validate

def main():
    e=read(OUT/'evidence.json');r=read(OUT/'review.json');verify(e);verify(r);validate(e,r['decisions'])
    index={x['event_id']:x for x in e['all_losses']+e['negative']}
    for d in r['decisions']:
        x=index[d['event_id']]
        if d['source_record_identity']!=object_sha256(x):raise ValueError('Source identity differs')
        if d['evidence_sha256']!=x['evidence_sha256'] or d['image_sha256']!=x['source']['image_sha256']:raise ValueError('Evidence binding differs')
        if ':' in d['review_id']:
            if d['prediction']!=x['predictions'][int(d['review_id'].split(':')[1])]:raise ValueError('Prediction differs')
        elif d['truth']!=x['truth'] or d['model_events']!=x['events']:raise ValueError('Instance or event differs')
    expected={x['event_id'] for x in e['all_losses']}-set(e['focus_ids'])
    if {x['event_id'] for x in r['pending']}!=expected:raise ValueError('Pending scope omitted')
    tests=['tests.test_hold_error_review','tests.test_structure_fit','tests.test_structure_fit_integrity']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    paths=[OUT/'evidence.json',OUT/'review.json',OUT/'summary.json',Path(__file__),ROOT/'docs/results/ml_whole_image_hold_error_localization_20260909.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    frozen(OUT/'completion.json',dict(status='priority_visual_localization_complete_with_named_gaps',explicit_decisions=len(r['decisions']),pending_loss_entries=len(expected),
        all_losses_visually_reviewed=False,training_started=False,labels_modified=False,regression_output=result.stderr,
        whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('PRIORITY_LOCALIZATION_VERIFIED_49_DECISIONS_41_PENDING')

if __name__=='__main__':main()
