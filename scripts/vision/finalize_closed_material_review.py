"""Summarize existing explicit decisions, retain all frame-level blockers."""
from pathlib import Path
import subprocess,sys
from scripts.vision.closed_body_material_pilot import OUT,ROOT,read,verify,frozen,file_sha256
from scripts.vision.record_closed_material_review import validate,frame_gates
from scripts.vision.test_body_material_applicability import baseline_verify


def main():
    paths=[OUT/p for p in ('protocol.json','technical-completion.json','review-evidence.json','review-pages.json','mask-coverage.json','semantic-review.json')]
    for p in paths:verify(read(p))
    q=read(OUT/'review-evidence.json');r=read(OUT/'semantic-review.json');m=read(OUT/'mask-coverage.json')
    validate(q,r['decisions'])
    if frame_gates(q,r['decisions'],m)!=r['frames']:raise ValueError('Whole-frame gates changed')
    dest=OUT/'review-completion.json'
    if dest.exists():verify(read(dest));print('VALID_REVIEW_REUSED_TRAINING_BLOCKED');return
    tests=['tests.test_closed_material_review','tests.test_closed_body_material_pilot','tests.test_body_visibility_replay','tests.test_instance_visibility_diagnosis']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    b=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [Path(__file__),ROOT/'docs/results/ml_closed_body_material_review_20260910.md']
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='review_complete_training_blocked_by_data_risks',labels_reviewed=len(r['decisions']),
        held_labels=sum(d['decision']=='held_content_uncertain' for d in r['decisions']),
        held_frames=sum(f['status']=='held_whole_image' for f in r['frames']),
        training_ready=False,training_started=False,historical_labels_changed=False,
        next_priority='Trace instance 128 box generation and edge handling before any data revision; resolve instance 118 supervision-content risk without automatic relabeling.',
        baseline=b,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print('46_REVIEWED; 4_FRAMES_HELD; TRAINING_BLOCKED; BASELINE_40_PASSED')


if __name__=='__main__':main()
