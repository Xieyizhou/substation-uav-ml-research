"""Sign the verified design blocker; no training readiness is issued."""
import subprocess,sys
from pathlib import Path
from scripts.vision.verify_material_view_design import OUT,prior,blockers
from scripts.vision.evaluate_scale_endpoints import baseline_verify

def main():
    vp=OUT/'verification.json';v=prior.read(vp);prior.verify(v)
    if blockers(v['candidates'])!=v['coverage']:raise ValueError('Changed coverage')
    suites=['tests.test_material_view_design','tests.test_closed_material_review','tests.test_material_member_fit']
    r=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths=[vp,Path(__file__).resolve(),prior.ROOT/'docs/material-view-diversity-design-v1.md']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in suites]
    prior.frozen(OUT/'completion.json',dict(status='design_and_known_candidate_verification_complete_training_blocked',
        named_blockers=['No usable new-pose group among the verified eight candidates','No usable changed-material reactor or transformer candidate in this inventory','Four candidate variants remain held','Concrete V/VM members, class-budget solution and loaders not frozen'],
        training_ready=False,training_started=False,collection_started=False,selected_candidate=None,
        baseline=b,regression_output=r.stdout+r.stderr,whole_repository_tests_claimed=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('DESIGN_VERIFIED_TRAINING_BLOCKED')

if __name__=='__main__':main()
