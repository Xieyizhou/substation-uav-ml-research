"""Finalize a blocked build honestly, with independent replay failures retained."""
from pathlib import Path
import subprocess
import sys
from scripts.vision.establish_material_view_candidates import OUT,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify
from scripts.vision import run_visibility_cleanup_validation as replay


def main():
    paths=[OUT/'source-review-completion-v1.json', OUT/'native-source-replay-v1/completion.json',
           OUT/'world-drafts-v1/manifest.json',Path(__file__),
           prior.ROOT/'docs/results/ml_material_view_materials_build_20260911.md']
    for p in paths[:3]:prior.verify(prior.read(p))
    attempts=[]; dependency_gaps=[]
    for n in range(1,4):
        p=OUT/f'native-source-replay-v1/replay/N04/attempt-{n:02}/receipt.json'
        r=prior.read(p);prior.verify(r)
        # Preserve a deep historical dependency failure as a blocker, never waive it.
        if n==1:
            try:replay.verify_tree(p)
            except ValueError as exc:dependency_gaps.append(str(exc))
        if not r['process_cleanup_complete']:raise ValueError('Incomplete cleanup')
        attempts.append(dict(attempt=n,status=r['status'],reason=r['reason'],process_cleanup_complete=True))
        paths.append(p)
    tests=['tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    prior.frozen(OUT/'materials-build-completion-v1.json',dict(status='world_drafts_complete_replay_and_provenance_blocked',
        draft_worlds=6,new_training_images=0,training_started=False,training_ready=False,
        attempts=attempts,clock_parser_repair_unit_tested=True,clock_parser_repair_live_verified=False,
        regression_output=t.stdout+t.stderr,baseline=b,whole_repository_tests_claimed=False,
        full_dependency_chain_verified=False,dependency_gaps=dependency_gaps,
        next_step='Separate explicitly bounded repaired N04 replay; do not reset consumed attempt budget.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('DRAFT6; IMAGE0; TEST16_PASS; PINNED40_PASS; REPLAY_BLOCKED',dependency_gaps)


if __name__=='__main__':main()
