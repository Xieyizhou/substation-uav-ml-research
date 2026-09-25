"""Fail-closed readiness publication only; never starts training."""
import argparse
import subprocess
import sys
from pathlib import Path
from scripts.vision.prepare_whole_image_hold import *
from scripts.vision.structure_fit import validate_review
from scripts.vision.supervision_risk_revision import OUT as RISK
from scripts.vision.supervision_risk_proposal import validate as validate_proposals

TESTS=['tests.test_whole_image_hold','tests.test_supervision_hold_design','tests.test_order_retention',
       'tests.test_retention450_adapter','tests.test_retention_real_quota',
       'tests.test_permission_risk_pilot','tests.test_supervision_risk_proposal']

def validate_units(p,units):
    if set(units)!=set(p['schedules']):raise ValueError('Missing loader unit')
    for key,u in units.items():
        verify(u)
        if u['protocol_identity']!=p['identity']:raise ValueError('Stale unit')
        if any(u.get(k) is not False for k in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Forbidden optimization')
        if u['checked_draws']!=2700 or u['checked_batches']!=450:raise ValueError('Incomplete loader')
        check_actual(p,key,u['actual'])

def finish():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'ready.json').exists():verify(read(OUT/'ready.json'));print('READY_REVALIDATED_NO_TRAINING');return
    paths=[OUT/'protocol.json',OUT/'manifest.json',FIT/'evidence.json',FIT/'review.json',RISK/'protocol.json',
           PILOT/'updated-proposal.json',Path(__file__),ROOT/'scripts/vision/prepare_whole_image_hold.py']
    for x in paths:
        if x.suffix=='.json':verify(read(x))
    validate_review(read(FIT/'evidence.json'),read(FIT/'review.json')['decisions'])
    validate_proposals(read(RISK/'protocol.json'),read(PILOT/'updated-proposal.json')['decisions'])
    for seed in (7,17,27):
        a=p['schedules'][f'reference-450-{seed}'];b=p['schedules'][f'hold-450-{seed}']
        original=read(FIT/'protocol.json')['models'][f'interleaved-450-{seed}']['draws']
        if a!=original or b!=replace_slots(p['pool_rows'],a,set(p['held_member_ids']),seed):raise ValueError('Sequence not reproducible')
    units={}
    for key in p['schedules']:
        path=OUT/'preflight'/key/'completion.json';units[key]=read(path);paths.append(path)
    validate_units(p,units)
    # Pin actual evaluation identities and references, not just their path strings.
    for field in ('paired_review','negative_review','historical_reference'):
        x=Path(p['evaluation'][field]);verify(read(x));paths.append(x)
    from scripts.vision.exposure_order_retention import PRIOR
    for seed in (7,17,27):
        x=PRIOR/f'evaluation-retained_reference-450-{seed}.json';verify(read(x));paths.append(x)
    if file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initialization changed')
    paths.append(Path(p['initialization']['path']))
    result=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    paths += [ROOT/(t.replace('.','/')+'.py') for t in TESTS]
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths.append(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    frozen(OUT/'verification.json',dict(status='related_tests_and_pinned40_passed',stderr=result.stderr,baseline=baseline,
        whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    frozen(OUT/'generation-note.json',dict(status='implementation_error_fixed_before_protocol_freeze',
        error='Initial source-check implementation used trace annotations without bbox_xyxy and raised KeyError before manifest/export.',
        resolution='Read uniquely resolved original collection receipt truth objects for complete-coordinate verification.',
        historical_files_modified=False,inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
    paths.extend([OUT/'verification.json',OUT/'generation-note.json',ROOT/'docs/results/ml_whole_image_hold_pretraining_20260909.md'])
    frozen(OUT/'ready.json',dict(status='ready_for_training_not_started',cells=sorted(units),training_started=False,
        optimizer_steps_executed=0,actual_loader_draws=16200,actual_loader_batches=2700,
        all_data_risks_resolved=False,historical_source_gaps_preserved=True,
        scope='Development data-policy control only; explicit training entry and authorization required.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print('READY_FOR_TRAINING_NOT_STARTED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');a=ap.parse_args()
    if a.finalize:finish()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
