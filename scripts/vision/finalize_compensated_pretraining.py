"""Final evidence audit and explicit-entry preflight, never starts training."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.train_compensated_double_dose import OUT,KEYS,contract,prior
from scripts.vision.compensated_pool_fit import OUT as FIT,KEYS as FIT_KEYS,check,validate_unit
from scripts.vision.build_compensated_error_review import DEST
from scripts.vision.record_compensated_error_review import validate as review_validate
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def verify_local_chain(path,seen=None):
    seen=set() if seen is None else seen;path=Path(path).resolve()
    if path in seen:return seen
    seen.add(path);r=prior.read(path);prior.verify(r)
    for name in r.get('inputs',{}):
        child=Path(name).resolve()
        if child.suffix=='.json' and child.is_relative_to(DEST.resolve()):verify_local_chain(child,seen)
    return seen


def main():
    if (OUT/'training-authorization.json').exists() or any((OUT/'training').glob('**/last.pt')):
        raise ValueError('Cannot certify not-started after training authorization/output')
    ep,rp=DEST/'evidence.json',DEST/'review.json';e,r=prior.read(ep),prior.read(rp)
    prior.verify(e);prior.verify(r);review_validate(e,r['decisions'])
    if len(r['decisions'])!=84 or len(r['pending_ids'])!=18:raise ValueError('Review scope changed')
    fit=prior.read(FIT/'protocol.json');check(fit)
    paths=[ep,rp,DEST/'review-summary.json',FIT/'protocol.json',FIT/'summary.json',OUT/'protocol.json',
           OUT/'counts.json',OUT/'decision-protocol.json',OUT/'loader-completion.json',Path(__file__).resolve()]
    for key in FIT_KEYS:
        fp=FIT/'inference'/f'{key}.json';record=prior.read(fp);validate_unit(record,key,fit);paths.append(fp)
    for key in KEYS:
        contract(key)
    for seed in (7,17,27):
        files=list((OUT/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(files)!=1:raise ValueError('Ambiguous actual loader preflight')
        paths+=files
    checked=set()
    for path in paths:
        if path.suffix=='.json':verify_local_chain(path,checked)
    suites=['tests.test_compensated_double_dose','tests.test_compensated_double_dose_entry',
        'tests.test_compensated_error_review','tests.test_material_member_fit','tests.test_compensated_loader_receipts',
        'tests.test_compensated_material_sequences','tests.test_compensated_material_evaluation',
        'tests.test_unified_lighting_evaluation','tests.test_order_diagnosis','tests.test_fixed_budget_diagnosis']
    result=subprocess.run([sys.executable,'-m','unittest',*suites],cwd=prior.ROOT,capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline40 identity failure')
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    paths += [prior.ROOT/'scripts/vision'/f for f in ('train_compensated_double_dose.py','preflight_compensated_double_dose.py',
        'freeze_compensated_double_dose.py','brightness_transfer_runtime.py','order_retention_runtime.py','frozen_multiscale_runtime.py',
        'validate_compensated_loader_receipts.py')]
    report=prior.ROOT/'docs/results/ml_compensated_next_pretraining_20260911.md';paths.append(report)
    entry=prior.frozen(OUT/'entry-ready.json',dict(status='training_entry_verified_not_started',cells=list(KEYS),
        baseline=baseline,regression_output=result.stdout+result.stderr,training_started=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    probe=subprocess.run([sys.executable,'-m','scripts.vision.train_compensated_double_dose'],cwd=prior.ROOT,capture_output=True,text=True)
    if probe.returncode or 'ENTRY_READY_NO_TRAINING_STARTED' not in probe.stdout:
        raise ValueError('Default entry failure: '+probe.stdout+probe.stderr)
    if (OUT/'training-authorization.json').exists() or (OUT/'training').exists():raise ValueError('Default entry unexpectedly started training')
    paths.append(OUT/'entry-ready.json')
    completed=prior.frozen(OUT/'completion.json',dict(status='ready_for_training_not_started',training_started=False,
        training_admitted=False,promotable=False,selected_candidate=None,
        explicit_review_decisions=84,pending_development_content_ids=r['pending_ids'],pending_targets_not_removed=True,
        existing_weight_fit_units=9,fit_pool_members=277,fit_image_model_combinations=2493,
        future_training_units=6,actual_preflight_exposures=16200,actual_preflight_batches=2700,
        optimizer_created=False,backward_executed=False,training_validation_executed=False,
        fresh_local_dependency_jsons_verified=len(checked),historical_whole_pool_recertified=False,
        new_member_dose=dict(previous=54,next=108,total=2700),rejected_threefold_design_preserved=True,
        default_entry_output=probe.stdout,regression_output=result.stdout+result.stderr,baseline=baseline,
        whole_repository_tests_claimed=False,report=str(report),
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    verify_local_chain(OUT/'completion.json')
    print(completed['status']);print(result.stdout+result.stderr);print('BASELINE40',baseline['integrity_passed'])
    return completed


if __name__=='__main__':main()
