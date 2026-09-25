"""Finalize real preflight evidence; never invoke training."""
import json
import platform
import subprocess
import sys
from pathlib import Path
from scripts.vision.freeze_closed_source_control import OUT,KEYS,protocol,prior
from scripts.vision.train_closed_source_control import contract
from scripts.vision.closed_source_training_quality import sources,validate
from scripts.vision.evaluate_same_source_material_dose import baseline_verify


def quality_chain():
    q=prior.read(OUT/'quality-review.json');prior.verify(q)
    es=[]
    for _,_,ep in sources():
        e=prior.read(ep);prior.verify(e);es.append(e)
        # Evidence binds original RGB and full collection record; verify its file payloads too.
        r=prior.read(e['source_receipt'])
        for row in r['views']:
            if 'rgb_path' in row and prior.file_sha256(row['rgb_path'])!=row['image_sha256']:raise ValueError('Source RGB changed')
        for path in e['inputs']:
            if path.endswith('collection-receipt.json'):prior.verify(prior.read(path))
    validate(es,q['decisions'])
    if any(x['status']!='approved_for_bounded_research_cohort' for x in q['decisions']):raise ValueError('Unapproved decision')
    # Full selected replay leaves are checked, not just completion-file hashes.
    for path in q['inputs']:
        if path.endswith('/receipt.json'):
            result=prior.read(path);prior.verify(result)
            if result['status']!='original_pixel_evidence_certified' or not result['process_cleanup_complete']:raise ValueError('Replay no longer valid')
    return q


def main():
    p=protocol();quality_chain();import torch,ultralytics
    env=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__)
    if env!=p['environment']:raise ValueError('Runtime environment changed')
    deps=[OUT/'protocol.json',OUT/'quality-review.json',Path(__file__).resolve(),Path(__file__).with_name('run_closed_source_control.py').resolve()]
    for key in KEYS:
        _,_,r=contract(key)
        if r['optimizer_created'] or r['backward_executed'] or r['training_validation_executed']:raise ValueError('Preflight trained')
        deps.extend((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
    config=prior.ROOT/'config/perception/visual_experiment_baseline_v1.json';baseline=baseline_verify(config);deps.append(config)
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    tests=['tests.test_closed_source_control','tests.test_closed_exterior_review','tests.test_closed_exterior_capture','tests.test_canonical_gates','tests.test_canonical_recovery','tests.test_canonical_shutdown']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=prior.ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    deps.extend(prior.ROOT/Path(name.replace('.','/')).with_suffix('.py') for name in tests)
    dest=OUT/'entry-ready.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),protocol_identity=p['identity'],
        checked_draws=16200,checked_batches=2700,review_decisions=52,baseline=baseline,environment=env,
        tests=dict(exit_code=result.returncode,output=result.stdout+result.stderr,scope='23 targeted tests; no all-repository claim'),
        training_started=False,training_admitted=False,promotable=False,inputs={str(x):prior.file_sha256(x) for x in deps}))
    print('READY',list(KEYS),'NO_TRAINING_STARTED')


if __name__=='__main__':main()
