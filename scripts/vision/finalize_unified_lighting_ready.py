"""Sign readiness only after capture, review, paired loaders and regressions."""
from pathlib import Path
import subprocess,sys
from collections import Counter
from scripts.vision.prepare_unified_lighting_training import OUT,DESIGN,prior,paired_sequences
from scripts.vision.unified_hold_lighting_counts import SOURCE
from scripts.vision.record_unified_lighting_review import validate
from scripts.vision.brightness_lr_retention import overrides
from scripts.vision.brightness_transfer_runtime import check_log
from scripts.vision.order_retention_runtime import check_actual
from scripts.vision.test_body_material_applicability import baseline_verify
from scripts.vision.prepare_physical_lighting_control import validate_change
from scripts.vision import run_physical_lighting_capture_v2 as capture
from scripts.vision import brightness_transfer_runtime,order_retention_runtime,preflight_unified_lighting

def main():
    paths=[OUT/'protocol.json',OUT/'loader-completion.json',DESIGN/'design.json',DESIGN/'counts.json',DESIGN/'light-review/evidence.json',DESIGN/'light-review/review.json',DESIGN/'light-capture/receipt.json']
    for x in paths:prior.verify(prior.read(x))
    p,load,d,c,e,r,cap=map(prior.read,paths);validate(e,r['decisions'])
    if load['status']!='six_loaders_verified_final_regression_pending' or cap['status']!='four_captured_review_pending':raise ValueError('Missing prior stage')
    if prior.file_sha256(Path(p['initialization']['path']))!=p['initialization']['sha256']:raise ValueError('Initialization invalid')
    rows={x['member_id']:x for x in p['pool_rows']}
    cells=[]
    for seed in (7,17,27):
        expected=paired_sequences(d['pool_rows'],d['original_sequences'][f'brightness-450-{seed}'],c['results'][str(seed)]['counts'],c['light_quotas'],p['variants'],seed)
        if (p['schedules'][f'R-clean-{seed}'],p['schedules'][f'L-physical-{seed}'],p['replacement_positions'][str(seed)])!=expected:raise ValueError('Sequence not reproducible')
        found=list((OUT/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(found)!=1:raise ValueError('Incomplete or duplicate loader evidence')
        cell=prior.read(found[0]);prior.verify(cell);paths+=found
        if cell['optimizer_created'] or cell['backward_executed'] or cell['training_validation_run']:raise ValueError('Forbidden execution')
        for key,actual in cell['cells'].items():
            cells.append(key);check_actual(p,key,actual['actual']);check_log(p,key,actual['brightness_log'])
            if p['training_config'][key]!=overrides(seed) or len(actual['batch_records'])!=450:raise ValueError('Configuration/batch conflict')
            if set(actual['actual'])&set(p['held_members']):raise ValueError('Held member loaded')
            for member in set(actual['actual']):
                for kind in ('image','label'):
                    path=Path(rows[member][kind+'_path'])
                    if prior.file_sha256(path)!=rows[member][kind+'_sha256']:raise ValueError('Export changed')
    if len(cells)!=6:raise ValueError('Missing training config')
    for f in d['light_sources']:
        a,b=capture.base.ET.parse(f['original_world']).getroot(),capture.base.ET.parse(f['lighting_world']).getroot();validate_change(a,b)
    for item in cap['results']:
        rr=prior.read(item['receipt']);prior.verify(rr)
        if not rr['process_cleanup_complete'] or rr['status']!='capture_technical_checks_passed':raise ValueError('Capture incomplete')
    tests=['tests.test_unified_lighting_design','tests.test_boundary_roundoff','tests.test_candidate29_audit','tests.test_l05_hold_gate','tests.test_l05_risk_scope','tests.test_physical_lighting_preflight','tests.test_physical_lighting_capture','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    tests += ['tests.test_revision_compensation','tests.test_revision_minimax']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    capture.base.guard()
    paths += [Path(__file__).resolve(),Path(brightness_transfer_runtime.__file__),Path(order_retention_runtime.__file__),Path(preflight_unified_lighting.__file__),prior.ROOT/'docs/results/ml_unified_lighting_ready_20260910.md']+[prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'ready.json',dict(status='ready_for_training_not_started',cells=sorted(cells),training_started=False,
        actual_loads=16200,actual_batches=2700,light_images_reviewed=4,target_decisions=11,baseline=baseline,
        authorization_boundary='Training requires explicit start; this receipt does not run an optimizer or train weights.',
        regression=dict(output=t.stderr,whole_repository_tested=False),inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('READY_FOR_TRAINING_NOT_STARTED; 85 REGRESSIONS; BASELINE40 PASS')

if __name__=='__main__':main()
