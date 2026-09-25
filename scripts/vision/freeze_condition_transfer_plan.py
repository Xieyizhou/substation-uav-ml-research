"""Freeze an actionable review queue and conditional design, never training readiness."""
import subprocess,sys
from pathlib import Path
from scripts.vision.condition_transfer_design import OUT,SOURCE,ROOT,run,read,verify,frozen,file_sha256,baseline_verify


def main():
    c=run();p=read(SOURCE/'protocol.json');verify(p);idx={r['member_id']:r for r in p['pool_rows']}
    train=[]
    for t in c['targets']:
        if t['panel_condition'] not in ('unknown','visible_low_contrast'):continue
        train.append(dict(target=t,source=idx[t['member_id']],priority='missing_condition' if t['panel_condition']=='unknown' else 'replay_eligibility_recheck',decision=None))
    sources={}
    for row in read(p['evaluation']['paired_review'])['frames']:sources[row['view_id'],row['variant']]=row
    dev=[dict(target=t,source=sources[t['view_id'],t['variant']],priority='material_persistent_miss' if t['variant']=='material' and all(not s['hit'] for s in t['states']) else 'lighting_switchgear_persistent_miss' if t['variant']=='lighting' and t['truth']['class_name']=='switchgear' and all(not s['hit'] for s in t['states']) else 'retain_all_outcomes',decision=None) for t in c['development_targets']]
    paths=[OUT/'coverage.json',SOURCE/'protocol.json',Path(p['evaluation']['paired_review']),Path(__file__),ROOT/'docs/condition-transfer-design-v1.md']
    for r in dev:
        src=r['source']
        if file_sha256(src['image_path'])!=src['image_sha256']:raise ValueError('Stale development RGB')
        paths.append(Path(src['image_path']))
    queue=OUT/'review-queue.json'
    if queue.exists():verify(read(queue))
    else:frozen(queue,dict(status='awaiting_explicit_condition_review',training_targets=train,development_targets=dev,automatic_decisions_created=0,
        inputs={str(x):file_sha256(x) for x in paths}))
    tests=['tests.test_condition_transfer_design','tests.test_switchgear_condition_review','tests.test_brightness_lr_retention','tests.test_brightness_lr_audit','tests.test_brightness_audit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    b=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Pinned integrity failure')
    paths += [queue,ROOT/'scripts/vision/condition_transfer_design.py']+[ROOT/(x.replace('.','/')+'.py') for x in tests]
    cp=OUT/'design-receipt.json'
    if cp.exists():verify(read(cp))
    else:frozen(cp,dict(status='design_frozen_condition_review_required',training_ready=False,training_started=False,new_renderings=0,new_inference_units=0,
        training_review_targets=len(train),development_review_targets=len(dev),named_gaps=['43 capacitor panel/body conditions not yet linked','1 switchgear panel condition unknown','5 low-contrast events need replay eligibility check','Exact pilot member identities and component map pending review','Unmodified replay RGB alignment not yet tested'],
        baseline=b,regression_output=t.stderr,whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('DESIGN_FROZEN_NOT_TRAINING_READY',len(train),len(dev));print(t.stderr)


if __name__=='__main__':main()
