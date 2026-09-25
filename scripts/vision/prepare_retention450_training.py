"""Bind exact-quota design to immutable existing exports; no training on prepare."""
from pathlib import Path
import yaml
from scripts.vision.run_visibility_repair_training_v2 import OUT as PRIOR,ROOT,read,save,file_sha256,verify_tree,baseline_verify
from scripts.vision.design_original_retention_optimization_v2 import OUT as DESIGN
from scripts.vision.exposure_protocol import exposures,NAMES

OUT=DESIGN/'training-adapter-v1'
KEYS=tuple(f'{arm}-450-{seed}' for seed in (7,17,27) for arm in ('retained_reference','retained_appearance'))

def validate_args(args):
    from scripts.vision.run_matched_appearance_training import validate_args as check
    if args.get('epochs')!=45:raise ValueError('Expected 45 epochs / 450 optimizer steps')
    check({**args,'epochs':30})

def prepare():
    dest=OUT/'protocol.json'
    if dest.exists():verify_tree(dest);return read(dest)
    paths=[PRIOR/'protocol.json',DESIGN/'design.json',PRIOR/'post-training-audit-v1/negative-review.json',
        PRIOR/'positive-regression-review-v1/review.json',Path(__file__),ROOT/'scripts/vision/run_retention450_training.py']
    seen=set()
    for path in paths[:4]:verify_tree(path,seen)
    p=read(PRIOR/'protocol.json');d=read(DESIGN/'design.json')
    lookup={r['member_id']:r for r in p['pool_rows']}
    inputs={str(path):file_sha256(path) for path in paths};datasets={}
    OUT.mkdir(parents=True,exist_ok=True)
    for key in KEYS:
        seq=d['schedules'][key]
        if len(seq)!=2700:raise ValueError('Wrong exposure budget')
        listing=OUT/f'{key}.txt';config=OUT/f'{key}.yaml'
        if listing.exists() or config.exists():raise ValueError('Incomplete export retained; use new version')
        listing.write_text('\n'.join(lookup[m]['image_path'] for m in sorted(set(seq)))+'\n')
        config.write_text(yaml.safe_dump(dict(path=str(OUT),train=str(listing),val=str(listing),names=list(NAMES))))
        datasets[key]=str(config)
        for path in (listing,config):inputs[str(path)]=file_sha256(path)
    return save(dest,dict(pool_rows=p['pool_rows'],datasets=datasets,schedules=d['schedules'],
        exposures={k:exposures(p['pool_rows'],v) for k,v in d['schedules'].items()},
        controls={**p['controls'],'optimizer_steps':[450]},acceptance_policy=p['acceptance_policy'],
        retention={**p['retention'],'additional_reference':'retained_reference-450',
            'historical_R300_role':'historical only; do not label R300 same-budget for this 450-step experiment'},
        initialization=p['initialization'],evaluation=p['evaluation'],max_attempts=3,
        candidate_policy='No automatic candidate. All seeds, explicit output review, frozen thresholds, historical A and matched retained_reference-450 retention required.',
        status='frozen_runtime_sampler_gate_required',inputs=inputs,
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')))
