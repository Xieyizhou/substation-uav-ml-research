"""Freeze a faithful repeat with observational checkpoints, not a new data recipe."""
from collections import Counter
from pathlib import Path
from scripts.vision import train_material_retention_coverage as old
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion

prior=old.prior
OUT=old.OUT.parent/'material-learning-trajectory-v1'
KEYS=tuple(f'T-{s}' for s in (7,17,27))


def checkpoints(sequence,rows):
    # Observe all material exposures, not only errors or gray members.
    steps={0,450,*range(50,451,50)}
    for i,mid in enumerate(sequence):
        if rows[mid].get('variant') in ('warm','cool','gray_target_body') and 'full_truth' in rows[mid]:
            step=i//6+1
            steps.update((step-1,step))
    return sorted(steps)


def main():
    paths=[Path(__file__).resolve(),old.OUT/'evaluation/completion.json',
           old.OUT/'fit-transfer-diagnosis-v1/summary.json',old.OUT/'protocol.json']
    for path in paths[1:]:prior.verify(prior.read(path))
    fit=prior.read(paths[2]);cells={}
    for module in (pilot,expansion):
        ep,rp=module.DEST/'evidence.json',module.DEST/'label-review.json'
        e,r=prior.read(ep),prior.read(rp)
        prior.verify(e);prior.verify(r)
        if not module.validate(e,r['decisions']):raise ValueError('Unknown source review')
        paths.extend((ep,rp))
    for key in KEYS:
        old.complete(key)
        p,_,expected=old.contract(key)
        rows={r['member_id']:r for r in p['pool_rows']}
        seq=p['schedules'][key]
        if seq!=expected['actual']:raise ValueError('Historical actual exposure drift')
        arm,seed=key.split('-')
        instances=[r for r in fit['instances'] if r['arm']==arm and r['seed']==int(seed)
                   and r['exact_member_exposures']>0]
        cells[key]=dict(source_protocol_identity=p['identity'],checkpoint_steps=checkpoints(seq,rows),
            image_exposures=dict(Counter(seq)),actual_exposed_instances=instances,
            planned_target_misses=[r for r in instances if r['planned_target'] and not r['hit']],
            checkpoint_selection='terminal_450_only_for_development_judgment',
            source_sequence=seq,brightness_factors=p['brightness_factors'][key],
            training_config=p['training_config'][key])
        cp=old.OUT/'training'/key/'completion.json';c=prior.read(cp)
        paths.extend((cp,Path(c['exposure_path']),Path(c['weights'])))
    for rel in ('docs/results/ml_exposure_order_posttraining_review_20260909.md',
                'docs/results/ml_material_retention_evaluation_20260912.md',
                'docs/plans/ml_material_learning_trajectory_20260912.md'):
        paths.append(prior.ROOT/rel)
    result=dict(status='trajectory_protocol_frozen_instrumentation_preflight_pending',cells=cells,
        interpretation='Same T training repeated with observational checkpoints. No efficacy claim from intermediate selection.',
        training_ready=False,training_started=False,training_admitted=False,promotable=False,
        source_review_scope='Reuse hash-valid full-label reviews; not new visual approval or pixel certification.',
        required_preflight=['checkpoint capture preserves model, EMA and RNG state',
                            'actual loader and brightness hashes match historical T',
                            'optimizer count and terminal state reproduction checked',
                            'three bounded attempts with semantic stop and process cleanup'],
        inputs={str(p):prior.file_sha256(p) for p in paths})
    dest=OUT/'protocol.json';OUT.mkdir(exist_ok=True)
    if dest.exists():
        frozen=prior.read(dest);prior.verify(frozen)
        if frozen['cells']!=cells:raise ValueError('Frozen trajectory changed')
        return frozen
    return prior.frozen(dest,result)


if __name__=='__main__':
    r=main();print(r['status'])
    for k,v in r['cells'].items():print(k,len(v['checkpoint_steps']),'checkpoints',len(v['planned_target_misses']),'exposed target misses')
