"""Bounded coverage diagnosis linking reviewed reactor conditions to current errors."""
from collections import Counter
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT as SCALE, prior
from scripts.vision.build_scale_reactor_coverage import OUT as INV

def run():
    inventory=INV/'inventory.json'; review=INV/'explicit-review.json'
    misses=SCALE/'evaluation/endpoint-review-v1/explicit-review.json'
    fp=SCALE/'evaluation/false-positive-review-v1/explicit-review.json'
    fit=SCALE/'same-member-scale-fit-v1/summary.json'
    paths=[inventory,review,misses,fp,fit,Path(__file__).resolve()]
    records=[prior.read(p) for p in paths[:5]]
    for r in records:prior.verify(r)
    inv,rv,mr,fr,fs=records[:5]
    if len(rv['decisions'])!=72 or len(mr['decisions'])!=30 or len(fr['decisions'])!=13:raise ValueError('Review population incomplete')
    train_state=Counter(d['visual_state'] for d in rv['decisions'])
    train_by_variant={}
    for v in sorted({d['variant'] for d in rv['decisions']}):
        train_by_variant[v]=dict(Counter(d['visual_state'] for d in rv['decisions'] if d['variant']==v))
    dev_reactor=[d for d in mr['decisions'] if d['truth']['class_name']=='reactor']
    dev_state=Counter(d['visual_state'] for d in dev_reactor)
    dev_by_state={s:sorted(d['review_id'] for d in dev_reactor if d['visual_state']==s) for s in sorted(dev_state)}
    # A source condition is considered represented only when the current reviewed
    # training inventory has an explicit matching visual state. This is descriptive,
    # not a quality threshold or a claim of causal equivalence.
    represented={s:train_state.get(s,0)>0 for s in dev_state}
    missing=[s for s,ok in represented.items() if not ok]
    fitagg=fs['aggregates']
    destination=INV/'coverage-summary.json'
    if destination.exists():r=prior.read(destination);prior.verify(r);return r
    return prior.frozen(destination,dict(status='coverage_diagnosis_complete_source_gap_identified',
        training_reactor_members=inv['reactor_targets'],training_registered_lineages=inv['registered_lineages'],
        training_condition_states=dict(train_state),training_state_by_variant=train_by_variant,
        development_reactor_new_miss_targets=len(dev_reactor),development_reactor_states=dict(dev_state),
        development_reactor_ids_by_state=dev_by_state,condition_states_missing_in_training=missing,
        condition_representation=represented,development_transition_counts=mr['all_transition_counts'],
        false_positive_structure_counts=dict(Counter(d['visual_structure'] for d in fr['decisions'])),
        fit_probe_aggregates=fitagg,
        findings=[
            'Training reactor inventory is dominated by clear_body frames; only three partial_occlusion and five small_clear frames were reviewed, with no explicit occluded or truncated training state.',
            'Six of eight reviewed development reactor new-miss target frames are occluded, truncated or occluded_small; two are clear_body and still fail, so coverage gap is supported but not a sole-cause proof.',
            'Same-member scale fit improves at 0.75 and 0.5 while retaining 59/60 target hits at 1.0; this does not certify development transfer or establish a pure scale cause.',
            'Asset identity remains source-resolved rather than inferred from visual shape; shared lineages and repeated seeds are not independent scenes.'
        ],
        next_priority='source_isolated_reactor_occlusion_truncation_coverage_design',
        training_admitted=False,promotable=False,selected_candidate=None,
        inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':
    r=run();print(r['status']);print(r['condition_states_missing_in_training']);print(r['development_reactor_states'])
