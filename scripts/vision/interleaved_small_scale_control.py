"""Independent six-cell order-only contrast against frozen SR/SM endpoints."""
import copy
from pathlib import Path
from scripts.vision import small_scale_control as source
from scripts.vision.small_scale_interleave import VERSION, permutation, reorder, validate
from scripts.vision.evaluate_small_scale import complete
from scripts.vision.record_small_scale_fp_review import run as review
from scripts.vision.freeze_closed_source_control import counts

prior = source.prior
OUT = source.OUT/'interleaved-tail-control-v1'
KEYS = tuple(f'{family}-{seed}' for seed in (7,17,27) for family in ('ISR1100','ISM1100'))
FIELDS = ('pool_rows','names','initialization','evaluation','held_members','acceptance_policy','environment')


def constraints(p, old):
    for f in FIELDS:
        if p[f] != old[f]: raise ValueError('Fixed input drift: '+f)
    rows = {r['member_id']:r for r in p['pool_rows']}
    if len(rows) != len(p['pool_rows']): raise ValueError('Duplicate member identity')
    for k in KEYS:
        before = k[1:]; order = p['batch_permutations'][k]
        validate(old['schedules'][before], old['brightness_factors'][before], p['schedules'][k], p['brightness_factors'][k], order)
        if p['training_config'][k] != old['training_config'][before]: raise ValueError('Training config drift')
        if p['listings'][k] != old['listings'][before]: raise ValueError('Listing drift')
        if p['totals'][k] != counts(p['schedules'][k],rows) or p['totals'][k] != old['totals'][before]:
            raise ValueError('Exposure totals differ')
        ledger = [dict(first_step=i//6+1,**counts(p['schedules'][k][i:i+300],rows)) for i in range(0,6600,300)]
        if ledger != p['ledger'][k]: raise ValueError('Window ledger drift')


def freeze():
    old = source.freeze(); r = review()
    if r['status'] != 'explicit_visual_review_complete' or len(r['decisions']) != 210:
        raise ValueError('Incomplete false-positive review')
    deps = [source.OUT/'design.json', source.OUT/'evaluation/false-positive-review-v1/review.json',
            source.OUT/'evaluation/false-positive-review-v1/evidence.json', Path(__file__).resolve(),
            Path(__file__).with_name('small_scale_interleave.py')]
    actual_audit = {}
    rows = {r['member_id']:r for r in old['pool_rows']}
    for key in source.KEYS:
        c = complete(key); xp = Path(c['exposure_path']); x = prior.read(xp); prior.verify(x)
        ep = source.OUT/'evaluation'/f'{key}.json'; prior.verify(prior.read(ep))
        deps += [source.OUT/'training'/key/'completion.json',xp,Path(c['weights']),ep]
        actual_audit[key] = dict(total=counts(x['actual'],rows),tail=counts(x['actual'][6000:],rows),
            negative_draws=sum(not rows[m]['class_instances'] for m in x['actual']),
            tail_negative_draws=sum(not rows[m]['class_instances'] for m in x['actual'][6000:]))
    OUT.mkdir(exist_ok=True); dest=OUT/'design.json'
    if dest.exists():
        p=prior.read(dest); prior.verify(p); constraints(p,old); return p
    p = {f:copy.deepcopy(old[f]) for f in FIELDS}
    p.update(schedules={},brightness_factors={},training_config={},listings={},totals={},ledger={},batch_permutations={})
    order=permutation()
    for k in KEYS:
        before=k[1:]; seq=reorder(old['schedules'][before],order)
        p['schedules'][k]=seq; p['brightness_factors'][k]=reorder(old['brightness_factors'][before],order)
        p['batch_permutations'][k]=order
        p['training_config'][k]=copy.deepcopy(old['training_config'][before])
        p['listings'][k]=old['listings'][before]
        deps.append(Path(p['listings'][k]))
        p['totals'][k]=counts(seq,rows)
        p['ledger'][k]=[dict(first_step=i//6+1,**counts(seq[i:i+300],rows)) for i in range(0,6600,300)]
    constraints(p,old)
    return prior.frozen(dest,dict(p,status='order_control_frozen_preflight_pending',version=VERSION,
        training_started=False,historical_actual_exposure=actual_audit,
        comparison='ISR1100 vs SR1100 and ISM1100 vs SM1100: identical 1100 batches with identical member-brightness combinations, only order changed; one appended batch follows each ten historical batches.',
        limitations=['Negative image counts stay fixed but negative exposure positions change as part of the order intervention.',
            'Positive-tail class imbalance and four-source-pose concentration remain unchanged.',
            'Does not isolate optimizer forgetting mechanism, total negative dose, or asset generalization.',
            'Two families and all three seeds are retained regardless of results; no threshold or checkpoint selection.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(freeze()['status'],'NO_TRAINING')
