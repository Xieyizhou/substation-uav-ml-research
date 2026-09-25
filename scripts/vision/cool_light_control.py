"""Same interleaved slots: neutral illumination versus viewed cool recipe."""
import copy
from pathlib import Path
from scripts.vision.cool_light_capture import OUT, prior
from scripts.vision.cool_light_dataset import export
from scripts.vision.interleaved_light_control import OUT as REF
from scripts.vision.freeze_closed_source_control import counts

KEYS = tuple(f'IC1000-{seed}' for seed in (7, 17, 27))


def substitution(ref, members):
    new = {r['source_member_id']: r for r in members}
    if len(new) != 32 or len({r['member_id'] for r in members}) != 32:
        raise ValueError('Incomplete or duplicate counterpart inventory')
    old = [r for r in ref['pool_rows'] if r.get('illumination') == 'physical_low_neutral']
    if len(old) != 32 or {r['source_member_id'] for r in old} != set(new):
        raise ValueError('Same-source counterpart mismatch')
    return {r['member_id']: new[r['source_member_id']]['member_id'] for r in old}


def constraints(p, ref, members):
    swap = substitution(ref, members)
    rows = {r['member_id']: r for r in p['pool_rows']}
    if len(rows) != len(p['pool_rows']):
        raise ValueError('Duplicate pool identity')
    if p['pool_rows'] != ref['pool_rows'] + members:
        raise ValueError('Pool altered beyond counterparts')
    for field in ('initialization', 'evaluation', 'held_members', 'environment', 'acceptance_policy', 'names'):
        if p[field] != ref[field]:
            raise ValueError('Non-lighting input drift: '+field)
    for r in rows.values():
        for kind in ('image', 'label'):
            if prior.file_sha256(r[kind+'_path']) != r[kind+'_sha256']:
                raise ValueError('Member content drift')
    for key in KEYS:
        oldkey = 'IL1000-'+key.split('-')[-1]
        before, after = ref['schedules'][oldkey], p['schedules'][key]
        if after != [swap.get(m, m) for m in before] or len(after) != 6000:
            raise ValueError('Sequence not exact substitution')
        if p['brightness_factors'][key] != ref['brightness_factors'][oldkey]:
            raise ValueError('Frozen brightness changed')
        cfg = p['training_config'][key]
        if cfg != ref['training_config'][oldkey] or cfg['epochs'] != 100 or cfg['lr0'] != .0005:
            raise ValueError('Training configuration changed')
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        batches = {i//6 for i in changed}
        if len(changed) != 300 or len(batches) != 50 or any(sum(i//6 == b for i in changed) != 6 for b in batches):
            raise ValueError('Not 50 intact treatment batches')
        for a, b in zip(before, after):
            if b in p['held_members']:
                raise ValueError('Held member restored')
            if rows[a]['subset'] == 'hard_negative' and a != b:
                raise ValueError('Negative position changed')
            if rows[a]['lineage_id'] != rows[b]['lineage_id'] or rows[a]['class_instances'] != rows[b]['class_instances']:
                raise ValueError('Supervision lineage/count changed')
            if Path(rows[a]['label_path']).read_bytes() != Path(rows[b]['label_path']).read_bytes():
                raise ValueError('Full labels changed')
        total, original = counts(after, rows), counts(before, rows)
        if any(total[k] != original[k] for k in ('classes', 'subsets', 'lineages')):
            raise ValueError('Exposure budget changed')
        ledger = [dict(first_step=i//6+1, **counts(after[i:i+300], rows)) for i in range(0, 6000, 300)]
        if p['totals'][key] != total or p['ledger'][key] != ledger:
            raise ValueError('Exposure ledger mismatch')


def freeze():
    manifest = export()
    ref = prior.read(REF/'design.json')
    prior.verify(ref)
    members = manifest['members']
    deps = [REF/'design.json', OUT/'capture-protocol.json', OUT/'condition-audit.json',
            OUT/'pose-role-audit.json', OUT/'quality-review.json', OUT/'export/manifest.json',
            REF/'evaluation/error-review-v1/completion.json', REF/'pool-fit-diagnosis-v1/summary.json',
            prior.ROOT/'docs/plans/ml_cool_light_control_20260914.md', Path(__file__).resolve()]
    for seed in (7, 17, 27):
        cp = REF/'training'/f'IL1000-{seed}'/'completion.json'
        c = prior.read(cp)
        prior.verify(c)
        xp = Path(c['exposure_path'])
        x = prior.read(xp)
        prior.verify(x)
        if c['optimizer_steps'] != 1000 or x['actual'] != ref['schedules'][f'IL1000-{seed}'] or prior.file_sha256(c['weights']) != c['weights_sha256']:
            raise ValueError('Historical control invalid')
        deps.extend([cp, xp, Path(c['weights'])])
    for path in deps:
        if path.suffix == '.json':
            prior.verify(prior.read(path))
    dest = OUT/'design.json'
    if dest.exists():
        p = prior.read(dest)
        prior.verify(p)
        constraints(p, ref, members)
        return p
    p = {k: copy.deepcopy(ref[k]) for k in ('pool_rows', 'names', 'initialization', 'evaluation', 'held_members', 'acceptance_policy', 'environment')}
    p['pool_rows'] += members
    p.update(schedules={}, brightness_factors={}, training_config={}, listings={}, totals={}, ledger={})
    rows = {r['member_id']: r for r in p['pool_rows']}
    swap = substitution(ref, members)
    for key in KEYS:
        old = 'IL1000-'+key.split('-')[-1]
        seq = [swap.get(m, m) for m in ref['schedules'][old]]
        p['schedules'][key] = seq
        p['brightness_factors'][key] = copy.deepcopy(ref['brightness_factors'][old])
        p['training_config'][key] = copy.deepcopy(ref['training_config'][old])
        listing = OUT/(key+'.txt')
        text = ''.join(rows[m]['image_path']+'\n' for m in sorted(set(seq)))
        if listing.exists() and listing.read_text() != text:
            raise ValueError('Listing drift')
        if not listing.exists():
            listing.write_text(text)
        p['listings'][key] = str(listing)
        deps.append(listing)
        p['totals'][key] = counts(seq, rows)
        p['ledger'][key] = [dict(first_step=i//6+1, **counts(seq[i:i+300], rows)) for i in range(0, 6000, 300)]
    constraints(p, ref, members)
    return prior.frozen(dest, dict(p, status='cool_recipe_interleaved_control_frozen_preflight_pending',
        comparison='IC1000 versus IL1000 at LR .0005: replace exactly 300 neutral-low exposures in 50 intact interleaved batches with same-source cool-light counterparts; all other slots, full labels and brightness factors preserved.',
        limitations=['Known viewed-development lighting recipe adaptation, not blind evaluation or independent-scene generalization.',
                    '32 derivatives of 8 existing training poses; shared layouts and assets.',
                    'The acquisition template experiment metadata is superseded by condition-audit.json and this protocol.',
                    'Latest .00025 family is background context, not the direct same-configuration control.'],
        training_started=False, inputs={str(d): prior.file_sha256(d) for d in deps}))


if __name__ == '__main__':
    print(freeze()['status'])
