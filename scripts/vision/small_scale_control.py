"""Freeze matched-budget small-pose material tails; historical inputs untouched."""
import copy
from pathlib import Path
from scripts.vision.small_scale_material_capture import OUT, prior
from scripts.vision.neutral_gray_control import OUT as REF
from scripts.vision.small_scale_dataset import export
from scripts.vision.small_scale_schedule import tails, validate_tail
from scripts.vision.freeze_closed_source_control import counts

KEYS = tuple(f'{family}-{seed}' for seed in (7, 17, 27) for family in ('SR1100', 'SM1100'))


def constraints(p, ref, members):
    if p['pool_rows'] != ref['pool_rows'] + members: raise ValueError('Pool drift')
    rows = {r['member_id']: r for r in p['pool_rows']}
    if len(rows) != len(p['pool_rows']): raise ValueError('Duplicate member')
    for field in ('names', 'initialization', 'evaluation', 'held_members', 'acceptance_policy', 'environment'):
        if p[field] != ref[field]: raise ValueError('Fixed contract drift: '+field)
    for r in rows.values():
        for kind in ('image', 'label'):
            if prior.file_sha256(r[kind+'_path']) != r[kind+'_sha256']: raise ValueError('Member hash drift')
    for seed in (7, 17, 27):
        old = f'ICG1000-{seed}'; before = ref['schedules'][old]
        a, b = tails(members, seed)
        validate_tail(a, b, members)
        for family, tail in (('SR1100', a), ('SM1100', b)):
            key = f'{family}-{seed}'; seq = p['schedules'][key]
            if len(before) != 6000 or seq != before + tail: raise ValueError('Prefix/tail sequence drift')
            if set(seq) & set(p['held_members']): raise ValueError('Held member restored')
            gains = ref['brightness_factors'][old]
            if p['brightness_factors'][key] != gains + gains[:600]: raise ValueError('Brightness drift')
            cfg = copy.deepcopy(ref['training_config'][old]); cfg['epochs'] = 110
            if p['training_config'][key] != cfg: raise ValueError('Training configuration drift')
            if p['totals'][key] != counts(seq, rows): raise ValueError('Total exposure ledger drift')
            ledger = [dict(first_step=i//6+1, **counts(seq[i:i+300], rows)) for i in range(0, 6600, 300)]
            if p['ledger'][key] != ledger: raise ValueError('Window ledger drift')
        for field in ('classes', 'lineages', 'subsets'):
            if p['totals'][f'SR1100-{seed}'][field] != p['totals'][f'SM1100-{seed}'][field]:
                raise ValueError('Unmatched family budget')


def freeze():
    manifest = export(); members = manifest['members']
    rp = REF/'design.json'; ref = prior.read(rp); prior.verify(ref)
    deps = [rp, OUT/'capture-protocol.json', OUT/'export/manifest.json', OUT/'quality-review.json',
            Path(__file__).resolve(), Path(__file__).with_name('small_scale_schedule.py')]
    for seed in (7, 17, 27):
        cp = REF/'training'/f'ICG1000-{seed}'/'completion.json'
        c = prior.read(cp); prior.verify(c)
        xp = Path(c['exposure_path']); x = prior.read(xp); prior.verify(x)
        if c['optimizer_steps'] != 1000 or x['actual'] != ref['schedules'][f'ICG1000-{seed}']:
            raise ValueError('Historical actual prefix mismatch')
        if prior.file_sha256(c['weights']) != c['weights_sha256']: raise ValueError('Historical weights drift')
        deps += [cp, xp, Path(c['weights'])]
    dest = OUT/'design.json'
    if dest.exists():
        p = prior.read(dest); prior.verify(p); constraints(p, ref, members); return p
    p = {k:copy.deepcopy(ref[k]) for k in ('pool_rows', 'names', 'initialization', 'evaluation', 'held_members', 'acceptance_policy', 'environment')}
    p['pool_rows'] += members
    p.update(schedules={}, brightness_factors={}, training_config={}, listings={}, totals={}, ledger={})
    rows = {r['member_id']:r for r in p['pool_rows']}
    for seed in (7, 17, 27):
        old = f'ICG1000-{seed}'
        for family, tail in zip(('SR1100', 'SM1100'), tails(members, seed)):
            key = f'{family}-{seed}'; seq = ref['schedules'][old] + tail
            p['schedules'][key] = seq
            p['brightness_factors'][key] = ref['brightness_factors'][old] + ref['brightness_factors'][old][:600]
            cfg = copy.deepcopy(ref['training_config'][old]); cfg['epochs'] = 110
            p['training_config'][key] = cfg
            listing = OUT/(key+'.txt'); text = ''.join(rows[m]['image_path']+'\n' for m in sorted(set(seq)))
            if listing.exists() and listing.read_text() != text: raise ValueError('Listing drift')
            if not listing.exists(): listing.write_text(text)
            p['listings'][key] = str(listing); deps.append(listing)
            p['totals'][key] = counts(seq, rows)
            p['ledger'][key] = [dict(first_step=i//6+1, **counts(seq[i:i+300], rows)) for i in range(0,6600,300)]
    constraints(p, ref, members)
    return prior.frozen(dest, dict(p, status='small_scale_control_frozen_preflight_pending', training_started=False,
        comparison='SR1100 versus SM1100: half of 600 appended small-pose exposures replaced with same-pose gray material; identical light positions, full labels, brightness and 6000-draw historical prefix.',
        limitations=['Four pose groups, three rearranged layouts, existing shared assets; not independent-asset generalization.',
                    'ICG1000 has a shorter budget; it is descriptive context, not the matched-budget material comparator.',
                    'Gray tail has 152 normal-light and 148 cool-light draws; all tail lighting totals remain 300/300.',
                    'No absolute material full-image recall threshold is retroactively introduced.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(freeze()['status'])
