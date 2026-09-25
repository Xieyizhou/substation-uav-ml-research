"""Read-only matched native training-fit comparison; never infer or train."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision import double_dose_pool_fit as high
from scripts.vision import compensated_pool_fit as low
from src.ml.artifacts import object_sha256


def paired_rows(a, b):
    def index(rows):
        result = {r['member_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate member')
        return result
    aa, bb = index(a), index(b)
    if set(aa) != set(bb):
        raise ValueError('Missing member')
    for mid in sorted(aa):
        x, y = aa[mid], bb[mid]
        if x['truth'] != y['truth'] or x['image_sha256'] != y['image_sha256']:
            raise ValueError('Image or complete label mismatch')
        if (x['actual_exposures'] > 0) != (y['actual_exposures'] > 0):
            raise ValueError('Matched exposed member set changed')
        yield mid, x, y


def main():
    pp = [low.OUT/'protocol.json', high.OUT/'protocol.json']
    lp, hp = [high.prior.read(x) for x in pp]
    low.check(lp); high.check(hp)
    paths = pp + [Path(__file__).resolve()]
    for folder in (low.OUT, high.OUT):
        path = folder/'summary.json'; high.prior.verify(high.prior.read(path)); paths.append(path)
    members = {m['member_id']: m for m in hp['members']}
    if object_sha256(lp['members']) != object_sha256(hp['members']):
        raise ValueError('Frozen pool changed')
    groups, details, seen = {}, [], {}
    for key in high.KEYS:
        units = []
        for folder, p in ((low.OUT, lp), (high.OUT, hp)):
            path = folder/'inference'/f'{key}.json'
            u = high.prior.read(path); high.validate_unit(u, key, p)
            paths.append(path); units.append(u)
        g = defaultdict(Counter); seen[key] = []
        for mid, a, b in paired_rows(units[0]['rows'], units[1]['rows']):
            m = members[mid]
            exposed = a['actual_exposures'] > 0
            if exposed: seen[key].append(mid)
            tags = ['exposed' if exposed else 'unexposed']
            if m['new_compensated_member']:
                tags += [('new_exposed' if exposed else 'new_unexposed')]
                if m.get('variant') in ('warm', 'cool'):
                    tags += [('warm_cool_exposed' if exposed else 'warm_cool_unexposed')]
            elif exposed:
                tags += ['old_exposed']
            ah = {v['truth_index'] for v in a['matches']}
            bh = {v['truth_index'] for v in b['matches']}
            for tag in tags:
                g[tag, 'all']['images'] += 1
                g[tag, 'all']['low_unmatched_predictions'] += a['unmatched_prediction_count']
                g[tag, 'all']['high_unmatched_predictions'] += b['unmatched_prediction_count']
            for i, t in enumerate(a['truth']):
                state = 'persistent_hit' if i in ah and i in bh else 'gain' if i in bh else 'loss' if i in ah else 'persistent_miss'
                for tag in tags:
                    for cls in ('all', t['class_name']):
                        g[tag, cls]['truth'] += 1
                        g[tag, cls]['low_hits'] += i in ah
                        g[tag, cls]['high_hits'] += i in bh
                        g[tag, cls][state] += 1
                details.append(dict(model=key, member_id=mid, truth=t, state=state,
                    low_actual_exposures=a['actual_exposures'], high_actual_exposures=b['actual_exposures'],
                    image_sha256=a['image_sha256'], variant=m.get('variant', 'unknown'),
                    lineage_id=m['lineage_id'], new_member=m['new_compensated_member'],
                    high_miss=next((z for z in b['misses'] if z['truth_index']==i), None)))
        groups[key] = [dict(group=t, category=c, **v) for (t,c),v in sorted(g.items())]
    body = dict(status='matched_native_fit_comparison_complete', groups=groups, instances=details,
        equal_exposed_members_per_seed=True, exposed_member_ids=seen,
        training_fit_not_generalization=True, selected_candidate=None,
        limits=['Native RGB, not every sampled brightness tensor',
                'Repeated seeds and derived materials are not independent scenes',
                'Unexposed rows excluded from training-fit groups',
                'Not whole-pool supervision re-certification'],
        inputs={str(p):high.prior.file_sha256(p) for p in paths})
    path=high.OUT/'dose-comparison.json'
    if path.exists():
        old=high.prior.read(path); high.prior.verify(old)
        for k,v in body.items():
            if old[k] != v: raise ValueError('Frozen comparison drift')
        return old
    return high.prior.frozen(path, body)


if __name__ == '__main__':
    print(main()['status'])
