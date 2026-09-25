"""Versioned paired tail generation, independent of model outcomes."""
import hashlib
from collections import Counter

VERSION = 'small-scale-material-tail-v1'


def digest(seed, *parts):
    return hashlib.sha256('|'.join(map(str, (VERSION, seed, *parts))).encode()).hexdigest()


def tails(members, seed):
    lookup = {}
    for r in members:
        key = (r['pair_id'], r['illumination'], r['variant'])
        if key in lookup: raise ValueError('Duplicate counterpart')
        lookup[key] = r['member_id']
    pairs = sorted({r['pair_id'] for r in members})
    if len(pairs) != 4 or len(lookup) != 16: raise ValueError('Incomplete inventory')
    for pair in pairs:
        for light in ('normal', 'cool'):
            for variant in ('original', 'gray035'):
                if (pair, light, variant) not in lookup: raise ValueError('Missing counterpart')
    # Twenty-five batches per pose; deterministic mixing of pose batches.
    blocks = sorted(((pair, n) for pair in pairs for n in range(25)),
                    key=lambda x: digest(seed, 'batch', *x))
    reference, material = [], []
    for pair, n in blocks:
        lights = sorted([(light, j) for light in ('normal', 'cool') for j in range(3)],
                        key=lambda x: digest(seed, 'within', pair, n, *x))
        # Three gray draws per batch; alternate 2/1 between light conditions.
        chosen = set()
        for light in ('normal', 'cool'):
            slots = [i for i, x in enumerate(lights) if x[0] == light]
            slots.sort(key=lambda i: digest(seed, 'gray', pair, n, i))
            count = 2 if (n % 2 == 0) == (light == 'normal') else 1
            chosen.update(slots[:count])
        for i, (light, _) in enumerate(lights):
            reference.append(lookup[pair, light, 'original'])
            material.append(lookup[pair, light, 'gray035' if i in chosen else 'original'])
    validate_tail(reference, material, members)
    return reference, material


def validate_tail(reference, material, members):
    rows = {r['member_id']: r for r in members}
    if len(rows) != len(members) or len(reference) != 600 or len(material) != 600:
        raise ValueError('Tail size/identity failure')
    quota = Counter(); gray = 0
    for a, b in zip(reference, material):
        x, y = rows[a], rows[b]
        if x['variant'] != 'original' or y['variant'] not in ('original', 'gray035'):
            raise ValueError('Unplanned material')
        for field in ('pair_id', 'illumination', 'lineage_id', 'class_instances', 'label_sha256'):
            if x[field] != y[field]: raise ValueError('Paired supervision/position changed: '+field)
        quota[x['pair_id'], x['illumination']] += 1
        gray += y['variant'] == 'gray035'
    if len(quota) != 8 or set(quota.values()) != {75} or gray != 300:
        raise ValueError('Exposure quota failure')
    for i in range(0, 600, 6):
        if sum(rows[m]['variant'] == 'gray035' for m in material[i:i+6]) != 3:
            raise ValueError('Batch material quota failure')
