"""Exact-quota correction, preserving the immutable initial draft."""
from collections import Counter
from pathlib import Path
import scripts.vision.design_original_retention_optimization as draft

OUT = draft.OUT / 'exact-quota-v2'

def build(original, lookup, variants):
    if len(original) != 1800:
        raise ValueError('Expected 1800 original exposures')
    # Derive the frozen block quotas, not a positional fraction of a shuffled block.
    quotas = Counter(lookup[m]['subset'] for m in original[:600])
    if quotas['bridge_positive'] != 120 or any(n % 2 for n in quotas.values()):
        raise ValueError('Unexpected original block quotas')
    remaining = Counter({s:n//2 for s,n in quotas.items()})
    tail = []
    for mid in original[600:1200]:
        subset = lookup[mid]['subset']
        if remaining[subset] > 0:
            tail.append(mid); remaining[subset] -= 1
    if any(remaining.values()) or len(tail) != 300:
        raise ValueError('Cannot satisfy half-block quota')
    extra = list(original[:600]) + tail
    reference = list(original) + extra
    treatment = list(reference); count = 0
    for i, mid in enumerate(extra, 1800):
        row = lookup[mid]
        if row['subset'] != 'bridge_positive': continue
        variant = ('neutral_bridge', 'background_bridge')[count % 2]; count += 1
        replacement = variants[(row['lineage_id'], variant)]
        other = lookup[replacement]
        if other['lineage_id'] != row['lineage_id'] or other['class_instances'] != row['class_instances']:
            raise ValueError('Paired supervision changed')
        # Exact exported coordinates and all co-occurring labels, not only counts.
        if Path(row['label_path']).read_bytes() != Path(other['label_path']).read_bytes():
            raise ValueError('Full paired label bytes differ')
        treatment[i] = replacement
    if count != 180: raise ValueError('Wrong appearance quota')
    return reference, treatment

def main():
    draft.OUT = OUT
    draft.build = build
    draft.main()
    p = OUT/'design.json'
    # Bind correction implementation as well as the original serialization code.
    d = draft.read(p)
    d['inputs'][str(Path(__file__))] = draft.file_sha256(Path(__file__))
    d.pop('identity', None)
    draft.save(p, d)

if __name__ == '__main__': main()
