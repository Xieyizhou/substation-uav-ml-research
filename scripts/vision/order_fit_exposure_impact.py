"""Exact arithmetic impact of quality holds, not a replacement sampler or gate."""
from collections import Counter
from math import ceil, gcd
from functools import reduce
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def subset_impact(members, counts, retained):
    budget=sum(counts.get(m['member_id'],0) for m in members)
    available=[m for m in members if m['member_id'] in retained]
    previous_positive=[m for m in available if counts.get(m['member_id'],0)>0]
    old=sum(counts.get(m['member_id'],0) for m in available)
    classes=Counter()
    for m in members:
        for name,n in m['class_instances'].items():classes[name]+=n*counts.get(m['member_id'],0)
    modular=[]
    for name,total in sorted(classes.items()):
        coeffs=[m['class_instances'].get(name,0) for m in available]
        divisor=reduce(gcd,coeffs,0)
        possible=(total==0) if divisor==0 else total%divisor==0
        modular.append(dict(class_name=name,old_instance_exposures=total,coefficient_gcd=divisor,
            same_subset_exact_preservation_passes_necessary_divisibility=possible,
            supporting_member_ids=[m['member_id'] for m in available if m['class_instances'].get(name,0)]))
    return dict(original_members=len(members),remaining_members=len(available),budget=budget,
        historical_draws_on_remaining=old,draws_requiring_reassignment=budget-old,
        old_max_member_exposure=max((counts.get(m['member_id'],0) for m in members),default=0),
        minimum_possible_max_draws_using_all_remaining=ceil(budget/len(available)) if available else None,
        minimum_possible_max_draws_without_restoring_zero_exposure=ceil(budget/len(previous_positive)) if previous_positive else None,
        formerly_zero_members=[m['member_id'] for m in available if not counts.get(m['member_id'],0)],
        same_subset_class_necessary_conditions=modular)


def run():
    dest=OUT/'quality-hold-exposure-impact.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'closeout-inventory-v2.json',Path(__file__).resolve()]
    p,q=[prior.read(path) for path in paths[:2]]
    for r in (p,q):prior.verify(r)
    retained={m['member_id'] for m in q['members'] if m['stage']=='quality_supported_source_role_gate_pending'}
    units=[]
    for key,counts in p['actual_exposures'].items():
        units.append(dict(key=key,subsets={subset:subset_impact([m for m in p['members'] if m['subset']==subset],counts,retained)
            for subset in sorted({m['subset'] for m in p['members']})}))
    return prior.frozen(dest,dict(status='exposure_impact_measured_no_schedule_or_training_started',units=units,
        interpretation=[
            'All remaining candidates are optimistically included before final source-role gates; further holds can only tighten these capacity bounds.',
            'GCD tests apply only to preserving each class within its original subset. They do not establish infeasibility of globally compensated class exposures.',
            'Zero-exposure preservation is reported as a separate scenario, not imposed as a new scientific constraint.',
            'Repeating a member or a seed does not increase independent source coverage.',
            'No original budget, label, member status, augmentation or threshold was changed.'],
        training_started=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':
    r=run()
    for unit in r['units']:
        if unit['key'].startswith('ISR'):
            s=unit['subsets']['regular'];print(unit['key'],s['draws_requiring_reassignment'],s['minimum_possible_max_draws_using_all_remaining'],s['minimum_possible_max_draws_without_restoring_zero_exposure'])
