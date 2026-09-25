"""Historical-dose and same-source replacement capacity audit. Never train."""
from collections import Counter
from pathlib import Path
from scripts.vision import train_material_retention_coverage as reference
from scripts.vision import train_compensated_double_dose as historical

prior=reference.prior
OUT=reference.OUT.parent/'material-dose-feasibility-v1'


def capacity(p,key):
    rows={r['member_id']:r for r in p['pool_rows']};counts=Counter(p['schedules'][key]);eligible=[]
    for mid,n in sorted(counts.items()):
        r=rows[mid]
        if r.get('variant')!='original' or 'full_truth' not in r:continue
        variants=[v for v in rows.values() if v.get('pair_id')==r.get('pair_id') and v.get('variant') in ('warm','cool','gray_target_body')]
        # This is structural feasibility, not permission to increase a member's exposure.
        valid=[v for v in variants if v['class_instances']==r['class_instances'] and Path(v['label_path']).read_bytes()==Path(r['label_path']).read_bytes()]
        if valid:
            eligible.append(dict(member_id=mid,pair_id=r['pair_id'],original_exposures=n,
                maximum_replacements=n,preserve_one_original_capacity=max(0,n-1),
                variants=[v['member_id'] for v in valid],
                positions=[i for i,m in enumerate(p['schedules'][key]) if m==mid]))
    return dict(cell=key,material_exposures=sum(n for m,n in counts.items() if rows[m].get('variant') in ('warm','cool','gray_target_body')),
        eligible=eligible,maximum_same_source_extra=sum(x['maximum_replacements'] for x in eligible),
        preserve_one_original_extra=sum(x['preserve_one_original_capacity'] for x in eligible))


def main():
    p,_,_=reference.contract('T-7');paths=[reference.OUT/'protocol.json',Path(__file__).resolve()]
    cells=[]
    for seed in (7,17,27):
        reference.complete(f'T-{seed}');cells.append(capacity(p,f'T-{seed}'))
        paths.append(reference.OUT/'training'/f'T-{seed}'/'completion.json')
    hp=historical.OUT/'protocol.json';h=prior.read(hp);prior.verify(h);paths.append(hp)
    from scripts.vision.freeze_compensated_double_dose import checks
    from scripts.vision import train_compensated_material as low
    lp=low.OUT/'protocol.json';l=prior.read(lp);prior.verify(l);checks(l,h);paths.append(lp)
    for seed in (7,17,27):
        for family in ('V','VM'):
            key=f'{family}-{seed}';historical.complete(key);low.complete(key)
            for root in (historical.OUT,low.OUT):
                cp=root/'training'/key/'completion.json';ep=root/'evaluation'/f'{key}.json'
                prior.verify(prior.read(ep));paths.extend((cp,ep))
    from scripts.vision.double_dose_pool_fit import OUT as FIT
    fit=FIT/'dose-comparison.json'
    prior.verify(prior.read(fit));paths.append(fit)
    OUT.mkdir(exist_ok=True);dest=OUT/'audit.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='capacity_and_history_checked_not_training_ready',cells=cells,
        historical_contrast='54 to 108 new-source exposures, replacing old bridge members; not fixed-source pure material dose',
        interpretation='Same-source replacement exchanges original appearance for material exposure; cannot claim pure additive effect.',
        quality_gate='Latest full-label review and member-specific risk caps must be checked before any new increase.',
        training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=main();print(r['status'])
    for c in r['cells']:print(c['cell'],c['material_exposures'],c['maximum_same_source_extra'],c['preserve_one_original_extra'])
