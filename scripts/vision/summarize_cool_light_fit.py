"""Evidence before the next material-only adaptation; no model or label changes."""
from collections import Counter
from pathlib import Path
from scripts.vision.cool_light_control import OUT, KEYS, prior
from scripts.vision.diagnose_material_late_rehearsal_fit import validate


def main():
    fit = OUT/'pool-fit-diagnosis-v1'
    p = prior.read(fit/'protocol.json'); prior.verify(p)
    members = {m['member_id']:m for m in p['members']}
    deps = [fit/'protocol.json', OUT/'evaluation/error-review-v1/completion.json', OUT/'evaluation/summary.json', Path(__file__).resolve()]
    units = []
    for key in KEYS:
        path = fit/(key+'.json'); r = prior.read(path); validate(r,key,p); deps.append(path)
        groups = {}
        for name in ('all_exposed','cool','normal_neutral','cool_neutral'):
            rows=[]
            for row in r['rows']:
                m=members[row['member_id']]
                if not p['actual_exposures'][key].get(row['member_id'],0):continue
                cool=m.get('illumination')=='physical_cool_viewed_development_recipe'
                if name=='cool' and not cool:continue
                if name=='normal_neutral' and (cool or m.get('variant')!='neutral'):continue
                if name=='cool_neutral' and (not cool or m.get('variant')!='neutral'):continue
                rows.append(row)
            groups[name]=dict(images=len(rows),matched=sum(len(x['matches']) for x in rows),truth=sum(len(x['truth']) for x in rows),
                unmatched_predictions=sum(x['unmatched_prediction_count'] for x in rows),
                misses=[dict(member_id=x['member_id'],miss=m) for x in rows for m in x['misses']])
        units.append(dict(cell=key,groups=groups))
    dest=fit/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='three_seed_fit_diagnostic_complete',units=units,
        finding='Normal-light neutral members are fitted; cool neutral has low-confidence misses in seeds 17 and 27. Development losses include clear material targets; more exposure is not established as the sole remedy.',
        next_direction='Bounded .42-to-.35 neutral-material counterpart test, preserving current cold-light exposure and all other training positions. Known viewed-condition adaptation, not structure invariance or generalization proof.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':print(main()['status'])
