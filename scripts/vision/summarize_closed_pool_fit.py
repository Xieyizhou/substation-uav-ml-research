"""Describe fit on actual exposures; never select a model or change data."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_closed_pool_fit import OUT,KEYS,freeze,validate,prior

def metrics(rows):
    truth=Counter();hit=Counter();reasons=Counter()
    for row in rows:
        for t in row['truth']:truth[t['class_name']]+=1
        for m in row['matches']:hit[row['truth'][m['truth_index']]['class_name']]+=1
        for m in row['misses']:reasons[m['reason']]+=1
    return dict(images=len(rows),instances=dict(truth),matched=dict(hit),recall={c:hit[c]/n for c,n in truth.items()},
        miss_reasons=dict(reasons),unmatched_predictions=sum(r['unmatched_prediction_count'] for r in rows),
        negative_images=sum(not r['truth'] for r in rows),negative_false_positive_images=sum(not r['truth'] and bool(r['predictions']) for r in rows))

def main():
    p=freeze();dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    records={};deps=[OUT/'protocol.json',Path(__file__).resolve()]
    for k in KEYS:
        path=OUT/(k+'.json');r=prior.read(path);validate(r,p,k);records[k]=r;deps.append(path)
    common=set.intersection(*[{r['member_id'] for r in v['rows'] if r['exposures']>0} for v in records.values()])
    members={m['member_id']:m for m in p['members']};results={}
    for key,r in records.items():
        exposed=[x for x in r['rows'] if x['exposures']>0]
        results[key]=dict(exposed=metrics(exposed),unexposed=metrics([x for x in r['rows'] if not x['exposures']]),
            common_exposed=metrics([x for x in exposed if x['member_id'] in common]),
            by_variant={v:metrics([x for x in exposed if members[x['member_id']]['variant']==v]) for v in sorted({m['variant'] for m in p['members']})},
            new_source_exposed=metrics([x for x in exposed if x['member_id'].startswith('new-')]))
    return prior.frozen(dest,dict(status='fit_diagnosis_not_acceptance',results=results,common_exposed_members=sorted(common),
        unknown_quality_not_filtered=True,interpretation='Full supervision fit, not audited-clean-only fit or generalization. No inference of independent scenes from member IDs.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    r=main()
    for k,v in r['results'].items():print(k,'EXPOSED',v['exposed'],'NEW',v['new_source_exposed'])
