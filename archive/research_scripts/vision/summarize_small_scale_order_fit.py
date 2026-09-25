"""Require all twelve fit units; distinguish exposure and paired members."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior,runtime


def stats(rows):
    truths=Counter(t['class_name'] for r in rows for t in r['truth'])
    hits=Counter(m['class_name'] for r in rows for m in r['matches'])
    negatives=[r for r in rows if not r['truth']]
    return dict(images=len(rows),truth=dict(truths),matched=dict(hits),
        recall={k:hits[k]/v for k,v in truths.items()},
        negative_frames=len(negatives),negative_frames_with_prediction=sum(bool(r['predictions']) for r in negatives),
        negative_fpr=sum(bool(r['predictions']) for r in negatives)/len(negatives) if negatives else None,
        miss_reasons=dict(Counter(m['reason'] for r in rows for m in r['misses'])),
        matching_competitions=sum(m['formal_matching_competition'] for r in rows for m in r['misses']))


def run():
    p=freeze();results={};units={};pairs={};deps=[OUT/'protocol.json',Path(__file__).resolve()]
    for key in p['models']:
        path=OUT/(key+'.json');r=prior.read(path);runtime.validate(r,key,p);deps.append(path)
        results[key]={x['member_id']:x for x in r['rows']}
        units[key]={status:stats([x for x in r['rows'] if bool(p['actual_exposures'][key].get(x['member_id'],0))==exposed])
                    for status,exposed in [('exposed_training_fit',True),('unexposed_not_fit',False)]}
    for key in p['models']:
        if not key.startswith('I'):continue
        before=key[1:];common=set(p['actual_exposures'][key])&set(p['actual_exposures'][before])
        transitions=Counter();events=[]
        for mid in sorted(common):
            a,b=results[before][mid],results[key][mid]
            if a['truth']!=b['truth']:raise ValueError('Paired full truth differs')
            ah={m['truth_index'] for m in a['matches']};bh={m['truth_index'] for m in b['matches']}
            for i,t in enumerate(a['truth']):
                state='retained_hit' if i in ah and i in bh else 'loss' if i in ah else 'gain' if i in bh else 'persistent_miss'
                transitions[state]+=1;events.append(dict(member_id=mid,truth=t,transition=state))
        pairs[key]=dict(reference=before,common_members=len(common),
            before=stats([results[before][m] for m in sorted(common)]),
            after=stats([results[key][m] for m in sorted(common)]),transitions=dict(transitions),events=events)
    dest=OUT/'fit-summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='twelve_model_fit_complete_not_dataset_admission',units=units,paired=pairs,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(run()['status'])
