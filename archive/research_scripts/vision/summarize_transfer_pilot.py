"""Keep all seeds/conditions, including genuine panel no-ops."""
from collections import defaultdict, Counter
from pathlib import Path
from scripts.vision.infer_transfer_pilot_v2 import OUT,freeze,base


def main():
    p=freeze();prior=base.prior;paths=[OUT/'protocol.json',Path(__file__).resolve()]
    records=[]; totals=defaultdict(Counter)
    for key in p['models']:
        fp=OUT/(key+'.json');r=prior.read(fp);base.validate(r,key,p);paths.append(fp)
        for member,row in zip(p['members'],r['rows']):
            hits={x['truth_index'] for x in row['matches']}
            for i,t in enumerate(row['truth']):
                planned=t['object_id']==member['target'];hit=i in hits
                miss=next((x for x in row['misses'] if x['truth_index']==i),None)
                records.append(dict(model=key,source_id=member['source_id'],condition=member['condition'],
                    object_id=t['object_id'],category=t['class_name'],planned=planned,hit=hit,miss=miss))
                for scope in ('all', 'planned' if planned else 'incidental'):
                    a=totals[key,member['condition'],scope];a['truth']+=1;a['hit']+=hit
    result=dict(status='pilot_existing_weight_diagnostic_complete',instances=records,
        totals=[dict(model=k[0],condition=k[1],scope=k[2],**v) for k,v in totals.items()],
        independent_source_poses=4,training_ready=False,training_started=False,
        limits=['One source pose per class: diagnostic pilot, not conclusive training route.',
                'Target-full/body can be identical when target has no panel; not independent replicates.',
                'Full-scene source audit and expansion review remain required before training-route freeze.'],
        inputs={str(x):prior.file_sha256(x) for x in paths})
    dest=OUT/'summary.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    return prior.frozen(dest,result)


if __name__=='__main__':
    r=main()
    for x in r['totals']:
        if x['scope']=='planned':print(x['model'],x['condition'],str(x['hit'])+'/'+str(x['truth']))
