"""Combine fixed pilot and expansion; instance-paired contrasts, never train."""
from collections import Counter,defaultdict
from pathlib import Path
from scripts.vision import infer_transfer_pilot_v2 as pilot
from scripts.vision import infer_transfer_expansion as expansion

OUT=pilot.base.DESIGN/'analysis-v1'
CONTRASTS=(('original_control','gray_target_body'),('gray_target_body','gray_target_full'),
           ('gray_target_body','gray_all_body'),('gray_all_body','gray_all_full'),('warm','gray_target_body'),('cool','gray_target_body'))


def main():
    prior=pilot.base.prior;records=[];paths=[Path(__file__).resolve()];totals=defaultdict(Counter)
    for mod in (pilot,expansion):
        p=mod.freeze();paths.append(mod.OUT/'protocol.json')
        for key in p['models']:
            fp=mod.OUT/(key+'.json');r=prior.read(fp);pilot.base.validate(r,key,p);paths.append(fp)
            for m,row in zip(p['members'],r['rows'],strict=True):
                hits={x['truth_index'] for x in row['matches']}
                for scope in ('all','planned'):
                    a=totals[key,m['condition'],scope]
                    if scope=='all':a['unmatched_predictions']+=row['unmatched_prediction_count'];a['predictions']+=len(row['predictions'])
                    for i,t in enumerate(row['truth']):
                        if scope=='planned' and t['object_id']!=m['target']:continue
                        a['truth']+=1;a['hits']+=i in hits
                for i,t in enumerate(row['truth']):
                    records.append(dict(model=key,source_id=m['source_id'],condition=m['condition'],object_id=t['object_id'],
                        category=t['class_name'],bbox_xyxy=t['bbox_xyxy'],planned=t['object_id']==m['target'],hit=i in hits,
                        miss=next((x for x in row['misses'] if x['truth_index']==i),None)))
    idx={tuple(x[k] for k in ('model','source_id','condition','object_id')):x for x in records}
    if len(idx)!=len(records):raise ValueError('Duplicate instance identity')
    changes=[]
    for before,after in CONTRASTS:
        for x in records:
            if x['condition']!=before:continue
            y=idx[x['model'],x['source_id'],after,x['object_id']]
            if x['category']!=y['category'] or x['bbox_xyxy']!=y['bbox_xyxy']:raise ValueError('Unpaired full truth')
            changes.append(dict(model=x['model'],source_id=x['source_id'],object_id=x['object_id'],category=x['category'],
                planned=x['planned'],before=before,after=after,
                outcome=('persistent_hit' if y['hit'] else 'loss') if x['hit'] else ('gain' if y['hit'] else 'persistent_miss'),
                before_miss=x['miss'],after_miss=y['miss']))
    OUT.mkdir(exist_ok=True);dest=OUT/'summary.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    return prior.frozen(dest,dict(status='twelve_pose_existing_weight_factorial_complete',instances=records,paired_changes=changes,
        totals=[dict(model=k[0],condition=k[1],scope=k[2],**v) for k,v in totals.items()],
        source_poses=12,training_ready=False,training_started=False,training_admitted=False,promotable=False,
        limits=['Same layout/assets; seeds and variants are repeated observations, not independent scenes.',
                'Structural no-op panel conditions retained; image changes do not establish panel visibility.',
                'This diagnoses existing weights, not efficacy of future training.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=main()
    for x in r['totals']:
        if x['scope']=='planned':print(x['model'],x['condition'],x['hits'],x['truth'])
