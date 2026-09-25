"""Full-instance same-source diagnostics with actual exposure classification."""
from pathlib import Path
from scripts.vision.diagnose_material_retention_transfer import OUT,PREVIOUS,prior,freeze,validate
from scripts.vision.summarize_gray_fit_transfer import target_index


def main():
    p=freeze();dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    rows=[];counts=[];paths=[OUT/'protocol.json',Path(__file__).resolve()]
    for key in p['models']:
        path=OUT/(key+'.json');r=prior.read(path);validate(r,key,p);paths.append(path)
        arm,seed=key.split('-');actual=p['actual_exposures'][key]
        for m,x in zip(p['members'],r['rows'],strict=True):
            j=target_index(m,x);hit={v['truth_index'] for v in x['matches']}
            for i,t in enumerate(x['truth']):
                rows.append(dict(arm=arm,seed=int(seed),source_id=m['source_id'],condition=m['condition'],member_id=m['member_id'],
                    object_id=t['object_id'],class_name=t['class_name'],bbox_xyxy=t['bbox_xyxy'],planned_target=i==j,
                    hit=i in hit,miss=next((v for v in x['misses'] if v['truth_index']==i),None),
                    exact_member_exposures=actual.get(m['member_id'],0),
                    exposure_caveat='Zero exact ID does not certify unseen pixels; identical same-source condition aliases remain grouped.'))
        for condition in sorted({m['condition'] for m in p['members']}):
            rs=[x for x in rows if x['arm']==arm and x['seed']==int(seed) and x['condition']==condition]
            targets=[x for x in rs if x['planned_target']]
            counts.append(dict(arm=arm,seed=int(seed),condition=condition,target_hits=sum(x['hit'] for x in targets),target_count=len(targets),
                full_hits=sum(x['hit'] for x in rs),full_count=len(rs),
                actual_exposed_targets=sum(x['exact_member_exposures']>0 for x in targets),
                actual_exposed_target_hits=sum(x['hit'] for x in targets if x['exact_member_exposures']>0)))
    previous=PREVIOUS/'summary.json';old=prior.read(previous);prior.verify(old);paths.append(previous)
    return prior.frozen(dest,dict(status='six_model_same_source_diagnostic_complete',counts=counts,instances=rows,
        historical_counts=old['counts'],independent_source_poses=12,training_admitted=False,promotable=False,
        interpretation='Full truth retained. Fit is only actual exposed members; same-source variants are not blind tests.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
