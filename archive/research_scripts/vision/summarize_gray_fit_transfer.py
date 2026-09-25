"""Summarize full truth and resolved target results without training or approval."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_gray_body_transfer import OUT, PILOT, EXPANSION, prior, freeze, validate
from scripts.vision.infer_transfer_pilot import validate as historical_validate
from scripts.vision import train_compensated_material as old_train


def target_index(member,row):
    ids=[i for i,t in enumerate(row['truth']) if t['object_id']==member['target']]
    if len(ids)!=1:raise ValueError('Ambiguous target identity')
    return ids[0]


def main():
    p=freeze();dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    members={r['member_id']:r for r in p['members']}
    paths=[OUT/'protocol.json',Path(__file__).resolve()]
    rows=[];counts=[];cross=[];exposure=[]
    old_protocol=prior.read(old_train.OUT/'protocol.json');prior.verify(old_protocol)
    for seed in (7,17,27):
        old_train.complete(f'VM-{seed}')
        cp=old_train.OUT/'training'/f'VM-{seed}'/'completion.json';c=prior.read(cp)
        xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        paths.extend([cp,xp])
        new_path=OUT/f'G-{seed}.json';new=prior.read(new_path);validate(new,f'G-{seed}',p);paths.append(new_path)
        historical=[]
        for directory in (PILOT,EXPANSION):
            pp=directory/'protocol.json';hp=prior.read(pp);prior.verify(hp)
            key=f'low-VM-{seed}';ep=directory/f'{key}.json';r=prior.read(ep)
            historical_validate(r,key,hp)
            if hp['models'][key]['weights']!=c['weights']:raise ValueError('Historical endpoint drift')
            paths.extend([pp,ep]);historical.extend(r['rows'])
        actual={'VM':Counter(x['actual']),'G':Counter(p['actual_exposures'][f'G-{seed}'])}
        for arm,result in (('VM',historical),('G',new['rows'])):
            if len(result)!=84 or {r['member_id'] for r in result}!=set(members):raise ValueError('Result matrix incomplete')
            for r in result:
                m=members[r['member_id']];j=target_index(m,r)
                hits={v['truth_index'] for v in r['matches']}
                for i,t in enumerate(r['truth']):
                    miss=next((v for v in r['misses'] if v['truth_index']==i),None)
                    rows.append(dict(seed=seed,arm=arm,source_id=m['source_id'],condition=m['condition'],
                        member_id=m['member_id'],object_id=t['object_id'],class_name=t['class_name'],
                        bbox_xyxy=t['bbox_xyxy'],planned_target=i==j,hit=i in hits,miss=miss))
            for condition in sorted({m['condition'] for m in p['members']}):
                selected=[r for r in rows if r['seed']==seed and r['arm']==arm and r['condition']==condition]
                targets=[r for r in selected if r['planned_target']]
                counts.append(dict(seed=seed,arm=arm,condition=condition,target_hits=sum(r['hit'] for r in targets),target_count=len(targets),
                    full_hits=sum(r['hit'] for r in selected),full_truth_count=len(selected),
                    target_per_class={cl:dict(hits=sum(r['hit'] for r in targets if r['class_name']==cl),total=sum(r['class_name']==cl for r in targets)) for cl in sorted({r['class_name'] for r in targets})}))
            for m in p['members']:
                if m['condition'] in ('gray_target_body','warm','cool'):
                    exposure.append(dict(seed=seed,arm=arm,source_id=m['source_id'],condition=m['condition'],member_id=m['member_id'],actual_image_exposures=actual[arm][m['member_id']]))
            for sid in sorted({m['source_id'] for m in p['members']}):
                ts={r['condition']:r for r in rows if r['seed']==seed and r['arm']==arm and r['source_id']==sid and r['planned_target']}
                base=ts['gray_target_body']
                for condition,t in ts.items():
                    if t['object_id']!=base['object_id'] or t['class_name']!=base['class_name'] or max(abs(a-b) for a,b in zip(t['bbox_xyxy'],base['bbox_xyxy']))>1:
                        raise ValueError('Paired target identity or coordinates drift')
                    cross.append(dict(seed=seed,arm=arm,source_id=sid,condition=condition,gray_hit=base['hit'],condition_hit=t['hit']))
    return prior.frozen(dest,dict(status='fit_and_same_source_condition_diagnosis_complete',counts=counts,instances=rows,
        cross_condition_targets=cross,actual_exposure=exposure,
        gray_fit_misses=[r for r in rows if r['arm']=='G' and r['condition']=='gray_target_body' and r['planned_target'] and not r['hit']],
        limits=['Only actual gray members are called training-fit; other variants are same-source diagnostics, not independent tests.',
                'Raw-image fit does not measure all frozen brightness-augmented tensors.',
                'Three seeds and seven variants do not expand twelve source poses.',
                'No training, threshold change, candidate selection, or automatic visual approval.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
