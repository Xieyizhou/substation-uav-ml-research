"""Describe all observed transitions and intervening exposures, never infer causality."""
from collections import Counter
from pathlib import Path
from scripts.vision.analyze_material_learning_trajectory import OUT,TRAIN,prior
from scripts.vision.structure_fit import iou
from scripts.vision.train_material_learning_trajectory import historical


def candidate_profile(truth,predictions):
    same=[p for p in predictions if p['class_name']==truth['class_name'] and iou(p['bbox_xyxy'],truth['bbox_xyxy'])>=.5]
    wrong=[p for p in predictions if p['class_name']!=truth['class_name'] and iou(p['bbox_xyxy'],truth['bbox_xyxy'])>=.5]
    def best(rows):
        if not rows:return None
        p=max(rows,key=lambda x:x['confidence'])
        return dict(**p,iou=iou(p['bbox_xyxy'],truth['bbox_xyxy']))
    return dict(same_class=best(same),other_class=best(wrong))


def window(sequence,rows,start,end,member):
    if not 0<=start<end<=450:raise ValueError('Invalid observed interval')
    mids=sequence[start*6:end*6]
    return dict(first_step=start+1,last_step=end,steps=end-start,image_count=len(mids),
        members=dict(Counter(mids)),subsets=dict(Counter(rows[m]['subset'] for m in mids)),
        variants=dict(Counter(rows[m].get('variant','unspecified') for m in mids)),
        class_instances=dict(sum((Counter(rows[m]['class_instances']) for m in mids),Counter())),
        target_member_exposures=mids.count(member),
        target_lineage_exposures=sum(rows[m]['lineage_id']==rows[member]['lineage_id'] for m in mids))


def main():
    sp=OUT/'summary.json';summary=prior.read(sp);prior.verify(summary)
    pp=historical.OUT/'protocol.json';source=prior.read(pp);prior.verify(source)
    rows={r['member_id']:r for r in source['pool_rows']}
    paths=[sp,pp,Path(__file__).resolve()];cache={};events=[];ends=[]
    def get(key,step,mode,member):
        k=(key,step,mode)
        if k not in cache:
            path=OUT/key/f'{step:03}-{mode}'/'complete.json';r=prior.read(path);prior.verify(r)
            cache[k]={x['member_id']:x for x in r['rows']};paths.append(path)
        return cache[k][member]
    for t in summary['trajectories']:
        # All actually exposed members, all full-image instances, both modes.
        if not t['exposure_steps']:continue
        key,mid,mode=t['cell'],t['member_id'],t['mode'];sequence=source['schedules'][key]
        common={k:t[k] for k in ('cell','mode','member_id','condition','source_id','truth','planned_target','image_sha256')}
        for a,b in zip(t['timeline'],t['timeline'][1:]):
            state=('hit' if a['hit'] else 'miss')+'_to_'+('hit' if b['hit'] else 'miss')
            events.append(dict(**common,state=state,from_step=a['step'],to_step=b['step'],
                exposure=window(sequence,rows,a['step'],b['step'],mid),
                before=candidate_profile(t['truth'],get(key,a['step'],mode,mid)['low_predictions']),
                after=candidate_profile(t['truth'],get(key,b['step'],mode,mid)['low_predictions']),
                miss=b['miss'],causal_attribution=False))
        if t['planned_target'] and not t['endpoint_hit']:
            hit_steps=[x['step'] for x in t['timeline'] if x['hit']]
            last=hit_steps[-1] if hit_steps else None
            ends.append(dict(**common,classification=t['classification'],exposure_steps=t['exposure_steps'],
                last_observed_hit=last,
                first_observed_miss_after_last_hit=next((x['step'] for x in t['timeline'] if last is not None and x['step']>last),None),
                after_last_hit=window(sequence,rows,last,450,mid) if last is not None else None,
                endpoint=candidate_profile(t['truth'],get(key,450,mode,mid)['low_predictions'])))
    counts=[]
    for key in source['schedules']:
        if not key.startswith('T-'):continue
        for mode in ('ema','raw'):
            subset=[e for e in events if e['cell']==key and e['mode']==mode and e['planned_target']]
            counts.append(dict(cell=key,mode=mode,transitions=dict(Counter(e['state'] for e in subset)),
                one_step_transitions=dict(Counter(e['state'] for e in subset if e['exposure']['steps']==1))))
    dest=OUT/'transition-windows.json'
    return prior.frozen(dest,dict(status='all_exposed_full_instance_windows_described',events=events,
        endpoint_target_misses=ends,counts=counts,selected_candidate=None,
        limits=['Temporal exposure overlap is not a causal intervention.',
                'All stable-hit, stable-miss and gain intervals are retained as descriptive controls.',
                'Different interval lengths and repeated targets prevent treating windows as independent samples.',
                'Low-threshold retained predictions remain limited by NMS and max_det.'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    r=main();print(r['status']);print(r['counts'])
