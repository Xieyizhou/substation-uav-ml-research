"""Full-coverage trajectory summary, refusing partial or stale inference."""
from collections import Counter
from pathlib import Path
from scripts.vision.analyze_material_learning_trajectory import OUT,TRAIN,prior,freeze,validate,MODES,DEV_STEPS,classify


def main():
    p=freeze();tp=TRAIN/'protocol.json';training=prior.read(tp);prior.verify(training)
    paths=[OUT/'protocol.json',tp,Path(__file__).resolve()];trajectories=[];development=[]
    for key,cs in p['cells'].items():
        seq=training['cells'][key]['source_sequence']
        for mode in MODES:
            results=[]
            for c in cs:
                path=OUT/key/f"{c['step']:03}-{mode}"/'complete.json'
                if not path.exists():raise ValueError('Incomplete trajectory: '+str(path))
                r=prior.read(path);validate(r,p,key,c,mode);results.append(r);paths.append(path)
            for m in p['members']:
                exposure_steps=[i//6+1 for i,x in enumerate(seq) if x==m['member_id']]
                for ti,truth in enumerate(m['truth']):
                    timeline=[]
                    for r in results:
                        row=next(x for x in r['rows'] if x['member_id']==m['member_id'])
                        if row['truth'][ti]!=truth:raise ValueError('Target changed across time')
                        hit=any(x['truth_index']==ti for x in row['matches'])
                        miss=next((x for x in row['misses'] if x['truth_index']==ti),None)
                        timeline.append(dict(step=r['step'],hit=hit,miss=miss))
                    hits=[t['hit'] for t in timeline]
                    trajectories.append(dict(cell=key,mode=mode,member_id=m['member_id'],
                        image_sha256=m['image_sha256'],condition=m['condition'],source_id=m['source_id'],truth=truth,
                        planned_target=truth['object_id']==m['target'],exposure_steps=exposure_steps,
                        classification=classify(hits),initial_hit=hits[0],endpoint_hit=hits[-1],
                        first_observed_hit_step=next((t['step'] for t in timeline if t['hit']),None),
                        observed_hit_after_first_exposure=bool(exposure_steps) and any(t['hit'] and t['step']>=min(exposure_steps) for t in timeline),
                        timeline=timeline))
            for r in results:
                if r['step'] not in DEV_STEPS:continue
                for condition in sorted({m['condition'] for m in p['development_members']}):
                    members=[m for m in p['development_members'] if m['condition']==condition]
                    rows={x['member_id']:x for x in r['rows']}
                    positives=[m for m in members if m['truth']];negatives=[m for m in members if not m['truth']]
                    total=sum(len(m['truth']) for m in positives)
                    development.append(dict(cell=key,mode=mode,step=r['step'],condition=condition,
                        truth_count=total,matched_truth=sum(len(rows[m['member_id']]['matches']) for m in positives),
                        negative_frames=len(negatives),negative_frames_with_predictions=sum(bool(rows[m['member_id']]['predictions']) for m in negatives),
                        unmatched_predictions=sum(rows[m['member_id']]['unmatched_prediction_count'] for m in members)))
    counts=[]
    for key in p['cells']:
        for mode in MODES:
            targets=[x for x in trajectories if x['cell']==key and x['mode']==mode and x['planned_target'] and x['exposure_steps']]
            misses=[x for x in targets if not x['endpoint_hit']]
            counts.append(dict(cell=key,mode=mode,actual_exposed_target_count=len(targets),endpoint_misses=len(misses),
                endpoint_miss_classes=dict(Counter(x['classification'] for x in misses)),
                initial_hit_then_endpoint_miss=sum(x['initial_hit'] for x in misses),
                initially_missed_then_hit_then_missed=sum(not x['initial_hit'] and x['classification']=='previously_hit_endpoint_miss' for x in misses)))
    return prior.frozen(OUT/'summary.json',dict(status='full_raw_ema_trajectory_summary_complete',counts=counts,
        trajectories=trajectories,development=development,selected_candidate=None,
        limits=p['limits'],inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=main();print(r['status']);print(r['counts'])
