"""Aggregate existing explicit observations; no invented review decisions."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision.record_closed_gamma_review import DEST, prior, validate


def main():
    ep,rp=DEST/'evidence.json',DEST/'review.json'
    e,r=prior.read(ep),prior.read(rp);prior.verify(e);prior.verify(r);validate(e,r['decisions'])
    ds={d['decision_id']:d for d in r['decisions']}
    loss_lookup={};fp_groups=defaultdict(list)
    for event in e['events']:
        if event['kind']=='LOSS':
            for transition in event['events']:
                k=(transition['comparison'],transition['seed'],transition['pair_id'],transition['variant'],
                   transition['object_id'],tuple(transition['truth']['bbox_xyxy']))
                if k in loss_lookup:raise ValueError('Duplicate loss identity')
                loss_lookup[k]=ds[event['event_id']+':0']
        else:
            for j,p in enumerate(event['events']):
                d=ds[f"{event['event_id']}:{j}"]
                fp_groups[event['source']['view_id']].append(dict(event_id=event['event_id'],decision_id=d['decision_id'],
                    variant=event['source']['variant'],image_sha256=d['image_sha256'],content=d['content'],
                    cell=p['cell'],seed=p['seed'],prediction=p['prediction']))
    counts=defaultdict(Counter);loss_details=[]
    for t in e['transitions']:
        group=(t['comparison'],t['variant'],t['truth']['class_name'])
        counts[group][t['state']]+=1
        if t['state']=='loss':
            k=(t['comparison'],t['seed'],t['pair_id'],t['variant'],t['object_id'],tuple(t['truth']['bbox_xyxy']))
            d=loss_lookup[k]
            counts[group]['loss:'+d['content']]+=1
            counts[group]['miss:'+t['miss']['reason']]+=1
            loss_details.append(dict(**t,review_decision=d['decision_id'],review_content=d['content'],
                                     pixel_visibility_certified=False))
    same_frame=[]
    for view_id,rows in fp_groups.items():
        for image in sorted({x['image_sha256'] for x in rows}):
            chunk=[x for x in rows if x['image_sha256']==image]
            same_frame.append(dict(view_id=view_id,image_sha256=image,seeds=sorted({x['seed'] for x in chunk}),
                cells=sorted({x['cell'] for x in chunk}),prediction_count=len(chunk),predictions=chunk))
    paths=[ep,rp,Path(__file__).resolve()]
    result=prior.frozen(DEST/'review-summary.json',dict(status='reviewed_error_inventory_with_content_gaps',
        unique_fp_images=len(same_frame),fp_source_poses=len(fp_groups),fp_predictions=sum(len(x) for x in fp_groups.values()),
        unique_loss_targets=sum(x['kind']=='LOSS' for x in e['events']),loss_events=len(loss_details),
        all_transition_count=len(e['transitions']),transitions_by_class=[dict(comparison=a,variant=b,category=c,**v) for (a,b,c),v in counts.items()],
        loss_details=loss_details,same_image_recurrence=same_frame,
        same_source_pose_variants=[dict(view_id=k,variants=sorted({x['variant'] for x in v}),
            unique_images=len({x['image_sha256'] for x in v}),predictions=v) for k,v in fp_groups.items()],
        unknown_content_ids=r['pending_ids'],full_metrics_unchanged=True,selected_candidate=None,
        limits=['Source pose grouping is not pixel overlap or asset-identity certification',
                'Per-condition RGB observations are not reliable instance masks',
                'Comparison/seed loss events share unique truth targets; neither count is independent scene count',
                'All gains and persistent hits/misses remain in the frozen evidence'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print(result['status'],'FP',result['fp_predictions'],'images',len(same_frame),'poses',len(fp_groups))
    return result


if __name__=='__main__':main()
