"""Execute paired, coverage and quota checks; do not synthesize training data."""
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from scripts.vision.material_transfer_controls import OUT,FIT,PRIOR,prior
from scripts.vision.analyze_recovery_paired_calibration import iou

def dimensions(box,width,height):
    gain=640/max(width,height);w=(box[2]-box[0])*gain;h=(box[3]-box[1])*gain
    return dict(width640=w,height640=h,short640=min(w,h),aspect=w/h)

def quota_triads(count):
    if count<3:return dict(feasible=False,reason='Need at least one original, warm and cool at fixed source count',count=count)
    n=count//3;return dict(feasible=True,original=count-2*n,warm=n,cool=n,count=count)

def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p)
    fp=FIT/'completion.json';fit=prior.read(fp);prior.verify(fit)
    dp=PRIOR/'recomputed-metrics.json';dev=prior.read(dp);prior.verify(dev)
    ep=PRIOR/'initial-gate.json';ev=prior.read(ep);prior.verify(ev)
    cp=PRIOR/'coverage-census.json';census=prior.read(cp);prior.verify(census)
    mp=FIT/'protocol.json';members=prior.read(mp);prior.verify(members)
    paths=[pp,fp,dp,ep,cp,mp,Path(__file__).resolve()]
    paired=[];index={(x['model'],x['lineage_id'],x['object_id'],x['source_variant']):x for x in fit['instance_results']}
    if len(index)!=len(fit['instance_results']):raise ValueError('Duplicate instance identity')
    for x in fit['instance_results']:
        if x['source_variant']=='original':continue
        key=(x['model'],x['lineage_id'],x['object_id'],'original')
        if key not in index:raise ValueError('Missing training pair')
        b=index[key]
        if max(abs(a-b) for a,b in zip(x['bbox_xyxy'],b['bbox_xyxy']))>1:raise ValueError('Paired box drift')
        paired.append(dict(model=x['model'],lineage=x['lineage_id'],object_id=x['object_id'],class_name=x['class_name'],variant=x['source_variant'],original_hit=b['hit'],variant_hit=x['hit'],original_exposure=b['actual_exposures'],variant_exposure=x['actual_exposures']))
    coverage=[];lookup={m['member_id']:m for m in members['members']}
    train=[]
    for x in fit['instance_results']:
        if x['model']!='R-clean-7' or not x['primary_gray']:continue
        m=lookup[x['member_id']]
        with Image.open(m['image_path']) as im:w,h=im.size
        train.append(dict(x,**dimensions(x['bbox_xyxy'],w,h)))
    for d in ev['corrected_target_records']:
        if d['variant']!='material':continue
        with Image.open(d['image_path']) as im:w,h=im.size
        size=dimensions(d['truth']['bbox_xyxy'],w,h);same=[x for x in train if x['class_name']==d['category']]
        mins={k:min(x[k] for x in same) for k in size};maxs={k:max(x[k] for x in same) for k in size}
        coverage.append(dict(evidence_id=d['evidence_id'],object_id=d['object_id'],class_name=d['category'],**size,
            same_class_training_members=sorted({x['member_id'] for x in same}),
            train_min=mins,train_max=maxs,short_within_observed_range=mins['short640']<=size['short640']<=maxs['short640'],
            interpretation='Observed bbox range only; not an independent viewpoint or visibility control.'))
    candidates=[]
    for c in census['candidates']:
        if c['variant']!='warm':continue
        candidates.append(dict(source=c['source_review_id'],status=c['status'],altered_class=c['altered_target_class'],
            altered_objects=c['altered_target_objects'],fixed_source_triads={k:quota_triads(n) for k,n in c['actual_source_exposures'].items()},
            proposed_member_class_counts=c['full_label_classes'],training_approved=False,
            qualification='Count-only test for a three-way original/warm/cool replacement, not an intake approval or universal infeasibility claim.'))
    margins=[]
    # Low-confidence evidence on native 640 predictions; do not choose a threshold.
    for key in p['models']:
        path=FIT/'inference'/f'{key}.json';r=prior.read(path);prior.verify(r);paths.append(path)
        for row in r['rows']:
            if not lookup[row['member_id']]['primary_gray']:continue
            for i,t in enumerate(row['truth']):
                same=[z for z in row['low_predictions'] if z['class_name']==t['class_name'] and iou(z['bbox_xyxy'],t['bbox_xyxy'])>=.5]
                wrong=[z for z in row['low_predictions'] if z['class_name']!=t['class_name'] and iou(z['bbox_xyxy'],t['bbox_xyxy'])>=.5]
                margins.append(dict(model=key,member_id=row['member_id'],truth_index=i,class_name=t['class_name'],
                    same_class_retained_max_confidence=max((z['confidence'] for z in same),default=None),
                    wrong_class_retained_max_confidence=max((z['confidence'] for z in wrong),default=None)))
    prior.frozen(OUT/'analytical-controls.json',dict(status='five_controls_executed_scale_control_pending',training_pairs=paired,
        development_pairs=dev['paired_transitions'],size_coverage=coverage,confidence_evidence=margins,candidate_triads=candidates,
        thresholds_unchanged=True,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('FIVE CONTROLS CHECKED',len(paired),'train instance pairs',len(coverage),'material targets')

if __name__=='__main__':main()
