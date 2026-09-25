"""Summarize existing decisions and fixed inference; no decisions synthesized."""
from collections import Counter,defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess,sys
from scripts.vision.structure_fit import OUT,SOURCE,ROOT,read,frozen,file_sha256,verify,unique,validate_review,STRUCTURES
from scripts.vision.run_structure_fit import validate_unit
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.review_structure_fit import OBS

def metrics(rows):
    out={}
    for cls in ('all','transformer','switchgear','capacitor_bank','reactor'):
        truth=sum(t['class_name']==cls or cls=='all' for r in rows for t in r['truth'])
        pred=sum(t['class_name']==cls or cls=='all' for r in rows for t in r['predictions'])
        matched=sum(t['class_name']==cls or cls=='all' for r in rows for t in r['matches'])
        out[cls]=dict(truth=truth,predictions=pred,matched=matched,recall=matched/truth if truth else None,precision=matched/pred if pred else None,unmatched_predictions=pred-matched)
    negatives=[r for r in rows if not r['truth']]
    return dict(frames=len(rows),per_class=out,no_target_frames=len(negatives),no_target_fp_frames=sum(bool(r['predictions']) for r in negatives),
        no_target_FPR=sum(bool(r['predictions']) for r in negatives)/len(negatives) if negatives else None,
        miss_reasons=dict(Counter(m['reason'] for r in rows for m in r['misses'])))

def source_audit(p):
    dest=OUT/'source-audit.json'
    if dest.exists():verify(read(dest));return read(dest)
    inputs={};rows=[]
    # Read actual saved world and per-frame receipt, not the plan's removed objects list.
    for e in p['negative']:
        f=e['source'];ip=Path(f['image_path']);pp=ip.parents[2]/'plan/plan.json';rp=ip.parents[1]/'collection-receipt.json'
        plan=read(pp);receipt=read(rp);wp=pp.parent/'world.sdf';actual=file_sha256(wp)
        if actual!=plan['files']['world.sdf'] or actual!=receipt['world_sha256']:raise ValueError('Source world mismatch')
        view=unique(receipt['views'],lambda x:x['view_id'])[f['view_id']]
        if view['image_sha256']!=f['image_sha256'] or view['truth']['objects']:raise ValueError('Negative source conflict')
        models=ET.parse(wp).getroot().findall('./world/model');assets=[]
        for model in models:
            if model.find('static') is not None:
                assets.append(dict(name=model.get('name'),pose=model.findtext('pose'),visuals=[dict(name=v.get('name'),material=v.findtext('material/diffuse'),box=v.findtext('geometry/box/size')) for v in model.findall('.//visual')]))
        for path in (pp,rp,wp,pp.parent/'sensor_source.sdf'):inputs[str(path)]=file_sha256(path)
        color=OBS[(int(e['event_id'][1:])+1)//2][2]
        candidate='control_building' if color=='gray' and f['map_id']=='complex' else 'cabinet_center' if color=='blue' and f['map_id']=='complex' else None
        if candidate and candidate not in {a['name'] for a in assets}:raise ValueError('Attribution asset missing')
        rows.append(dict(event_id=e['event_id'],source_view_id=f['view_id'],world_sha256=actual,saved_actual_pose=view['actual_pose'],camera_position=view['camera_position'],actual_annotation_mode=receipt['actual_annotation_mode'],
            source_layout=f.get('source_layout_id','unknown'),derived_layout=f.get('derived_layout_id','unknown'),pair_id=f['pair_id'],map_id=f['map_id'],assets=assets,
            primary_visual_attribution=candidate or 'unknown',attribution_basis='visual appearance plus saved world, not instance-mask certification'))
    np=Path(p['development'][48]['image_path']) if len(p['development'])>48 else None
    # Development worlds resolved from the frozen negative review's matrix.
    old=read(SOURCE/'protocol.json');matrixpath=Path(old['evaluation']['negative_review']).parent/'matrix.json';matrix=read(matrixpath);inputs[str(matrixpath)]=file_sha256(matrixpath)
    devworlds=[]
    for run in matrix['runs']:
        pp=Path(run['plan_path']);plan=read(pp);wp=pp.parent/'world.sdf'
        if file_sha256(wp)!=run['world_sha256']:raise ValueError('Development world changed')
        models=ET.parse(wp).getroot().findall('./world/model');devworlds.append(dict(lighting=run['lighting_id'],world_sha256=file_sha256(wp),
            assets=[dict(name=m.get('name'),visuals=[dict(name=v.get('name'),diffuse=v.findtext('material/diffuse'),size=v.findtext('geometry/box/size')) for v in m.findall('.//visual')]) for m in models if m.get('name') in ('control_building','cabinet_center')]))
        for path in (pp,wp):inputs[str(path)]=file_sha256(path)
    return frozen(dest,dict(status='negative_sources_reverified_with_visual_attribution_limits',rows=rows,development_worlds=devworlds,
        corrigendum='Previous visual-review cabinet counts include gray cabinet-shaped control_building. Preserve historical decisions; do not interpret visual appearance counts as simulator asset categories.',
        positive_instance_gap='26 base/regular reactor member lineage records are member-only; no independent-instance claim. Pixel-level certification unavailable.',inputs=inputs))

def main():
    pp=OUT/'protocol.json';ep=OUT/'evidence.json';rp=OUT/'review.json';p=read(pp);review=read(rp)
    for r in (p,read(ep),review):verify(r)
    validate_review(read(ep),review['decisions']);dec={d['event_id']:d for d in review['decisions']}
    source_audit(p);models={}
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';r=read(path);validate_unit(r,key,p);models[key]=r
    common=set.intersection(*[{r['member_id'] for r in m['pool'] if r['actual_exposures']>0} for key,m in models.items() if key!='v2.11'])
    reactor_by_member={r['member']['member_id']:r for r in p['reactors']};neg_by_member={r['member']['member_id']:r for r in p['negative']}
    summaries={};ledger={};reactor_events=[]
    for key,m in models.items():
        seen=[r for r in m['pool'] if r['actual_exposures']>0];cohort=[r for r in m['pool'] if r['member_id'] in common]
        percontent={}
        for state in ('clear_body','partial_identifiable','insufficient_or_uncertain'):
            chosen=[r for r in seen if r['member_id'] in reactor_by_member and dec[reactor_by_member[r['member_id']]['event_id']]['identifiable_content']==state]
            percontent[state]=metrics(chosen)
        summaries[key]=dict(role='training_fit_diagnostic_only' if key!='v2.11' else 'initialization_comparator_not_training_fit',
            seen=metrics(seen),unexposed=metrics([r for r in m['pool'] if not r['actual_exposures']]),common_seen_cohort=metrics(cohort),reactor_by_reviewed_content=percontent,
            development_by_variant={v:metrics([r for r in m['development'] if r['variant']==v]) for v in sorted({r['variant'] for r in m['development']})})
        memberrows={r['member_id']:r for r in p['rows']};draws=p['models'][key]['draws'];windows=[]
        for start in range(0,len(draws),300):
            c=Counter(draws[start:start+300]);classes=Counter();structs=Counter();content=Counter();lineages=Counter();sizes=Counter()
            for mid,count in c.items():
                r=memberrows[mid];classes.update({k:v*count for k,v in r['class_instances'].items()});lineages[r['lineage_id']]+=count
                if mid in neg_by_member:
                    d=dec[neg_by_member[mid]['event_id']]
                    for s,v in d['structures'].items():
                        if v['state']=='present':structs[s]+=count
                if mid in reactor_by_member:
                    e=reactor_by_member[mid];d=dec[e['event_id']];content[d['identifiable_content']]+=count
                    from PIL import Image
                    with Image.open(r['image_path']) as im:w,h=im.size
                    b=e['truth']['bbox_xyxy'];short=min(b[2]-b[0],b[3]-b[1])*640/max(w,h);sizes['lt32' if short<32 else '32to64' if short<64 else 'ge64']+=count
            windows.append(dict(step_start=start//6+1,step_end=(start+300)//6,image_exposures=sum(c.values()),class_instance_exposures=dict(classes),
                negative_structure_image_exposures=dict(structs),reactor_reviewed_content_exposures=dict(content),reactor_size_exposures=dict(sizes),registered_lineage_exposures=dict(lineages)))
        ledger[key]=windows
        for r in m['pool']:
            if r['member_id'] in reactor_by_member:
                e=reactor_by_member[r['member_id']];idx=next(i for i,t in enumerate(r['truth']) if t['annotation_id']==e['truth']['annotation_id'])
                reactor_events.append(dict(model=key,event_id=e['event_id'],member_id=r['member_id'],actual_exposures=r['actual_exposures'],content=dec[e['event_id']]['identifiable_content'],
                    hit=any(x['truth_index']==idx for x in r['matches']),misses=[x for x in r['misses'] if x['truth_index']==idx]))
    # Every development reactor context and seed retained, not only losses.
    dev_reactor=[]
    for e in p['dev_reactors']:
        rowkey=(e['frame']['view_id'],e['frame']['variant']);by_model={}
        for key,m in models.items():
            row=next(r for r in m['development'] if (r['view_id'],r['variant'])==rowkey)
            idx=next(i for i,t in enumerate(row['truth']) if t['annotation_id']==e['truth']['annotation_id'])
            by_model[key]=dict(hit=any(x['truth_index']==idx for x in row['matches']),misses=[x for x in row['misses'] if x['truth_index']==idx])
        changes={}
        for seed in (7,17,27):
            a=by_model[f'retained_reference-450-{seed}']['hit']
            for arm in ('staged','interleaved'):
                b=by_model[f'{arm}-450-{seed}']['hit'];changes[f'{arm}-{seed}']='retained' if a and b else 'gain' if b else 'loss' if a else 'persistent_miss'
        dev_reactor.append(dict(event_id=e['event_id'],instance_identity=e['instance_identity'],content=dec[e['event_id']]['identifiable_content'],by_model=by_model,changes=changes))
    inputs={str(x):file_sha256(x) for x in (pp,ep,rp,OUT/'source-audit.json',Path(__file__))}
    inputs.update({str(OUT/'inference'/f'{k}.json'):file_sha256(OUT/'inference'/f'{k}.json') for k in models})
    frozen(OUT/'analysis.json',dict(status='numerical_diagnosis_complete_with_named_review_gaps',summaries=summaries,exposure_windows=ledger,
        common_seen_member_ids=sorted(common),reactor_training_events=reactor_events,development_reactor_comparisons=dev_reactor,
        reviewed_training_reactor_counts=dict(Counter(dec[e['event_id']]['identifiable_content'] for e in p['reactors'])),
        unknown_structure_slots=sum(v['state']=='unknown' for d in review['decisions'] if 'structures'in d for v in d['structures'].values()),inputs=inputs))
    print('ANALYSIS_COMPLETE',len(common),'common members',flush=True)

if __name__=='__main__':main()
