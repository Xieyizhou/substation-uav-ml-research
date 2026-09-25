"""Recompute all six fixed evaluations against corrected, reviewed identities."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision.material_control_feasibility import OUT, RUN, prior, paired_truth
from scripts.vision.evaluate_unified_lighting import validate_record
from scripts.vision.exposure_metrics import score, summary
from scripts.vision.record_material_feasibility_review import validate_decisions

def ratio(a,b):return a/b if b else None

def main():
    ep,rp=OUT/'initial-gate.json',OUT/'independent-review.json'
    e,r=prior.read(ep),prior.read(rp)
    for x in (e,r):prior.verify(x)
    validate_decisions(e['corrected_target_records'],r['decisions'])
    config=prior.read(RUN/'protocol.json');pp=Path(config['evaluation']['paired_review'])
    p=prior.read(pp);prior.verify(p);pairs,raw=paired_truth(p['frames'])
    frames={(f['view_id'],f['variant']):(f,t) for f,t in pairs}
    decisions={(d['view_id'],d['variant'],d['truth']['annotation_id']):d for d in r['decisions']}
    originals={(d['view_id'],d['object_id']):d for d in r['decisions'] if d['variant']=='original'}
    paths=[ep,rp,pp,RUN/'protocol.json',OUT/'risk-containment.json',Path(__file__).resolve(),
           prior.ROOT/'scripts/vision/exposure_metrics.py',prior.ROOT/'scripts/vision/record_material_feasibility_review.py']+list(map(Path,raw))
    cells={};events=[];bins=defaultdict(Counter)
    for family in ('R-clean','L-physical'):
        for seed in (7,17,27):
            key=f'{family}-{seed}';path=RUN/'evaluation'/f'{key}.json';old=prior.read(path);validate_record(old,key);paths.append(path)
            rescored=[]
            for row in old['rows']:
                frame,truth=frames[row['view_id'],row['variant']]
                new=score(frame,truth,row['predictions'],row['low_predictions'])
                if new!=row:raise ValueError(f'{key}: stale truth, identity, or matching')
                rescored.append(new);hits={m['truth_index'] for m in new['matches']}
                for i,t in enumerate(truth):
                    d=decisions[row['view_id'],row['variant'],t['annotation_id']]
                    if d['truth']!=t:raise ValueError('Full truth/review mismatch')
                    original=originals[row['view_id'],d['object_id']]
                    if original['category']!=d['category'] or max(abs(a-b) for a,b in zip(original['truth']['bbox_xyxy'],t['bbox_xyxy']))>1:
                        raise ValueError('Paired identity/coordinate conflict')
                    miss=next((x for x in row['misses'] if x['truth_index']==i),None)
                    ev=dict(cell=key,seed=seed,family=family,pair_id=d['pair_id'],view_id=d['view_id'],variant=d['variant'],
                        object_id=d['object_id'],category=d['category'],bbox_xyxy=t['bbox_xyxy'],
                        hit=i in hits,original_group=original['stratum'],condition_review=d['stratum'],
                        miss=miss,evidence_id=d['evidence_id'],planned=i==row['planned_truth_index'])
                    events.append(ev)
                    for mode,group in (('original_fixed_group',original['stratum']),('independent_condition_review',d['stratum'])):
                        for cat in ('all',d['category']):
                            b=bins[key,d['variant'],mode,group,cat];b['truth']+=1;b['hit']+=i in hits
                            if miss:b[miss['reason']]+=1
            full={v:summary([x for x in rescored if x['variant']==v]) for v in ('original','material','background','lighting')}
            if full!=old['summary']:raise ValueError('Summary recomputation differs')
            cells[key]=dict(summary=full,negative_summary=old['negative_summary'],matching_conflicts=old['matching_conflicts'])
    transitions=[]
    index={(x['cell'],x['pair_id'],x['object_id'],x['variant']):x for x in events}
    for x in events:
        if x['variant']=='original':continue
        o=index[x['cell'],x['pair_id'],x['object_id'],'original']
        state={(True,True):'persistent_hit',(True,False):'loss',(False,True):'gain',(False,False):'persistent_miss'}[o['hit'],x['hit']]
        transitions.append(dict(**x,transition=state))
    strata=[dict(cell=k[0],variant=k[1],grouping=k[2],stratum=k[3],category=k[4],**v,recall=ratio(v['hit'],v['truth'])) for k,v in sorted(bins.items())]
    prior.frozen(OUT/'recomputed-metrics.json',dict(status='six_units_rescored_with_named_visual_unknowns',
        cells=cells,events=events,strata=strata,paired_transitions=transitions,
        independent_pose_groups=12,independent_images=48,seeds_are_repeated_measurements=True,
        original_group_is_not_variant_certification=True,unmatched_predictions_not_assigned_to_truth_strata=True,
        unknowns_retained=True,selected_candidate=None,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('RESCORED',len(cells),'events',len(events),'conflicts',sum(x['matching_conflicts'] for x in cells.values()))

if __name__=='__main__':main()
