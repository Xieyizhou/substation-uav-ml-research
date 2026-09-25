"""Bounded existing same-source variants; no new training or images."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_reviewed_target_fit import OUT as ANCHOR,freeze as anchors,prior,runtime,DEST,KEYS
from scripts.vision.structure_fit import truth_for

OUT=DEST/'source-variant-probe-v1'
def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    a=anchors();d=prior.read(DEST/'design.json');prior.verify(d)
    groups=sorted({m['pair_id'] for m in a['members'] if m.get('pair_id')});members=[];deps=[ANCHOR/'completion.json',ANCHOR/'protocol.json',DEST/'design.json',Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    comparisons=[]
    for gid in groups:
        group=[dict(m,truth=truth_for(m)) for m in d['pool_rows'] if m.get('pair_id')==gid]
        sig=lambda m:sorted((t['class_name'],tuple(round(v,4) for v in t['bbox_xyxy'])) for t in m['truth'])
        same=all(sig(m)==sig(group[0]) for m in group)
        comparisons.append(dict(pair_id=gid,members=[m['member_id'] for m in group],full_class_counts_equal=all(m['class_instances']==group[0]['class_instances'] for m in group),full_box_coordinates_equal_rounded_4dp=same,
            geometry_certification='not_inferred_from_equal_boxes',viewpoint_control='not_established_by_this_probe'))
        members+=group
    for m in members:
        for kind in ('image','label'):
            path=Path(m[kind+'_path'])
            if prior.file_sha256(path)!=m[kind+'_sha256']:raise ValueError('Input drift')
            deps.append(path)
    if len(groups)!=4 or len(members)!=56:raise ValueError('Bounded probe scope changed')
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='existing_variant_fit_probe_frozen',members=members,models=a['models'],actual_exposures=a['actual_exposures'],environment=a['environment'],inference=a['inference'],comparisons=comparisons,
        scope='Four registered source groups from original ten anchors; 56 existing variants are not independent scenes. Per-model zero exposure is not fit.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))

def finish(p):
    deps=[OUT/'protocol.json'];results=[]
    for k in KEYS:
        path=OUT/(k+'.json');r=prior.read(path);runtime.validate(r,k,p);deps.append(path)
        for m,row in zip(p['members'],r['rows']):
            results.append(dict(model=k,member_id=m['member_id'],pair_id=m['pair_id'],variant=m['variant'],
                exposures=p['actual_exposures'][k].get(m['member_id'],0),truth_count=len(row['truth']),matched=len(row['matches']),
                unmatched_predictions=row['unmatched_prediction_count'],misses=row['misses']))
    dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='existing_variant_probe_complete',rows=results,
        exposed=dict(member_model_events=sum(r['exposures']>0 for r in results),truth_count=sum(r['truth_count'] for r in results if r['exposures']>0),matched=sum(r['matched'] for r in results if r['exposures']>0)),
        zero_exposure_events=sum(r['exposures']==0 for r in results),
        interpretation='Fit across registered source variants, not viewpoint/scale/context causality; no new training decision auto-generated.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze();print('FROZEN',len(p['members']),flush=True)
    if a.infer:
        runtime.OUT=OUT
        for k in KEYS:runtime.infer(k,p)
        r=finish(p);print(r['status'],r['exposed'],'zero_exposure',r['zero_exposure_events'],flush=True)
