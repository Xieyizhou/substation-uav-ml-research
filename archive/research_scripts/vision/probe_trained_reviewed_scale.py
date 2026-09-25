"""Existing frozen scale images, current endpoints; explicit inference only."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT as TRAIN, KEYS, prior
from scripts.vision.train_reviewed_scale_control import complete
from scripts.vision import probe_reviewed_scale as old
from scripts.vision import diagnose_material_late_rehearsal_fit as runtime

OUT = TRAIN/'same-member-scale-fit-v1'

def freeze():
    source=old.freeze()
    deps=[old.OUT/'protocol.json', old.OUT/'summary.json', Path(__file__).resolve(),
          Path(runtime.__file__).resolve()]
    models={}; exposures={}
    for key in KEYS:
        c=complete(key); xp=Path(c['exposure_path']); x=prior.read(xp); prior.verify(x)
        models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'])
        counts=Counter(x['actual'])
        exposures[key]={m['parent_member_id']:counts[m['parent_member_id']] for m in source['members']}
        deps.extend([TRAIN/'training'/key/'completion.json',xp,Path(c['weights'])])
        r=prior.read(old.OUT/(key+'.json'));runtime.validate(r,key,source)
        deps.append(old.OUT/(key+'.json'))
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p)
        if p['models']!=models or p['actual_parent_exposures']!=exposures:raise ValueError('Exposure or model drift')
        return p
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='same_member_scale_fit_frozen',members=source['members'],
        anchor_targets=source['anchor_targets'],transform=source['transform'],models=models,
        environment=source['environment'],inference=source['inference'],actual_parent_exposures=exposures,
        interpretation='Same frozen diagnostic derivatives, not new independent scenes. Parent exposure is not exact derivative exposure; loader resize order differs from this probe.',
        inputs={**source['inputs'],**{str(p):prior.file_sha256(p) for p in deps}}))

def finish(p):
    results=[];deps=[OUT/'protocol.json']; oldp=old.freeze()
    targets={t['member_id']:t for t in p['anchor_targets']}
    for key in KEYS:
        current=prior.read(OUT/(key+'.json'));runtime.validate(current,key,p)
        previous=prior.read(old.OUT/(key+'.json'));runtime.validate(previous,key,oldp)
        deps += [OUT/(key+'.json'),old.OUT/(key+'.json')]
        for member,now,before in zip(p['members'],current['rows'],previous['rows']):
            t=targets[member['parent_member_id']]
            j=next(i for i,v in enumerate(member['truth']) if v['label_line_index']==t['truth']['label_line_index'])
            nh=any(v['truth_index']==j for v in now['matches']); bh=any(v['truth_index']==j for v in before['matches'])
            results.append(dict(model=key,member_id=member['member_id'],target_id=t['target_id'],
                scale=member['scale'],class_name=t['class_name'],parent_exposures=p['actual_parent_exposures'][key][member['parent_member_id']],
                target_hit=nh,previous_target_hit=bh,state=('persistent_hit' if bh else 'gain') if nh else ('new_miss' if bh else 'persistent_miss'),
                full_truth=len(now['truth']),full_matches=len(now['matches']),previous_full_matches=len(before['matches']),
                unmatched_predictions=len(now['predictions'])-len(now['matches']),
                miss=next((v for v in now['misses'] if v['truth_index']==j),None)))
    aggregates={str(s):dict(target_events=sum(v['scale']==s for v in results),
        target_hits=sum(v['target_hit'] for v in results if v['scale']==s),
        previous_target_hits=sum(v['previous_target_hit'] for v in results if v['scale']==s),
        full_truth=sum(v['full_truth'] for v in results if v['scale']==s),
        full_matches=sum(v['full_matches'] for v in results if v['scale']==s),
        previous_full_matches=sum(v['previous_full_matches'] for v in results if v['scale']==s)) for s in old.SCALES}
    dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='same_member_scale_fit_complete',rows=results,aggregates=aggregates,
        inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--cell',choices=KEYS);ap.add_argument('--finish',action='store_true');a=ap.parse_args()
    if (a.cell or a.finish) and not a.infer:ap.error('Explicit --infer required')
    p=freeze();runtime.OUT=OUT
    if a.cell:runtime.infer(a.cell,p)
    elif a.finish:print(finish(p)['aggregates'])
    elif a.infer:
        for k in KEYS:runtime.infer(k,p)
        print(finish(p)['aggregates'])
    else:print('PREFLIGHT_ONLY',len(p['members']))
