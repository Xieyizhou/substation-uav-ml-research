"""Read-only saved exposure/loss/checkpoint diagnostics; never mutate a model."""
from collections import Counter
from pathlib import Path
from statistics import mean
import torch
from scripts.vision.physical_low_light_capture import OUT, prior

DEST=OUT/'tail-direction-diagnosis-v1'

def main():
    torch.set_num_threads(4)
    dp=OUT/'design.json';p=prior.read(dp);prior.verify(p)
    lookup={r['member_id']:r for r in p['pool_rows']}
    deps=[dp,OUT/'report-completion.json',Path(__file__).resolve()]
    prior.verify(prior.read(deps[1]))
    results=[]
    for seed in (7,17,27):
        models={};xs={}
        for family,root in (('B900',OUT.parent),('R1000',OUT),('L1000',OUT)):
            cp=root/'training'/f'{family}-{seed}'/'completion.json';c=prior.read(cp);prior.verify(c)
            wp=Path(c['weights']);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
            if prior.file_sha256(wp)!=c['weights_sha256']:raise ValueError('Weight identity changed')
            deps.extend([cp,wp,xp]);xs[family]=x
            # Trusted local experiment artifacts, not externally supplied checkpoints.
            ck=torch.load(wp,map_location='cpu',weights_only=False)
            model=ck['model'];sd=model.state_dict()
            if not all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()):raise ValueError('Nonfinite saved tensor')
            models[family]=sd
            if ck.get('optimizer') is not None or ck.get('ema') is not None:raise ValueError('Unexpected unstripped checkpoint')
        for family in ('R1000','L1000'):
            x=xs[family];b=xs['B900'];windows=[]
            if [s['loss_items'] for s in x['step_records'][:900]]!=[s['loss_items'] for s in b['step_records']]:raise ValueError('Prefix loss drift')
            for start,end in ((800,900),(900,950),(950,1000)):
                seq=x['actual'][start*6:end*6];steps=x['step_records'][start:end]
                windows.append(dict(first_step=start+1,last_step=end,images=len(seq),members=len(set(seq)),
                    registered_lineages=len({lookup[m]['lineage_id'] for m in seq}),
                    subsets=dict(Counter(lookup[m]['subset'] for m in seq)),
                    class_instances={c:sum(lookup[m]['class_instances'].get(c,0) for m in seq) for c in ('transformer','switchgear','capacitor_bank','reactor')},
                    loss_mean={k:mean(s['loss_items'][k] for s in steps) for k in steps[0]['loss_items']}))
            aa,bb=models[family],models['B900'];groups={}
            if set(aa)!=set(bb):raise ValueError('State schema mismatch')
            for group in ('bn_running_mean','bn_running_var','other_floating_state'):
                names=[n for n,v in aa.items() if v.is_floating_point() and (('bn_running_mean' if n.endswith('running_mean') else 'bn_running_var' if n.endswith('running_var') else 'other_floating_state')==group)]
                delta=sum(float((aa[n].double()-bb[n].double()).square().sum()) for n in names)
                norm=sum(float(bb[n].double().square().sum()) for n in names)
                groups[group]=dict(tensors=len(names),changed=sum(not torch.equal(aa[n],bb[n]) for n in names),relative_l2=(delta/norm)**.5 if norm else None)
            results.append(dict(cell=f'{family}-{seed}',windows=windows,saved_EMA_drift_from_B900=groups,prefix_recorded_losses_exact=True))
    import ultralytics.engine.trainer as trainer
    import ultralytics.utils.torch_utils as tu
    deps.extend([Path(trainer.__file__),Path(tu.__file__)])
    return prior.frozen(DEST/'analysis.json',dict(status='read_only_diagnosis_complete',units=results,
        limits=['Relative tensor drift is descriptive, not causal attribution or quality threshold.',
               'Registered lineage count outside the tail is not independent scene certification.',
               'Only stripped EMA endpoints exist here; raw model/optimizer/900-step continuation state is not recovered.'],
        next_direction='Bounded no-gradient BN-running-statistic counterfactual diagnostic, before more optimization; require separate authorization.',
        inputs={str(f):prior.file_sha256(f) for f in deps}))

if __name__=='__main__':
    r=main()
    for u in r['units']:print(u['cell'],u['saved_EMA_drift_from_B900'],u['windows'][-1]['class_instances'])
