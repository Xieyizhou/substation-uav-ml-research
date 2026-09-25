"""Six existing CPU weights, full-label fitting diagnostics only."""
import argparse,os,signal,subprocess,sys,traceback
from pathlib import Path
from collections import Counter
from scripts.vision.train_reviewed_hold import OUT as SOURCE,ready,complete,prior
from scripts.vision import check_lineage_training_fit as base
from scripts.vision import check_lineage_training_fit_v2 as runner
from scripts.vision.brightness_lr_retention import runtime_check

OUT=SOURCE.parent.parent/'reviewed-hold-fit-diagnosis-v1'
OLD_FIT=base.OUT

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    p=ready();prior.verify(prior.read(SOURCE.parent/'completion.json'))
    tp=OLD_FIT/'protocol.json';t=prior.read(tp);prior.verify(t)
    rows=p['pool_rows'];idx={r['member_id']:r for r in rows}
    paths=[SOURCE/'protocol.json',SOURCE.parent/'completion.json',tp,Path(__file__).resolve(),Path(base.__file__),Path(runner.__file__)]
    paths += [prior.ROOT/'scripts/vision'/n for n in ('structure_fit.py','evaluate_exposure_diagnosis.py','evaluate_paired_visual_factors.py','exposure_metrics.py')]
    for r in rows:
        for kind in ('image','label'):
            path=Path(r[kind+'_path'])
            if prior.file_sha256(path)!=r[kind+'_sha256']:raise ValueError('Stale pool input')
            paths.append(path)
        if Counter(x['class_name'] for x in base.truth_for(r))!=Counter(r['class_instances']):raise ValueError('Class count drift')
    for target in t['targets']:
        d=target['review'];r=idx[d['member_id']]
        base.check_review(d,r,base.truth_for(r)[d['truth']['label_line_index']])
    models={}
    for seed in (7,17,27):
        key=f'brightness-450-{seed}';complete(key,p)
        for arm,cp,dev in [('reference',Path(p['baselines'][str(seed)]['training_receipt']),Path(p['baselines'][str(seed)]['evaluation'])),
                           ('reviewed',SOURCE/'training'/key/'completion.json',SOURCE/'evaluation'/f'{key}.json')]:
            c=prior.read(cp);prior.verify(c);runtime_check(c,seed,.0005)
            ep=Path(c['exposure_path']);e=prior.read(ep);prior.verify(e);prior.verify(prior.read(dev))
            if len(e['draws'])!=2700 or set(e['draws'])-set(idx):raise ValueError('Exposure identity mismatch')
            if prior.file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Weights changed')
            models[f'{arm}-{seed}']=dict(weights=c['weights'],weights_sha256=c['weights_sha256'],draws=e['draws'],development=str(dev))
            paths += [cp,ep,dev,Path(c['weights'])]
    OUT.mkdir(exist_ok=True);(OUT/'inference').mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='frozen_inference_only',rows=rows,targets=t['targets'],models=models,
        selection='All frozen pool members; clear targets reused from pre-existing hash-valid explicit reviews, never selected by new predictions.',
        inference=t['inference'],inputs={str(x):prior.file_sha256(x) for x in paths}))

def setup():
    base.OUT=OUT

def run():
    p=freeze();setup()
    for key in p['models']:
        dest=OUT/'inference'/f'{key}.json'
        if dest.exists():base.valid(prior.read(dest),key,p);continue
        folder=OUT/'inference'/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
        if n>3:raise ValueError('Attempt cap')
        a=folder/f'attempt-{n:03}';a.mkdir();proc=None
        try:
            with (a/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.check_reviewed_hold_fit','--worker',key,'--attempt',str(a)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                proc.wait(timeout=1800)
                if proc.returncode:raise RuntimeError('Inference failure: '+str(a))
            base.valid(prior.read(dest),key,p)
        except BaseException:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            prior.frozen(a/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
        print('FIT_COMPLETE',key,flush=True)

def summarize():
    p=freeze();setup();rs={};paths=[OUT/'protocol.json',Path(__file__).resolve()]
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';u=prior.read(path);base.valid(u,key,p);rs[key]={r['member_id']:r for r in u['rows']};paths.append(path)
    common={m for m in rs[next(iter(rs))] if all(d[m]['exposures']>0 for d in rs.values())}
    clear={(t['review']['member_id'],t['review']['truth']['annotation_id']) for t in p['targets'] if t['clear_primary']}
    details=[];metrics={}
    for key,d in rs.items():
        events=[]
        for row in p['rows']:
            mid=row['member_id'];s=d[mid]['scoring'];hits={m['truth_index'] for m in s['matches']}
            for i,t in enumerate(s['truth']):
                events.append(dict(model=key,member_id=mid,truth=t,common_exposed=mid in common,clear_primary=(mid,t['annotation_id']) in clear,
                    subset=row['subset'],exposures=d[mid]['exposures'],hit=i in hits,miss=next((m for m in s['misses'] if m['truth_index']==i),None)))
        details+=events;metrics[key]={}
        for scope in ('common_all','common_clear','common_bridge','own_exposed'):
            selected=[e for e in events if (e['exposures']>0 if scope=='own_exposed' else e['common_exposed']) and (scope!='common_clear' or e['clear_primary']) and (scope!='common_bridge' or e['subset']=='bridge_positive')]
            metrics[key][scope]={}
            for cls in ('all','reactor','capacitor_bank','switchgear','transformer'):
                es=[e for e in selected if cls=='all' or e['truth']['class_name']==cls]
                metrics[key][scope][cls]=dict(instances=len(es),hits=sum(e['hit'] for e in es),recall=sum(e['hit'] for e in es)/len(es) if es else None,
                    reasons=dict(Counter(e['miss']['reason'] for e in es if not e['hit'])))
    prior.frozen(OUT/'summary.json',dict(status='fitting_complete_visual_crosscheck_pending',common_members=len(common),metrics=metrics,details=details,
        limitation='Common exposure is not equal count; all pool is not all quality-approved. Clear reviewed primary subset is restricted to reactor/capacitor. Bridge aggregate is not material-only.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('FIT_SUMMARY_COMPLETE',len(common))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--summarize',action='store_true');ap.add_argument('--worker');ap.add_argument('--attempt',type=Path);a=ap.parse_args()
    if a.worker:
        if a.attempt is None:ap.error('attempt required')
        setup();runner.worker(a.worker,a.attempt)
    elif a.infer:run();summarize()
    elif a.summarize:summarize()
    else:freeze();print('PREFLIGHT_ONLY_NO_TRAINING')
