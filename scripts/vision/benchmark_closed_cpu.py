"""Explicit inference-only CPU benchmark with strict output equivalence."""
import argparse,json,time,subprocess,sys,resource,traceback
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from scripts.vision import train_closed_source_control as train
from scripts.vision.run_fixed_budget_diagnosis import predict,VARIANTS
from scripts.vision.evaluate_unified_lighting import forbidden

prior=train.prior;OUT=train.OUT/'cpu-inference-benchmark-v1'
CONFIGS={'single4':(4,1),'single6':(6,1),'single8':(8,1),'dual4':(4,2)}


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    OUT.mkdir(exist_ok=True);p,_,_=train.contract('E-7');models={};deps=[train.OUT/'protocol.json',Path(__file__).resolve()]
    for key in train.KEYS:
        train.complete(key);cp=train.OUT/'training'/key/'completion.json';c=prior.read(cp);deps.extend([cp,Path(c['weights'])]);models[key]=c['weights']
    reviews=[Path(p['evaluation'][x]) for x in ('paired_review','negative_review')]
    paired,negative=[prior.read(x) for x in reviews];deps+=reviews
    sample=[]
    for variant in VARIANTS:
        sample.append(sorted([r for r in paired['frames'] if r['variant']==variant],key=lambda r:r['view_id'])[0])
    sample+=sorted(negative['frames'],key=lambda r:r['view_id'])[:4]
    for r in sample:
        if r['decision']!='accepted' or prior.file_sha256(r['image_path'])!=r['image_sha256']:raise ValueError('Stale benchmark input')
        deps.append(Path(r['image_path']))
    return prior.frozen(dest,dict(status='frozen_inference_only',models=models,sample=sample,keys=['E-7','M-7'],repetitions=3,
        configurations=CONFIGS,evaluation=p['evaluation'],selection='Fastest total wall time among configurations with exactly identical ordered predictions at both thresholds and all repeats. Fallback single4.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))


def worker(config,key):
    p=freeze();dest=OUT/config/(key+'.json')
    if dest.exists():prior.verify(prior.read(dest));return
    import torch
    from ultralytics import YOLO
    threads=CONFIGS[config][0];start=time.monotonic();cpu=time.process_time();outputs=[]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
        model=YOLO(p['models'][key]);predict(model,p['sample'][0]['image_path'],.37)
        # Ultralytics CPU setup may reset thread count. Set and verify AFTER setup.
        setup_threads=torch.get_num_threads();torch.set_num_threads(threads)
        for row in p['sample'][:1]:predict(model,row['image_path'],.001)
        timed=time.monotonic()
        for repeat in range(p['repetitions']):
            result=[]
            for row in p['sample']:
                for confidence in (.37,.001):
                    pred=predict(model,row['image_path'],confidence)
                    if torch.get_num_threads()!=threads:raise ValueError('Actual thread setting changed')
                    result.append(dict(image_sha256=row['image_sha256'],confidence=confidence,predictions=pred))
            outputs.append(result)
    elapsed=time.monotonic()-timed
    if any(x!=outputs[0] for x in outputs):raise ValueError('Within-configuration predictions changed')
    prior.frozen(dest,dict(status='complete',key=key,config=config,threads=threads,threads_after_backend_setup=setup_threads,
        inference_seconds=elapsed,total_worker_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,outputs=outputs,
        optimizer_created=False,backward_executed=False,validation_run=False,
        inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}))


def launch(config,keys):
    procs=[]
    try:
        for key in keys:
            log=(OUT/config/(key+'.log')).open('x')
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.benchmark_closed_cpu','--worker',config,key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append((proc,log))
        for proc,_ in procs:
            proc.wait(timeout=600)
            if proc.returncode:raise RuntimeError('Benchmark worker failed')
    finally:
        for proc,log in procs:train.cleanup(proc);log.close()


def run():
    p=freeze();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    results=[];deps=[OUT/'protocol.json']
    for config,(_,parallel) in CONFIGS.items():
        folder=OUT/config;folder.mkdir(exist_ok=True);cp=folder/'completion.json'
        if cp.exists():r=prior.read(cp);prior.verify(r)
        else:
            if list(folder.iterdir()):raise ValueError('Incomplete benchmark preserved; investigate before retry')
            start=time.monotonic()
            for i in range(0,2,parallel):launch(config,p['keys'][i:i+parallel])
            r=prior.frozen(cp,dict(config=config,wall_seconds=time.monotonic()-start,inputs={str(folder/(k+'.json')):prior.file_sha256(folder/(k+'.json')) for k in p['keys']}))
        records=[prior.read(folder/(k+'.json')) for k in p['keys']]
        for record in records:prior.verify(record)
        baseline=[prior.read(OUT/'single4'/(k+'.json')) for k in p['keys']]
        exact=all(a['outputs']==b['outputs'] for a,b in zip(records,baseline,strict=True))
        results.append(dict(config=config,wall_seconds=r['wall_seconds'],exact_predictions=exact,
            worker_inference_seconds=[x['inference_seconds'] for x in records],peak_worker_rss_bytes=[x['peak_rss_bytes'] for x in records],
            cpu_seconds=[x['cpu_seconds'] for x in records],effective_threads=[x['threads'] for x in records],backend_setup_threads=[x['threads_after_backend_setup'] for x in records]))
        deps.append(cp);print(results[-1],flush=True)
    winner=min((r for r in results if r['exact_predictions']),key=lambda r:r['wall_seconds'])
    prior.frozen(dest,dict(status='benchmark_complete',results=results,selected=winner['config'],
        speedup=results[0]['wall_seconds']/winner['wall_seconds'],selection_requires_full_evaluation_pilot_recheck=True,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--worker',nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    elif a.benchmark:run()
    else:print(freeze()['status'],'NO_INFERENCE')
