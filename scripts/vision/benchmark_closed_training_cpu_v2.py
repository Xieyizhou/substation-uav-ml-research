"""Bounded actual-optimizer throughput probe. Never produces candidate weights."""
import argparse,subprocess,sys,time,traceback
from pathlib import Path
from scripts.vision.closed_budget_design import OUT as BUDGET,freeze,prior
from scripts.vision.preflight_closed_budget import receipt
from scripts.vision.closed_budget_engine_v2 import execute
from scripts.vision.train_closed_source_control import cleanup
OUT=BUDGET/'cpu-training-benchmark-v2'
CONFIGS={'single4':(4,1),'single6':(6,1),'single8':(8,1),'dual4':(4,2)}

def protocol():
    p=freeze();dest=OUT/'protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return p,r
    prior.verify(prior.read(BUDGET/'loader-completion.json'))
    OUT.mkdir(exist_ok=True)
    deps=[BUDGET/'design.json',BUDGET/'loader-completion.json',Path(__file__).resolve(),Path(__file__).with_name('closed_budget_engine_v2.py'),BUDGET/'cpu-training-benchmark-v1/single4/0/attempt-001/failure.json']
    r=prior.frozen(dest,dict(status='frozen_cpu_probe_not_scientific_training',configs=CONFIGS,repetitions=2,steps_per_probe=10,cell='B450-7',
        selection='Fastest wall for two identical 10-step jobs including startup. Require finite losses, exact frozen inputs, effective thread verification; repeated same-config model states must match. Cross-thread state differences reported, never silently treated as identical.',
        limit='Short probe may overestimate startup contribution; no guarantee of sustained throughput. Both experiment budgets use selected execution policy.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))
    return p,r

def worker(config,index):
    p,b=protocol();root=OUT/config/str(index);root.mkdir(parents=True,exist_ok=True);dest=root/'complete.json'
    if dest.exists():prior.verify(prior.read(dest));return
    n=len(list(root.glob('attempt-*')))+1
    if n>2:raise ValueError('Probe attempt cap including prior-version failure')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        expected,ep=receipt('B450-7',p)
        r=execute(p,'B450-7',attempt,CONFIGS[config][0],expected,probe_steps=10)
        deps=[OUT/'protocol.json',ep]
        prior.frozen(dest,dict(**r,config=config,index=index,inputs={str(d):prior.file_sha256(d) for d in deps}))
    except BaseException:prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc()));raise

def run():
    _,b=protocol();results=[]
    for config,(threads,parallel) in CONFIGS.items():
        folder=OUT/config;folder.mkdir(exist_ok=True);dest=folder/'timing.json'
        if dest.exists():r=prior.read(dest);prior.verify(r);results.append(r);continue
        if any((folder/str(i)/'complete.json').exists() for i in range(2)):raise ValueError('Incomplete timing group requires explicit new benchmark version')
        start=time.monotonic();procs=[];logs=[]
        try:
            for i in range(2):
                log=(folder/f'worker-{i}.log').open('x');logs.append(log)
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.benchmark_closed_training_cpu_v2','--worker',config,str(i)],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                procs.append(proc)
                if parallel==1:
                    proc.wait(timeout=600)
                    if proc.returncode:raise RuntimeError('Probe failed '+str(folder))
            for proc in procs:
                proc.wait(timeout=600)
                if proc.returncode:raise RuntimeError('Probe failed '+str(folder))
        finally:
            for proc in procs:cleanup(proc)
            for log in logs:log.close()
        wall=time.monotonic()-start;paths=[folder/str(i)/'complete.json' for i in range(2)];records=[prior.read(x) for x in paths]
        for record in records:
            prior.verify(record)
            if record['optimizer_steps']!=10 or record['effective_threads']!=threads:raise ValueError('Probe actual config mismatch')
        exact=records[0]['terminal_model_state_identity']==records[1]['terminal_model_state_identity']
        r=prior.frozen(dest,dict(config=config,wall_seconds=wall,repeat_exact_state=exact,states=[x['terminal_model_state_identity'] for x in records],
            mean_step_seconds=sum(s['seconds'] for r in records for s in r['step_records'])/20,
            inputs={str(x):prior.file_sha256(x) for x in paths}));results.append(r);print('CPU_BENCHMARK',config,wall,'REPEAT_EXACT',exact,flush=True)
    valid=[x for x in results if x['repeat_exact_state']]
    if not valid:raise ValueError('No reproducible CPU configuration')
    winner=min(valid,key=lambda x:x['wall_seconds'])
    deps=[OUT/'protocol.json']+[OUT/c/'timing.json' for c in CONFIGS]
    prior.frozen(OUT/'completion.json',dict(status='cpu_training_policy_selected_not_full_training',selected=winner['config'],results=results,
        cross_thread_exact={r['config']:r['states'][0]==results[0]['states'][0] for r in results},inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--worker',nargs=2);a=ap.parse_args()
    if a.worker:worker(a.worker[0],int(a.worker[1]))
    elif a.benchmark:run()
    else:print('NO_OPTIMIZER_NO_TRAINING')
