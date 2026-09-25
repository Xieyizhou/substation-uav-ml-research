"""Bounded parallel inference selected by a frozen CPU benchmark; no training."""
import argparse,fcntl,subprocess,sys,time,traceback
from pathlib import Path
from scripts.vision import evaluate_closed_budget as evaluation
from scripts.vision.benchmark_closed_cpu import OUT as BENCH,CONFIGS
from scripts.vision.train_closed_source_control import cleanup

OUT=evaluation.OUT;prior=evaluation.prior


def run():
    benchmark=prior.read(BENCH/'completion.json');prior.verify(benchmark)
    selected=benchmark['selected'];threads,parallel=CONFIGS[selected]
    if not next(r for r in benchmark['results'] if r['config']==selected)['exact_predictions']:raise ValueError('Unverified speed setting')
    root=OUT/'evaluation-runner';root.mkdir(exist_ok=True)
    dest=root/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    keys=[f'{family}-{seed}' for seed in (7,17,27) for family in ('B450','B900')]
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
        for i in range(0,len(keys),parallel):
            procs=[]
            try:
                for key in keys[i:i+parallel]:
                    ep=OUT/'evaluation'/(key+'.json')
                    if ep.exists():evaluation.evaluate(key);continue
                    folder=root/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Three inference launches exhausted')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_closed_budget','--evaluate','--cell',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    procs.append((proc,log,attempt,key));print('INFERENCE_STARTED',key,proc.pid,flush=True)
                for proc,_,attempt,key in procs:
                    proc.wait(timeout=1200)
                    if proc.returncode:raise RuntimeError('Inference failed: '+str(attempt))
                    evaluation.evaluate(key);print('INFERENCE_VERIFIED',key,flush=True)
            except BaseException:
                for proc,_,attempt,key in procs:
                    cleanup(proc)
                    prior.frozen(attempt/'failure.json',dict(status='group_stopped',cell=key,error=traceback.format_exc(),process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_,_ in procs:cleanup(proc);log.close()
        result=evaluation.finish()
        deps=[BENCH/'completion.json',OUT/'evaluation/summary.json',Path(__file__).resolve()]
        prior.frozen(dest,dict(status='six_models_evaluated_error_review_pending',wall_seconds=time.monotonic()-start,
            threads_per_worker=threads,concurrent_workers=parallel,matching_conflicts=result['matching_conflicts'],
            inputs={str(x):prior.file_sha256(x) for x in deps}))
        print('NUMERICAL_EVALUATION_COMPLETE_REVIEW_PENDING',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    if a.evaluate:run()
    else:print('NO_INFERENCE_NO_TRAINING')
