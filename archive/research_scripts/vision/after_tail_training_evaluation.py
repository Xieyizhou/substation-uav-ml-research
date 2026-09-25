"""One-shot dependency on the live training lock; not a timer or recurring job."""
import argparse,fcntl,signal,subprocess,sys,traceback
from pathlib import Path
from scripts.vision import evaluate_tail_interleaving as evaluation
from scripts.vision.train_closed_source_control import cleanup
OUT=evaluation.OUT;prior=evaluation.prior

def run():
    root=OUT/'evaluation-runner';root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        evaluation.protocol()
        print('WAITING_FOR_TRAINING_COMPLETION_LOCK',flush=True)
        with (OUT/'training-runner/run.lock').open('r') as training:
            fcntl.flock(training,fcntl.LOCK_SH)
            cp=OUT/'training-runner/completion.json'
            if not cp.exists():raise RuntimeError('Training stopped without complete receipt; evaluation refused')
            prior.verify(prior.read(cp))
            for key in evaluation.KEYS:evaluation.complete(key)
        print('ALL_TRAINING_VERIFIED_STARTING_EVALUATION',flush=True)
        for group in (evaluation.KEYS[:2],evaluation.KEYS[2:]):
            jobs=[]
            try:
                for key in group:
                    if (OUT/'evaluation'/f'{key}.json').exists():evaluation.evaluate(key);continue
                    folder=root/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Inference launch cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_tail_interleaving','--evaluate','--cell',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt));print('EVALUATION_STARTED',key,proc.pid,flush=True)
                for proc,_,_ in jobs:
                    proc.wait(timeout=1200)
                    if proc.returncode:raise RuntimeError('Evaluation worker failed')
            except BaseException:
                for proc,_,attempt in jobs:
                    cleanup(proc);prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_ in jobs:cleanup(proc);log.close()
        r=evaluation.finish()
        deps=[cp,OUT/'evaluation/summary.json',Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='numerical_complete_AI_review_pending',selected_candidate=None,review_complete=False,inputs={str(d):prior.file_sha256(d) for d in deps}))
        print('EVALUATION_COMPLETE_EXPLICIT_AI_REVIEW_PENDING',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--after-training',action='store_true');a=ap.parse_args()
    if a.after_training:run()
    else:print('NO_WAIT_NO_EVALUATION_NO_TRAINING')
