"""Explicit six-cell launcher, revalidating quality leaves before every worker."""
import argparse
import fcntl
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision import train_closed_source_control as training
from scripts.vision import ready_closed_source_control as gate

OUT=training.OUT;KEYS=training.KEYS;prior=training.prior


def run():
    gate.main();training.ready()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=OUT/'training-authorization.json'
        if not auth.exists():prior.frozen(auth,dict(status='explicit_train_command_received',cells=list(KEYS),
            scope='User authorized supplementation and continued progress until training starts, after quality and real loader gates',
            inputs={str(OUT/'entry-ready.json'):prior.file_sha256(OUT/'entry-ready.json'),str(Path(__file__).resolve()):prior.file_sha256(__file__)}))
        else:prior.verify(prior.read(auth))
        for key in KEYS:
            if (OUT/'training'/key/'completion.json').exists():training.complete(key);continue
            folder=OUT/'workers'/key;folder.mkdir(parents=True,exist_ok=True)
            number=len(list(folder.glob('attempt-*')))+1
            if number>3:raise ValueError('Worker budget exhausted')
            attempt=folder/f'attempt-{number:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.run_closed_source_control','--train','--worker',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    prior.frozen(attempt/'launch.json',dict(status='worker_launched_not_optimizer_confirmation',cell=key,pid=proc.pid,
                        training_admitted=False,promotable=False,inputs={str(auth):prior.file_sha256(auth)}))
                    print('STARTED',key,proc.pid,flush=True);proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Worker failed: '+str(attempt))
                training.complete(key)
            except BaseException:
                if proc is not None:training.cleanup(proc)
                prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            finally:
                if proc is not None:training.cleanup(proc)
            print('TRAINED',key,flush=True)
        prior.frozen(OUT/'training/completion.json',dict(status='six_cells_trained_not_evaluated',cells=list(KEYS),
            inputs={str(OUT/'training'/k/'completion.json'):prior.file_sha256(OUT/'training'/k/'completion.json') for k in KEYS}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('--worker requires explicit --train')
    if a.worker:gate.main();training.worker(a.worker)
    elif a.train:run()
    else:gate.main();print('NO_TRAINING_STARTED')
