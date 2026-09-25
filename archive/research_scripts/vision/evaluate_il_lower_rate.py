"""One-shot post-training fixed evaluation; never approves visual reviews."""
import argparse,fcntl,subprocess,sys,signal
from pathlib import Path
from scripts.vision.il_lower_rate_control import OUT,REF,KEYS,prior
from scripts.vision import evaluate_tail_interleaving as adapter
from scripts.vision.train_closed_source_control import cleanup

def protocol():
    p=prior.read(OUT/'design.json');prior.verify(p);return p

def configure():
    adapter.OUT=OUT;adapter.KEYS=KEYS;adapter.protocol=protocol

def evaluate(key):
    configure();r=adapter.evaluate(key)
    path=OUT/'evaluation'/f'{key}-light-binding.json'
    if path.exists():prior.verify(prior.read(path))
    else:
        deps=[Path(__file__).resolve(),OUT/'evaluation'/f'{key}.json',OUT/'evaluation'/f'{key}-adapter-binding.json']
        prior.frozen(path,dict(status='current_adapter_bound',inputs={str(d):prior.file_sha256(d) for d in deps}))
    return r

def finish():
    records=[evaluate(k) for k in KEYS];fixed=adapter.fixed
    sp=REF/'evaluation/summary.json';old=prior.read(sp);prior.verify(old)
    refs=[];deps=[sp,Path(__file__).resolve()]
    for key in KEYS:
        rp=REF/'evaluation'/('IL1000-'+key.split('-')[-1]+'.json');r=prior.read(rp);prior.verify(r);refs.append(r)
        deps.extend([rp,OUT/'evaluation'/f'{key}.json',OUT/'evaluation'/f'{key}-light-binding.json'])
    p=protocol();hp=Path(p['evaluation']['historical_reference']);h=prior.read(hp);prior.verify(h);deps.append(hp)
    historical=[]
    for seed in (7,17,27):
        path=fixed.PRIOR/f'evaluation-retained_reference-450-{seed}.json';r=prior.read(path);prior.verify(r);historical.append(r);deps.append(path)
    _,policy,_=fixed.reference_contract('fixed-7')
    baseline=fixed.baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    g=fixed.aggregate(records);ref=fixed.aggregate(refs)
    return prior.frozen(OUT/'evaluation/summary.json',dict(status='numerical_complete_explicit_review_pending',aggregate=g,IL1000=ref,B900=old['B900'],
        policy=fixed.policy_checks(g,fixed.aggregate(historical),h['historical_A'],policy),relative_IL1000=fixed.direct_checks(g,ref),relative_B900=fixed.direct_checks(g,old['B900']),
        paired_changes={str(s):fixed.paired_change(a['rows'],b['rows']) for s,a,b in zip((7,17,27),refs,records)},matching_conflicts=sum(r['matching_conflicts'] for r in records),
        selected_candidate=None,review_complete=False,baseline=baseline,inputs={str(d):prior.file_sha256(d) for d in deps}))

def after_training():
    protocol();root=OUT/'evaluation-runner';root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        print('WAITING_FOR_THREE_SEED_TRAINING_LOCK',flush=True)
        with (OUT/'training-runner/run.lock').open('r') as training:
            fcntl.flock(training,fcntl.LOCK_SH)
            cp=OUT/'training-runner/completion.json';prior.verify(prior.read(cp));configure()
            for key in KEYS:adapter.complete(key)
        for group in (KEYS[:2],KEYS[2:]):
            jobs=[]
            try:
                for key in group:
                    if (OUT/'evaluation'/f'{key}.json').exists():evaluate(key);continue
                    folder=root/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Attempt cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_il_lower_rate','--cell',key,'--evaluate'],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log));print('EVALUATION_STARTED',key,proc.pid,flush=True)
                for proc,_ in jobs:
                    proc.wait(timeout=1200)
                    if proc.returncode:raise RuntimeError('Evaluation failed; see independent attempt log')
            finally:
                for proc,log in jobs:cleanup(proc);log.close()
        finish();deps=[cp,OUT/'evaluation/summary.json',Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='numerical_complete_AI_review_pending',review_complete=False,selected_candidate=None,inputs={str(d):prior.file_sha256(d) for d in deps}))
        print('NUMERICAL_EVALUATION_COMPLETE_REVIEW_PENDING',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--after-training',action='store_true');ap.add_argument('--evaluate',action='store_true');ap.add_argument('--cell',choices=KEYS);a=ap.parse_args()
    if a.after_training:after_training()
    elif a.evaluate:
        if a.cell:evaluate(a.cell)
        else:finish()
    else:protocol();print('PREFLIGHT_ONLY_NO_EVALUATION_NO_TRAINING')
