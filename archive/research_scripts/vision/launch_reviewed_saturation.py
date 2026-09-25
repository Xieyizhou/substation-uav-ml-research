"""Launch gate binding tests, fixed-40 integrity and development input identities."""
import argparse
from pathlib import Path
import subprocess
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import reviewed_saturation_control as experiment
from scripts.vision.verify_experiment_baseline import verify, ROOT

TESTS = ('tests.test_reviewed_saturation_control','tests.test_reviewed_negative_order',
         'tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')


def development():
    paired, negatives, deps = experiment.evaluation.ev._load_inputs()
    if len(paired)!=48 or len(negatives)!=48:raise ValueError('Development population drift')
    return deps


def ready(create=False):
    path=experiment.OUT/'launch-readiness.json'
    if not path.exists():
        if not create:raise ValueError('Launch gate missing')
        experiment.preflight()
        baseline=ROOT/'config/perception/visual_experiment_baseline_v1.json'
        integrity=verify(baseline)
        if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Baseline integrity failure')
        command=[sys.executable,'-m','unittest',*TESTS,'-q']
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        deps=development()
        for p in [Path(__file__),Path(experiment.__file__),baseline,experiment.OUT/'protocol.json',experiment.OUT/'entry-ready.json',experiment.OUT/'plan-zh.md']+[ROOT/('tests/'+name.split('.')[-1]+'.py') for name in TESTS]:
            deps[str(p.resolve())]=file_sha256(p)
        write_record(path,dict(status='verified_for_explicit_training_and_automatic_evaluation',
            test_command=command,test_returncode=0,test_output=result.stdout+result.stderr,integrity=integrity,
            development_inputs=development(),training_admitted=False,promotable=False,inputs=deps))
    r=experiment.checked(path)
    if development()!=r['development_inputs']:raise ValueError('Frozen evaluation inputs changed')
    return r


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true')
    parser.add_argument('--worker',choices=experiment.KEYS);parser.add_argument('--eval-worker',choices=experiment.KEYS)
    args=parser.parse_args()
    if args.worker:
        ready();experiment.worker(args.worker)
    elif args.eval_worker:
        ready();experiment.eval_worker(args.eval_worker)
    elif args.train:
        ready(create=True)
        experiment.MODULE='scripts.vision.launch_reviewed_saturation'
        experiment.train()
    else:
        ready(create=True);print('LAUNCH_PREFLIGHT_COMPLETE_NO_TRAINING')
