"""One predeclared lower-rate contrast against I-300; no threshold search."""
import argparse
import csv
import fcntl
from pathlib import Path
from collections import Counter
import yaml
from scripts.vision.prepare_instance_exposure_balance import OUT as PRIOR,read,save,file_sha256,verify_tree,ROOT
from scripts.vision.analyze_visibility_quality_results import paired_change
import scripts.vision.run_visibility_quality_training as evaluator
import scripts.vision.train_lower_rate_control as trainer

OUT=PRIOR/'lower-rate-control-v1'

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(PRIOR/'completion.json');p=read(PRIOR/'protocol.json');OUT.mkdir(parents=True,exist_ok=True)
    inputs={str(x):file_sha256(x) for x in (PRIOR/'protocol.json',PRIOR/'completion.json',Path(__file__),ROOT/'scripts/vision/train_lower_rate_control.py',ROOT/'tests/test_lower_rate_control.py')}
    paired={};prefix={};errors={};datasets={};schedules={};expected={}
    for seed in (7,17,27):
        records={};loss={}
        for steps in (100,300):
            cp=PRIOR/f'I-{steps}-{seed}/completion.json';cell=evaluator.trainer.checked_cell(cp,p)
            ep=PRIOR/f'evaluation-I-{steps}-{seed}.json';r=read(ep);records[steps]=r
            for x in (cp,ep):inputs[str(x)]=file_sha256(x)
            with (Path(cell['exposure_path']).parent/'results.csv').open() as f:
                loss[steps]=[{k:v for k,v in row.items() if 'train/' in k} for row in csv.DictReader(f)]
            if r['matching_conflicts'] or any(x['matching_conflict'] for x in r['rows']):raise ValueError('Unresolved matching conflict')
            counts=Counter(f"{row['variant']}:{m['class_name']}:{m['reason']}" for row in r['rows'] for m in row['misses'])
            errors[f'I-{steps}-{seed}']=dict(counts)
        prefix[str(seed)]=p['schedules'][f'I-300-{seed}'][:600]==p['schedules'][f'I-100-{seed}'] and loss[300][:len(loss[100])]==loss[100]
        if not prefix[str(seed)]:raise ValueError('I prefix mismatch')
        paired[str(seed)]=paired_change(records[100]['rows'],records[300]['rows'])
        oldkey=f'I-300-{seed}';key=f'J-300-{seed}'
        schedules[key]=p['schedules'][oldkey];expected[key]=p['exposures'][oldkey];datasets[key]=p['datasets'][oldkey]
    evidence=save(OUT/'prior-analysis.json',dict(status='six_I_cells_verified_no_candidate',paired_budget_change=paired,miss_events=errors,training_prefix_verified=prefix,inputs=inputs))
    controls={**p['controls'],'learning_rate':0.0003,'lr0':0.0003}
    return save(path,dict(status='frozen_before_training',pool_rows=p['pool_rows'],schedules=schedules,exposures=expected,datasets=datasets,controls=controls,acceptance_policy=p['acceptance_policy'],retention=p['retention'],seeds=[7,17,27],steps=300,candidate_priority=['J-300'],max_attempts=3,inputs={**inputs,str(OUT/'prior-analysis.json'):file_sha256(OUT/'prior-analysis.json')},sole_training_change='constant AdamW lr0 0.001 -> 0.0003; exact I-300 schedules, membership, initialization, CPU, augmentations and endpoints retained',rationale='I-300 improved original planned recall and FPR but full-image retention still fails, with many same-class low-confidence misses. Test less aggressive adaptation at fixed budget; not proof that excessive LR is the unique cause. No learning-rate grid or favorable-seed selection.'))

def checked_runtime(key,p):
    cell=trainer.checked_cell(OUT/key/'completion.json',p)
    args=yaml.safe_load((Path(cell['exposure_path']).parent/'args.yaml').read_text())
    if args['lr0']!=.0003 or args['lrf']!=1 or args['batch']!=6 or args['epochs']!=30:raise ValueError('Runtime training settings mismatch')
    return cell

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    p=prepare()
    if args.prepare_only:print('LOWER_RATE_FROZEN',OUT);return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trainer.OUT=OUT;evaluator.OUT=OUT
        records=[];inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}
        for seed in (7,17,27):
            key=f'J-300-{seed}';verify_tree(OUT/'protocol.json')
            if not (OUT/key/'completion.json').exists() and len(list((OUT/key).glob('attempt-*')))>=3:raise ValueError('Attempt budget exhausted')
            trainer.train(key,{**p,'datasets':{'J':p['datasets'][key]}});checked_runtime(key,p)
            records.append(evaluator.evaluate(key,p));ep=OUT/f'evaluation-{key}.json';inputs[str(ep)]=file_sha256(ep)
            print('EVALUATED',key,records[-1]['negative_summary'],flush=True)
        hist=read(evaluator.HIST_EVAL/'completion.json');group=evaluator.aggregate(records)
        checks=evaluator.policy_checks(group,hist['aggregate']['R-300'],hist['historical_A'],p)
        from scripts.vision.verify_experiment_baseline import verify
        baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
        verify_tree(OUT/'protocol.json')
        result=save(OUT/'completion.json',dict(status='development_candidate_selected' if checks['passed'] else 'complete_no_candidate',selected_family='J-300' if checks['passed'] else None,aggregate={'J-300':group},policy_results={'J-300':checks},baseline=baseline,inputs=inputs))
        print('FINAL',result['status'],flush=True)

if __name__=='__main__':main()
