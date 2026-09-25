"""Independent six-cell runner reusing frozen training and evaluation behavior."""
import fcntl
from scripts.vision.prepare_instance_exposure_balance import OUT,read,verify_tree
import scripts.vision.run_visibility_quality_training as legacy

def finalize(p):
    hist=read(legacy.HIST_EVAL/'completion.json');groups={};checks={}
    inputs={str(OUT/'protocol.json'):legacy.file_sha256(OUT/'protocol.json')}
    for steps in (100,300):
        records=[]
        for seed in (7,17,27):
            key=f'I-{steps}-{seed}';legacy.trainer.checked_cell(OUT/key/'completion.json',p)
            path=OUT/f'evaluation-{key}.json';verify_tree(path);records.append(read(path));inputs[str(path)]=legacy.file_sha256(path)
        name=f'I-{steps}';groups[name]=legacy.aggregate(records)
        checks[name]=legacy.policy_checks(groups[name],hist['aggregate'][f'R-{steps}'],hist['historical_A'],p)
    selected=next((name for name in p['candidate_priority'] if checks[name]['passed']),None)
    from scripts.vision.verify_experiment_baseline import verify
    baseline=verify(legacy.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    verify_tree(OUT/'protocol.json')
    return legacy.save(OUT/'completion.json',dict(status='development_candidate_selected' if selected else 'complete_no_candidate',selected_family=selected,aggregate=groups,policy_results=checks,baseline=baseline,inputs=inputs))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_tree(OUT/'protocol.json');p=read(OUT/'protocol.json')
        legacy.OUT=OUT;legacy.trainer.OUT=OUT
        for steps in (100,300):
            for seed in (7,17,27):
                key=f'I-{steps}-{seed}';verify_tree(OUT/'protocol.json')
                if not (OUT/key/'completion.json').exists() and len(list((OUT/key).glob('attempt-*')))>=3:raise ValueError('Attempt budget exhausted')
                legacy.trainer.train(key,{**p,'datasets':{'I':p['datasets'][key]}})
                r=legacy.evaluate(key,p);print('EVALUATED',key,r['negative_summary'],flush=True)
        print('FINAL',finalize(p)['status'],flush=True)

if __name__=='__main__':main()
