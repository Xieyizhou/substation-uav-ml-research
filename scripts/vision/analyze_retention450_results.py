"""Post-training numerical audit, never an automatic visual review or promotion."""
import csv
import math
from pathlib import Path
import yaml
from scripts.vision.prepare_retention450_training import OUT,KEYS,ROOT,read,save,file_sha256,verify_tree,baseline_verify,validate_args
from scripts.vision.run_exposure_diagnosis import checked_cell
from scripts.vision.run_matched_appearance_training import validate_evaluation
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks

def main():
    dest=OUT/'numerical-audit-v1.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'execution.json')
    p=read(OUT/'protocol.json');records={};inputs={str(OUT/'execution.json'):file_sha256(OUT/'execution.json')}
    for key in KEYS:
        cp=OUT/key/'completion.json';cell=checked_cell(cp,p)
        if cell['optimizer_steps']!=450:raise ValueError('Wrong endpoint')
        folder=Path(cell['exposure_path']).parent
        args=yaml.safe_load((folder/'args.yaml').read_text());validate_args(args)
        if args['seed']!=int(key.split('-')[-1]):raise ValueError('Wrong seed')
        rows=list(csv.DictReader((folder/'results.csv').open()))
        if len(rows)!=45:raise ValueError('Incomplete curve')
        for row in rows:
            if not any(k.strip().startswith('lr/') for k in row):raise ValueError('Missing LR')
            for k,v in row.items():
                if k.strip().startswith('train/') and not math.isfinite(float(v)):raise ValueError('Nonfinite loss')
                if k.strip().startswith('lr/') and abs(float(v)-.001)>1e-12:raise ValueError('LR changed')
        ep=OUT/f'evaluation-{key}.json';r=read(ep);validate_evaluation(r,key)
        for path,digest in r['inputs'].items():
            if file_sha256(path)!=digest:raise ValueError('Stale evaluation input')
        records[key]=r
        for path in (cp,ep,folder/'sampler-preflight.json',folder/'results.csv'):
            inputs[str(path)]=file_sha256(path)
    groups={arm:aggregate([records[f'{arm}-450-{s}'] for s in (7,17,27)]) for arm in ('retained_reference','retained_appearance')}
    hp=Path(p['evaluation']['historical_reference']);hist=read(hp);inputs[str(hp)]=file_sha256(hp)
    gate=policy_checks(groups['retained_appearance'],groups['retained_reference'],hist['historical_A'],p)
    for c in gate['checks']:
        if c.get('reference')=='same_budget_R':c['reference']='matched_retained_reference-450'
    paired={}
    for seed in (7,17,27):
        a={(x['pair_id'],x['variant']):x for x in records[f'retained_reference-450-{seed}']['rows']}
        for v in ('original','material','background','lighting'):
            b=[x for x in records[f'retained_appearance-450-{seed}']['rows'] if x['variant']==v]
            paired[f'{seed}:{v}']={'gains':sum(x['planned_assigned_hit'] and not a[x['pair_id'],v]['planned_assigned_hit'] for x in b),
                'losses':sum(not x['planned_assigned_hit'] and a[x['pair_id'],v]['planned_assigned_hit'] for x in b)}
    events=[dict(cell=key,view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=row['predictions'])
        for key,r in records.items() for row in r['negative_rows'] if row['predictions']]
    inputs[str(Path(__file__))]=file_sha256(Path(__file__))
    save(dest,dict(status='numerically_rejected_visual_review_pending',selected_family=None,aggregate=groups,policy=gate,paired=paired,
        negative_frames_to_review=events,negative_prediction_events=sum(len(e['predictions']) for e in events),
        unique_negative_images=len({e['image_sha256'] for e in events}),runtime_units_verified=6,
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs))
    print('AUDIT_SAVED',dest,flush=True)

if __name__=='__main__':main()
