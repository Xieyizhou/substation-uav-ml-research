"""Pre-frozen six-cell visibility-quality intervention; unchanged development gates."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from scripts.vision.run_visibility_cleanup_validation import OUT as REVIEW,ROOT,read,save,file_sha256,verify_tree
from scripts.vision.train_hard_negative_coverage import TRAIN as HIST_TRAIN
from scripts.vision.exposure_protocol import OUT as HIST_EVAL,exposures,SEEDS
from scripts.vision.canonical_batch_order import canonicalize
from scripts.vision.run_fixed_budget_diagnosis import predict,checked_rows,paired_truth,score,summary,VARIANTS,BASE
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
import scripts.vision.run_exposure_diagnosis as trainer

OUT=REVIEW/'visibility-quality-training-v1'

def filtered_draws(rows,old,excluded,seed):
    lookup={r['member_id']:r for r in rows};counts=Counter(x for x in old if x not in excluded)
    eligible={s:sorted(r['member_id'] for r in rows if r['subset']==s and r['member_id'] not in excluded) for s in {r['subset'] for r in rows}}
    rng=random.Random('visibility-quality-v1:'+str(seed));tie={}
    for s,ids in eligible.items():
        rng.shuffle(ids);tie.update({x:i for i,x in enumerate(ids)})
    result=[];swaps=[]
    for i,mid in enumerate(old):
        if mid in excluded:
            new=min(eligible[lookup[mid]['subset']],key=lambda x:(counts[x],tie[x]));counts[new]+=1;swaps.append(dict(position=i,old=mid,new=new));result.append(new)
        else:result.append(mid)
    if len(result)!=600 or set(result)&excluded:raise ValueError('Held members or incorrect draw count')
    if Counter(lookup[x]['subset'] for x in old)!=Counter(lookup[x]['subset'] for x in result):raise ValueError('Changed subset quotas')
    return canonicalize(result),swaps

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(REVIEW/'completion.json');review=read(REVIEW/'review.json')
    hist=read(HIST_TRAIN/'protocol.json');rows=hist['pool_rows'];held=set(review['held_members'])
    group=lambda r:r.get('derivation_group') or r['lineage_id']
    excluded_groups={group(r) for r in rows if r['member_id'] in held}
    excluded={r['member_id'] for r in rows if group(r) in excluded_groups}
    retained=[r for r in rows if r['member_id'] not in excluded]
    OUT.mkdir(parents=True,exist_ok=True);schedules={};datasets={};swaps={};inputs={}
    paths=[REVIEW/'completion.json',HIST_TRAIN/'protocol.json',HIST_EVAL/'protocol.json',HIST_EVAL/'completion.json',Path(__file__),ROOT/'scripts/vision/run_exposure_diagnosis.py',ROOT/'tests/test_visibility_quality_training.py']
    for r in retained:
        for k in ('image','label'):
            if file_sha256(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('Stale retained member')
            inputs[r[k+'_path']]=r[k+'_sha256']
    for seed in SEEDS:
        sequence,changes=filtered_draws(rows,hist['schedules'][f'N-100-{seed}'],excluded,seed);swaps[str(seed)]=changes
        for steps in (100,300):
            key=f'Q-{steps}-{seed}';schedules[key]=sequence*(steps//100)
            listing=OUT/f'{key}.txt';lookup={r['member_id']:r for r in retained}
            listing.write_text('\n'.join(lookup[x]['image_path'] for x in sorted(set(sequence)))+'\n')
            dataset=OUT/f'{key}.yaml';dataset.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n')
            datasets[key]=str(dataset);paths.extend([listing,dataset])
    inputs.update({str(p):file_sha256(p) for p in paths})
    policy=read(HIST_EVAL/'protocol.json')
    return save(path,dict(status='frozen',pool_rows=retained,excluded_members=sorted(excluded),excluded_group_count=len(excluded_groups),
        original_pool_count=len(rows),retained_pool_count=len(retained),retained_subset_counts=dict(Counter(r['subset'] for r in retained)),
        schedules=schedules,datasets=datasets,swaps_before_canonical_sort=swaps,exposures={k:exposures(retained,v) for k,v in schedules.items()},
        controls=hist['controls'],acceptance_policy=policy['acceptance_policy'],retention=policy['retention'],
        max_attempts=3,candidate_priority=['Q-100','Q-300'],inputs=inputs,
        interpretation='Conservative whole-image/derivation-group exclusion plus within-subset least-exposed replacement. Not a pure single-label causal effect. No thresholds, optimizer or augmentation changes; per-batch canonical order. 300-step sequence repeats 100-step draws exactly three times.',
        retained_admission='Reuse existing audited development membership; not a new all-label review or formal training admission.',
        endpoint_policy='All six independent cells run; only last endpoint. Validation on training members is fit diagnostic only. Keep all seeds.',
        protected_test='sealed_not_evaluated'))

def evaluate(key,protocol):
    from ultralytics import YOLO
    cp=OUT/key/'completion.json';cell=trainer.checked_cell(cp,protocol);ep=OUT/f'evaluation-{key}.json'
    if ep.exists():verify_tree(ep);return read(ep)
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    np=BASE/'hard-negative-isolated-v2/semantic-review.json';negative=read(np)
    if negative['status']!='reviewed' or negative['accepted']!=48 or negative['held']:raise ValueError('Incomplete development review')
    inputs={str(p):file_sha256(p) for p in (OUT/'protocol.json',cp,rpath,np)};inputs.update(receipt_inputs)
    for row in reviewed+negative['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale image review')
        inputs[row['image_path']]=row['image_sha256']
    model=YOLO(cell['weights']);results=[score(row,t,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)) for row,t in paired]
    neg=[]
    for row in negative['frames']:
        preds=predict(model,row['image_path'],.37);low=predict(model,row['image_path'],.001)
        neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=preds,diagnostic_predictions=low,frame_has_prediction=bool(preds)))
    if len(results)!=48 or len(neg)!=48 or any(r['matching_conflict'] for r in results):raise ValueError('Incomplete/conflicting evaluation')
    return save(ep,dict(status='complete',cell=key,rows=results,negative_rows=neg,matching_conflicts=0,
        summary={v:summary([r for r in results if r['variant']==v]) for v in VARIANTS},
        negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in neg)/48,unmatched_predictions=sum(len(r['predictions']) for r in neg)),inputs=inputs))

def finalize(protocol):
    historical=read(HIST_EVAL/'completion.json');groups={};checks={};inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}
    for steps in (100,300):
        records=[]
        for seed in SEEDS:
            key=f'Q-{steps}-{seed}';cp=OUT/key/'completion.json';trainer.checked_cell(cp,protocol)
            ep=OUT/f'evaluation-{key}.json';verify_tree(ep);records.append(read(ep));inputs[str(ep)]=file_sha256(ep)
        name=f'Q-{steps}';groups[name]=aggregate(records)
        checks[name]=policy_checks(groups[name],historical['aggregate'][f'R-{steps}'],historical['historical_A'],protocol)
    selected=next((name for name in protocol['candidate_priority'] if checks[name]['passed']),None)
    from scripts.vision.verify_experiment_baseline import verify
    baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    verify_tree(OUT/'protocol.json')
    return save(OUT/'completion.json',dict(status='development_candidate_selected' if selected else 'complete_no_candidate',selected_family=selected,
        aggregate=groups,policy_results=checks,baseline=baseline,inputs=inputs,
        unseen_scene_status='sealed_not_evaluated',training_admitted=False,promotable=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();p=prepare()
    print('QUALITY_PROTOCOL',p['retained_subset_counts'],'EXCLUDED',len(p['excluded_members']),flush=True)
    if args.prepare_only:return
    trainer.OUT=OUT
    for steps in (100,300):
        for seed in SEEDS:
            key=f'Q-{steps}-{seed}'
            verify_tree(OUT/'protocol.json')
            if not (OUT/key/'completion.json').exists() and len(list((OUT/key).glob('attempt-*')))>=3:raise ValueError('Training attempt budget exhausted')
            trainer.train(key,{**p,'datasets':{'Q':p['datasets'][key]}})
            result=evaluate(key,p)
            print('EVALUATED',key,result['negative_summary'],flush=True)
    print('QUALITY_RESULT',finalize(p)['status'],flush=True)

if __name__=='__main__':main()
