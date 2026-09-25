"""Pre-frozen 72 old / 36 new negative exposure experiment."""
import sys,json,random,subprocess
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.train_hard_negative_coverage as trainer
from scripts.vision.exposure_protocol import read,save,file_sha256,SEEDS,exposures
from scripts.vision.evaluate_hard_negative_coverage import EVAL as PREV,OLD,VARIANTS,aggregate,policy_checks,predict,checked_rows,paired_truth,score,summary
from scripts.vision.train_hard_negative_coverage import verify_tree
OUT=trainer.OUT/'negative-anchor-v1'
SOURCE=trainer.TRAIN

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(PREV/'negative-diagnosis-v1/completion.json')
    prior=read(SOURCE/'protocol.json');rows=prior['pool_rows'];lookup={r['member_id']:r for r in rows}
    schedules={};datasets={}
    OUT.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        old=sorted(r['member_id'] for r in rows if r['subset']=='hard_negative' and not r['member_id'].startswith('coverage:'))
        groups=defaultdict(list)
        for mid in prior['schedules'][f'N-100-{seed}']:
            if mid.startswith('coverage:'):groups[lookup[mid]['lineage_id']].append(mid)
        keys=sorted(groups);rng=random.Random(f'anchor-v1:{seed}');rng.shuffle(keys)
        fresh=[m for k in keys[:18] for m in groups[k]]
        negatives=old*3+fresh;rng.shuffle(negatives);it=iter(negatives)
        draws=[next(it) if lookup[mid]['subset']=='hard_negative' else mid for mid in prior['schedules'][f'O-100-{seed}']]
        assert len(fresh)==36 and len(draws)==600
        key=f'H-100-{seed}';schedules[key]=draws
        listing=OUT/f'{key}.txt';listing.write_text('\n'.join(lookup[m]['image_path'] for m in sorted(set(draws)))+'\n')
        dataset=OUT/f'{key}.yaml';dataset.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n')
        datasets[key]=str(dataset)
    paths=[Path(__file__),SOURCE/'protocol.json',PREV/'negative-diagnosis-v1/completion.json']
    paths += [OUT/f'{k}.{ext}' for k in schedules for ext in ('txt','yaml')]
    return save(path,dict(status='frozen',pool_rows=rows,schedules=schedules,datasets=datasets,controls=prior['controls'],
        exposures={k:exposures(rows,v) for k,v in schedules.items()},
        quota=dict(base=216,regular=156,bridge_positive=120,old_negative=72,new_negative=36),
        interpretation='Fixed-budget redistribution, not full old-exposure restoration or pure causal isolation.',
        judgment='FPR mean below N and at least two seeds strictly improve; original/lighting overall and each class mean recall decline <= .05 vs N. Candidate separately requires all historical gates.',
        inputs={str(p):file_sha256(p) for p in paths}))

def main():
    p=prepare();trainer.TRAIN=OUT
    for key in p['schedules']:trainer.train(key,p)
    from ultralytics import YOLO
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=trainer.BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['accepted']!=48 or neg['held']:raise ValueError('Incomplete negative review')
    records={};inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',rpath,npath)};inputs.update(receipt_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale review')
        inputs[row['image_path']]=row['image_sha256']
    for key in p['schedules']:
        cp=OUT/key/'completion.json';cell=trainer.checked_cell(cp,p);ep=OUT/f'evaluation-{key}.json'
        sources={**inputs,str(cp):file_sha256(cp)}
        if ep.exists():
            verify_tree(ep);record=read(ep)
            if record['inputs']!=sources:raise ValueError('Stale evaluation')
        else:
            print('EVALUATE',key,flush=True);model=YOLO(cell['weights'])
            rows=[score(r,t,predict(model,r['image_path'],.37),predict(model,r['image_path'],.001)) for r,t in paired]
            negatives=[]
            for r in neg['frames']:
                predictions=predict(model,r['image_path'],.37)
                negatives.append(dict(view_id=r['view_id'],variant=r['variant'],image_sha256=r['image_sha256'],predictions=predictions,frame_has_prediction=bool(predictions)))
            record=save(ep,dict(status='complete',inputs=sources,rows=rows,negative_rows=negatives,
                summary={v:summary([r for r in rows if r['variant']==v]) for v in VARIANTS},
                negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in negatives)/48,unmatched_predictions=sum(len(r['predictions']) for r in negatives)),
                matching_conflicts=sum(r['matching_conflict'] for r in rows)))
        if len(record['rows'])!=48 or len(record['negative_rows'])!=48 or record['matching_conflicts']:raise ValueError('Incomplete/conflicting evaluation')
        records[key]=record
    verify_tree(PREV/'completion.json');old=read(PREV/'completion.json');h=aggregate(list(records.values()));n=old['groups']['N']
    gates=policy_checks(h,old['same_budget_R'],old['historical_A'],read(OLD/'protocol.json'))
    retention=[]
    for v in ('original','lighting'):
        for name in (None,'transformer','switchgear','capacitor_bank','reactor'):
            a=h[v] if name is None else h[v]['per_class'][name];b=n[v] if name is None else n[v]['per_class'][name]
            retention.append(dict(variant=v,category=name,delta=a['instance_recall']['mean']-b['instance_recall']['mean']))
    fpr=h['no_target']['frame_false_positive_rate'];previous=n['no_target']['frame_false_positive_rate']
    improves=sum(a<b for a,b in zip(fpr['values'],previous['values']))
    supported=fpr['mean']<previous['mean'] and improves>=2 and all(r['delta']>=-.05-1e-12 for r in retention)
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_hard_negative_coverage','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation'],capture_output=True,text=True)
    baseline=subprocess.run([sys.executable,'scripts/vision/verify_experiment_baseline.py'],capture_output=True,text=True)
    b=json.loads(baseline.stdout) if baseline.returncode==0 else {}
    if tests.returncode or not b.get('integrity_passed') or b.get('pinned_files_verified')!=40:raise ValueError('Verification failed')
    verify_tree(OUT/'protocol.json')
    result=save(OUT/'completion.json',dict(status='development_complete',sampling_hypothesis_supported=supported,improving_seeds=improves,
        group=h,relative_N_retention=retention,candidate_gates=gates,selected_candidate='H-all-three-seeds' if gates['passed'] else None,
        tests_output=tests.stdout+tests.stderr,baseline=b,
        inputs={str(q):file_sha256(q) for q in [OUT/'protocol.json',PREV/'completion.json']+[OUT/f'evaluation-{k}.json' for k in records]}))
    print('JUDGMENT',json.dumps({k:result[k] for k in ('sampling_hypothesis_supported','improving_seeds','selected_candidate','group','relative_N_retention')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
