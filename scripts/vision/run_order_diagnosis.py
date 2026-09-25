"""One-seed exposure-order controls, never an across-seed candidate family."""
import sys,random,json,subprocess
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.train_hard_negative_coverage as trainer
from scripts.vision.run_fixed_sequence_diagnosis import OUT as REFERENCE,parameter_hash
from scripts.vision.exposure_protocol import read,save,file_sha256,exposures,NAMES
from scripts.vision.evaluate_hard_negative_coverage import VARIANTS,predict,checked_rows,paired_truth,score,summary
from scripts.vision.train_hard_negative_coverage import verify_tree
from scripts.vision.finalize_hard_negative_coverage_training import losses
from src.ml.artifacts import object_sha256
OUT=trainer.OUT/'exposure-order-diagnosis-v1'
KEYS=('W-100-7','B-100-7','G-100-7')

def batches(draws):return [draws[i:i+6] for i in range(0,len(draws),6)]

def transform(draws):
    within=[];rng=random.Random('exposure-order-v1:within')
    for batch in batches(draws):
        batch=list(batch);rng.shuffle(batch);within+=batch
    whole=batches(draws);random.Random('exposure-order-v1:whole').shuffle(whole)
    regroup=list(draws);random.Random('exposure-order-v1:regroup').shuffle(regroup)
    return dict(zip(KEYS,(within,[m for b in whole for m in b],regroup)))

def validate_transforms(reference,schedules):
    if set(schedules)!=set(KEYS) or len(reference)!=600:raise ValueError('Incomplete order design')
    for sequence in schedules.values():
        if len(sequence)!=600 or Counter(sequence)!=Counter(reference):raise ValueError('Exposure multiset changed')
    if any(Counter(a)!=Counter(b) for a,b in zip(batches(reference),batches(schedules['W-100-7']))):raise ValueError('W changes batch membership')
    if Counter(map(tuple,batches(reference)))!=Counter(map(tuple,batches(schedules['B-100-7']))):raise ValueError('B changes batch contents or internal order')
    if any(s==reference for s in schedules.values()):raise ValueError('Transformation is identity')

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(REFERENCE/'report-receipt.json');ref=read(REFERENCE/'protocol.json')
    original=ref['schedules']['F-100-7'];schedules=transform(original);validate_transforms(original,schedules)
    paths=[Path(__file__),ROOT/'tests/test_order_diagnosis.py',REFERENCE/'report-receipt.json']
    return save(path,dict(status='frozen',reference_cell='F-100-7',pool_rows=ref['pool_rows'],schedules=schedules,
        datasets={k:ref['datasets']['F-100-7'] for k in KEYS},controls=ref['controls'],
        exposures={k:exposures(ref['pool_rows'],s) for k,s in schedules.items()},
        transforms=dict(W='Shuffle positions within each six-image batch; preserve batch order and membership.',
            B='Permute 100 complete batches; preserve internal order and membership.',
            G='Shuffle all 600 draws; preserve member multiplicity, change grouping and time order together.'),
        judgment=dict(fpr_delta=2/48,planned_delta=2/12,recall_delta=.05,variants=['original','lighting'],
            rule='Material difference if any absolute delta vs reference reaches the inclusive threshold. Operational, not statistical significance.'),
        candidate_policy='One seed per order arm: diagnostic only; never select a three-seed candidate by pooling these arms.',
        scope='No new members, quotas, architecture, augmentation, threshold or sealed-scene evaluation.',
        inputs={str(p):file_sha256(p) for p in paths}))

def deltas(result,reference,policy):
    checks=[dict(metric='no_target.FPR',delta=result['negative_summary']['frame_false_positive_rate']-reference['negative_summary']['frame_false_positive_rate'],threshold=policy['fpr_delta'])]
    for v in policy['variants']:
        checks.append(dict(metric=f'{v}.planned_hit',delta=result['summary'][v]['planned_instance_hit_rate']-reference['summary'][v]['planned_instance_hit_rate'],threshold=policy['planned_delta']))
        for name in (None,*NAMES):
            a=result['summary'][v] if name is None else result['summary'][v]['per_class'][name]
            b=reference['summary'][v] if name is None else reference['summary'][v]['per_class'][name]
            checks.append(dict(metric=f'{v}.{name or "all"}.recall',delta=a['instance_recall']-b['instance_recall'],threshold=policy['recall_delta']))
    for c in checks:c['material']=abs(c['delta'])+1e-12>=c['threshold']
    return dict(material_difference=any(c['material'] for c in checks),checks=checks)

def main():
    from ultralytics import YOLO
    p=prepare();trainer.TRAIN=OUT
    original=read(REFERENCE/'protocol.json')['schedules']['F-100-7'];validate_transforms(original,p['schedules'])
    for key in KEYS:trainer.train(key,p)
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=trainer.BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Negative review incomplete')
    rp=REFERENCE/'evaluation-F-100-7.json';verify_tree(rp);reference=read(rp)
    inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',rpath,npath,rp)};inputs.update(receipt_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale review')
        inputs[row['image_path']]=row['image_sha256']
    ref_model=YOLO(read(REFERENCE/'F-100-7/completion.json')['weights']);ref_state=ref_model.model.state_dict()
    records={};curves={}
    for key in KEYS:
        cp=OUT/key/'completion.json';cell=trainer.checked_cell(cp,p);ep=OUT/f'evaluation-{key}.json';sources={**inputs,str(cp):file_sha256(cp)}
        actual=read(cell['exposure_path'])
        if Counter(actual['draws'])!=Counter(original):raise ValueError('Actual member exposures differ')
        curves[key]=losses(Path(cell['exposure_path']).parent/'results.csv')
        if ep.exists():
            verify_tree(ep);r=read(ep)
            if r['inputs']!=sources:raise ValueError('Cached inference changed')
        else:
            print('EVALUATE',key,flush=True);model=YOLO(cell['weights']);digest=parameter_hash(model)
            changes={}
            for name,t in model.model.state_dict().items():
                difference=(t.detach().cpu().double()-ref_state[name].detach().cpu().double()).abs()
                changes[name]=dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()),changed=int((difference!=0).sum()))
            rows=[score(row,truth,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)) for row,truth in paired]
            negatives=[]
            for row in neg['frames']:
                preds=predict(model,row['image_path'],.37)
                negatives.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=preds,frame_has_prediction=bool(preds)))
            r=save(ep,dict(status='complete',cell=key,inputs=sources,rows=rows,negative_rows=negatives,parameter_sha256=digest,parameter_differences=changes,
                summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
                negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negatives)/48,unmatched_predictions=sum(len(x['predictions']) for x in negatives)),matching_conflicts=sum(x['matching_conflict'] for x in rows)))
            del model
        if r['status']!='complete' or r['cell']!=key or len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicting evaluation')
        for field,expected in [('rows',[x for x,_ in paired]),('negative_rows',neg['frames'])]:
            if {(x['view_id'],x['variant'],x['image_sha256']) for x in r[field]}!={(x['view_id'],x['variant'],x['image_sha256']) for x in expected}:raise ValueError('Evaluation membership changed')
        records[key]=r
    comparisons={};refrows={(r['pair_id'],r['variant']):r for r in reference['rows']};refneg={(r['view_id'],r['variant']):r for r in reference['negative_rows']}
    for key,r in records.items():
        comparisons[key]=dict(**deltas(r,reference,p['judgment']),identical_parameters=r['parameter_sha256']==reference['parameter_sha256'],
            identical_predictions=object_sha256(dict(rows=r['rows'],negative_rows=r['negative_rows']))==object_sha256(dict(rows=reference['rows'],negative_rows=reference['negative_rows'])),
            paired_deltas=[dict(pair_id=x['pair_id'],variant=x['variant'],category=x['category'],planned_hit_delta=int(x['planned_instance_hit'])-int(refrows[x['pair_id'],x['variant']]['planned_instance_hit']),matched_instance_delta=len(x['matches'])-len(refrows[x['pair_id'],x['variant']]['matches'])) for x in r['rows']],
            negative_transitions=[dict(view_id=x['view_id'],variant=x['variant'],reference=refneg[x['view_id'],x['variant']]['frame_has_prediction'],current=x['frame_has_prediction']) for x in r['negative_rows']])
    suites=['tests.test_order_diagnosis','tests.test_fixed_sequence_diagnosis','tests.test_negative_anchor','tests.test_hard_negative_coverage_evaluation','tests.test_hard_negative_coverage','tests.test_canonical_gates','tests.test_canonical_diagnostic_absence','tests.test_canonical_recovery','tests.test_canonical_shutdown','tests.test_exposure_diagnosis','tests.test_visual_bridge_training','tests.test_visual_bridge_supplement','tests.test_paired_visual_factors']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True);b=subprocess.run([sys.executable,'scripts/vision/verify_experiment_baseline.py'],capture_output=True,text=True)
    baseline=json.loads(b.stdout) if b.returncode==0 else {}
    if t.returncode or not baseline.get('integrity_passed') or baseline.get('pinned_files_verified')!=40:raise ValueError('Verification failed')
    verify_tree(OUT/'protocol.json')
    save(OUT/'completion.json',dict(status='diagnosis_complete',selected_candidate=None,comparisons=comparisons,loss_curves=curves,
        optimizer_steps=300,image_exposures=1800,tests_output=t.stdout+t.stderr,baseline=baseline,
        inputs={str(q):file_sha256(q) for q in [OUT/'protocol.json',rp]+[OUT/f'evaluation-{k}.json' for k in KEYS]+[ROOT/(s.replace('.','/')+'.py') for s in suites]}))
    print('JUDGMENT',json.dumps({k:{'material_difference':v['material_difference'],'identical_parameters':v['identical_parameters'],'identical_predictions':v['identical_predictions'],'FPR':records[k]['negative_summary']['frame_false_positive_rate']} for k,v in comparisons.items()}),flush=True)

if __name__=='__main__':main()
