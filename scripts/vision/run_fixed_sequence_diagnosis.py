"""Isolate training seed under one fresh, outcome-independent exposure sequence."""
import sys,json,random,hashlib,subprocess
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.train_hard_negative_coverage as trainer
from scripts.vision.exposure_protocol import read,save,file_sha256,SEEDS,exposures,NAMES
from scripts.vision.evaluate_hard_negative_coverage import EVAL as PREV,OLD,VARIANTS,aggregate,policy_checks,predict,checked_rows,paired_truth,score,summary
from scripts.vision.train_hard_negative_coverage import verify_tree
from src.ml.artifacts import object_sha256
SOURCE=trainer.TRAIN
ANCHOR=trainer.OUT/'negative-anchor-v1'
OUT=trainer.OUT/'fixed-sequence-seed-diagnosis-v1'
KEYS=tuple(f'F-100-{s}' for s in SEEDS)

def schedule(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['subset']].append(r['member_id'])
    draws=[]
    for subset,n in (('base',216),('regular',156),('bridge_positive',120)):
        rng=random.Random(f'fixed-sequence-v1:{subset}');stream=[]
        while len(stream)<n:
            block=sorted(groups[subset]);rng.shuffle(block);stream.extend(block)
        draws.extend(stream[:n])
    old=sorted(r['member_id'] for r in rows if r['subset']=='hard_negative' and not r['member_id'].startswith('coverage:'))
    pairs=defaultdict(list)
    for r in rows:
        if r['member_id'].startswith('coverage:'):pairs[r['lineage_id']].append(r['member_id'])
    if len(old)!=24 or len(pairs)!=48 or any(len(x)!=2 for x in pairs.values()):raise ValueError('Negative pools incomplete')
    keys=sorted(pairs);random.Random('fixed-sequence-v1:new-pairs').shuffle(keys)
    draws+=old*3+[m for k in keys[:18] for m in sorted(pairs[k])]
    random.Random('fixed-sequence-v1:positions').shuffle(draws)
    if len(draws)!=600:raise ValueError('Quota mismatch')
    return draws,keys[:18]

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(ANCHOR/'error-diagnosis-v1/completion.json')
    source=read(SOURCE/'protocol.json');rows=source['pool_rows'];draws,pairs=schedule(rows)
    OUT.mkdir(parents=True,exist_ok=True);lookup={r['member_id']:r for r in rows}
    listing=OUT/'members.txt';listing.write_text('\n'.join(lookup[m]['image_path'] for m in sorted(set(draws)))+'\n')
    data=OUT/'dataset.yaml';data.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n')
    paths=[Path(__file__),ROOT/'tests/test_fixed_sequence_diagnosis.py',SOURCE/'protocol.json',ANCHOR/'error-diagnosis-v1/completion.json',listing,data]
    paths += [ROOT/'scripts/vision'/f'{n}.py' for n in ('train_hard_negative_coverage','evaluate_hard_negative_coverage','evaluate_exposure_diagnosis','exposure_metrics','finalize_exposure_diagnosis')]
    return save(path,dict(status='frozen',pool_rows=rows,schedules={k:draws for k in KEYS},datasets={k:str(data) for k in KEYS},
        exposures={k:exposures(rows,draws) for k in KEYS},controls=source['controls'],selected_new_pairs=pairs,
        selection='Fresh deterministic string-seeded sampling; not selected from historical seed performance.',
        judgment=dict(fpr_range_threshold=2/48,planned_range_threshold=2/12,recall_range_threshold=.05,
            variants=['original','lighting'],rule='Material seed variation if any inclusive threshold is reached. Otherwise limited observed variation; not proof for other sequences.',
            parameter_and_prediction_equality='Also report exact identity separately; candidate still requires all existing R/A gates.'),
        inputs={str(p):file_sha256(p) for p in paths}))

def parameter_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.model.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def variation(group,policy):
    def span(stats):return stats['max_seed']-stats['min_seed']
    checks=[dict(metric='no_target.FPR',range=span(group['no_target']['frame_false_positive_rate']),threshold=policy['fpr_range_threshold'])]
    for v in policy['variants']:
        checks.append(dict(metric=f'{v}.planned_hit',range=span(group[v]['planned_instance_hit_rate']),threshold=policy['planned_range_threshold']))
        for c in (None,*NAMES):
            r=group[v] if c is None else group[v]['per_class'][c]
            checks.append(dict(metric=f'{v}.{c or "all"}.recall',range=span(r['instance_recall']),threshold=policy['recall_range_threshold']))
    for c in checks:c['material']=c['range']+1e-12>=c['threshold']
    return dict(material_seed_variation=any(c['material'] for c in checks),checks=checks)

def main():
    p=prepare();trainer.TRAIN=OUT
    for key in KEYS:trainer.train(key,p)
    from ultralytics import YOLO
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=trainer.BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Incomplete negative review')
    inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',rpath,npath)};inputs.update(receipt_inputs)
    for r in reviewed+neg['frames']:
        if r['decision']!='accepted' or file_sha256(r['image_path'])!=r['image_sha256']:raise ValueError('Stale image or review')
        inputs[r['image_path']]=r['image_sha256']
    records={}
    for key in KEYS:
        cp=OUT/key/'completion.json';cell=trainer.checked_cell(cp,p);ep=OUT/f'evaluation-{key}.json'
        sources={**inputs,str(cp):file_sha256(cp)}
        if ep.exists():
            verify_tree(ep);r=read(ep)
            if r['inputs']!=sources:raise ValueError('Stale cached evaluation')
        else:
            print('EVALUATE',key,flush=True);model=YOLO(cell['weights']);digest=parameter_hash(model)
            rows=[score(row,truth,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)) for row,truth in paired]
            negatives=[]
            for row in neg['frames']:
                preds=predict(model,row['image_path'],.37)
                negatives.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=preds,frame_has_prediction=bool(preds)))
            r=save(ep,dict(status='complete',cell=key,inputs=sources,rows=rows,negative_rows=negatives,parameter_sha256=digest,
                summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
                negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negatives)/48,unmatched_predictions=sum(len(x['predictions']) for x in negatives)),
                matching_conflicts=sum(x['matching_conflict'] for x in rows)))
            del model
        if r['status']!='complete' or r['cell']!=key or len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete evaluation or matching conflict')
        if {(x['view_id'],x['variant'],x['image_sha256']) for x in r['rows']}!={(x['view_id'],x['variant'],x['image_sha256']) for x,_ in paired}:raise ValueError('Paired membership mismatch')
        if {(x['view_id'],x['variant'],x['image_sha256']) for x in r['negative_rows']}!={(x['view_id'],x['variant'],x['image_sha256']) for x in neg['frames']}:raise ValueError('Negative membership mismatch')
        records[key]=r
    group=aggregate([records[k] for k in KEYS]);verify_tree(PREV/'completion.json');old=read(PREV/'completion.json')
    gates=policy_checks(group,old['same_budget_R'],old['historical_A'],read(OLD/'protocol.json'))
    v=variation(group,p['judgment'])
    suites=['tests.test_fixed_sequence_diagnosis','tests.test_negative_anchor','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    b=subprocess.run([sys.executable,'scripts/vision/verify_experiment_baseline.py'],capture_output=True,text=True);baseline=json.loads(b.stdout) if b.returncode==0 else {}
    if t.returncode or not baseline.get('integrity_passed') or baseline.get('pinned_files_verified')!=40:raise ValueError('Verification failed')
    verify_tree(OUT/'protocol.json')
    result=save(OUT/'completion.json',dict(status='development_complete',seed_order=list(SEEDS),group=group,variation=v,candidate_gates=gates,
        selected_candidate='F-all-three-seeds' if gates['passed'] else None,
        identical_parameters=len({r['parameter_sha256'] for r in records.values()})==1,
        identical_predictions=len({object_sha256(dict(rows=r['rows'],negative_rows=r['negative_rows'])) for r in records.values()})==1,
        tests_output=t.stdout+t.stderr,baseline=baseline,
        inputs={str(q):file_sha256(q) for q in [OUT/'protocol.json',PREV/'completion.json']+[OUT/f'evaluation-{k}.json' for k in KEYS]}))
    print('JUDGMENT',json.dumps({k:result[k] for k in ('variation','identical_parameters','identical_predictions','selected_candidate','group')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
