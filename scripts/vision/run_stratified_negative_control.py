"""One frozen 100-step coverage-stratified negative comparison; no promotion."""
import sys,json,random
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.train_hard_negative_coverage as trainer
from scripts.vision.run_fixed_budget_diagnosis import REFERENCE,BASE,read,save,file_sha256,verify_tree,exposures,predict,checked_rows,paired_truth,score,summary,VARIANTS,parameter_hash,retention
from scripts.vision.inspect_budget_errors import OUT as REVIEW
OUT=REFERENCE.parent/'stratified-negative-control-v1'
KEY='S-100-7'
QUOTAS={'B1':4,'B2':4,'G1':1,'G2':1,'M1':4,'C1':2,'C2':1,'P1':1}

def schedule(rows,reference):
    groups=defaultdict(lambda:defaultdict(list));lookup={r['member_id']:r for r in rows}
    for r in rows:
        if r.get('coverage_unit'):groups[r['coverage_unit']][r['lineage_id']].append(r['member_id'])
    selected=[];pairs={}
    for unit,n in QUOTAS.items():
        available=groups[unit]
        if len(available)<n or any(len(v)!=2 for v in available.values()):raise ValueError('Incomplete pair inventory')
        keys=sorted(available);random.Random('stratified-negative-v1:'+unit).shuffle(keys);pairs[unit]=keys[:n]
        selected.extend(mid for k in keys[:n] for mid in sorted(available[k]))
    random.Random('stratified-negative-v1:slot-assignment').shuffle(selected)
    slots=[i for i,m in enumerate(reference) if lookup[m].get('coverage_unit')]
    if len(reference)!=600 or len(slots)!=36 or len(set(selected))!=36:raise ValueError('Invalid reference/new quota')
    result=list(reference)
    for i,mid in zip(slots,selected):result[i]=mid
    validate(rows,reference,result)
    return result,pairs,slots

def validate(rows,reference,result):
    lookup={r['member_id']:r for r in rows}
    if len(reference)!=600 or len(result)!=600:raise ValueError('Wrong budget')
    selected=[]
    for a,b in zip(reference,result):
        if not lookup[a].get('coverage_unit'):
            if a!=b:raise ValueError('Shared position changed')
        else:
            if not lookup[b].get('coverage_unit'):raise ValueError('Negative slot changed role')
            selected.append(b)
    if len(set(selected))!=36:raise ValueError('Duplicate new exposure')
    if Counter(lookup[m]['coverage_unit'] for m in selected)!=Counter({k:2*v for k,v in QUOTAS.items()}):raise ValueError('Coverage quota mismatch')
    if any(v!=2 for v in Counter(lookup[m]['lineage_id'] for m in selected).values()):raise ValueError('Incomplete pair')
    if exposures(rows,reference)['class_instance_exposure']!=exposures(rows,result)['class_instance_exposure']:raise ValueError('Positive supervision changed')

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(REVIEW/'completion.json');verify_tree(REFERENCE/'completion.json')
    ref=read(REFERENCE/'protocol.json');rows=ref['pool_rows'];reference=ref['schedules']['CF-100-7'];draws,pairs,slots=schedule(rows,reference)
    OUT.mkdir(parents=True,exist_ok=True);lookup={r['member_id']:r for r in rows}
    listing=OUT/'members.txt';listing.write_text('\n'.join(lookup[m]['image_path'] for m in sorted(set(draws)))+'\n')
    dataset=OUT/'dataset.yaml';dataset.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n')
    paths=[Path(__file__),ROOT/'tests/test_stratified_negative_control.py',REVIEW/'completion.json',REFERENCE/'completion.json',listing,dataset,ROOT/'scripts/vision/train_hard_negative_coverage.py']
    return save(path,dict(status='frozen',pool_rows=rows,schedules={KEY:draws},datasets={KEY:str(dataset)},controls=ref['controls'],exposures={KEY:exposures(rows,draws)},
        reference_schedule=reference,reference_cell='CF-100-7',selected_pairs=pairs,new_negative_slots=slots,pair_quotas=QUOTAS,
        scope='One-sequence seed 7 diagnostic. Only replace the 36 predefined new-negative slots; preserve all other members AND positions. No post-replacement sorting.',
        interpretation='Fixed-budget coverage substitution, not pure addition. No new data, augmentation, thresholds, architecture or sealed test.',
        candidate_policy='Never select a candidate from this one-sequence experiment. Keep all historical three-seed R/A and absolute gates.',
        retention_tolerance=.05,inputs={str(q):file_sha256(q) for q in paths}))

def main():
    from ultralytics import YOLO
    p=prepare();validate(p['pool_rows'],p['reference_schedule'],p['schedules'][KEY]);trainer.TRAIN=OUT
    cell=trainer.train(KEY,p);cp=OUT/KEY/'completion.json'
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Incomplete review')
    rp=REFERENCE/'evaluation-CF-100-7.json';verify_tree(rp);ref=read(rp)
    sources={str(q):file_sha256(q) for q in (OUT/'protocol.json',cp,rpath,npath,rp)};sources.update(receipt_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale image review')
        sources[row['image_path']]=row['image_sha256']
    ep=OUT/'evaluation.json'
    if ep.exists():
        verify_tree(ep);r=read(ep)
        if r['inputs']!=sources:raise ValueError('Stale evaluation')
    else:
        print('EVALUATING_ENDPOINT',flush=True);model=YOLO(cell['weights']);digest=parameter_hash(model)
        rows=[score(row,t,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)) for row,t in paired]
        negatives=[]
        for row in neg['frames']:
            predictions=predict(model,row['image_path'],.37)
            negatives.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=predictions,frame_has_prediction=bool(predictions)))
        r=save(ep,dict(status='complete',inputs=sources,rows=rows,negative_rows=negatives,parameter_sha256=digest,
            summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negatives)/48,unmatched_predictions=sum(len(x['predictions']) for x in negatives)),
            matching_conflicts=sum(x['matching_conflict'] for x in rows)))
    if len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicting evaluation')
    gains={}
    for v in VARIANTS:
        a={x['pair_id']:x for x in r['rows'] if x['variant']==v};b={x['pair_id']:x for x in ref['rows'] if x['variant']==v}
        if set(a)!=set(b) or len(a)!=12:raise ValueError('Pair mismatch')
        gains[v]=dict(gained=[k for k in a if a[k]['planned_assigned_hit'] and not b[k]['planned_assigned_hit']],lost=[k for k in a if b[k]['planned_assigned_hit'] and not a[k]['planned_assigned_hit']])
    from scripts.vision.finalize_hard_negative_coverage_training import losses
    curve=losses(Path(cell['exposure_path']).parent/'results.csv')
    save(OUT/'completion.json',dict(status='stratified_comparison_complete',selected_candidate=None,optimizer_steps=100,image_exposures=600,
        reference_summary=ref['summary'],stratified_summary=r['summary'],reference_negative=ref['negative_summary'],stratified_negative=r['negative_summary'],
        retention_vs_reference=retention(r['summary'],ref['summary']),paired_gains_losses=gains,loss_curves=curve,training_validation_role='training fit only',
        inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',ep,rp,cp)}))
    print('COMPLETE',r['negative_summary'],flush=True)

if __name__=='__main__':main()
