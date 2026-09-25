"""Independent development export and frozen evaluation/reference identities."""
import shutil
from collections import Counter
from pathlib import Path
import yaml
from scripts.vision.design_visibility_repair_contrast import OUT as DESIGN,SOURCE,ROOT,REFERENCE,read,save,file_sha256,verify_tree,baseline_verify
from scripts.vision.exposure_protocol import exposures,NAMES,BASE
import scripts.vision.run_visibility_quality_training as evaluation

OUT=DESIGN/'training-v1'
KEYS=tuple(f'{arm}-300-{seed}' for seed in (7,17,27) for arm in ('original_only','three_variant'))

def prepare():
    dest=OUT/'protocol.json'
    if dest.exists():verify_tree(dest);return read(dest)
    verify_tree(DESIGN/'design-validation.json');design=read(DESIGN/'design.json');old=read(REFERENCE/'protocol.json')
    lookup={r['member_id']:r for r in old['pool_rows']};ids=sorted(set().union(*(set(x) for x in design['schedules'].values())))
    reviewed,rpath=evaluation.checked_rows();paired,truth_inputs=evaluation.paired_truth(reviewed)
    np=BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(np)
    if len(paired)!=48 or neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Development evaluation review incomplete')
    hist=evaluation.HIST_EVAL/'completion.json';verify_tree(hist);historical=read(hist)
    if 'R-300' not in historical['aggregate'] or not historical['historical_A']:raise ValueError('Reference groups missing')
    paths=[DESIGN/'design-validation.json',REFERENCE/'protocol.json',rpath,np,hist,Path(__file__),ROOT/'scripts/vision/run_visibility_repair_training.py']
    inputs={str(p):file_sha256(p) for p in paths};inputs.update(truth_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale development review')
        inputs[row['image_path']]=row['image_sha256']
    eval_hashes={r['image_sha256'] for r in reviewed+neg['frames']}
    weight=Path(design['controls']['initial_weights'])
    if file_sha256(weight)!=design['controls']['initial_weights_sha256']:raise ValueError('Initialization changed')
    inputs[str(weight)]=file_sha256(weight)
    export=OUT/'export-attempt-001'
    if export.exists():raise ValueError('Incomplete prior export preserved; resolve in a new export version')
    (export/'images').mkdir(parents=True);(export/'labels').mkdir()
    rows=[]
    for i,mid in enumerate(ids):
        source=lookup[mid]
        if source['image_sha256'] in eval_hashes:raise ValueError('Development evaluation image in training')
        row={**source,'source_image_path':source['image_path'],'source_label_path':source['label_path']}
        for kind,folder,suffix in [('image','images',Path(source['image_path']).suffix),('label','labels','.txt')]:
            src=Path(source[kind+'_path'])
            if file_sha256(src)!=source[kind+'_sha256']:raise ValueError('Source member changed')
            target=export/folder/f'{i:03}{suffix}';shutil.copy2(src,target)
            if file_sha256(target)!=source[kind+'_sha256']:raise ValueError('Export byte mismatch')
            row[kind+'_path']=str(target);inputs[str(target)]=source[kind+'_sha256']
        rows.append(row)
    member={r['member_id']:r for r in rows};datasets={}
    for key in KEYS:
        listing=export/f'{key}.txt';listing.write_text('\n'.join(member[mid]['image_path'] for mid in sorted(set(design['schedules'][key])))+'\n')
        config=export/f'{key}.yaml';config.write_text(yaml.safe_dump(dict(path=str(export),train=str(listing),val=str(listing),names=list(NAMES))))
        datasets[key]=str(config)
        for path in (listing,config):inputs[str(path)]=file_sha256(path)
    return save(dest,dict(status='frozen_before_training_runtime_sampler_gate_required',pool_rows=rows,datasets=datasets,
        schedules=design['schedules'],exposures={k:exposures(rows,v) for k,v in design['schedules'].items()},
        controls=design['controls'],acceptance_policy=design['acceptance_policy'],retention=design['retention'],max_attempts=3,
        inherited_common_review=dict(source_protocol=str(REFERENCE/'protocol.json'),source_identity=old['identity'],
            membership_and_hash_chain_revalidated=True,new_visual_review=False,subset_counts=dict(Counter(r['subset'] for r in rows if r['subset']!='bridge_positive'))),
        bridge_review=str(SOURCE/'diagnosis.json'),evaluation=dict(paired_review=str(rpath),negative_review=str(np),historical_reference=str(hist),confidence=.37,diagnostic_confidence=.001,imgsz=640,device='cpu',nms_iou=.7,max_det=300,matching_iou=.5,agnostic_nms=False),
        initialization=dict(path=str(weight),sha256=file_sha256(weight)),candidate_policy='three_variant only after all six units, formal evaluation, matched retention and explicit prediction review; no promotion',
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs))

if __name__=='__main__':print('FROZEN',prepare()['identity'],flush=True)
