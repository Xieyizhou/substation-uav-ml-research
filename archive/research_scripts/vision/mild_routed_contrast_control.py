"""Single half-amplitude material-routing control; optimizer-free default."""
import argparse,copy
from pathlib import Path
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import dose_matched_contrast_control as dose
from scripts.vision import contrast_cpu4_control as cpu
from scripts.vision import contrast_transfer_control as transform_runtime
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT
SOURCE_KEYS=tuple(routed.KEYS)
DOSE=dose.OUT
DOSE_KEYS=tuple(dose.KEYS)
GLOBAL=routed.SOURCE/'contrast'
GLOBAL_KEYS=tuple(routed.PRIOR_KEYS)
OUT=SOURCE.parent/'mild-routed-contrast-control-v1'
KEYS=tuple(f'mild-routed-480-{s}' for s in (7,17,27))
VERSION='material-routing-half-contrast-amplitude-v1'
checked=routed.checked

def coefficients(values):
    if len(values)!=2880 or any(x not in (.75,1.,1.25) for x in values):raise ValueError('Invalid routed factors')
    return [1.+(x-1.)*.5 for x in values]

def transform(images,values):
    import torch
    if images.dtype!=torch.uint8 or images.ndim!=4 or len(images)!=len(values):raise ValueError('Invalid contrast input')
    if any(x not in (.875,1.,1.125) for x in values):raise ValueError('Unfrozen mild factor')
    x=images.float();mean=x.mean(dim=(2,3),keepdim=True)
    factor=torch.tensor(values,dtype=x.dtype,device=x.device).reshape(-1,1,1,1)
    raw=(x-mean)*factor+mean
    return raw.round().clamp(0,255).to(torch.uint8),((raw<0)|(raw>255)).sum(dim=(1,2,3)).tolist()

def validate(old,p):
    for field in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if p[field]!=old[field]:raise ValueError('Non-amplitude change '+field)
    config=copy.deepcopy(old['configuration']);config['augmentation']=VERSION
    if p['configuration']!=config:raise ValueError('Configuration drift')
    for key,prior in zip(KEYS,SOURCE_KEYS):
        for field in ('schedules','exposures','windows','listings'):
            if p[field][key]!=old[field][prior]:raise ValueError('Exposure or full-label drift')
        if p['contrast'][key]!=coefficients(old['contrast'][prior]):raise ValueError('Amplitude or position drift')

def reviews_valid(root):
    ep=root/'audit-v1/evidence.json';rp=root/'audit-v1/review.json';np=root/'evaluation-v1/error-review-v1/evidence.json'
    e,r,n=checked(ep),checked(rp),checked(np);validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions']);checked(root/'audit-v1/completion.json')
    return (ep,rp,np,root/'audit-v1/completion.json')

def freeze():
    source=SOURCE/'protocol.json';old=checked(source);review_paths=(*reviews_valid(SOURCE),*reviews_valid(DOSE));dest=OUT/'protocol.json'
    if dest.exists():
        p=checked(dest);validate(old,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(old)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['contrast']={k:coefficients(old['contrast'][o]) for k,o in zip(KEYS,SOURCE_KEYS)};p['configuration']['augmentation']=VERSION
    p['amplitude_ledger']={k:dict(augmented_draws=sum(v!=1 for v in p['contrast'][k]),identity_positions_unchanged=True,
        windows=[dict(first_step=i//6+1,last_step=min(i+300,2880)//6,contracted=p['contrast'][k][i:i+300].count(.875),expanded=p['contrast'][k][i:i+300].count(1.125)) for i in range(0,2880,300)]) for k in KEYS}
    deps=dict(old['inputs'])
    for root,keys in ((SOURCE,SOURCE_KEYS),(DOSE,DOSE_KEYS)):
        protocol=checked(root/'protocol.json');deps[str((root/'protocol.json').resolve())]=file_sha256(root/'protocol.json')
        for key in keys:
            for name in ('completion.json','tensor-verification.json','thread-verification.json'):
                f=root/'training'/key/name;c=checked(f);deps[str(f.resolve())]=file_sha256(f)
                if name=='completion.json':
                    xp=Path(c['exposure_path']);ex=checked(xp)
                    if c['optimizer_steps']!=480 or ex['actual']!=protocol['schedules'][key]:raise ValueError('Historical actual exposure mismatch')
                    for q in (xp,Path(c['weights'])):deps[str(q.resolve())]=file_sha256(q)
            ec=root/'evaluation-v1/units'/key/'completion.json';c=checked(ec);checked(c['result'])
            for q in (ec,Path(c['result'])):deps[str(q.resolve())]=file_sha256(q)
    for q in (source,*review_paths,Path(__file__),OUT/'research-plan-zh.md'):deps[str(q.resolve())]=file_sha256(q)
    p.update(status='half_amplitude_material_routing_frozen_preflight_pending',arm='mild_material_routed',direct_control=str(SOURCE),
        design='Only amplitude changes versus material routing: 0.75->0.875, 1->1, 1.25->1.125. Same per-position route/sign/count, 392 members, full labels, 2880 draws, 480 batches, CPU4, LR0.00025 and independent v2.11 initialization.',
        rationale='Nominal-dose matched non-targeted allocation lost lighting recall in all seeds. Keep routing fixed and test whether smaller perturbations reduce capability losses while retaining material benefit. This is a hypothesis, not an established cause.',
        limits='Whole material frames including their background/padding transform; not object segmentation or physical rendering. Shared scenes, three seeds not independent data. Half nominal amplitude need not halve clipped/rounded pixel distance. No new data, labels, threshold or checkpoint search.',
        contrast_definition='round(clamp(channel_mean + factor*(pixel-channel_mean),0,255)), factors 0.875/1/1.125 in exact old routed positions.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(old,p);return write_record(dest,p)

def summarize():
    records=[];transitions=[];queue=[];deps={}
    for seed,key,prior,dkey,gkey in zip((7,17,27),KEYS,SOURCE_KEYS,DOSE_KEYS,GLOBAL_KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        for name,root,oldkey in (('routed',SOURCE,prior),('dose_matched',DOSE,dkey),('reference',GLOBAL.parent/'reference',gkey),('global',GLOBAL,gkey)):
            op=root/'evaluation-v1/units'/oldkey/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
            for row in r['rows']:transitions.append(dict(seed=seed,reference=name,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
            for q in (op,Path(o['result'])):deps[str(q.resolve())]=file_sha256(q)
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for q in (cp,Path(c['result']),OUT/'training'/key/'thread-verification.json'):checked(q);deps[str(q.resolve())]=file_sha256(q)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_fixed_retention_gates_pending',group=runtime.evaluation.aggregate(records),instance_comparisons=transitions,
        negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    cpu.OUT=OUT;cpu.KEYS=KEYS;transform_runtime.OUT=OUT;transform_runtime.KEYS=KEYS;transform_runtime.transform=transform
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=transform_runtime.loader;runtime.MODULE='scripts.vision.mild_routed_contrast_control'
    runtime.TESTS+=('tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control','tests.test_contrast_cpu4_review','tests.test_mild_routed_contrast_control')
    runtime.summarize=summarize

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:cpu.worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
