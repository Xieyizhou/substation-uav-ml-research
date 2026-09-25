"""Nominal dose/window matched, non-member-targeted contrast control."""
import argparse,copy,hashlib
from collections import Counter
from pathlib import Path
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import contrast_cpu4_control as cpu
from scripts.vision import contrast_transfer_control as transform_runtime
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT
SOURCE_KEYS=tuple(routed.KEYS)
GLOBAL=routed.SOURCE/'contrast'
GLOBAL_KEYS=tuple(routed.PRIOR_KEYS)
OUT=SOURCE.parent/'dose-matched-contrast-control-v1'
KEYS=tuple(f'dose-matched-480-{s}' for s in (7,17,27))
VERSION='dose-window-matched-nontargeted-sha256-v1'
checked=routed.checked

def coefficients(seed,global_values,target):
    if len(global_values)!=2880 or len(target)!=2880:raise ValueError('Expected 2880 positions')
    if any(v not in (.75,1.,1.25) for v in global_values+target):raise ValueError('Unknown contrast factor')
    if any(t!=1 and t!=g for t,g in zip(target,global_values)):raise ValueError('Target not subset of global factors')
    values=[1.]*2880
    for start in range(0,2880,300):
        stop=min(start+300,2880)
        for factor in (.75,1.25):
            count=target[start:stop].count(factor)
            eligible=[i for i in range(start,stop) if global_values[i]==factor]
            eligible.sort(key=lambda i:hashlib.sha256(f'{VERSION}|{seed}|{start}|{factor}|{i}'.encode()).hexdigest())
            if count>len(eligible):raise ValueError('Insufficient eligible positions')
            for i in eligible[:count]:values[i]=factor
        if Counter(values[start:stop])!=Counter(target[start:stop]):raise ValueError('Window dose mismatch')
    return values

def validate(old,global_p,p):
    for field in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if p[field]!=old[field]:raise ValueError('Non-dose change '+field)
    expected=copy.deepcopy(old['configuration']);expected['augmentation']=VERSION
    if p['configuration']!=expected:raise ValueError('Configuration drift')
    for seed,key,prior,gkey in zip((7,17,27),KEYS,SOURCE_KEYS,GLOBAL_KEYS):
        for field in ('schedules','exposures','windows','listings'):
            if p[field][key]!=old[field][prior] or old[field][prior]!=global_p[field][gkey]:raise ValueError('Exposure or batch drift')
        if p['contrast'][key]!=coefficients(seed,global_p['contrast'][gkey],old['contrast'][prior]):raise ValueError('Coefficient allocation drift')

def freeze():
    source=SOURCE/'protocol.json';gp=GLOBAL/'protocol.json';old,g=checked(source),checked(gp)
    ep=SOURCE/'audit-v1/evidence.json';rp=SOURCE/'audit-v1/review.json';np=SOURCE/'evaluation-v1/error-review-v1/evidence.json'
    e,r,n=checked(ep),checked(rp),checked(np);validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions'])
    dest=OUT/'protocol.json'
    if dest.exists():
        p=checked(dest);validate(old,g,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(old)
    for field in ('identity','inputs','routing_ledger'):p.pop(field,None)
    for field in ('schedules','exposures','windows','listings'):p[field]={k:copy.deepcopy(old[field][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['contrast']={k:coefficients(s,g['contrast'][gk],old['contrast'][o]) for s,k,o,gk in zip((7,17,27),KEYS,SOURCE_KEYS,GLOBAL_KEYS)}
    p['configuration']['augmentation']=VERSION
    by={r['member_id']:r for r in p['pool_rows']};p['dose_ledger']={}
    for key,prior in zip(KEYS,SOURCE_KEYS):
        v=p['contrast'][key];t=old['contrast'][prior];seq=p['schedules'][key]
        p['dose_ledger'][key]=dict(nonidentity_draws=sum(x!=1 for x in v),changed_positions=sum(a!=b for a,b in zip(v,t)),
            windows=[dict(first_step=i//6+1,last_step=min(i+300,2880)//6,factors=dict(Counter(v[i:i+300]))) for i in range(0,2880,300)],
            per_variant={variant:sum(by[m]['variant']==variant and f!=1 for m,f in zip(seq,v)) for variant in sorted({r['variant'] for r in p['pool_rows']})})
    deps=dict(old['inputs'])
    for key in SOURCE_KEYS:
        for name in ('completion.json','tensor-verification.json','thread-verification.json'):
            q=SOURCE/'training'/key/name;c=checked(q);deps[str(q.resolve())]=file_sha256(q)
            if name=='completion.json':
                xp=Path(c['exposure_path']);ex=checked(xp)
                if c['optimizer_steps']!=480 or ex['actual']!=old['schedules'][key]:raise ValueError('Historical exposure mismatch')
                for a in (xp,Path(c['weights'])):deps[str(a.resolve())]=file_sha256(a)
        ec=SOURCE/'evaluation-v1/units'/key/'completion.json';c=checked(ec);checked(c['result'])
        for a in (ec,Path(c['result'])):deps[str(a.resolve())]=file_sha256(a)
    for q in (source,gp,ep,rp,np,Path(__file__),OUT/'research-plan-zh.md'):deps[str(q.resolve())]=file_sha256(q)
    p.update(status='dose_matched_frozen_preflight_pending',arm='dose_matched_non_targeted',direct_control=str(SOURCE),
        design='Same 392 members, 2880 draws, 480 batches, full labels and per-position global eligible factors. Match routed counts for each factor in every 50-step window (last window 30). SHA256 selection independent of member/class/variant and evaluation scores.',
        rationale='Routed augmentation recovered some common-condition recall versus global, but nominal dose fell. Match dose and coarse temporal exposure to distinguish these explanations.',
        limits='Nominal dose and factor counts, not pixel-distance dose, are matched. Within-window positions and member/augmentation combinations differ. Negatives may now be transformed but membership and exposure positions do not change. Shared scenes and variants not independent samples. No threshold, checkpoint or seed search.',
        contrast_definition=VERSION,training_admitted=False,promotable=False,inputs=deps)
    validate(old,g,p);return write_record(dest,p)

def summarize():
    records=[];transitions=[];queue=[];deps={}
    for seed,key,prior,gkey in zip((7,17,27),KEYS,SOURCE_KEYS,GLOBAL_KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        for name,root,oldkey in (('routed',SOURCE,prior),('reference',GLOBAL.parent/'reference',gkey),('global',GLOBAL,gkey)):
            op=root/'evaluation-v1/units'/oldkey/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
            for row in r['rows']:transitions.append(dict(seed=seed,reference=name,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
            for q in (op,Path(o['result'])):deps[str(q.resolve())]=file_sha256(q)
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for q in (cp,Path(c['result']),OUT/'training'/key/'thread-verification.json'):checked(q);deps[str(q.resolve())]=file_sha256(q)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_fixed_retention_gates_pending',group=runtime.evaluation.aggregate(records),instance_comparisons=transitions,
        negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    cpu.OUT=OUT;cpu.KEYS=KEYS;transform_runtime.OUT=OUT;transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=transform_runtime.loader;runtime.MODULE='scripts.vision.dose_matched_contrast_control'
    runtime.TESTS+=('tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control','tests.test_contrast_cpu4_review','tests.test_dose_matched_contrast_control')
    runtime.summarize=summarize

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:cpu.worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
