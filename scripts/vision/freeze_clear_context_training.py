"""Exact integer construction: new original views replace original bridge draws.

This is a feasibility witness, not a claim of global count optimality. No solver
dependency, score-based selection, old-member increase or budget relaxation.
"""
from collections import Counter
import hashlib
from pathlib import Path
import platform
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.prepare_clear_context_increment import OUT as DATA,checked
from scripts.vision.reviewed_negative_order_control import OUT as CONTROL
from scripts.vision.prepare_negative_rehearsal_control import _exposure
from scripts.vision.review_clear_context_increment import validate

OUT=DATA/'training-control-v1'
SEEDS=(7,17,27)
KEYS=tuple(f'clear-context-480-{s}' for s in SEEDS)
NAMES=('transformer','switchgear','capacitor_bank','reactor')
# Sum=120 images, full-label counts=(40,50,60,40). Derived from frozen 12x19-label set.
DONOR_VECTOR_QUOTAS={(0,0,0,1):28,(1,0,0,1):12,(0,0,1,0):32,
                     (1,0,1,0):28,(0,3,0,0):10,(0,2,0,0):10}


def rank(seed,*parts):
    return hashlib.sha256('|'.join(map(str,('clear-context-substitution-v1',seed,*parts))).encode()).hexdigest()


def vector(row):return tuple(int(row['class_instances'].get(c,0)) for c in NAMES)


def construct(rows,new,sequence,seed):
    counts=Counter(sequence);by={r['member_id']:r for r in rows}
    if len(sequence)!=2880 or len(new)!=12:raise ValueError('Unexpected population')
    required=Counter()
    for r in new:
        for c,n in r['class_instances'].items():required[c]+=10*n
    donor_classes=Counter({c:sum(v[j]*n for v,n in DONOR_VECTOR_QUOTAS.items()) for j,c in enumerate(NAMES)})
    if required!=donor_classes or sum(DONOR_VECTOR_QUOTAS.values())!=120:raise ValueError('New supervision does not fit frozen witness')
    decrements=Counter()
    for signature,quota in DONOR_VECTOR_QUOTAS.items():
        options=sorted((r['member_id'] for r in rows if r['subset']=='bridge_positive' and r['variant']=='original' and vector(r)==signature),key=lambda m:rank(seed,'donor-order',m))
        if sum(max(0,counts[m]-1) for m in options)<quota:raise ValueError('Insufficient legal donor capacity')
        while quota:
            for member in options:
                if counts[member]-decrements[member]>1:
                    decrements[member]+=1;quota-=1
                    if quota==0:break
    positions=[]
    for member,n in decrements.items():
        eligible=sorted((i for i,m in enumerate(sequence) if m==member),key=lambda i:rank(seed,'remove-position',member,i))
        positions.extend(eligible[:n])
    replacement=sorted((r['member_id'] for r in new),key=lambda m:rank(seed,'new-member-order',m))*10
    result=list(sequence)
    for position,member in zip(sorted(positions),replacement,strict=True):result[position]=member
    validate_sequence(rows,new,sequence,result)
    return result,dict(decrements),sorted(positions)


def validate_sequence(rows,new,old,sequence):
    all_rows=rows+new;by={r['member_id']:r for r in all_rows};a=Counter(old);b=Counter(sequence)
    if len(by)!=len(all_rows) or len(sequence)!=len(old) or set(b)!=set(by):raise ValueError('Duplicate/missing member or budget drift')
    for r in new:
        if b[r['member_id']]!=10:raise ValueError('New member exposure drift')
    for r in rows:
        m=r['member_id'];eligible=r['subset']=='bridge_positive' and r.get('variant')=='original'
        if b[m]<1 or b[m]>a[m] or (not eligible and b[m]!=a[m]):raise ValueError('Old member increased/removed or protected condition changed')
    for i,(x,y) in enumerate(zip(old,sequence)):
        if x!=y and not (by[x]['subset']=='bridge_positive' and by[x].get('variant')=='original' and y in {r['member_id'] for r in new}):raise ValueError('Non-allowed replacement position')
    before=_exposure(rows,old);after=_exposure(all_rows,sequence)
    for key in ('draws','optimizer_steps','class_instance_exposure','subset_exposure'):
        if before[key]!=after[key]:raise ValueError('Exposure ledger drift: '+key)


def freeze():
    old=checked(CONTROL/'protocol.json');manifest=checked(DATA/'dataset/manifest.json')
    review=checked(DATA/'review.json');e=checked(DATA/'evidence/manifest-v2.json');validate(e,review)
    if manifest['status']!='reviewed_lossless_export_and_reference_exclusion_passed':raise ValueError('Candidate exclusion incomplete')
    import torch,ultralytics
    environment=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,platform=platform.platform())
    if old['environment']!=environment:raise ValueError('Historical control environment differs')
    if file_sha256(old['initialization']['path'])!=old['initialization']['sha256']:raise ValueError('Initialization changed')
    for row in old['pool_rows']+manifest['members']:
        for kind in ('image','label'):
            if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Member bytes changed')
    dest=OUT/'protocol.json'
    if dest.exists():
        p=checked(dest)
        for seed,key in zip(SEEDS,KEYS):validate_sequence(old['pool_rows'],manifest['members'],old['schedules'][f'reviewed-interleaved-480-{seed}'],p['schedules'][key])
        return p
    diagnosis=checked(DATA/'pretraining-fit.json')
    if len(diagnosis['rows'])!=36:raise ValueError('Candidate diagnosis incomplete')
    new=manifest['members'];rows=old['pool_rows']+new;OUT.mkdir(parents=True,exist_ok=True)
    p={k:old[k] for k in ('names','initialization','training_config','evaluation','environment','held_members')}
    p.update(pool_rows=rows,schedules={},exposures={},listings={},windows={},replacement_audit={})
    deps=[CONTROL/'protocol.json',DATA/'dataset/manifest.json',DATA/'review.json',DATA/'evidence/manifest-v2.json',DATA/'pretraining-fit.json',Path(__file__)]
    from scripts.vision import train_reactor_visibility_expansion,order_retention_runtime,evaluate_reviewed_negative_order,clear_context_training
    deps += [Path(x.__file__) for x in (train_reactor_visibility_expansion,order_retention_runtime,evaluate_reviewed_negative_order,clear_context_training)]
    from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
    _,_,evaluation_deps=_load_inputs();deps += [Path(x) for x in evaluation_deps]
    for seed,key in zip(SEEDS,KEYS):
        ref=f'reviewed-interleaved-480-{seed}';seq,removed,positions=construct(old['pool_rows'],new,old['schedules'][ref],seed)
        p['schedules'][key]=seq;p['exposures'][key]=_exposure(rows,seq)
        p['replacement_audit'][key]=dict(decrements=removed,replaced_positions=positions,new_member_draws=120,
            maximum_old_member_decrement=max(removed.values()),construction='Exact integer witness, no optimality claim')
        p['windows'][key]=[dict(first_step=i//6+1,last_step=min(i+300,len(seq))//6,**_exposure(rows,seq[i:i+300])) for i in range(0,len(seq),300)]
        listing=OUT/f'{key}.txt';by={r['member_id']:r for r in rows}
        with listing.open('x') as stream:stream.write(''.join(by[m]['image_path']+'\n' for m in sorted(set(seq))))
        p['listings'][key]=str(listing.resolve());deps.append(listing)
        cp=CONTROL/'training'/ref/'completion.json';c=checked(cp);actual=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or actual['actual']!=old['schedules'][ref]:raise ValueError('Control exposure invalid')
        deps += [cp,Path(c['weights']),Path(c['exposure_path'])]
    p.update(status='frozen_preflight_pending',configuration=dict(steps=480,batch=6,nbs=6,lr=.0005,optimizer='AdamW',augmentation='off',cpu_threads_per_worker=4,max_parallel_workers=2,checkpoint='last.pt',initialization='independent_v2.11'),
        design='12 reviewed new original-appearance poses x 10 exposures replace 120 original bridge-positive draws. Same total/subset/full-class-instance exposure; all material/lighting members and all negative members/positions unchanged. Old members never increase and retain at least one exposure.',
        limits='A data replacement strategy, not pure independent-scene or class-exposure causal effect. Camera pose, context, instance composition and some batch composition differ. Shared extreme layout/primitive assets; simple boxes can remain visually ambiguous. No claim that new data fixes all error structures.',
        acceptance='All existing fixed development and retention gates unchanged; no threshold/checkpoint/seed selection.',
        automatic_followup='After all three endpoints, fixed confidence .37/.001 evaluation on 48 paired + 48 negative images; error queue remains pending AI review.',
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in deps})
    return write_record(dest,p)


if __name__=='__main__':print(freeze()['status'])
