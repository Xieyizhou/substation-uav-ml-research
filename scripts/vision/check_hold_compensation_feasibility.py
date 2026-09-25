"""Integer counting proof and non-training count witness; no schedule mutation."""
from collections import Counter
from pathlib import Path
from scripts.vision.train_whole_image_hold import OUT as RUN,ready,read,verify,frozen,file_sha256,ROOT

OUT=RUN/'instance-exposure-compensation-feasibility-v1'
NAMES=('reactor','capacitor_bank','transformer','switchgear')

def count(rows):
    return {k:sum(r['class_instances'].get(k,0) for r in rows) for k in NAMES}

def certificate(eligible,slots,target):
    # Every available member has at most one reactor. To restore one reactor per
    # replacement slot, ALL replacement images must have exactly one reactor.
    if any(r['class_instances'].get('reactor',0)>1 for r in eligible):raise ValueError('Reactor bound invalid')
    if target['reactor']!=sum(slots.values()):raise ValueError('Saturation precondition absent')
    maximum={}
    for subset in slots:
        rows=[r for r in eligible if r['subset']==subset and r['class_instances'].get('reactor',0)==1]
        if not rows:raise ValueError('No reactor source in subset')
        maximum[subset]=max(2*r['class_instances'].get('transformer',0)-r['class_instances'].get('switchgear',0) for r in rows)
    lhs=2*target['transformer']-target['switchgear'];rhs=sum(slots[s]*maximum[s] for s in slots)
    return dict(expression='2*transformer-switchgear',per_image_upper_bound=maximum,
        required=lhs,maximum_possible=rhs,violation=lhs-rhs,exact_four_class_infeasible=lhs>rhs)

def build():
    p=ready();prior=RUN/'error-localization-v1/completion.json';verify(read(prior))
    held=set(p['held_member_ids']);idx={r['member_id']:r for r in p['pool_rows']}
    eligible=[r for r in idx.values() if r['member_id'] not in held and r['subset'] in ('base','regular')]
    joint=[r for r in eligible if r['class_instances'].get('reactor',0)==1 and r['class_instances'].get('capacitor_bank',0)==1]
    results={}
    for seed in (7,17,27):
        removed=[idx[m] for m in p['schedules'][f'reference-450-{seed}'] if m in held]
        slots=Counter(r['subset'] for r in removed);target=count(removed);proof=certificate(eligible,slots,target)
        if not proof['exact_four_class_infeasible']:raise ValueError('No certificate; do not claim infeasibility')
        # A deliberately simple integer witness proves ONLY R/C count feasibility.
        # It is not an approved member selection, schedule, or optimization optimum.
        selected=[];witness={}
        for subset,n in slots.items():
            cap=sum(r['class_instances'].get('capacitor_bank',0) for r in removed if r['subset']==subset)
            both=sorted([r for r in joint if r['subset']==subset],key=lambda r:(r['class_instances'].get('switchgear',0),r['member_id']))
            only=sorted([r for r in eligible if r['subset']==subset and r['class_instances'].get('reactor',0)==1 and not r['class_instances'].get('capacitor_bank',0)],key=lambda r:r['member_id'])
            if cap and not both or n>cap and not only:raise ValueError('No count witness')
            for r,k in ([(both[0],cap)] if cap else [])+([(only[0],n-cap)] if n>cap else []):
                selected.extend([r]*k);witness[r['member_id']]=k
        actual=count(selected)
        if any(actual[k]!=target[k] for k in ('reactor','capacitor_bank')):raise ValueError('Invalid count witness')
        results[str(seed)]=dict(replacement_slots=dict(slots),target_instances=target,certificate=proof,
            joint_R_C_support_members=len(joint),minimum_some_joint_member_extra_exposures=(target['capacitor_bank']+len(joint)-1)//len(joint),
            R_C_only_count_witness=witness,witness_instances=actual,witness_delta={k:actual[k]-target[k] for k in NAMES},
            witness_not_a_training_recommendation=True)
    paths=[RUN/'protocol.json',RUN/'ready.json',prior,Path(__file__),ROOT/'docs/results/ml_hold_compensation_feasibility_20260909.md']
    OUT.mkdir(exist_ok=True)
    return frozen(OUT/'analysis.json',dict(status='strict_local_replacement_infeasible_relaxation_not_authorized',results=results,
        joint_support=[dict(member_id=r['member_id'],subset=r['subset'],class_instances=r['class_instances'],lineage_id=r['lineage_id']) for r in joint],
        scope='Only replace original held slots within original subset; all other positions including negatives stay unchanged.',
        independence_certified=False,existing_review_only_not_new_admission=True,training_started=False,new_schedule_created=False,
        solver_note='An exploratory SymPy LP returned a witness violating equalities; rejected by direct substitution. Conclusions use exhaustive finite per-member upper bounds, not that solver result.',
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=build()
    for seed,v in r['results'].items():print(seed,v['certificate'],v['witness_delta'])
