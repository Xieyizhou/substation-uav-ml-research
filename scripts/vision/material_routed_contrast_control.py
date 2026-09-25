"""One bounded material-member routing control; default optimizer-free preflight."""
import argparse,copy
from collections import Counter
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import contrast_cpu4_control as previous
from scripts.vision import contrast_transfer_control as transform_runtime
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review

SOURCE=previous.ROOT
OUT=SOURCE.parent/'material-routed-contrast-control-v1'
KEYS=tuple(f'material-routed-480-{s}' for s in (7,17,27))
PRIOR_KEYS=tuple(previous.KEYS)
MATERIAL=frozenset(('neutral','cool','warm','gray_all_body','gray_target_body','gray035'))
UNCHANGED=frozenset(('original','physical-lighting','clear_near','occluded','edge_candidate','clear_far'))
VERSION='registered-material-only-contrast-v1'
checked=previous.checked

def route(rows,sequence,values):
    by={r['member_id']:r for r in rows}
    if len(by)!=len(rows) or len(sequence)!=len(values):raise ValueError('Duplicate member or coefficient length')
    if any(r['variant'] not in MATERIAL|UNCHANGED for r in rows):raise ValueError('Unresolved variant')
    if any(v not in (.75,1.,1.25) for v in values):raise ValueError('Unfrozen factor')
    try:return [v if by[m]['variant'] in MATERIAL else 1. for m,v in zip(sequence,values)]
    except KeyError as exc:raise ValueError('Unknown member') from exc

def validate(old,new):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if new[f]!=old[f]:raise ValueError('Non-routing change '+f)
    expected=copy.deepcopy(old['configuration']);expected['augmentation']=VERSION
    if new['configuration']!=expected:raise ValueError('Configuration drift')
    for key,prior in zip(KEYS,PRIOR_KEYS):
        for f in ('schedules','exposures','windows','listings'):
            if new[f][key]!=old[f][prior]:raise ValueError('Exposure drift '+f)
        if new['contrast'][key]!=route(old['pool_rows'],old['schedules'][prior],old['contrast'][prior]):raise ValueError('Routing drift')

def freeze():
    source=SOURCE/'contrast/protocol.json';old=checked(source);dest=OUT/'protocol.json'
    review_root=SOURCE/'contrast/audit-v1';e=checked(review_root/'evidence.json');r=checked(review_root/'review.json')
    n=checked(SOURCE/'contrast/evaluation-v1/error-review-v1/evidence.json');validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions'])
    if dest.exists():
        p=checked(dest);validate(old,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(old)
    for f in ('identity','inputs'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,PRIOR_KEYS)}
    p['contrast']={k:route(old['pool_rows'],old['schedules'][o],old['contrast'][o]) for k,o in zip(KEYS,PRIOR_KEYS)}
    p['configuration']['augmentation']=VERSION
    p['routing_ledger']={}
    by={r['member_id']:r for r in p['pool_rows']}
    for key,prior in zip(KEYS,PRIOR_KEYS):
        seq=p['schedules'][key];values=p['contrast'][key];previous_values=old['contrast'][prior]
        p['routing_ledger'][key]=dict(material_members=sum(r['variant'] in MATERIAL for r in p['pool_rows']),
            changed_vs_global=sum(a!=b for a,b in zip(values,previous_values)),
            actual_nonidentity_draws=sum(v!=1 for v in values),global_nonidentity_draws=sum(v!=1 for v in previous_values),
            per_variant={v:dict(draws=sum(by[m]['variant']==v for m in seq),augmented=sum(by[m]['variant']==v and f!=1 for m,f in zip(seq,values))) for v in sorted(MATERIAL|UNCHANGED)},
            windows=[dict(first_step=i//6+1,last_step=(i+300)//6 if i+300<=2880 else 480,nonidentity_draws=sum(v!=1 for v in values[i:i+300])) for i in range(0,2880,300)])
    deps=dict(old['inputs'])
    for arm in ('reference','contrast'):
        for key in PRIOR_KEYS:
            for name in ('completion.json','tensor-verification.json','thread-verification.json'):
                f=SOURCE/arm/'training'/key/name;c=checked(f);deps[str(f.resolve())]=file_sha256(f)
                if name=='completion.json':
                    xp=Path(c['exposure_path']);ex=checked(xp)
                    if c['optimizer_steps']!=480 or ex['actual']!=old['schedules'][key]:raise ValueError('Historical actual exposure drift')
                    for q in (xp,Path(c['weights'])):deps[str(q.resolve())]=file_sha256(q)
            ec=SOURCE/arm/'evaluation-v1/units'/key/'completion.json';c=checked(ec);checked(c['result'])
            for q in (ec,Path(c['result'])):deps[str(q.resolve())]=file_sha256(q)
    for q in (source,review_root/'completion.json',review_root/'review.json',review_root/'evidence.json',Path(__file__)):
        if q.suffix=='.json':checked(q)
        deps[str(q.resolve())]=file_sha256(q)
    p.update(status='material_routing_frozen_preflight_pending',arm='material_routed',direct_control=str(SOURCE/'reference'),global_contrast_control=str(SOURCE/'contrast'),
        design='Same frozen 392 members, 2880 draws, 480 batches and LR 0.00025. Reuse exact global-arm factors only at registered material variant positions; all other positions identity. All 3 seeds independently initialized from v2.11.',
        rationale='Global contrast improved material recall in all seeds but generated 40 original/lighting losses, 33 with low-confidence same-class candidates. Test whether limiting perturbation preserves common conditions while retaining some material benefit.',
        limits='Material variants include cool/warm lighting combinations; membership routing is not segmentation. Whole frames/background/padding are transformed on routed members. Total augmentation dose decreases, so this is not a dose-matched pure routing causal test. Shared sources are not independent scenes. One bounded arm, no coefficient/seed/checkpoint search.',
        contrast_definition='Use the prior frozen per-position coefficient on neutral/cool/warm/gray_all_body/gray_target_body/gray035 members; coefficient 1 on all other members. Identical channel-mean transform and complete labels.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(old,p);return write_record(dest,p)

def summarize():
    records=[];transitions=[];queue=[];deps={}
    for seed,key,prior in zip((7,17,27),KEYS,PRIOR_KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        for arm in ('reference','contrast'):
            op=SOURCE/arm/'evaluation-v1/units'/prior/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
            for row in r['rows']:transitions.append(dict(seed=seed,reference=arm,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
            for q in (op,Path(o['result'])):deps[str(q.resolve())]=file_sha256(q)
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for q in (cp,Path(c['result']),OUT/'training'/key/'thread-verification.json'):checked(q);deps[str(q.resolve())]=file_sha256(q)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_fixed_retention_gates_pending',group=runtime.evaluation.aggregate(records),
        instance_comparisons=transitions,negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},
        selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    previous.OUT=OUT;previous.KEYS=KEYS;transform_runtime.OUT=OUT;transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=transform_runtime.loader;runtime.MODULE='scripts.vision.material_routed_contrast_control'
    runtime.TESTS=runtime.TESTS+('tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control','tests.test_contrast_cpu4_review','tests.test_material_routed_contrast_control')
    runtime.summarize=summarize

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:previous.worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
