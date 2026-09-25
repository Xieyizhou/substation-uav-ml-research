"""Bounded grayscale-addition control on existing routed contrast positions."""
import argparse, copy
from pathlib import Path
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import mild_routed_contrast_control as mild
from scripts.vision import contrast_transfer_control as augmentation
from scripts.vision import contrast_cpu4_control as cpu
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT
SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-gray-transfer-control-v1'
KEYS=tuple(f'routed-gray-480-{s}' for s in (7,17,27))
VERSION='routed-contrast-plus-rgb-mean-gray-v1'
checked=routed.checked
_contrast=augmentation.transform

def transform(images,values):
    import torch
    if images.ndim!=4 or images.shape[1]!=3:raise ValueError('Expected RGB BCHW')
    out,clipped=_contrast(images,values)
    mask=torch.tensor([v!=1 for v in values],device=images.device,dtype=torch.bool)
    gray=out.float().mean(dim=1,keepdim=True).round().to(torch.uint8)
    return torch.where(mask[:,None,None,None],gray.expand_as(out),out),clipped

def validate(old,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if p[f]!=old[f]:raise ValueError('Unexpected non-transform change '+f)
    config=copy.deepcopy(old['configuration']);config['augmentation']=VERSION
    if p['configuration']!=config:raise ValueError('Training configuration drift')
    for key,prior in zip(KEYS,SOURCE_KEYS):
        for f in ('schedules','exposures','windows','listings','contrast'):
            if p[f][key]!=old[f][prior]:raise ValueError('Exposure, order, label or contrast drift')
        if p['grayscale'][key]!=[v!=1 for v in old['contrast'][prior]]:raise ValueError('Grayscale routing drift')
        if len(p['grayscale'][key])!=2880:raise ValueError('Incomplete sequence')

def review_gate():
    paths=[mild.OUT/'audit-v1/evidence.json',mild.OUT/'audit-v1/review.json',mild.OUT/'evaluation-v1/error-review-v1/evidence.json',
           mild.OUT/'audit-v1/material-evidence.json',mild.OUT/'audit-v1/material-review.json',mild.OUT/'audit-v1/completion.json']
    e,r,n,m,mr,c=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,mr['positive_decisions']);validate_review(n,r['negative_decisions'])
    if not c['integrity']['integrity_passed']:raise ValueError('Missing integrity')
    ep=OUT/'preview-v1/evidence.json';rp=OUT/'preview-v1/review.json'
    preview,review=checked(ep),checked(rp)
    expected={(r['member_id'],r['page_sha256']) for r in preview['rows']}
    actual=[(r['member_id'],r['page_sha256']) for r in review['decisions']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Missing or duplicate preview review')
    if any(r['decision']!='bounded_transform_test_allowed' or not r['reason'] for r in review['decisions']):raise ValueError('Unresolved transform preview')
    paths += [ep,rp]
    return paths

def freeze():
    old=checked(SOURCE/'protocol.json');dest=OUT/'protocol.json'
    review_paths=review_gate()
    if dest.exists():
        p=checked(dest);validate(old,p);return p
    OUT.mkdir(exist_ok=True);p=copy.deepcopy(old)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):
        p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['augmentation']=VERSION
    p['grayscale']={k:[v!=1 for v in p['contrast'][k]] for k in KEYS}
    paths=[SOURCE/'protocol.json',Path(__file__),OUT/'research-plan-zh.md',Path(augmentation.__file__),Path(cpu.__file__),Path(runtime.__file__),
           Path('tests/test_routed_gray_transfer.py'),*review_paths]
    fit=SOURCE.parent/'routed-amplitude-fit-diagnosis-v1'
    paths += [fit/'protocol.json',fit/'summary.json',fit/'residual-review-v1/evidence.json',fit/'residual-review-v1/review.json']
    for root,keys in ((SOURCE,SOURCE_KEYS),(mild.OUT,mild.KEYS)):
        q=checked(root/'protocol.json')
        for key in keys:
            cp=root/'training'/key/'completion.json';c=checked(cp);xp=Path(c['exposure_path']);ex=checked(xp)
            if c['optimizer_steps']!=480 or ex['actual']!=q['schedules'][key]:raise ValueError('Historical comparison invalid')
            paths += [cp,xp,Path(c['weights']),root/'training'/key/'tensor-verification.json',root/'training'/key/'thread-verification.json']
            ep=root/'evaluation-v1/units'/key/'completion.json';ec=checked(ep);paths += [ep,Path(ec['result'])]
    deps=dict(old['inputs'])
    for path in paths:
        if path.suffix=='.json':checked(path)
        deps[str(path.resolve())]=file_sha256(path)
    p.update(status='frozen_grayscale_addition_preflight_pending',arm='routed_grayscale_addition',direct_control=str(SOURCE),
        design='Only add equal-RGB-channel-mean grayscale after existing routed contrast on its nonidentity positions. Same all draws, batches, labels, factors and 480 optimization steps.',
        rationale='Current material fitting nearly saturates while development material recall remains low. Test additional suppression of chromatic cues under bounded material-only routing, not more optimization or contrast amplitude.',
        limits='Whole selected frames, including background and padding. Not physical material simulation, segmentation, source independence or guaranteed structural learning. Achromatic contrasts can be reduced; original training positions and all negative positions remain unchanged.',
        contrast_definition='Existing round(clamp(channel spatial mean + factor*(pixel-mean))) followed on nonidentity factors by rounded arithmetic RGB mean repeated into 3 channels.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(old,p);return write_record(dest,p)

def summarize():
    # Reuse fixed formal/diagnostic aggregation without changing historical files.
    mild.OUT=OUT;mild.KEYS=KEYS
    return mild.summarize()

def configure():
    cpu.OUT=OUT;cpu.KEYS=KEYS;augmentation.OUT=OUT;augmentation.KEYS=KEYS;augmentation.transform=transform
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=augmentation.loader
    runtime.MODULE='scripts.vision.routed_gray_transfer_control';runtime.summarize=summarize
    runtime.TESTS+=('tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control','tests.test_contrast_cpu4_review','tests.test_routed_gray_transfer')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS)
    a=ap.parse_args();configure()
    if a.worker:cpu.worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
