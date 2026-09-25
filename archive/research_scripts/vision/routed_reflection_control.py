"""One fixed horizontal-reflection control; default is loader preflight only."""
import argparse
import copy
import hashlib
from pathlib import Path
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_gray_transfer_control as gray
from scripts.vision import contrast_transfer_control as augmentation
from scripts.vision import contrast_cpu4_control as cpu
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT
SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-reflection-control-v1'
KEYS=tuple(f'routed-reflection-480-{s}' for s in (7,17,27))
VERSION='routed-contrast-plus-horizontal-reflection-v1'
checked=routed.checked

def positions(seed):
    order=sorted(range(2880),key=lambda i:hashlib.sha256(f'{VERSION}|{seed}|{i}'.encode()).hexdigest())
    selected=set(order[:1440])
    return [i in selected for i in range(2880)]

def reflect(batch,flags):
    import torch
    image,boxes,index=batch['img'],batch['bboxes'],batch['batch_idx']
    if image.dtype!=torch.uint8 or image.ndim!=4 or image.shape[1]!=3 or len(flags)!=len(image) or any(type(x)!=bool for x in flags):raise ValueError('Invalid reflection input')
    if boxes.ndim!=2 or boxes.shape[1]!=4 or index.ndim!=1 or len(index)!=len(boxes):raise ValueError('Invalid normalized xywh labels')
    if not torch.isfinite(boxes).all() or not ((boxes>=0)&(boxes<=1)).all():raise ValueError('Non-normalized boxes')
    if not torch.isfinite(index).all() or not (index==index.long()).all() or not ((index>=0)&(index<len(image))).all():raise ValueError('Invalid image index')
    mask=torch.tensor(flags,dtype=torch.bool,device=image.device)
    result=dict(batch);result['img']=torch.where(mask[:,None,None,None],image.flip(-1),image)
    result['bboxes']=boxes.clone();selected=mask[index.long()]
    result['bboxes'][selected,0]=1-boxes[selected,0]
    return result

def validate(old,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if old[f]!=p[f]:raise ValueError('Unexpected change '+f)
    config=copy.deepcopy(old['configuration']);config['augmentation']=VERSION
    if p['configuration']!=config:raise ValueError('Configuration drift')
    for seed,k,o in zip((7,17,27),KEYS,SOURCE_KEYS):
        for f in ('schedules','exposures','windows','listings','contrast'):
            if p[f][k]!=old[f][o]:raise ValueError('Exposure, order, labels or contrast drift')
        if p['reflection'][k]!=positions(seed):raise ValueError('Reflection schedule drift')

def review_gate():
    a=gray.OUT/'audit-v1'
    paths=[a/'evidence.json',a/'review.json',a/'material-evidence.json',gray.OUT/'evaluation-v1/error-review-v1/evidence.json',a/'completion.json']
    e,r,m,n,c=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions'])
    if c['status']!='review_complete_not_candidate_passed' or not c['integrity']['integrity_passed']:raise ValueError('Previous audit incomplete')
    ep=OUT/'preview-v1/evidence.json';rp=OUT/'preview-v1/review.json';e,r=map(checked,(ep,rp))
    expected={(x['member_id'],x['page_sha256']) for x in e['rows']};actual=[(x['member_id'],x['page_sha256']) for x in r['decisions']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Missing or duplicate preview')
    if any(x['decision']!='bounded_transform_test_allowed' or not x['reason'] or x['review_nature']!='AI辅助审核' for x in r['decisions']):raise ValueError('Unresolved preview')
    return paths+[ep,rp]

def freeze():
    old=checked(SOURCE/'protocol.json');path=OUT/'protocol.json';reviews=review_gate()
    if path.exists():
        p=checked(path);validate(old,p);return p
    p=copy.deepcopy(old)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['reflection']={k:positions(s) for k,s in zip(KEYS,(7,17,27))};p['configuration']['augmentation']=VERSION
    paths=[SOURCE/'protocol.json',Path(__file__),OUT/'research-plan-zh.md',Path('tests/test_routed_reflection.py'),Path(augmentation.__file__),Path(cpu.__file__),Path(runtime.__file__),*reviews]
    for key in SOURCE_KEYS:
        cp=SOURCE/'training'/key/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=old['schedules'][key]:raise ValueError('Direct control invalid')
        paths += [cp,Path(c['exposure_path']),Path(c['weights']),SOURCE/'training'/key/'tensor-verification.json',SOURCE/'training'/key/'thread-verification.json']
        ep=SOURCE/'evaluation-v1/units'/key/'completion.json';ec=checked(ep);paths += [ep,Path(ec['result'])]
    deps=dict(old['inputs'])
    for x in paths:
        if x.suffix=='.json':checked(x)
        deps[str(x.resolve())]=file_sha256(x)
    p.update(status='frozen_reflection_preflight_pending',arm='horizontal_reflection',direct_control=str(SOURCE),
        design='Same routed contrast, members, complete labels, 2880 draws/order/batches, LR .00025 and 480 steps. Only reflect whole frames horizontally on 1440 hash-frozen positions per seed, with every normalized box cx mapped to 1-cx.',
        rationale='Material training fitting is nearly saturated, while reduced contrast and additional grayscale did not improve transfer. Test left-right orientation/context sensitivity without further removing chromatic information.',
        limits='Mirrored frames are derived views, not independent poses or new 3D assets. This changes positive and negative image tensors at selected positions, not their exposure positions. It is not evidence of structure-only recognition or unique causality. No crop, resize, new occlusion, label deletion, new collection or sealed testing.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(old,p);return write_record(path,p)

def loader(p,key,owner):
    dataset,raw=runtime.baseline.loader(p,key,owner)
    ep=OUT/'actual-preflight'/f'{key}.json';expected=checked(ep)['tensor_records'] if ep.exists() else None
    reference=checked(augmentation.SOURCE/'actual-preflight'/f'{augmentation.previous.KEYS[KEYS.index(key)]}.json')['tensor_records']
    records=runtime.OBSERVED.setdefault(key,[])
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,name):return getattr(raw,name)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*60+j*6;before={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')}
                ref=reference[pos//6]
                if before!=ref['tensors'] or list(b['im_file'])!=ref['images']:raise ValueError('Raw member/full-label drift')
                factors=p['contrast'][key][pos:pos+6];flags=p['reflection'][key][pos:pos+6]
                b=dict(b);b['img'],clipped=augmentation.transform(b['img'],factors);b=reflect(b,flags)
                record=dict(position=pos,images=list(b['im_file']),raw_tensors=before,coefficients=factors,reflection=flags,clipped_channels=clipped,
                    tensors={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')})
                if any(record['tensors'][k]!=before[k] for k in ('cls','batch_idx')):raise ValueError('Instance/class changed')
                if expected is not None and record!=expected[pos//6]:raise ValueError('Transformed tensor or label drift')
                records.append(record);yield b
    return dataset,Wrapped()

def summarize():
    from scripts.vision import mild_routed_contrast_control as report
    report.OUT=OUT;report.KEYS=KEYS
    return report.summarize()

def configure():
    cpu.OUT=OUT;cpu.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader
    runtime.MODULE='scripts.vision.routed_reflection_control';runtime.summarize=summarize
    runtime.TESTS+=('tests.test_routed_reflection','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:cpu.worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
