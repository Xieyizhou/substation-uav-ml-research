"""Independent bounded gamma control; default is real preflight, never training."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_small_backbone_control as reporting
from scripts.vision import routed_gamma_transform as gamma
from scripts.vision.preview_routed_gamma import OUT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT; SOURCE_KEYS=tuple(routed.KEYS)
KEYS=tuple(f'routed-gamma-480-{s}' for s in (7,17,27))
runtime=routed.runtime; checked=routed.checked

def preview_gate():
    ep=OUT/'preview-v1/evidence.json'; rp=OUT/'preview-v1/review.json'
    e=checked(ep); r=checked(rp)
    if {x['variant'] for x in e['rows']}!=routed.MATERIAL:raise ValueError('Missing variant')
    decisions={x['member_id']:x for x in r['decisions']}
    if len(decisions)!=len(e['rows']) or len(r['decisions'])!=len(decisions):raise ValueError('Review coverage')
    for x in e['rows']:
        d=decisions[x['member_id']]
        if d['decision']!='bounded_transform_test_allowed' or d['review_nature']!='AI辅助审核' or not d['reason']:raise ValueError('Unapproved')
        for f in ('image_sha256','label_sha256','page_sha256'):
            if x[f]!=d[f]:raise ValueError('Stale review')
    return [ep,rp]

def validate(s,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if s[f]!=p[f]:raise ValueError('Unallowed change '+f)
    cfg=copy.deepcopy(s['configuration']);cfg['gamma_augmentation']=gamma.VERSION
    if p['configuration']!=cfg:raise ValueError('Config drift')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:s[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Exposure drift')
    for k in KEYS:
        if p['gamma_factors'][k]!=gamma.factors(p['contrast'][k]):raise ValueError('Gamma drift')

def freeze():
    s=checked(SOURCE/'protocol.json'); reviews=preview_gate(); dest=OUT/'protocol.json'
    if dest.exists():
        p=checked(dest);validate(s,p);return p
    p=copy.deepcopy(s)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):
        p[f]={k:copy.deepcopy(s[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['gamma_augmentation']=gamma.VERSION
    p['gamma_factors']={k:gamma.factors(p['contrast'][k]) for k in KEYS}
    fit=SOURCE.parent/'routed-retention-fit-diagnosis-v1'
    paths=[SOURCE/'protocol.json',SOURCE/'audit-v1/completion.json',fit/'protocol.json',fit/'summary.json',fit/'judgment-zh.md',Path(__file__),Path(gamma.__file__),Path(reporting.__file__),Path('tests/test_routed_gamma_transform.py'),*reviews]
    for k in SOURCE_KEYS:
        cp=SOURCE/'training'/k/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=s['schedules'][k]:raise ValueError('Reference invalid')
        paths.extend([cp,Path(c['weights']),Path(c['exposure_path']),SOURCE/'actual-preflight'/f'{k}.json'])
    deps=dict(s['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='gamma_frozen_preflight_pending',arm='material_routed_gamma',direct_control=str(SOURCE),
        design='Same members, labels, batches, contrast and 480 steps. Apply gamma .8/1/1.25 after contrast .75/1/1.25. Normal BN, constant LR .00025, independent v2.11 initialization, seeds 7/17/27. CPU4, two workers. Existing evaluation gates unchanged.',
        limits='Correlated contrast/gamma strength intervention, not physical relighting or new independent scenes. Six representative previews do not certify all-pool quality. No threshold, seed, checkpoint or sealed-test selection.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(s,p);return write_record(dest,p)

def loader(p,key,owner):
    dataset,raw=runtime.baseline.loader(p,key,owner)
    ref=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    ep=OUT/'actual-preflight'/f'{key}.json';expected=checked(ep)['tensor_records'] if ep.exists() else None
    records=runtime.OBSERVED.setdefault(key,[])
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*60+j*6;r=ref[pos//6]
                hashes={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')}
                if hashes!=r['raw_tensors'] or list(b['im_file'])!=r['images']:raise ValueError('Raw drift')
                before=dict(b);before['img'],_=routed.transform_runtime.transform(b['img'],p['contrast'][key][pos:pos+6])
                if {k:runtime.tensor_hash(before[k]) for k in hashes}!=r['tensors']:raise ValueError('Contrast drift')
                values=p['gamma_factors'][key][pos:pos+6];after=dict(before);after['img']=gamma.transform(before['img'],values)
                if any(not after[k].equal(before[k]) for k in ('cls','bboxes','batch_idx')):raise ValueError('Label drift')
                for i,g in enumerate(values):
                    if g==1 and not after['img'][i].equal(before['img'][i]):raise ValueError('Identity changed')
                record=dict(position=pos,images=list(b['im_file']),routed_tensors=r['tensors'],gamma_factors=values,tensors={k:runtime.tensor_hash(after[k]) for k in hashes},full_label_count=len(after['bboxes']))
                if expected is not None and record!=expected[pos//6]:raise ValueError('Tensor mismatch')
                records.append(record);yield after
    return dataset,Wrapped()

def verified_unit(k):
    c=checked(OUT/'training'/k/'completion.json');t=checked(OUT/'training'/k/'tensor-verification.json');checked(OUT/'training'/k/'thread-verification.json')
    if c['optimizer_steps']!=480 or t['records']!=checked(OUT/'actual-preflight'/f'{k}.json')['tensor_records']:raise ValueError('Incomplete unit')
    return c

def worker(k):
    from ultralytics.models.yolo.detect import DetectionTrainer
    freeze();runtime.launch_gate()
    if (OUT/'training'/k/'completion.json').exists():return verified_unit(k)
    original=DetectionTrainer.optimizer_step;rates=[]
    def step(t,*a,**kw):
        if any(g['lr']!=.00025 for g in t.optimizer.param_groups):raise ValueError('LR drift')
        result=original(t,*a,**kw);rates.append(.00025)
        if len(rates)==10:
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='real_gamma_training_started',optimizer_steps=10,learning_rates=rates.copy(),observed_at=datetime.now(timezone.utc).isoformat(),training_admitted=False,promotable=False,inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        return result
    with patch.object(DetectionTrainer,'optimizer_step',step):return routed.previous.worker(k)

def summarize():
    with patch.object(reporting,'OUT',OUT),patch.object(reporting,'KEYS',KEYS),patch.object(reporting,'SOURCE',SOURCE),patch.object(reporting,'SOURCE_KEYS',SOURCE_KEYS),patch.object(reporting,'verified_unit',verified_unit):return reporting.summarize()

def configure():
    routed.previous.OUT=OUT;routed.previous.KEYS=KEYS;routed.transform_runtime.OUT=OUT;routed.transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize;runtime.MODULE='scripts.vision.routed_gamma_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_gamma_transform','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.launch_gate(create=True);print('PREFLIGHT_ONLY_NO_TRAINING')
