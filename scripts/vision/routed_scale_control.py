"""Material-only no-crop scale experiment; explicit training only."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_small_backbone_control as reporting
from scripts.vision import routed_scale_transform as scale
from scripts.vision.preview_routed_scale import OUT,EXPECTED
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT;SOURCE_KEYS=tuple(routed.KEYS)
KEYS=tuple(f'routed-scale-480-{s}' for s in (7,17,27))
runtime=routed.runtime;checked=routed.checked

def preview_gate():
    ep=OUT/'preview-v1/evidence.json';rp=OUT/'preview-v1/review.json';e=checked(ep);r=checked(rp)
    if {x['group'] for x in e['rows']}!=EXPECTED or len(r['decisions'])!=len(e['rows']):raise ValueError('Incomplete preview review')
    by={x['member_id']:x for x in r['decisions']}
    if len(by)!=len(e['rows']):raise ValueError('Duplicate decision')
    for x in e['rows']:
        d=by[x['member_id']]
        if d['decision']!='bounded_transform_test_allowed' or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Unapproved preview')
        for f in ('image_sha256','label_sha256','page_sha256'):
            if d[f]!=x[f]:raise ValueError('Stale decision')
    return [ep,rp]

def validate(s,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if s[f]!=p[f]:raise ValueError('Non-scale change '+f)
    expected=copy.deepcopy(s['configuration']);expected['scale_augmentation']=scale.VERSION
    if p['configuration']!=expected:raise ValueError('Unexpected config')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:s[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Exposure change')
    for seed,k in zip((7,17,27),KEYS):
        if p['scale_factors'][k]!=scale.factors(p['pool_rows'],p['schedules'][k],seed):raise ValueError('Scale positions changed')

def freeze():
    s=checked(SOURCE/'protocol.json');reviews=preview_gate();path=OUT/'protocol.json'
    if path.exists():p=checked(path);validate(s,p);return p
    p=copy.deepcopy(s)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:copy.deepcopy(s[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['scale_augmentation']=scale.VERSION
    p['scale_factors']={k:scale.factors(p['pool_rows'],p['schedules'][k],seed) for seed,k in zip((7,17,27),KEYS)}
    coverage=SOURCE.parent/'routed-transfer-coverage-v1'
    paths=[SOURCE/'protocol.json',SOURCE/'audit-v1/completion.json',coverage/'coverage.json',coverage/'development-scale.json',coverage/'next-control-design-zh.md',Path(__file__),Path(scale.__file__),Path(reporting.__file__),Path('tests/test_routed_scale_transform.py'),*reviews]
    for k in SOURCE_KEYS:
        cp=SOURCE/'training'/k/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=s['schedules'][k]:raise ValueError('Reference invalid')
        paths += [cp,Path(c['weights']),Path(c['exposure_path']),SOURCE/'actual-preflight'/f'{k}.json']
    deps=dict(s['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='scale_frozen_preflight_pending',arm='material_routed_scale',direct_control=str(SOURCE),design='Same original routed configuration plus frozen half-material .75 center shrink after contrast; full transformed labels, no crop.',limits='Scale, interpolation and padding change together; not new independent scenes or physical camera-distance intervention.',training_admitted=False,promotable=False,inputs=deps)
    validate(s,p);return write_record(path,p)

def loader(p,key,owner):
    dataset,raw=runtime.baseline.loader(p,key,owner)
    reference=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    ep=OUT/'actual-preflight'/f'{key}.json';expected=checked(ep)['tensor_records'] if ep.exists() else None
    records=runtime.OBSERVED.setdefault(key,[])
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*60+j*6;ref=reference[pos//6]
                rawhash={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')}
                if rawhash!=ref['raw_tensors'] or list(b['im_file'])!=ref['images']:raise ValueError('Raw input changed')
                before=dict(b);before['img'],_=routed.transform_runtime.transform(b['img'],p['contrast'][key][pos:pos+6])
                if {k:runtime.tensor_hash(before[k]) for k in rawhash}!=ref['tensors']:raise ValueError('Routed input changed')
                values=p['scale_factors'][key][pos:pos+6];after=scale.transform(before,values)
                record=dict(position=pos,images=list(b['im_file']),routed_tensors=ref['tensors'],scale_factors=values,tensors={k:runtime.tensor_hash(after[k]) for k in rawhash},full_label_count=len(after['bboxes']))
                if len(after['bboxes'])!=len(before['bboxes']) or any(not after[k].equal(before[k]) for k in ('cls','batch_idx')):raise ValueError('Instance loss')
                if expected is not None and record!=expected[pos//6]:raise ValueError('Scale tensor mismatch')
                records.append(record);yield after
    return dataset,Wrapped()

def verified_unit(k):
    c=checked(OUT/'training'/k/'completion.json');t=checked(OUT/'training'/k/'tensor-verification.json');checked(OUT/'training'/k/'thread-verification.json')
    if c['optimizer_steps']!=480 or t['records']!=checked(OUT/'actual-preflight'/f'{k}.json')['tensor_records']:raise ValueError('Incomplete unit')
    return c

def summarize():
    with patch.object(reporting,'OUT',OUT),patch.object(reporting,'KEYS',KEYS),patch.object(reporting,'SOURCE',SOURCE),patch.object(reporting,'SOURCE_KEYS',SOURCE_KEYS),patch.object(reporting,'verified_unit',verified_unit):return reporting.summarize()

def configure():
    routed.previous.OUT=OUT;routed.previous.KEYS=KEYS;routed.transform_runtime.OUT=OUT;routed.transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize;runtime.MODULE='scripts.vision.routed_scale_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_scale_transform','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:freeze();routed.previous.worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
