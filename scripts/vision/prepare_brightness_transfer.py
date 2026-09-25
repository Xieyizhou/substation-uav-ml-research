"""Freeze a brightness-only comparison using the latest risk-capped sequence."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
import cv2,numpy as np
from scripts.vision.lineage_capped_control import OUT as SOURCE,ROOT,ready as prior_ready,read,verify,frozen,file_sha256
from scripts.vision.train_lineage_capped_v2 import completed
from scripts.vision.check_lineage_training_fit import OUT as FIT
from scripts.vision.brightness_transfer_runtime import factor,brighten,preflight
from scripts.vision.exposure_protocol import exposures

OUT=SOURCE/'brightness-transfer-control-v1'
KEYS=tuple(f'brightness-450-{s}' for s in (7,17,27))

def freeze():
    if (OUT/'protocol.json').exists():verify(read(OUT/'protocol.json'));return
    old=prior_ready();verify(read(FIT/'completion.json'));OUT.mkdir(exist_ok=True)
    paths=[SOURCE/'protocol.json',SOURCE/'completion.json',FIT/'completion.json',Path(__file__),ROOT/'scripts/vision/brightness_transfer_runtime.py',ROOT/'scripts/vision/order_retention_runtime.py']
    schedules={};factors={};listings={};ledger={};baselines={}
    for seed in (7,17,27):
        src=f'lineage-capped-450-{seed}';completed(src,old);cp=SOURCE/'training'/src/'completion.json';c=read(cp);ep=SOURCE/'evaluation'/f'{src}.json';verify(read(ep));paths.extend([cp,Path(c['exposure_path']),Path(c['weights']),ep])
        baselines[str(seed)]=dict(cell=src,training_receipt=str(cp),evaluation=str(ep))
        for arm in ('noaug','brightness'):
            key=f'{arm}-450-{seed}';schedules[key]=old['schedules'][src];listings[key]=old['listings'][src]
            factors[key]=[1.0 if arm=='noaug' else factor(seed,m,i) for i,m in enumerate(schedules[key])]
            ledger[key]=old['ledger'][src];paths.append(Path(listings[key]))
    import ultralytics,torch
    import ultralytics.data.augment as aug
    paths.append(Path(aug.__file__))
    for r in old['pool_rows']:
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
            paths.append(Path(r[kind+'_path']))
    frozen(OUT/'protocol.json',dict(status='frozen_preview_review_pending',pool_rows=old['pool_rows'],names=old['names'],schedules=schedules,listings=listings,ledger=ledger,
        initialization=old['initialization'],evaluation=old['evaluation'],acceptance_policy=old['acceptance_policy'],retention=old['retention'],brightness_factors=factors,baselines=baselines,
        environment=dict(ultralytics=ultralytics.__version__,torch=torch.__version__),
        augmentation='Per-draw hash: half expected exact identity, otherwise HSV V gain uniform [0.8,1.2], before standard geometric/letterbox path; H/S unchanged in HSV; no global RNG consumed. Integer RGB conversion may introduce quantization.',
        scope='Same latest risk-capped members, exact batch sequence and complete labels; no new training imagery, no threshold/optimizer/model changes, no sealed scene.',
        baseline_reuse='Only if original loader tensors match identity adapter and complete historical runtime/configuration remain verifiable.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print('FROZEN_SIX_CONFIGS_THREE_NEW_TRAININGS',flush=True)

def preview():
    p=read(OUT/'protocol.json');verify(p);idx={r['member_id']:r for r in p['pool_rows']};targets=read(FIT/'protocol.json')['targets'];chosen=[]
    for eid in ('C04-L3','C25-L7','C03-L2','C05-L0','C08-L4','C39-L7'):
        chosen.append(next(idx[t['review']['member_id']] for t in targets if t['review']['event_id']==eid))
    neg=sorted((r for r in p['pool_rows'] if r['subset']=='hard_negative' and r['member_id'] in p['schedules']['noaug-450-7']),key=lambda r:r['member_id'])
    chosen.extend([neg[i] for i in (0,20,40,60,80,100)])
    dest=OUT/'preview';dest.mkdir(exist_ok=True);rows=[];paths=[OUT/'protocol.json',Path(__file__)]
    for i,r in enumerate(chosen,1):
        bgr=cv2.imread(r['image_path']);page=Image.new('RGB',(1800,390),'white')
        for j,g in enumerate((.8,1.,1.2)):
            rgb=Image.fromarray(cv2.cvtColor(brighten(bgr,g),cv2.COLOR_BGR2RGB));rgb.thumbnail((600,350));page.paste(rgb,(j*600,35));ImageDraw.Draw(page).text((j*600+5,5),f'P{i:02} gain={g}',fill='black')
        path=dest/f'P{i:02}.png';page.save(path);paths.extend([path,Path(r['image_path'])]);rows.append(dict(event_id=f'P{i:02}',member=r,evidence_path=str(path),evidence_sha256=file_sha256(path)))
    frozen(dest/'evidence.json',dict(status='awaiting_explicit_AI_review',rows=rows,sample_scope='Six positive and six negative previews; extremal gains are diagnostic, not new training members or complete frame admission.',inputs={str(x):file_sha256(x) for x in paths}))

def ready():
    p=read(OUT/'protocol.json');verify(p);review=read(OUT/'preview/review.json');verify(review)
    e=read(OUT/'preview/evidence.json');verify(e)
    ds=review['decisions']
    if len(ds)!=12 or {d['event_id'] for d in ds}!={r['event_id'] for r in e['rows']}:raise ValueError('Missing/duplicate preview review')
    for d in ds:
        row=next(r for r in e['rows'] if r['event_id']==d['event_id'])
        if d['evidence_sha256']!=row['evidence_sha256'] or d['decision']!='bounded_preview_usable' or d['review_nature']!='AI辅助审核' or not d['reason']:raise ValueError('Preview review invalid')
    old=prior_ready()
    for seed in (7,17,27):
        completed(f'lineage-capped-450-{seed}',old)
        path=OUT/'preflight'/f'{seed}.json';u=read(path);verify(u)
        if not u['baseline_tensor_bytes_identical'] or not u['complete_labels_identical'] or any(u[x] for x in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Invalid preflight')
        from scripts.vision.brightness_transfer_runtime import check_log
        for arm in ('noaug','brightness'):
            k=f'{arm}-450-{seed}'
            if u['draws']!=p['schedules'][k] or p['schedules'][k]!=old['schedules'][f'lineage-capped-450-{seed}']:raise ValueError('Sequence mismatch')
            check_log(p,k,u['logs'][k])
    return p

def precheck():
    p=read(OUT/'protocol.json');verify(p);(OUT/'preflight').mkdir(exist_ok=True)
    for seed in (7,17,27):
        path=OUT/'preflight'/f'{seed}.json'
        if path.exists():verify(read(path));continue
        r=preflight(p,seed)
        frozen(path,dict(**r,inputs={str(x):file_sha256(x) for x in [OUT/'protocol.json',Path(__file__),ROOT/'scripts/vision/brightness_transfer_runtime.py']}));print('ACTUAL_LOADER_PASS',seed,flush=True)
    ready()
    paths=[OUT/'protocol.json',OUT/'preview/review.json']+[OUT/'preflight'/f'{s}.json' for s in (7,17,27)]
    frozen(OUT/'ready.json',dict(status='ready_for_training_not_started',new_training_units=3,reused_baselines=3,inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--preview',action='store_true');ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    elif a.preview:preview()
    elif a.preflight:precheck()
    else:print('READ_ONLY_NO_TRAINING')
