"""One intervention: frozen whole-image shrink/padding on unchanged exposures."""
import hashlib
from collections import Counter
from pathlib import Path
from scripts.vision.train_order_fit_reviewed import DEST as REFERENCE,KEYS,contract,bind,prior
from scripts.vision.probe_reviewed_scale import OUT as PROBE

OUT=REFERENCE/'scale-padding-control-v1'
VERSION='reviewed-scale-padding-control-v1'
def factors(seed,length):
    if length%300:raise ValueError('Incomplete 50-step window')
    result=[]
    for start in range(0,length,300):
        order=sorted(range(300),key=lambda i:hashlib.sha256(f'{VERSION}:{seed}:{start+i}'.encode()).digest())
        window=[None]*300
        for n,i in enumerate(order):window[i]=1. if n<150 else .75 if n<225 else .5
        result+=window
    return result

def freeze():
    dest=OUT/'design.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    p,_,deps,b=contract();probe=prior.read(PROBE/'summary.json');prior.verify(probe)
    if probe['aggregates']['1.0']['target_hits']!=60:raise ValueError('Identity probe failure')
    if not all(probe['aggregates'][str(s)]['target_hits']<60 for s in (.75,.5)):raise ValueError('Scale direction unsupported by frozen probe')
    for k in KEYS:bind().complete(k);deps.append(REFERENCE/'training'/k/'completion.json')
    result={k:v for k,v in p.items() if k not in ('identity','inputs','status','comparison','limitations')}
    scales={k:factors(int(k.split('-')[-1]),len(p['schedules'][k])) for k in KEYS}
    for k in KEYS:
        if Counter(scales[k])!=Counter({1.:3300,.75:1650,.5:1650}):raise ValueError('Scale quota mismatch')
    result.update(status='frozen_scale_control_actual_preflight_pending',version=VERSION,scale_factors=scales,
        direct_reference=str(REFERENCE),training_started=False,
        scale_transform=dict(interpolation='PIL_BILINEAR',fill_rgb=[114,114,114],placement='center_integer_offset',stage='after_frozen_brightness_before_standard_letterbox',full_label_transform=True,no_crop=True),
        comparison='Six new endpoints versus corresponding reviewed ISR/ISM endpoints: same source members, order, full-label instance exposure, brightness gains, initialization and budget; only whole-image scale/interpolation/padding differs.',
        limitations=['Not a pure camera-distance intervention; padding and spatial context occupancy also change.',
            'Negative member positions unchanged, but their image tensors undergo the same frozen scale augmentation.',
            'No new independent scenes; no restored risk members; no checkpoint or seed selection.'],baseline=b)
    OUT.mkdir(exist_ok=True);deps += [PROBE/'summary.json',PROBE/'protocol.json',Path(__file__).resolve()]
    return prior.frozen(dest,dict(result,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(freeze()['status'])
