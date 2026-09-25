"""Single added photometric factor; no new samples, training or threshold search."""
import copy
from hashlib import sha256
from collections import Counter
from pathlib import Path
from scripts.vision.closed_budget_design import OUT as SOURCE,freeze as source_design,prior
from scripts.vision.analyze_closed_budget_fit import OUT as FIT,main as analysis
from scripts.vision.train_closed_budget import complete

OUT=SOURCE/'gamma-transfer-control-v1'
KEYS=tuple(f'G900-{s}' for s in (7,17,27))
VERSION='gamma-transfer-control-v1'

def schedule(seed):
    result=[]
    for window in range(18):
        values=[1.0]*25+[.8]*(12+window%2)+[1.2]*(13-window%2)
        positions=sorted(range(50),key=lambda i:sha256(f'{VERSION}|{seed}|{window}|{i}'.encode()).hexdigest())
        out=[None]*50
        for i,v in zip(positions,values,strict=True):out[i]=v
        result.extend(out)
    return result

def check(p,old):
    for key in KEYS:
        seed=int(key.split('-')[-1]);ref=f'B900-{seed}'
        for field in ('schedules','brightness_factors','training_config','listings'):
            if p[field][key]!=old[field][ref]:raise ValueError('Non-gamma factor changed: '+field)
        if p['gamma_factors'][key]!=schedule(seed) or Counter(p['gamma_factors'][key])!=Counter({1.:450,.8:225,1.2:225}):raise ValueError('Gamma quota/order drift')
    for field in ('pool_rows','held_members','names','initialization','evaluation','acceptance_policy','environment'):
        if p[field]!=old[field]:raise ValueError('Source policy changed')

def freeze():
    old=source_design();analysis();dest=OUT/'design.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);check(r,old);return r
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','held_members','names','initialization','evaluation','acceptance_policy','environment')}
    p.update(schedules={},brightness_factors={},training_config={},listings={},gamma_factors={})
    paths=[SOURCE/'design.json',SOURCE/'evaluation-completion.json',FIT/'paired-analysis.json',Path(__file__).resolve()]
    for seed in (7,17,27):
        ref=f'B900-{seed}';complete(ref);key=f'G900-{seed}';paths.append(SOURCE/'training'/ref/'completion.json')
        for field in ('schedules','brightness_factors','training_config','listings'):p[field][key]=copy.deepcopy(old[field][ref])
        p['gamma_factors'][key]=schedule(seed)
    p.update(status='frozen_gamma_design_real_preflight_and_visual_check_pending',
        gamma_definition='After frozen brightness on resized BGR image, convert to HSV and replace V using uint8 round(255*(V/255)**gamma); gamma=1 returns an exact copy, no HSV round-trip. Then unchanged dataset formatting/letterbox.',
        gamma_budget='Per 900 batches: identity 450, gamma .8 225, gamma 1.2 225. Every 50-step window has 25 identity, remaining split 12/13 alternating. All six images in a batch share gamma. Deterministic versioned SHA-256 order.',
        scope='Compare three new G900 seeds against verified B900 seeds. All 5400 exposures, full labels, negative positions, base brightness, initialization, optimizer, LR and epochs unchanged. Only added gamma transformation differs, including negatives.',
        rationale='Training reactor fit improves 107/120 to118/120 with no new losses, but viewed-development reactor recall declines. Probe photometric condition transfer, not claim sole cause or mimic physical lighting.',
        limits=['Gamma is image-domain augmentation, not source-independent data or a new asset/layout.','Quantization can merge tone values; inspect transformed evidence before training.','Previous lower constant LR and broad multiscale results do not establish this gamma factor as effective.','No extra budget/threshold search, collection, label editing, sealed testing or favorable-seed selection.'],
        cpu_policy='Reuse verified dual4 training/inference execution; confirm unchanged gamma=1 tensor path and actual thread count before training.',
        training_started=False,training_admitted=False,promotable=False,inputs={str(d):prior.file_sha256(d) for d in paths})
    OUT.mkdir(exist_ok=True);check(p,old);return prior.frozen(dest,p)

if __name__=='__main__':print(freeze()['status'])
