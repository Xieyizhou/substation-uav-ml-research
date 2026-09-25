"""Independent pre-training checks against frozen Q and label files."""
from collections import Counter
from pathlib import Path
import math
from scripts.vision.prepare_instance_exposure_balance import OUT,PRIOR,QUOTAS,read,save,verify_tree,file_sha256,ROOT
from scripts.vision.exposure_protocol import NAMES,exposures

def main():
    verify_tree(OUT/'protocol.json');p=read(OUT/'protocol.json');q=read(PRIOR/'protocol.json');rows=p['pool_rows'];lookup={r['member_id']:r for r in rows}
    for r in rows:
        if file_sha256(r['label_path'])!=r['label_sha256']:raise ValueError('Stale label')
        actual=dict(Counter(NAMES[int(line.split()[0])] for line in Path(r['label_path']).read_text().splitlines() if line.strip()))
        if actual!=r['class_instances']:raise ValueError('Class count differs from actual labels')
    for seed in (7,17,27):
        short=p['schedules'][f'I-100-{seed}'];long=p['schedules'][f'I-300-{seed}'];old=q['schedules'][f'Q-100-{seed}']
        if long!=short*3:raise ValueError('Prefix or repeat mismatch')
        counts=Counter(short)
        if len(short)!=600 or Counter(lookup[x]['subset'] for x in short)!={**QUOTAS,'hard_negative':108}:raise ValueError('Quota mismatch')
        for i,mid in enumerate(old):
            if lookup[mid]['subset']=='hard_negative' and short[i]!=mid:raise ValueError('Negative position mismatch')
        for r in rows:
            subset=r['subset']
            if subset in QUOTAS:
                cap=2*math.ceil(QUOTAS[subset]/sum(x['subset']==subset for x in rows))
                if not 1<=counts[r['member_id']]<=cap:raise ValueError('Member bounds violated')
        for steps in (100,300):
            key=f'I-{steps}-{seed}'
            if exposures(rows,p['schedules'][key])!=p['exposures'][key]:raise ValueError('Frozen exposure differs')
        classes=p['exposures'][f'I-100-{seed}']['class_instance_exposure']
        if max(classes.values())-min(classes.values())!=p['optimization'][str(seed)]['optimal_range']:raise ValueError('Range mismatch')
    from scripts.vision.verify_experiment_baseline import verify
    baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    save(OUT/'preflight.json',dict(status='passed',labels_verified=len(rows),seeds=[7,17,27],baseline=baseline,checks=['actual label class counts','member lower and upper bounds','subset quota','negative exact positions','100/300 prefix and repeats','actual schedule exposure','solver range'],inputs={str(path):file_sha256(path) for path in (OUT/'protocol.json',Path(__file__))}))
    print('PREFLIGHT_PASSED',OUT,flush=True)

if __name__=='__main__':main()
