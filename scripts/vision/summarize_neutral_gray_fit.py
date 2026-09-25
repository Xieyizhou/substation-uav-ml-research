"""Validate fit outputs and compare actual supervision scale with development."""
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.neutral_gray_control import OUT,KEYS,prior
from scripts.vision.diagnose_material_late_rehearsal_fit import validate


def bucket(truth, size):
    b=truth['bbox_xyxy'];short=min(b[2]-b[0],b[3]-b[1])*640/max(size)
    return '<32' if short<32 else '32-64' if short<64 else '>=64'


def main():
    root=OUT/'pool-fit-diagnosis-v1';p=prior.read(root/'protocol.json');prior.verify(p)
    members={m['member_id']:m for m in p['members']};units=[]
    deps=[root/'protocol.json',OUT/'evaluation/analysis-v1.json',Path(__file__).resolve()]
    sizes={k:Image.open(m['image_path']).size for k,m in members.items()}
    for key in KEYS:
        path=root/(key+'.json');r=prior.read(path);validate(r,key,p);deps.append(path)
        groups={}
        for name in ('all_exposed','gray035','normal_gray035','cool_gray035','negative'):
            rows=[]
            for x in r['rows']:
                m=members[x['member_id']]
                if not p['actual_exposures'][key].get(x['member_id'],0):continue
                gray=m.get('material_setting')=='neutral_0.35'
                cool=m.get('illumination')=='physical_cool_viewed_development_recipe'
                if name=='gray035' and not gray:continue
                if name=='normal_gray035' and (not gray or cool):continue
                if name=='cool_gray035' and (not gray or not cool):continue
                if name=='negative' and m['subset']!='hard_negative':continue
                rows.append(x)
            groups[name]=dict(images=len(rows),matched=sum(len(x['matches']) for x in rows),truth=sum(len(x['truth']) for x in rows),
                unmatched_predictions=sum(x['unmatched_prediction_count'] for x in rows),
                misses=[dict(member_id=x['member_id'],miss=m) for x in rows for m in x['misses']],
                size_bins=dict(Counter(bucket(t,sizes[x['member_id']]) for x in rows for t in x['truth'])))
        units.append(dict(cell=key,groups=groups))
    gray=[m for m in p['members'] if m.get('material_setting')=='neutral_0.35']
    coverage={c:dict(instances=sum(t['class_name']==c for m in gray for t in m['truth']),
                    size_bins=dict(Counter(bucket(t,sizes[m['member_id']]) for m in gray for t in m['truth'] if t['class_name']==c)))
              for c in ('transformer','switchgear','capacitor_bank','reactor')}
    dest=root/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='three_seed_training_fit_verified',units=units,gray_supervision_coverage=coverage,
        interpretation='Training-fit diagnosis only. Normal-light gray has complete fit in all seeds, while cool gray retains some low-confidence misses. The development nonplanned gap cannot be explained solely by failure to memorize these normal-light gray training views.',
        next_direction_pending='Inspect supervision scale and viewing-angle gaps before bounded new-pose collection; keep all original capability gates.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':
    r=main();print(r['gray_supervision_coverage'])
    for x in r['units']:print(x['cell'],{k:(v['matched'],v['truth']) for k,v in x['groups'].items()})
