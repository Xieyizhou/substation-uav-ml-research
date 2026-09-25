"""Bounded 32-frame lower-light counterpart acquisition on reviewed training poses."""
import argparse,asyncio,copy,fcntl,re,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.closed_gamma_design import OUT as GAMMA,SOURCE,prior
from scripts.vision.closed_source_training_quality import OUT as COHORT,PAIRS,export
from scripts.vision.finalize_closed_gamma_review import main as feedback
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.collect import collect

OUT=SOURCE/'cool-light-coverage-v1'
LIGHT={'.//world/scene/ambient':'0.38 0.42 0.48 1','.//world/light[@name="sun"]/diffuse':'0.58 0.66 0.78 1'}

def lighting_only(original,changed):
    check=copy.deepcopy(changed)
    for path,value in LIGHT.items():
        a=original.findall(path);b=check.findall(path)
        if len(a)!=1 or len(b)!=1 or b[0].text!=value:raise ValueError('Missing/duplicate/unfrozen lighting')
        b[0].text=a[0].text
    if ET.tostring(original)!=ET.tostring(check):raise ValueError('Non-lighting world change')

def freeze():
    dest=OUT/'capture-protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    manifest=export();prior.verify(manifest)
    rows=sorted(manifest['members'],key=lambda m:(m['pair_id'],m['variant']))
    if len(rows)!=32 or {r['pair_id'] for r in rows}!=set(PAIRS):raise ValueError('Frozen cohort differs')
    OUT.mkdir(exist_ok=True);deps=[COHORT/'export/manifest.json',COHORT/'quality-review.json',Path(__file__).resolve()];units=[]
    for n,m in enumerate(rows,1):
        uid=f'L{n:02}';e=prior.read(m['evidence_path']);prior.verify(e);sp=Path(e['unit']['plan_path']);source=read_record(sp)
        v=next(copy.deepcopy(v) for v in source['calibration_views'] if v['view_id']==e['unit']['view_id'])
        folder=OUT/'plans'/uid;folder.mkdir(parents=True)
        for name,digest in source['files'].items():
            src=sp.parent/name
            if prior.file_sha256(src)!=digest:raise ValueError('Source file drift')
            shutil.copy2(src,folder/name);deps.append(src)
        original=ET.parse(sp.parent/'world.sdf').getroot();changed=copy.deepcopy(original)
        for path,value in LIGHT.items():
            nodes=changed.findall(path)
            if len(nodes)!=1:raise ValueError('Ambiguous illumination field')
            nodes[0].text=value
        lighting_only(original,changed);ET.ElementTree(changed).write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
        lighting_only(original,ET.parse(folder/'world.sdf').getroot())
        p={k:copy.deepcopy(value) for k,value in source.items() if k!='identity'}
        p.update(calibration_views=[v],pilot_views=[],diagnostic_purpose='cool_light_training_counterpart',data_role='bounded_development_training_candidate',training_admitted=False,promotable=False)
        p['files']={name:prior.file_sha256(folder/name) for name in p['files']};p=write_record(folder/'plan.json',p);validate_preflight(p,folder,[v])
        deps.extend(folder.iterdir());deps.append(Path(m['evidence_path']))
        units.append(dict(unit_id=uid,source_member=m,view=v,plan_path=str(folder/'plan.json'),source_plan=str(sp)))
    pilot=[next(u['unit_id'] for u in units if u['source_member']['layout']=='layout-A' and u['source_member']['planned_category']==c and u['source_member']['variant']=='original') for c in ('transformer','switchgear','capacitor_bank','reactor')]
    return prior.frozen(dest,dict(status='32_lower_light_candidates_frozen_quality_pending',units=units,pilot_units=pilot,lighting=LIGHT,
        experiment=dict(cells=6,seeds=[7,17,27],initialization='independent_v2.11',base_steps=900,tail_steps=100,total_exposures=6000,
            reference_tail='600 source images under original lighting',treatment_tail='same source/label/batch order; 50 of 100 added batches use lower-light counterparts, 50 original',
            original_base_5400_unchanged=True,gamma_enabled=False,tail_brightness=1.,lr=.0005,cpu_threads=4,cpu_parallel=2),
        collection_policy='One frozen frame per unit, canonical maximum three technical view attempts, no outer capture retry. At most three exact-replay attempts. Semantic failures stop; no favorable-score replacement.',
        limitations=['Eight already-reviewed training pose groups, two layouts, existing shared assets. No new independent scene from relighting.','Known development lighting recipe applied only to existing training poses; viewed-condition adaptation, not blind generalization.','Appended stage budget effect requires paired reference; comparison tests this data/stage strategy, not unique cause.','No protected labels, sealed scenes, historical edits or promotion.'],
        inputs={str(p):prior.file_sha256(p) for p in deps}))

async def run(pilot=False):
    p=freeze()
    if not pilot:
        gate=prior.read(OUT/'pilot-quality.json');prior.verify(gate)
        if gate['status']!='pilot_explicitly_reviewed_expand_frozen_32':raise ValueError('Pilot not reviewed')
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for u in p['units']:
            if pilot and u['unit_id'] not in p['pilot_units']:continue
            folder=OUT/'captures'/u['unit_id'];cp=folder/'collection-receipt.json'
            if cp.exists():r=read_record(cp)
            elif folder.exists():raise ValueError('Incomplete capture preserved '+str(folder))
            else:
                print('CAPTURING',u['unit_id'],u['source_member']['pair_id'],u['source_member']['variant'],flush=True)
                r=await collect(u['plan_path'],folder,mode='calibration')
            if r['status']!='complete_pending_review' or len(r['views'])!=1 or r['views'][0]['status']!='captured':raise ValueError('Capture blocked '+u['unit_id']+' '+str(r.get('error')))
            print('CAPTURED',u['unit_id'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run(a.pilot))
    else:print(freeze()['status'],'NO_CAPTURE_NO_TRAINING')
