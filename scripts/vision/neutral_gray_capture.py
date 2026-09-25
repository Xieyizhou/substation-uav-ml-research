"""Sixteen same-pose .35 neutral counterparts; no automatic training."""
import argparse
import asyncio
import copy
import fcntl
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.cool_light_capture import OUT as SOURCE, COHORT, prior
from src.vision.canonical.collect import collect
from src.vision.canonical.plan import read_record, write_record
from src.vision.canonical.gates import validate_preflight

OUT=SOURCE/'neutral-gray-calibration-v1'
FIELDS=('ambient','diffuse')


def material_nodes(root):
    nodes=[]
    for model in root.findall('.//model'):
        for visual in model.findall('link/visual'):
            if visual.attrib.get('name') not in ('body','front_panel','reactor'):continue
            if visual.find('plugin/label') is None:continue
            for field in FIELDS:
                xs=visual.findall('material/'+field)
                if len(xs)!=1:raise ValueError('Missing or duplicate material field')
                nodes.append(xs[0])
    if not nodes:raise ValueError('No target material nodes')
    return nodes


def material_only(original,changed):
    restored=copy.deepcopy(changed)
    a,b=material_nodes(original),material_nodes(restored)
    if len(a)!=len(b):raise ValueError('Material topology changed')
    for x,y in zip(a,b):
        if list(map(float,x.text.split())) != [.42,.42,.42,1.] or list(map(float,y.text.split())) != [.35,.35,.35,1.]:
            raise ValueError('Unfrozen material values')
        y.text=x.text
    if ET.tostring(original)!=ET.tostring(restored):raise ValueError('Non-material world change')


def freeze():
    dest=OUT/'capture-protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    cp=SOURCE/'capture-protocol.json';mp=SOURCE/'export/manifest.json'
    p,m=prior.read(cp),prior.read(mp);prior.verify(p);prior.verify(m)
    cool={r['source_member_id']:r for r in m['members']}
    sources=[u for u in p['units'] if u['source_member']['variant']=='neutral']
    if len(sources)!=8:raise ValueError('Expected eight neutral source poses')
    deps=[cp,mp,SOURCE/'quality-review.json',SOURCE/'pool-fit-diagnosis-v1/summary.json',
          SOURCE/'evaluation/error-review-v1/completion.json',Path(__file__).resolve()]
    for d in deps:
        if d.suffix=='.json':prior.verify(prior.read(d))
    units=[];OUT.mkdir(exist_ok=True)
    for u in sources:
        for condition in ('normal','cool'):
            source_member=u['source_member'] if condition=='normal' else cool[u['source_member']['member_id']]
            sp=Path(u['source_plan'] if condition=='normal' else u['plan_path'])
            source=read_record(sp)
            v=next(copy.deepcopy(x) for x in source['calibration_views'] if x['view_id']==u['view']['view_id'])
            uid=f'N{len(units)+1:02}';folder=OUT/'plans'/uid;folder.mkdir(parents=True)
            for name,digest in source['files'].items():
                src=sp.parent/name
                if prior.file_sha256(src)!=digest:raise ValueError('Source drift')
                shutil.copy2(src,folder/name);deps.append(src)
            original=ET.parse(sp.parent/'world.sdf').getroot();changed=copy.deepcopy(original)
            for node in material_nodes(changed):node.text='0.35 0.35 0.35 1'
            material_only(original,changed)
            ET.ElementTree(changed).write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
            material_only(original,ET.parse(folder/'world.sdf').getroot())
            plan={k:copy.deepcopy(x) for k,x in source.items() if k!='identity'}
            plan.update(calibration_views=[v],pilot_views=[],diagnostic_purpose='neutral_gray_calibration_counterpart',
                data_role='bounded_development_training_candidate',training_admitted=False,promotable=False)
            plan['files']={name:prior.file_sha256(folder/name) for name in plan['files']}
            plan=write_record(folder/'plan.json',plan);validate_preflight(plan,folder,[v])
            deps.extend(folder.iterdir());deps.extend([sp,Path(source_member['evidence_path'])])
            units.append(dict(unit_id=uid,source_member=source_member,source_plan=str(sp),plan_path=str(folder/'plan.json'),view=v,illumination_condition=condition))
    pilot=[next(u['unit_id'] for u in units if u['illumination_condition']=='cool' and u['source_member']['layout']=='layout-A' and u['source_member']['planned_category']==c) for c in ('capacitor_bank','reactor','switchgear','transformer')]
    return prior.frozen(dest,dict(status='16_gray_counterparts_frozen_quality_pending',units=units,pilot_units=pilot,
        change='Only labelled target body/front_panel/reactor ambient and diffuse .42 -> .35; bases, accessories, background, lighting, geometry, mapping and camera unchanged.',
        training_design='ICG1000 seeds 7/17/27 from independent v2.11, same 1000 steps, LR .0005, brightness and positions as IC1000; only the 16 neutral source identities substituted after review. No increased dose, no new thresholds.',
        limitations=['Eight existing poses and shared assets; no new independent scene.','Material value borrowed from viewed development recipe; known-condition calibration, not structure recognition proof.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


async def run(pilot=False):
    p=freeze()
    if not pilot:
        gate=prior.read(OUT/'pilot-quality.json');prior.verify(gate)
        if gate['status']!='pilot_explicitly_reviewed_expand_frozen_16':raise ValueError('Pilot not reviewed')
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for u in p['units']:
            if pilot and u['unit_id'] not in p['pilot_units']:continue
            folder=OUT/'captures'/u['unit_id'];cp=folder/'collection-receipt.json'
            if cp.exists():r=read_record(cp)
            elif folder.exists():raise ValueError('Incomplete capture retained')
            else:
                print('CAPTURING',u['unit_id'],u['illumination_condition'],flush=True)
                r=await collect(u['plan_path'],folder,mode='calibration')
            if r['status']!='complete_pending_review' or len(r['views'])!=1 or r['views'][0]['status']!='captured':raise ValueError('Capture blocked '+u['unit_id'])
            print('CAPTURED',u['unit_id'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run(a.pilot))
    else:print(freeze()['status'])
