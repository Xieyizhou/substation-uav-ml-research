"""Static source/renderer/quota preflight. No sensor launch or training entry."""
from pathlib import Path
from collections import Counter
import copy,hashlib,os,subprocess,xml.etree.ElementTree as ET
import numpy as np
from scripts.vision.check_reviewed_hold_fit import OUT as FIT,SOURCE,prior,base as fitbase
from scripts.vision.reviewed_hold_control import prior as history
from scripts.vision.run_depth_clip_test import OUT as DEPTH,BUILD,base as runtime
from scripts.vision.test_body_material_applicability import equal_rgb,label_correspondence,model_mapping
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world,validate_point

OUT=FIT.parent/'physical-lighting-control-preflight-v1'
VERSION='physical-lighting-control-preflight-v1'

def change_lighting(root):
    r=copy.deepcopy(root);scene=r.find('.//scene');sun=r.find(".//light[@name='sun']")
    if scene is None or sun is None:raise ValueError('Missing scene/sun')
    for el,text in [(scene.find('ambient'),'0.38 0.42 0.48 1'),(sun.find('diffuse'),'0.58 0.66 0.78 1')]:
        if el is None:raise ValueError('Missing explicit lighting field')
        el.text=text
    return r

def validate_change(original,changed):
    restored=copy.deepcopy(changed)
    for q in ('.//scene/ambient',".//light[@name='sun']/diffuse"):
        a,b=original.find(q),restored.find(q)
        if a is None or b is None:raise ValueError('Missing frozen light field')
        b.text=a.text
    if ET.tostring(original)!=ET.tostring(restored):raise ValueError('Non-lighting world change')
    if ET.tostring(change_lighting(original))!=ET.tostring(changed):raise ValueError('Unfrozen lighting value')

def slots(sequence,member,quota,seed):
    options=[i for i,m in enumerate(sequence) if m==member]
    if quota>len(options):raise ValueError('Insufficient source exposures')
    return sorted(sorted(options,key=lambda i:hashlib.sha256(f'{VERSION}:{seed}:{member}:{i}'.encode()).hexdigest())[:quota])

def main():
    p=prior.read(FIT/'protocol.json');prior.verify(p);training=prior.read(SOURCE/'protocol.json');prior.verify(training)
    ep=history.CONTROL/'evidence.json';rp=history.CONTROL/'review.json'
    e,r=prior.read(ep),prior.read(rp)
    for x in (e,r):prior.verify(x)
    events={x['member']['member_id']:x for x in e['events']};idx={x['member_id']:x for x in p['rows']}
    common=set.intersection(*(set(m['draws']) for m in p['models'].values()))
    candidates=sorted({t['review']['member_id'] for t in p['targets'] if t['clear_primary'] and t['review']['truth']['class_name']=='reactor' and t['review']['member_id'] in common})
    OUT.mkdir(exist_ok=True);paths=[FIT/'protocol.json',SOURCE/'protocol.json',ep,rp,Path(__file__).resolve()];selected=[];excluded=[]
    for mid in candidates:
        f=events[mid];row=idx[mid];truth=fitbase.truth_for(row);ds=[x for x in r['decisions'] if x['member_id']==mid]
        if len(ds)!=len(truth) or len({d['truth']['annotation_id'] for d in ds})!=len(truth):raise ValueError('Missing/duplicate full-label decisions')
        for d in ds:fitbase.check_review(d,row,truth[d['truth']['label_line_index']])
        risks=[d for d in ds if d['status']!='identifiable_geometry_with_recorded_limits']
        if risks:
            excluded.append(dict(member_id=mid,reason='existing_other_label_quality_risk',decisions=risks,historical_exposures_unchanged=True));continue
        plan=prior.read(f['source_plan']);rec=prior.read(f['source_receipt']);wp=Path(f['source_world']);config=prior.read(wp.parent/'obstacles.json')
        if prior.file_sha256(wp)!=plan['files']['world.sdf'] or annotation_mode_from_world(wp)!='full_2d':raise ValueError('World/mode changed')
        equal_rgb(f['source_image'],row['image_path']);label_correspondence(truth,rec['truth']['objects'])
        mapping={str(k):v for k,v in instance_mapping(plan).items()};wm,_=model_mapping(ET.parse(wp))
        if len({v['object_id'] for v in mapping.values()})!=len(mapping):raise ValueError('Instance collision')
        if any(wm.get(k)!=v['object_id'] for k,v in mapping.items()):raise ValueError('World/plan mapping mismatch')
        root=ET.parse(wp).getroot();camera=root.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
        if camera is None:raise ValueError('Camera frame unresolved')
        offset=list(map(float,camera.findtext('pose').split()))[:3];pose=rec['actual_pose']
        optical=(np.array(pose['position'])+runtime.rotate(pose['orientation'],offset)).tolist()
        validate_point(pose['position'],config,role='saved carrier');validate_point(optical,config,role='saved optical center')
        name=f'L{len(selected)+1:02}';folder=OUT/name;folder.mkdir(exist_ok=True);dark=change_lighting(root);validate_change(root,dark)
        for variant,tree in [('original',root),('physical-lighting',dark)]:
            dest=folder/(variant+'.sdf')
            if dest.exists():
                if ET.tostring(ET.parse(dest).getroot())!=ET.tostring(tree):raise ValueError('Existing draft changed')
            else:ET.ElementTree(tree).write(dest,encoding='utf-8',xml_declaration=True)
            paths.append(dest)
        selected.append(dict(id=name,member_id=mid,source=f,actual_pose={k:pose[k] for k in ('position','orientation')},optical_center=optical,
            instance_mapping=mapping,existing_label_review_count=len(ds),existing_review_nature='AI-assisted; saved-plan mapping checked posthoc',
            original_world=str(folder/'original.sdf'),lighting_world=str(folder/'physical-lighting.sdf'),
            status='static_preflight_passed_actual_replay_and_full_frame_review_pending'))
        paths += [Path(f[k]) for k in ('source_world','source_plan','source_receipt','source_image')]+[wp.parent/'obstacles.json',Path(row['image_path']),Path(row['label_path'])]
    if not selected:raise ValueError('No eligible whole frames')
    # Identical quota per source across seeds; no member gains exposure.
    quotas={s['member_id']:min(training['schedules'][f'brightness-450-{seed}'].count(s['member_id'])//2 for seed in (7,17,27)) for s in selected}
    if any(q<=0 for q in quotas.values()):raise ValueError('Zero eligible paired quota')
    schedules={}
    for seed in (7,17,27):
        key=f'brightness-450-{seed}';seq=training['schedules'][key];positions={i:m for m,q in quotas.items() for i in slots(seq,m,q,seed)}
        if any(idx[m]['subset']=='hard_negative' for m in positions.values()):raise ValueError('Negative position altered')
        schedules[str(seed)]=dict(position_to_source={str(i):m for i,m in sorted(positions.items())},
            original_sequence=seq,brightness_factors=training['brightness_factors'][key],
            replaced_exposures=len(positions),replaced_class_instances=dict(sum((Counter(idx[m]['class_instances']) for m in positions.values()),Counter())),
            windows_50_steps=[sum(i in positions for i in range(start,start+300)) for start in range(0,2700,300)],
            equality='Conditional on replay proving unchanged complete labels; pending images are not a loadable training dataset')
    runtime.guard();dp=DEPTH/'protocol.json';prior.verify(prior.read(dp));paths.append(dp)
    probe=BUILD/'resolve-plugin-prefix';expected=DEPTH/'variants/depth/libgz-rendering8-ogre2.8.2.3.dylib'
    probe_result=subprocess.check_output([str(probe)],env={**os.environ,'GZ_RENDERING_PLUGIN_PATH':str(expected.parent),'GZ_RENDERING_INSTALL_PREFIX':str(DEPTH/'runtime-prefix/depth')},text=True,timeout=30)
    resolved=next(x[7:] for x in probe_result.splitlines() if x.startswith('result='))
    if Path(resolved).resolve()!=expected.resolve():raise ValueError('Wrong isolated renderer')
    for x in (probe,expected,DEPTH/'diagnostic-server',DEPTH/'gz_visibility_capture_cleanup_fixed'):
        if not x.exists():raise ValueError('Missing runtime component')
        paths.append(x)
    runtime.guard()
    prior.frozen(OUT/'protocol.json',dict(status='static_design_complete_replay_not_started',candidates=candidates,selected=selected,excluded=excluded,
        selection='All clear common reactor sources whose complete existing label reviews are identifiable; no score-based selection',
        quotas_per_member=quotas,schedules=schedules,loader_probe=probe_result,training_ready=False,collection_started=False,training_started=False,
        next_boundary='Five original same-frame alignment replays and five physical-lighting captures with full-frame review required. No rendering performed by this tool.',
        stop_rules=['RGB alignment failure in original replay','label or instance mapping conflict','new visibility/content risk','three technical failures per replay unit'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('STATIC_READY',len(selected),'EXCLUDED',len(excluded),'QUOTA',sum(quotas.values()))

if __name__=='__main__':main()
