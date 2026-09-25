"""Saved-asset/source diagnostic only. No rendering, model inference or training."""
from collections import Counter
from pathlib import Path
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from scripts.vision.record_condition_target_review import OUT as PRIOR, normalize_mapping
from scripts.vision.condition_transfer_design import ROOT,read,verify,frozen,file_sha256,baseline_verify
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence
from scripts.vision.structure_fit import truth_for
from src.vision.canonical.plan import read_record

OUT=PRIOR/'body-material-applicability-v1'


def numbers(text,n):
    values=list(map(float,text.split()))
    if len(values)!=n or not all(math.isfinite(x) for x in values):raise ValueError('Invalid numeric geometry')
    return values


def bounds(visual):
    pose=visual.find('pose')
    if pose is not None and pose.attrib:raise ValueError('Unsupported relative geometry frame')
    p=numbers(visual.findtext('pose','0 0 0 0 0 0'),6)
    if any(abs(x)>1e-12 for x in p[3:]):raise ValueError('Rotated geometry requires another solver')
    g=visual.find('geometry')
    if g is None or len(g)!=1:raise ValueError('Ambiguous geometry')
    if g.find('box') is not None:half=[x/2 for x in numbers(g.findtext('box/size'),3)]
    elif g.find('cylinder') is not None:
        radius=numbers(g.findtext('cylinder/radius'),1)[0]
        half=[radius,radius,numbers(g.findtext('cylinder/length'),1)[0]/2]
    else:raise ValueError('Unsupported primitive')
    if min(half)<=0:raise ValueError('Nonpositive extent')
    return [[p[i]-half[i] for i in range(3)],[p[i]+half[i] for i in range(3)]]


def strictly_inside(outer,inner):
    margins=[inner[0][i]-outer[0][i] for i in range(3)]+[outer[1][i]-inner[1][i] for i in range(3)]
    return dict(strictly_inside=min(margins)>1e-9,min_clearance_m=min(margins),face_clearances_m=margins)


def signature(model):
    links=model.findall('link')
    if len(links)!=1:raise ValueError('Multiple links need frame resolution')
    visuals=links[0].findall('visual');byname={v.get('name'):v for v in visuals}
    if len(byname)!=len(visuals) or 'body' not in byname:raise ValueError('Missing or ambiguous body')
    body=byname['body']
    if body.find('geometry/box') is None:raise ValueError('Body is not a solid box primitive')
    bb=bounds(body);materials={}
    for name,v in byname.items():
        materials[name]={k:v.findtext('material/'+k) for k in ('ambient','diffuse','specular')}
    alpha=[numbers(body.findtext('material/'+k),4)[3] for k in ('ambient','diffuse')]
    transparency=float(body.findtext('transparency','0'))
    opaque=alpha==[1,1] and transparency==0 and body.find('material/pbr') is None and body.find('material/script') is None
    cylinders=[]
    for name,v in byname.items():
        if name.startswith('capacitor_'):
            if v.find('geometry/cylinder') is None:raise ValueError('Named capacitor not cylindrical')
            b=bounds(v);cylinders.append(dict(name=name,bounds_m=b,**strictly_inside(bb,b)))
    return dict(body_bounds_m=bb,body_opaque_in_saved_configuration=opaque,materials=materials,
        front_panel_present='front_panel' in byname,capacitor_cylinders=cylinders,
        six_cylinders_strictly_enclosed=opaque and len(cylinders)==6 and all(c['strictly_inside'] for c in cylinders),
        interpretation='Common-link local coordinates; analytic containment, not rendered pixel visibility certification.')


def model_mapping(tree):
    mapping={};models={}
    for model in tree.findall('.//world/model'):
        name=model.get('name')
        if name in models:raise ValueError('Duplicate model name')
        models[name]=model
        labels=set()
        for plugin in model.findall('.//visual/plugin'):
            if plugin.get('name')=='gz::sim::systems::Label':
                label=plugin.findtext('label')
                if label is None or not label.isdigit():raise ValueError('Unparseable visual label')
                labels.add(str(int(label)))
        for label in labels:
            if label in mapping:raise ValueError('Visual label belongs to multiple models')
            mapping[label]=name
    return mapping,models


def source_index():
    base=ROOT/'data/research/ml_training_recovery_v1';rows={};paths=[]
    specs=[('trusted-training-base-v1/frozen-ledger.json','base:','view_id','entries'),
        ('visual-augmentation-240-v2/frozen-intake-ledger.json','candidate:','candidate_id','entries'),
        ('visual-bridge-supplement-v2/pilot-v1/review-v1/semantic-review.json','bridge:','view_id','frames'),
        ('visual-bridge-supplement-v2/remaining-positive-v1/review-v1/semantic-review.json','bridge:','view_id','frames')]
    for name,prefix,key,coll in specs:
        path=base/name;record=read_record(path);paths.append(path)
        for r in record[coll]:
            mid=prefix+r[key]
            if mid in rows:raise ValueError('Duplicate source member')
            rows[mid]=r
    return rows,paths


def trace_training(e):
    sources,paths=source_index();results=[];cache={}
    for r in e['training']:
        member=r['source'];source=sources[member['member_id']]
        ip=Path(source.get('source_image_path',source.get('image_path')))
        expected=source.get('source_image_sha256',source.get('image_sha256'))
        if file_sha256(ip)!=expected:raise ValueError('Stale original RGB')
        equal_rgb(ip,member['image_path'])
        if file_sha256(member['label_path'])!=member['label_sha256']:raise ValueError('Changed complete label')
        cp=ip.parents[1]/'collection-receipt.json';pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
        if cp not in cache:
            rec=read_record(cp);plan=read_record(pp);digest=file_sha256(wp)
            if plan.get('files',{}).get('world.sdf')!=digest:raise ValueError('Plan world hash mismatch')
            if rec.get('world_sha256') not in (None,digest):raise ValueError('Receipt world hash mismatch')
            if rec.get('plan_identity') not in (None,plan['identity']):raise ValueError('Receipt plan identity mismatch')
            mapping,models=model_mapping(ET.parse(wp))
            receipt_mapping=normalize_mapping(rec.get('collection_checks',{}).get('instance_mapping',{}))
            if len({v['object_id'] for v in receipt_mapping.values()})!=len(receipt_mapping):raise ValueError('Receipt mapping collision')
            for label,identity in receipt_mapping.items():
                if mapping.get(label)!=identity['object_id']:raise ValueError('Saved visual and receipt instance disagree')
            cache[cp]=(rec,mapping,models,receipt_mapping,digest)
        rec,mapping,models,receipt_mapping,digest=cache[cp]
        views=[v for v in rec['views'] if v['view_id']==ip.parent.name]
        if len(views)!=1 or views[0]['image_sha256']!=expected:raise ValueError('Image/capture identity mismatch')
        full=truth_for(member);original=label_correspondence(full,views[0]['truth']['objects'])
        t=r['target']['truth'];indices=[i for i,x in enumerate(full) if x==t]
        if len(indices)!=1:raise ValueError('Target not unique in complete labels')
        original_target=original[indices[0]];match=re.search(r'instance-(\d+)-',original_target['annotation_id'])
        if not match:raise ValueError('Missing target instance label')
        label=str(int(match[1]));name=mapping[label]
        if receipt_mapping and receipt_mapping[label]['category']!=t['class_name']:raise ValueError('Category mismatch')
        gaps=[]
        if not rec.get('world_sha256'):gaps.append('historical_capture_world_hash_missing')
        if not receipt_mapping:gaps.append('historical_capture_instance_mapping_missing')
        if not rec.get('actual_annotation_mode'):gaps.append('historical_actual_annotation_mode_missing')
        results.append(dict(review_id=r['review_id'],member_id=member['member_id'],class_name=t['class_name'],
            source_image_path=str(ip),source_image_sha256=expected,world_path=str(wp),world_sha256=digest,
            capture_path=str(cp),native_rgb_equal=True,complete_labels_correspond=True,
            source_annotation_id=original_target['annotation_id'],object_id=name,runtime_label=label,
            source_status='historical_posthoc_world_mapping_only' if gaps else 'recorded_world_and_mapping_verified',
            gaps=gaps,actual_exposures=r['target']['actual_exposures'],signature=signature(models[name]),
            original_gate_retroactively_certified=False,replay_eligible=False,pixel_visibility_certified=False))
        paths += [ip,cp,pp,wp,Path(member['image_path']),Path(member['label_path'])]
    return results,paths


def run():
    OUT.mkdir(exist_ok=True);dest=OUT/'diagnosis.json'
    e=read(PRIOR/'evidence.json');review=read(PRIOR/'review.json')
    for p in ('evidence.json','review.json','completion.json'):verify(read(PRIOR/p))
    if dest.exists():verify(read(dest));print('REUSED_VALID_BODY_DIAGNOSTIC');return
    training,paths=trace_training(e);development=[]
    for world in review['saved_development_component_audit']:
        wp=Path(world['world_path'])
        if file_sha256(wp)!=world['world_sha256']:raise ValueError('Changed development world')
        _,models=model_mapping(ET.parse(wp));paths.append(wp)
        for name in sorted({t['object_id'] for t in world['targets']}):
            development.append(dict(variant=world['variant'],object_id=name,world_path=str(wp),signature=signature(models[name])))
    tests=['tests.test_body_material_applicability','tests.test_condition_target_review','tests.test_condition_transfer_design']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths += [PRIOR/p for p in ('evidence.json','review.json','completion.json')]
    paths += [Path(__file__),ROOT/'scripts/vision/check_structure_fit_sources.py',ROOT/'scripts/vision/structure_fit.py']
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='analytic_test_complete_asset_visibility_confound_confirmed',training=training,development=development,
        counts=dict(training_targets=len(training),source_status=dict(Counter(r['source_status'] for r in training)),
            capacitor_targets=sum(r['class_name']=='capacitor_bank' for r in training),
            capacitor_targets_with_enclosed_cylinders=sum(r['class_name']=='capacitor_bank' and r['signature']['six_cylinders_strictly_enclosed'] for r in training)),
        limits=['Analytic primitive containment is not a renderer test or instance-mask certificate.',
            'Targeted 49-label source audit, not full-pool coverage or generalization evidence.',
            'Historical missing capture evidence remains a gap. No replay eligibility is certified.',
            'Geometry appearance is a confound, not a proven unique cause of model errors.'],
        next_priority='Freeze a separate asset-visibility diagnostic before changing geometry or training; preserve old datasets and weights.',
        training_started=False,replay_started=False,inference_started=False,training_ready=False,
        baseline=baseline,regression=dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print(read(dest)['counts'])


if __name__=='__main__':run()
