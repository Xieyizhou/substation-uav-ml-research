"""Fail-closed explicit review, source/dedup checks and development-only admission."""
import argparse
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT, BASE, read, save, verify, file_sha256, prepare
from src.vision.canonical.plan import read_record
from src.vision.canonical.gates import validate_preflight, validate_view_pose
from scripts.vision.verify_pixel_duplicates import rgb_digest
from src.vision.training.hard_example_curator import dhash64

PROTECTED = BASE/'reference-group-audit-v1/supplemental-protected-fingerprints.jsonl'


def normalized_world(path):
    root = ET.parse(path).getroot()
    world = root.find('world')
    # Original worlds may rely on implicit ambient. Addition of scene/ambient is
    # an allowed light change; no other added scene field is ignored.
    scene = world.find('scene')
    if scene is not None:
        ambient = scene.find('ambient')
        if ambient is not None:
            scene.remove(ambient)
        if not list(scene) and not scene.attrib and not (scene.text or '').strip():
            world.remove(scene)
    diffuse = world.find("light[@name='sun']/diffuse")
    if diffuse is None:
        raise ValueError('Required sun field absent')
    diffuse.text = 'ALLOWED_LIGHT_VALUE'
    def value(node):
        return (node.tag, sorted(node.attrib.items()), (node.text or '').strip(), [value(c) for c in node])
    return value(root)


def validate_decisions(frames, decisions):
    expected = {r['view_id']: r for r in frames}
    if len(decisions) != len(expected) or {r['view_id'] for r in decisions} != set(expected):
        raise ValueError('Missing or duplicated explicit decisions')
    for d in decisions:
        r=expected[d['view_id']]
        if d['image_sha256'] != r['image_sha256'] or file_sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('Stale image review')
        if d['decision'] != 'accepted' or d['no_target_visible'] is not True or not d['coverage_confirmed']:
            raise ValueError('Semantic review held')
        if d['review_nature'] != 'AI-assisted' or not d['reason'] or not d['reviewed_at'] or not d['rois']:
            raise ValueError('Incomplete review metadata')
        for roi in d['rois']:
            x0,y0,x1,y1=roi['bbox_xyxy']
            if not (0<=x0<x1<=1920 and 0<=y0<y1<=1080) or roi['content']=='unknown':
                raise ValueError('Unconfirmed ROI')


def audit(stage):
    matrix=prepare(); manifest_path=OUT/f'{stage}-review-manifest.json'
    manifest=read(manifest_path); verify(manifest)
    inputs={str(manifest_path):file_sha256(manifest_path),str(PROTECTED):file_sha256(PROTECTED),str(Path(__file__)):file_sha256(Path(__file__))}
    protected=[__import__('json').loads(line) for line in PROTECTED.read_text().splitlines() if line]
    refs=[]
    prior=read(BASE/'exposure-controlled-diagnosis-v1/protocol.json'); verify(prior)
    paths=[r['image_path'] for r in prior['pool_rows']]
    for p in (BASE/'paired-visual-factors-v1/semantic-review.json',BASE/'hard-negative-isolated-v2/semantic-review.json'):
        doc=read(p);inputs[str(p)]=file_sha256(p)
        paths.extend(r['image_path'] for r in doc['frames'])
    if stage=='remaining':
        paths.extend(r['image_path'] for r in read(OUT/'pilot-review-manifest.json')['frames'])
    for path in sorted(set(paths)):
        digest=file_sha256(path);inputs[path]=digest
        pixel,size=rgb_digest(Path(path).read_bytes())
        refs.append(dict(path=path,image_sha256=digest,pixel_sha256=pixel,perceptual_hash=dhash64(path)))
    frames=[]
    for r in manifest['frames']:
        receipt=read_record(r['receipt_path']); plan_path=Path(r['receipt_path']).parent.parent/'plan/plan.json'
        plan=read_record(plan_path); _,config,_=validate_preflight(plan,plan_path.parent,plan['calibration_views'])
        captured=next(v for v in receipt['views'] if v['view_id']==r['view_id'])
        validate_view_pose(r,config,actual_carrier=captured['actual_pose']['position'])
        if captured['truth']['objects'] or captured['target_checks']['observed_instance_labels']:
            raise ValueError('Nonempty truth')
        pixel,size=rgb_digest(Path(r['image_path']).read_bytes()); ph=dhash64(r['image_path'])
        frames.append({**r,'pixel_sha256':pixel,'perceptual_hash':ph,'image_size':list(size),
            'world_sha256':plan['files']['world.sdf'],'sensor_sha256':plan['files']['sensor_source.sdf'],
            'source_layout_id':plan['source_layout_id'],'derived_layout_id':plan['derived_layout_id'],
            'actual_annotation_mode':receipt['gate_checks']['actual_annotation_mode'] if 'gate_checks' in receipt else 'full_2d_revalidated',
            'asset_ids':sorted(r['projected_structure_boxes']), 'label_object_count':0})
    checks=[]
    for r in frames:
        peers=refs+[p for p in frames if p['pair_id']!=r['pair_id']]
        exact=[p.get('path',p.get('image_path')) for p in peers if p['image_sha256']==r['image_sha256'] or p['pixel_sha256']==r['pixel_sha256']]
        near=sorted((dict(path=p.get('path',p.get('image_path')),distance=(int(p['perceptual_hash'],16)^int(r['perceptual_hash'],16)).bit_count()) for p in peers),key=lambda p:p['distance'])[:3]
        protected_exact=any(p['image_sha256']==r['image_sha256'] or p['pixel_sha256']==r['pixel_sha256'] for p in protected)
        protected_min=min((int(p['perceptual_hash'],16)^int(r['perceptual_hash'],16)).bit_count() for p in protected)
        checks.append(dict(view_id=r['view_id'],exact_matches=exact,nearest=near,protected_exact=protected_exact,protected_min_distance=protected_min))
    for pair in {r['pair_id'] for r in frames}:
        group=[r for r in frames if r['pair_id']==pair]
        if len(group)!=2 or {r['variant'] for r in group}!={'light_normal','light_cool_low'}:
            raise ValueError('Incomplete lighting pair')
        worlds=[Path(r['receipt_path']).parent.parent/'plan/world.sdf' for r in group]
        if normalized_world(worlds[0])!=normalized_world(worlds[1]):
            raise ValueError('Non-light world differences')
    result=save(OUT/f'{stage}-audit.json',dict(status='audited_pending_explicit_review',frames=frames,checks=checks,inputs=inputs))
    print('AUDIT',len(frames),'exact',sum(bool(r['exact_matches']) for r in checks),'near_le2',sum(r['nearest'][0]['distance']<=2 for r in checks),'protected',sum(r['protected_exact'] or r['protected_min_distance']<=2 for r in checks))
    return result


def admit(stage):
    manifest=read(OUT/f'{stage}-review-manifest.json');verify(manifest)
    audit_path=OUT/f'{stage}-audit.json';audited=read(audit_path);verify(audited)
    decisions_path=OUT/f'{stage}-decisions.json';decisions=read(decisions_path);verify(decisions)
    validate_decisions(manifest['frames'],decisions['decisions'])
    for check in audited['checks']:
        if check['exact_matches'] or check['protected_exact'] or check['protected_min_distance']<=2:
            raise ValueError('Exact or protected duplication requires exclusion')
        if check['nearest'][0]['distance']<=2:
            raise ValueError('Near-duplicate needs separate evidence-bound review before admission')
    result=save(OUT/f'{stage}-admission.json',dict(status='accepted',accepted=len(manifest['frames']),
        frames=audited['frames'],decisions=decisions['decisions'],development_training_eligible=True,
        inputs={str(p):file_sha256(p) for p in (audit_path,decisions_path,OUT/f'{stage}-review-manifest.json',Path(__file__))}))
    print('ADMITTED',result['accepted'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('audit','admit'));p.add_argument('--stage',default='pilot',choices=('pilot','remaining'));a=p.parse_args()
    (audit if a.action=='audit' else admit)(a.stage)
