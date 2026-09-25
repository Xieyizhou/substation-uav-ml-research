"""Independent hard-negative coverage planning and gated canonical acquisition."""
import argparse
import asyncio
import copy
import itertools
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import BASE, read, save, verify, file_sha256
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE as SOURCE
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record, write_record, rotate
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import validate_view_pose, validate_preflight

OUT = BASE / 'hard-negative-coverage-v1'
COUNTS = {'B1': 8, 'B2': 8, 'G1': 6, 'G2': 6, 'M1': 8, 'C1': 4, 'C2': 4, 'P1': 4}
PILOT = dict(zip(COUNTS, (2, 2, 1, 1, 2, 1, 1, 2)))
TARGETS = {'transformer', 'switchgear', 'capacitor_bank', 'reactor'}


def projected(row, bounds):
    q = row['orientation']; inverse = [-q[0], -q[1], -q[2], q[3]]
    pts = []
    for point in itertools.product(bounds[:2], bounds[2:4], bounds[4:]):
        x, y, z = rotate(inverse, [p-c for p, c in zip(point, row['camera_position'])])
        if x <= .05:
            return None
        pts.append((.5-y/(2*x*math.tan(1.466/2)), .5-z/(2*x*math.tan(1.466/2)*1080/1920)))
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def visible_rect(box):
    return box and max(0, min(1, box[2])-max(0, box[0])) * max(0, min(1, box[3])-max(0, box[1])) > .004


def candidates(unit, plan, config):
    map_id = plan['map_id']
    objs = [o for o in plan['objects'] if o['category'] not in TARGETS]
    allowed = ('control_building',) if unit[0] in 'BM' else ('cabinet',) if unit[0] == 'C' else ('pole',) if unit == 'P1' else ('control_building', 'pole')
    subjects = [o for o in objs if o['category'] in allowed]
    results = []
    for obj in subjects:
        b = obj['bounds']; center = [(b[i]+b[i+1])/2 for i in (0, 2, 4)]
        for bearing in range(3, 360, 15):
            for distance in (4.7, 6.3, 8.7, 11.3):
                for height in (2.3, 3.7):
                    a = math.radians(bearing)
                    camera = [center[0]+distance*math.cos(a), center[1]+distance*math.sin(a), height]
                    look = list(center)
                    if unit == 'G1':
                        look = [camera[0]+2.5*math.cos(a+math.pi), camera[1]+2.5*math.sin(a+math.pi), 0]
                    elif unit == 'G2':
                        look[2] = -.6
                    for offset in ((-35, 35) if unit in ('B2', 'C2') else (0,)):
                        position, orientation = _pose(camera, look, offset)
                        row = dict(map_id=map_id, object_id=obj['name'], category='no_target',
                            coverage_unit=unit, bearing=bearing, distance=distance, height=height,
                            offset=offset, camera_position=camera, position=position, orientation=orientation,
                            family=f'{map_id}:coverage:{unit}:{obj["name"]}:{bearing}', look_at=look)
                        try:
                            validate_view_pose(row, config)
                        except ValueError:
                            continue
                        boxes = {o['name']: projected(row, o['bounds']) for o in objs}
                        box = boxes[obj['name']]
                        if unit not in ('G1', 'G2') and not visible_rect(box):
                            continue
                        if unit in ('B1', 'C1') and not (box[0]>.03 and box[1]>.03 and box[2]<.97 and box[3]<.97):
                            continue
                        if unit in ('B2', 'C2') and not (box[0]<0<box[2] or box[0]<1<box[2]):
                            continue
                        if unit == 'M1' and sum(bool(visible_rect(v)) for v in boxes.values()) < 3:
                            continue
                        row['projected_structure_boxes'] = {k: v for k, v in boxes.items() if visible_rect(v)}
                        row['pose_id'] = object_sha256({'round': 301, **row})
                        results.append(row)
    return results


def prepare():
    path = OUT/'matrix.json'
    if path.exists():
        result = read(path); verify(result); return result
    source_matrix = read_record(SOURCE/'matrix.json')
    inputs = {str(SOURCE/'matrix.json'): file_sha256(SOURCE/'matrix.json'), str(Path(__file__)): file_sha256(Path(__file__)),
              str(ROOT/'docs/hard-negative-coverage-v1.md'): file_sha256(ROOT/'docs/hard-negative-coverage-v1.md')}
    sources = {}
    for run in source_matrix['runs']:
        p = Path(run['plan_path']); plan = read_record(p)
        inputs[str(p)] = file_sha256(p)
        for name, digest in plan['files'].items():
            if file_sha256(p.parent/name) != digest:
                raise ValueError('Source changed')
            inputs[str(p.parent/name)] = digest
        world = ET.parse(p.parent/'world.sdf')
        names = {m.get('name') for m in world.iter('model')}
        if names & set(plan['removed_target_models']):
            raise ValueError('Isolated world contains removed targets')
        sources[(run['map_id'], run['lighting_id'])] = (p, plan)
    # Only explicit, already-viewed canonical plans; never enumerate sealed sources.
    old = []
    for directory in (ROOT/'data/research/canonical_views_v1', SOURCE, BASE/'hard-negative-isolated-v2'):
        for p in sorted(directory.rglob('plan.json')):
            plan = read_record(p)
            inputs[str(p)] = file_sha256(p)
            old.extend((plan['map_id'], v['camera_position'], v['orientation']) for v in plan.get('calibration_views', [])+plan.get('pilot_views', []))
    selected = []
    inventory = {}
    for unit, count in COUNTS.items():
        pool = []
        maps = ('complex',) if unit[0] in 'BM' else ('simple', 'medium', 'complex')
        for map_id in maps:
            p, plan = sources[map_id, 'light_normal']
            inventory[map_id] = [o for o in plan['objects'] if o['category'] not in TARGETS]
            pool.extend(candidates(unit, plan, read(p.parent/'obstacles.json')))
        pool = [r for r in pool if not any(m == r['map_id'] and math.dist(c, r['camera_position']) < .75
                 and abs(sum(a*b for a, b in zip(q, r['orientation']))) > math.cos(math.radians(10)/2) for m, c, q in old)]
        chosen = []
        # Spread bearings, distances, maps and physical camera positions without model scores.
        while len(chosen) < count:
            available = [r for r in pool if r not in chosen and not any(r['map_id'] == s['map_id'] and math.dist(r['camera_position'], s['camera_position']) < 1.0 for s in selected+chosen)]
            if not available:
                raise ValueError(f'Coverage shortfall {unit}: {len(chosen)}/{count}')
            def rank(r):
                reuse = sum(s['map_id']==r['map_id'] and s['object_id']==r['object_id'] for s in chosen)
                distances = [math.dist(r['camera_position'], s['camera_position']) for s in chosen if s['map_id']==r['map_id']]
                return (reuse, -min(distances, default=100), r['pose_id'])
            chosen.append(min(available, key=rank))
        for i, r in enumerate(chosen):
            r['stage'] = 'pilot' if i < PILOT[unit] else 'remaining'
            r['ordinal'] = len(selected)
            selected.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for stage in ('pilot', 'remaining'):
        for (map_id, light), (source_path, source) in sorted(sources.items()):
            views = []
            for row in selected:
                if row['stage'] != stage or row['map_id'] != map_id:
                    continue
                v = copy.deepcopy(row)
                v.update(view_id=object_sha256({'coverage_v1': row['pose_id'], 'variant': light}),
                    pair_id=row['pose_id'], variant=light, lighting_id=light,
                    derivation_group=f'coverage-v1:{row["pose_id"]}',
                    data_role='training_candidate_development_only', training_admitted=False, promotable=False)
                views.append(v)
            if not views:
                continue
            run_id = f'{stage}-{map_id}-{light}'
            folder = OUT/'runs'/run_id/'plan'; folder.mkdir(parents=True, exist_ok=False)
            for name in source['files']:
                shutil.copyfile(source_path.parent/name, folder/name)
            plan = {k: copy.deepcopy(v) for k, v in source.items() if k != 'identity'}
            plan.update(calibration_views=views, pilot_views=[], parent_plan_identity=source['identity'],
                        diagnostic_purpose='hard_negative_coverage_v1', recording_group=run_id)
            written = write_record(folder/'plan.json', plan)
            validate_preflight(written, folder, views)
            inputs[str(folder/'plan.json')] = file_sha256(folder/'plan.json')
            runs.append(dict(run_id=run_id, stage=stage, plan_path=str(folder/'plan.json'), frame_count=len(views)))
    return save(path, dict(status='frozen_pending_pilot', counts=COUNTS, pilot_counts=PILOT, inventory=inventory,
        poses=selected, runs=runs, inputs=inputs, source_relation='existing canonical target-isolated layouts and assets; new poses, not independent scenes',
        source_exclusion_rule='exclude camera distance <0.75m AND orientation difference <10deg from enumerated viewed plans',
        light_values={'light_normal': 'unchanged hash-bound source', 'light_cool_low': {'ambient': '.30 .34 .40 1', 'sun_diffuse': '.48 .56 .68 1'}}))


async def capture(stage):
    from src.vision.canonical.collect import collect
    matrix = prepare()
    if stage == 'remaining':
        review = read(OUT/'pilot-admission.json'); verify(review)
        if review['status'] != 'accepted' or review['accepted'] != 24:
            raise ValueError('Pilot admission required')
    rows = []
    for run in matrix['runs']:
        if run['stage'] != stage:
            continue
        root = Path(run['plan_path']).parent.parent
        completed = root/'capture-completion.json'
        if completed.exists():
            cached = read(completed); verify(cached); rows.append(cached); continue
        for attempt in range(1, 4):
            output = root/f'capture-attempt-{attempt:03}'
            if output.exists():
                receipt = read_record(output/'collection-receipt.json') if (output/'collection-receipt.json').exists() else None
            else:
                print('CAPTURE', run['run_id'], attempt, flush=True)
                receipt = await collect(run['plan_path'], output)
            if receipt and any(r['status']=='rejected' for r in receipt['views']):
                raise ValueError('Semantic rejection; do not retry without investigation')
            if receipt and receipt['status']=='complete_pending_review' and len(receipt['views']) == run['frame_count']:
                rows.append(save(completed, dict(status='captured_pending_review', run_id=run['run_id'],
                    receipt_path=str(output/'collection-receipt.json'), inputs={str(output/'collection-receipt.json'): file_sha256(output/'collection-receipt.json'),
                    run['plan_path']: file_sha256(run['plan_path'])})))
                break
        else:
            raise ValueError('Three technical attempts exhausted')
    save(OUT/f'{stage}-capture.json', dict(status='complete_pending_review', runs=rows,
        inputs={str(OUT/'matrix.json'): file_sha256(OUT/'matrix.json')}))


def evidence(stage):
    from PIL import Image, ImageDraw
    progress = read(OUT/f'{stage}-capture.json'); verify(progress)
    frames = []; inputs = {str(OUT/f'{stage}-capture.json'): file_sha256(OUT/f'{stage}-capture.json')}
    for run in progress['runs']:
        verify(run)
        path = Path(run['receipt_path']); receipt = read_record(path)
        plan_path = path.parent.parent/'plan/plan.json'; plan = read_record(plan_path)
        planned = {r['view_id']: r for r in plan['calibration_views']}
        inputs[str(path)] = file_sha256(path); inputs[str(plan_path)] = file_sha256(plan_path)
        for r in receipt['views']:
            if r['status'] != 'captured' or r['truth']['objects'] or r['target_checks']['observed_instance_labels']:
                raise ValueError('Not valid empty truth')
            if file_sha256(r['rgb_path']) != r['image_sha256']:
                raise ValueError('Image changed')
            inputs[r['rgb_path']] = r['image_sha256']
            frames.append({**planned[r['view_id']], 'image_path': r['rgb_path'], 'image_sha256': r['image_sha256'],
                'truth_sha256': object_sha256(r['truth']), 'receipt_path': str(path), 'decision': 'pending'})
    frames.sort(key=lambda r: (r['ordinal'], r['variant']))
    folder = OUT/f'{stage}-evidence'; folder.mkdir(exist_ok=True)
    for i, r in enumerate(frames):
        r['review_index'] = i
        with Image.open(r['image_path']) as image:
            canvas = Image.new('RGB', (1280, 756), 'white')
            canvas.paste(image.resize((1280,720)), (0,36))
            ImageDraw.Draw(canvas).text((8,10), f'{i} {r["coverage_unit"]} pose {r["ordinal"]} {r["map_id"]} {r["variant"]}', fill='black')
            p = folder/f'{i:03}.jpg'; canvas.save(p, quality=95)
        r['evidence_path']=str(p); inputs[str(p)]=file_sha256(p)
    save(OUT/f'{stage}-review-manifest.json', dict(status='pending_explicit_review', frames=frames, inputs=inputs))
    print('REVIEW', len(frames), folder)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare','capture','evidence'))
    parser.add_argument('--stage', choices=('pilot','remaining'), default='pilot'); args=parser.parse_args()
    if args.action=='prepare':
        result=prepare(); print(result['identity'], Counter(r['stage'] for r in result['poses']))
    elif args.action=='capture':
        asyncio.run(capture(args.stage))
    else:
        evidence(args.stage)
