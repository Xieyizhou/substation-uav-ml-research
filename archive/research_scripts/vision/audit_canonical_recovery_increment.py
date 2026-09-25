#!/usr/bin/env python3
"""Audit captured pilot increments and prepare a bound semantic review queue.

Calibration captures are inventoried, not silently promoted to training data.
No semantic decision is inferred from source truth, expected class or dHash.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import pose_close, read_record
from src.vision.canonical.visible_admission import adapt_visible_truth
from src.vision.collection.gazebo_truth import _box_corners
from src.vision.collection.simulator_labels import class_for_simulator_label
from scripts.vision.audit_recovery_reference_groups import discover, fingerprint, read_rows
from scripts.vision.audit_reviewed_near_duplicates import neighbors

BASE = ROOT / 'data/research/ml_training_recovery_v1'
OUT = BASE / 'canonical-increment-audit-v1'
COLORS = {'transformer': '#ffb347', 'switchgear': '#20e0ff', 'capacitor_bank': '#ff69df', 'reactor': '#7bff4b'}


def adapt_checked(view, plan):
    """Require one-to-one raw box provenance before changing conventions."""
    raw = view['raw_truth'].get('annotatedBox', view['raw_truth'].get('annotated_box', []))
    objects = view['truth']['objects']
    if len(raw) != len(objects):
        raise ValueError('raw_truth_object_count_mismatch')
    for source, parsed in zip(raw, objects):
        if tuple(_box_corners(source)) != tuple(parsed['bbox_xyxy']):
            raise ValueError('raw_truth_bbox_mismatch')
        if class_for_simulator_label(int(source['label'])) != parsed['class_name']:
            raise ValueError('raw_truth_class_mismatch')
        if f'-instance-{int(source["label"]):04d}-' not in parsed['annotation_id']:
            raise ValueError('raw_truth_instance_mismatch')
    return adapt_visible_truth(view['truth'], annotation_mode=plan.get('annotation_mode'))


def framing_reasons(adapted, rules):
    width, height = adapted['image_width'], adapted['image_height']
    margin = rules['minimum_border_margin_px']
    reasons = []
    for obj in adapted['objects']:
        x1, y1, x2, y2 = obj['bbox_xyxy']
        if x1 <= margin or y1 <= margin or x2 >= width-margin or y2 >= height-margin:
            reasons.append({'annotation_id': obj['annotation_id'], 'reason': 'bbox_touches_frame_boundary'})
        elif x2-x1 > width*rules['maximum_bbox_width_fraction'] or y2-y1 > height*rules['maximum_bbox_height_fraction']:
            reasons.append({'annotation_id': obj['annotation_id'], 'reason': 'bbox_exceeds_framing_limit'})
    return reasons


def index_rows(rows):
    indexes = {k: defaultdict(list) for k in ('image_sha256', 'pixel_sha256', 'perceptual_hash')}
    for row in rows:
        add_row(indexes, row)
    return indexes


def add_row(indexes, row):
    for key, index in indexes.items():
        if row.get(key):
            value = int(row[key], 16) if key == 'perceptual_hash' else row[key]
            index[value].append(row)


def matches(row, indexes):
    result = []
    for key in ('image_sha256', 'pixel_sha256'):
        hits = indexes[key].get(row[key], [])
        if hits:
            result.append({'method': key, 'count': len(hits), 'example_id': hits[0]['id']})
    hits = list(neighbors(int(row['perceptual_hash'], 16), indexes['perceptual_hash']))
    if hits:
        distance, example = min(hits, key=lambda x: (x[0], x[1]['id']))
        result.append({'method': 'dhash64_hamming_le_2', 'count': len(hits),
                       'distance': distance, 'example_id': example['id']})
    return result


def main():
    OUT.mkdir(exist_ok=True)
    inputs, files = {}, {}
    def bind(path):
        path = Path(path)
        inputs[str(path)] = file_sha256(path)
        return path
    def save(name, value, jsonl=False):
        path = OUT / name
        if jsonl:
            path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in value))
        else:
            write_json(path, value)
        files[name] = {'path': str(path), 'sha256': file_sha256(path)}
    for path in (__file__, ROOT/'src/vision/canonical/visible_admission.py', ROOT/'src/vision/collection/gazebo_truth.py',
                 ROOT/'src/vision/collection/simulator_labels.py', ROOT/'scripts/vision/audit_recovery_reference_groups.py',
                 ROOT/'scripts/vision/audit_reviewed_near_duplicates.py'):
        bind(path)
    group_root = BASE/'reference-group-audit-v1'
    group_report = read_record(bind(group_root/'report.json'))
    for spec in group_report['files'].values():
        if file_sha256(bind(spec['path'])) != spec['sha256']:
            raise ValueError('Reference audit output changed')
    inventory_source = json.loads((group_root/'source-inventory.json').read_text())
    pixel_root = BASE/'pixel-dedup-v1'
    pixel_report = read_record(bind(pixel_root/'report.json'))
    cache_path = bind(pixel_root/'pixel-fingerprints.jsonl')
    if file_sha256(cache_path) != pixel_report['files']['pixel-fingerprints.jsonl']['sha256']:
        raise ValueError('Pixel fingerprint cache changed')
    pixels = {r['image_sha256']: r['pixel_sha256'] for r in read_rows(cache_path)}
    protected = []
    for name in ('protected-v2-image-index-v1', 'protected-blind-image-index-v1', 'protected-qualification-index-v1'):
        folder = ROOT/'data/research/canonical_views_v1'/name
        receipt = json.loads(bind(folder/'receipt.json').read_text())
        path = bind(folder/'images.jsonl')
        if file_sha256(path) != receipt['index_sha256'] or file_sha256(path) != pixel_report['inputs'][str(path)]:
            raise ValueError('Protected reference changed')
        protected.extend({**r, 'id': f'{name}:{r["sample_id"]}', 'pixel_sha256': pixels[r['image_sha256']]} for r in read_rows(path))
    protected.extend({**r, 'id': r['image_path']} for r in read_rows(group_root/'supplemental-protected-fingerprints.jsonl'))
    protected_index = index_rows(protected)
    # Keep historical exclusions monotone; compare to all prior accepted source
    # frames and all cleaned replay, not only the newly selected subset.
    reviewed_path = bind(pixel_root/'reviewed-decisions.jsonl')
    replay_path = bind(pixel_root/'replay-decisions.jsonl')
    for path in (reviewed_path, replay_path):
        if file_sha256(path) != pixel_report['files'][path.name]['sha256']:
            raise ValueError('Prior development decisions changed')
    development = [{**r, 'id': r['frame_id']} for r in read_rows(reviewed_path)]
    replay_dhash_path = bind(BASE/'near-dedup-v1/replay-fingerprints.jsonl')
    # Its dHash values are cross-checked for the 8,748 paths rehashed by the
    # previous audit; remaining values are historical exclusion-only evidence.
    replay_hashes = {r['sample_id']: r for r in read_rows(replay_dhash_path)}
    for row in read_rows(group_root/'decisions.jsonl'):
        if row['kind'] == 'replay' and replay_hashes[row['id']]['perceptual_hash'] != row['perceptual_hash']:
            raise ValueError('Replay dHash mismatch')
    development.extend({**r, 'id': r['sample_id'], 'image_sha256': r['payload_sha256'],
                        'perceptual_hash': replay_hashes[r['sample_id']]['perceptual_hash']} for r in read_rows(replay_path))
    development_index = index_rows(development)
    roots = [ROOT/'data/research/canonical_views_v1', BASE]
    plans = {}
    for path in discover(roots, ['plan.json']):
        plan = read_record(bind(path))
        plans[plan['identity']] = (Path(path), plan)
    policy_path = bind(ROOT/'config/perception/visual_hard_examples_v2_12_run18_run19_full_strict_v1.json')
    policy = json.loads(policy_path.read_text())['bbox_policy']['development']
    inventory, captures, visited = [], [], []
    for path in discover(roots, ['collection-receipt.json']):
        receipt = read_record(bind(path))
        if 'views' not in receipt:
            continue
        plan_path, plan = plans[receipt['plan_identity']]
        for filename, expected_hash in plan['files'].items():
            if file_sha256(bind(plan_path.parent/filename)) != expected_hash:
                raise ValueError('Plan snapshot changed')
        inventory.append({'path': path, 'identity': receipt['identity'], 'plan_identity': receipt['plan_identity'],
                          'mode': receipt['mode'], 'status': receipt['status'], 'map_id': receipt['map_id'],
                          'view_count': receipt['view_count'], 'captured': sum(v['status']=='captured' for v in receipt['views']),
                          'resumed_from_collection_identity': receipt.get('resumed_from_collection_identity'),
                          'annotation_mode': plan.get('annotation_mode'), 'plan_path': str(plan_path)})
        if receipt['view_count'] != len(receipt['views']):
            raise ValueError('Capture view count mismatch')
        for view in receipt['views']:
            visited.append({'collection_identity': receipt['identity'], 'view_id': view['view_id'],
                            'mode': receipt['mode'], 'status': view['status'], 'reason': view.get('reason')})
            if receipt['mode'] != 'pilot' or view['status'] != 'captured':
                continue
            planned = [v for v in plan['pilot_views'] if v['view_id']==view['view_id']]
            if len(planned)!=1 or not pose_close(view['actual_pose'], planned[0]):
                raise ValueError('Pilot source pose mismatch')
            if view['split']!='development' or view['map_id']!=plan['map_id']:
                raise ValueError('Pilot source split mismatch')
            if max(abs(view['rgb_timestamp']-view[k]) for k in ('depth_timestamp', 'truth_timestamp')) > .033334+1e-9:
                raise ValueError('Unsynchronized pilot frame')
            if file_sha256(bind(view['depth_path']))!=view['depth_sha256']:
                raise ValueError('Pilot depth changed')
            fp = fingerprint({'image_path': view['rgb_path'], 'image_sha256': view['image_sha256']})
            bind(view['rgb_path'])
            if fp['perceptual_hash']!=view['perceptual_hash']:
                raise ValueError('Pilot dHash changed')
            adapted = adapt_checked(view, plan)
            if list(fp['size']) != [adapted['image_width'], adapted['image_height']]:
                raise ValueError('Pilot image dimensions differ from truth')
            captures.append({**fp, 'id': receipt['identity']+':'+view['view_id'], 'view_id': view['view_id'],
                             'collection_identity': receipt['identity'], 'collection_path': path,
                             'collection_status': receipt['status'], 'plan_identity': plan['identity'], 'map_id': view['map_id'],
                             'family': view['family'], 'camera_position': view['camera_position'],
                             'source_truth': view['truth'], 'adapted_truth': adapted,
                             'framing_reasons': framing_reasons(adapted, policy),
                             'source_label_mode': plan['label_mode'], 'source_annotation_mode': plan['annotation_mode'],
                             'training_admitted': False})
    print(f'Inventoried {len(inventory)} collections, {len(captures)} captured pilot aliases', flush=True)
    # Prefer complete captures so recovery aliases do not keep an obsolete
    # incomplete receipt as the representative of an identical RGB frame.
    captures.sort(key=lambda r: (r['collection_status']!='complete_pending_review', r['image_sha256'], r['id']))
    seen, decisions, queue = {}, [], []
    retained_index = index_rows([])
    for row in captures:
        reason, hits = None, []
        if row['pixel_sha256'] in seen:
            reason = 'duplicate_canonical_pixels'
            hits = [{'representative': seen[row['pixel_sha256']]}]
        else:
            seen[row['pixel_sha256']] = row['id']
            if row['framing_reasons']:
                reason = 'exclude_entire_frame_framing_policy'
            for role, index in [('protected', protected_index), ('historical_development', development_index), ('canonical_retained', retained_index)]:
                if reason:
                    break
                hits = matches(row, index)
                if hits:
                    reason = 'exclude_'+role+'_exact_or_near'
        decision = {**row, 'screen_decision': reason or 'pending_semantic_review', 'matches': hits}
        decisions.append(decision)
        if reason is None:
            queue.append(decision)
            add_row(retained_index, row)
    queue.sort(key=lambda r: (r['map_id'], bool(r['adapted_truth']['objects']), r['image_sha256']))
    for index, row in enumerate(queue, 1):
        row['review_index'] = index
    save('source-inventory.json', {'collections': inventory, 'visited': visited, 'prior_inventory_sha256': file_sha256(group_root/'source-inventory.json')})
    save('screen-decisions.jsonl', decisions, True)
    save('review-queue.jsonl', queue, True)
    sheets = ROOT/'outputs/research/ml_training_recovery_v1/canonical-increment-review'
    sheets.mkdir(parents=True, exist_ok=True)
    sheet_files = {}
    for start in range(0, len(queue), 12):
        page = Image.new('RGB', (1920, 1600), '#202020')
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(queue[start:start+12]):
            x, y = (offset%3)*640, (offset//3)*400
            with Image.open(row['image_path']) as image:
                page.paste(image.resize((640,360)), (x,y+40))
            objects = row['adapted_truth']['objects']
            draw.text((x+5,y+3), f'{row["review_index"]:03d} {row["map_id"]} {row["view_id"][:9]} | {len(objects)} objects', fill='white')
            draw.text((x+5,y+20), 'T=transformer S=switchgear C=capacitor R=reactor' if objects else 'SOURCE EMPTY: confirm all four classes absent', fill='white')
            for n, obj in enumerate(objects, 1):
                x1,y1,x2,y2=obj['bbox_xyxy']
                box=(x+x1/3,y+40+y1/3,x+x2/3,y+40+y2/3)
                color=COLORS[obj['class_name']]
                draw.rectangle(box,outline=color,width=2)
                draw.text((box[0]+2,max(y+40,box[1]-12)),f'{n}{obj["class_name"][0].upper()}',fill=color)
        path=sheets/f'page-{start//12+1:02}.jpg'
        page.save(path, quality=95)
        sheet_files[str(path)] = file_sha256(path)
    coverage=Counter(c for r in queue for c in {o['class_name'] for o in r['adapted_truth']['objects']})
    coverage['no_target']=sum(not r['adapted_truth']['objects'] for r in queue)
    report={'schema_version':1, 'status':'structural_and_joint_screen_complete_semantic_review_pending',
            'inputs':inputs,'files':files,'sheet_files':sheet_files,'collection_count':len(inventory),
            'captured_pilot_aliases':len(captures),'unique_pilot_pixels':len(seen),
            'screen_decision_counts':dict(Counter(r['screen_decision'] for r in decisions)),
            'review_queue_count':len(queue),'potential_coverage_before_semantic_review':dict(coverage),
            'training_admitted':False,
            'limits':['Calibration and rejected captures were inventoried but not promoted.',
                      'All source annotations are preserved. Explicit visible-extrema adaptation adds one to inclusive maxima and does not establish full-object extent.',
                      'No visual review acceptance is inferred by this script.',
                      'Incomplete collections remain incomplete; individual captured-frame evidence does not release the collection.',
                      'Protected historical world/trajectory lineage gaps remain. Canonical plans do not have an independent random seed.',
                      'Historical replay dHash cache is rechecked against the previous rehashed subset; remaining historical cache values are exclusion-only evidence.',
                      'Mixed annotation revisions, full reference completeness, source isolation and quotas remain training gates.']}
    report['identity']=object_sha256(report)
    write_json(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','files','sheet_files')},indent=2),flush=True)


if __name__=='__main__':
    main()
