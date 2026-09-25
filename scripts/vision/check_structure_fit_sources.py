"""Additional fail-closed source and output checks; no historical mutation."""
import argparse
from collections import Counter
from pathlib import Path
import math
import re
from PIL import Image
from scripts.vision.structure_fit import OUT, ROOT, read, verify, frozen, file_sha256, pixels, unique, truth_for, scoring, validate_review


def strict_review(evidence, decisions):
    validate_review(evidence, decisions)
    events = unique(evidence['events'], lambda r: r['event_id'])
    for d in decisions:
        if d.get('pixel_visibility_certified') is not False:
            raise ValueError('No pixel-level certification authorized')
        if events[d['event_id']]['group'] == 'negative':
            with Image.open(events[d['event_id']]['image_path']) as im:
                w, h = im.size
            for s in d['structures'].values():
                if not s.get('reason'):
                    raise ValueError('Unnamed structure gap')
                if s['state'] == 'present':
                    x1, y1, x2, y2 = s['roi_xyxy']
                    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                        raise ValueError('Invalid evidence ROI')
                    if not all(s.get(k) for k in ('color_appearance','occlusion','truncation','asset_identity')):
                        raise ValueError('Incomplete structure observation')
        elif d.get('identifiable_content') not in ('clear_body','partial_identifiable','insufficient_or_uncertain'):
            raise ValueError('Missing reactor observation')


def equal_rgb(a, b):
    if pixels(a) != pixels(b):
        raise ValueError('Original dimensions or RGB pixels differ')


def label_correspondence(truth, original, tolerance=1e-4):
    """Tolerate numeric YOLO serialization only, not geometric relabeling."""
    if len(truth) != len(original):
        raise ValueError('Complete label count differs from source')
    used = set(); matched = []
    for t in truth:
        candidates = [i for i, s in enumerate(original) if s['class_name'] == t['class_name']
                      and max(abs(x-y) for x,y in zip(s['bbox_xyxy'],t['bbox_xyxy'])) <= tolerance]
        if len(candidates) != 1 or candidates[0] in used:
            raise ValueError('Label/source identity ambiguous or coordinates differ')
        i = candidates[0]; used.add(i); matched.append(original[i])
    return matched


def validate_mapping(mapping):
    unique(mapping.values(), lambda v: v['object_id'])
    for k,v in mapping.items():
        if not str(k).isdigit() or not v.get('category') or not v.get('object_id'):
            raise ValueError('Unparseable instance mapping')


def strict_unit(record, key, p):
    from scripts.vision.run_structure_fit import validate_unit
    validate_unit(record, key, p)
    counts = Counter(p['models'][key]['draws'])
    members = unique(p['rows'], lambda r:r['member_id'])
    if set(counts)-set(members): raise ValueError('Unknown actual exposure member')
    pool = unique(record['pool'], lambda r:r['member_id'])
    dev = unique(record['development'], lambda r:(r['view_id'],r['variant']))
    pairs = [(pool[mid], truth_for(r), r) for mid,r in members.items()]
    pairs += [(dev[(r['view_id'],r['variant'])],r['truth'],r) for r in p['development']]
    for row, truth, source in pairs:
        if row['image_sha256'] != source['image_sha256'] or row['truth'] != truth:
            raise ValueError('Inference image or full truth changed')
        expected = scoring(truth,row['predictions'],row['low_predictions'])
        if any(row[k] != v for k,v in expected.items()): raise ValueError('Matching result changed')
        if 'member_id' in row and row['actual_exposures'] != counts[row['member_id']]:
            raise ValueError('Actual exposure differs')
    if any(record.get(k) is not False for k in ('optimizer_created','backward_executed','training_validation_executed')):
        raise ValueError('Forbidden training operation')


def trace_sources():
    p = read(OUT/'protocol.json'); verify(p)
    base = ROOT/'data/research/ml_training_recovery_v1'
    bp = base/'trusted-training-base-v1/frozen-ledger.json'
    rp = base/'visual-augmentation-240-v2/frozen-intake-ledger.json'
    inputs = {str(x):file_sha256(x) for x in (OUT/'protocol.json',Path(__file__),bp,rp)}
    bases = unique(read(bp)['entries'], lambda r:'base:'+r['view_id'])
    regular = unique(read(rp)['entries'], lambda r:'candidate:'+r['candidate_id'])
    bridge = {}
    for stage in ('pilot-v1','remaining-positive-v1'):
        path = base/'visual-bridge-supplement-v2'/stage/'review-v1/semantic-review.json'
        inputs[str(path)] = file_sha256(path)
        for r in read(path)['frames']:
            mid = 'bridge:'+r['view_id']
            if mid in bridge: raise ValueError('Duplicate bridge source')
            bridge[mid] = r
    negative = {r['member']['member_id']:r['source'] for r in p['negative']}
    decisions = unique(read(OUT/'review.json')['decisions'],lambda r:r['event_id'])
    inputs[str(OUT/'review.json')] = file_sha256(OUT/'review.json')
    reactor_ids = {r['member']['member_id']:r['event_id'] for r in p['reactors']}
    rows = []; receipts = {}; plans = {}
    for r in p['rows']:
        # Verify the immediate export predecessor as well as the original RGB.
        equal_rgb(r['image_path'], r['source_image_path'])
        if file_sha256(r['source_label_path']) != r['label_sha256']:
            raise ValueError('Export full label changed')
        for k in ('source_image_path','source_label_path'): inputs[r[k]] = file_sha256(r[k])
        source = (bases if r['subset']=='base' else regular if r['subset']=='regular' else bridge if r['subset']=='bridge_positive' else negative)[r['member_id']]
        ip = Path(source.get('source_image_path',source.get('image_path')))
        expected = source.get('source_image_sha256',source.get('image_sha256'))
        if file_sha256(ip) != expected: raise ValueError('Stale original image')
        equal_rgb(ip,r['image_path']); inputs[str(ip)] = expected
        cp = ip.parents[1]/'collection-receipt.json'
        if cp not in receipts: receipts[cp] = read(cp)
        receipt = receipts[cp]; inputs[str(cp)] = file_sha256(cp)
        view = unique(receipt['views'],lambda v:v['view_id'])[ip.parent.name]
        if view['image_sha256'] != expected: raise ValueError('Original receipt image mismatch')
        original = label_correspondence(truth_for(r),view['truth']['objects'])
        checks = receipt.get('collection_checks',{})
        mapping = checks.get('instance_mapping',{}); validate_mapping(mapping)
        pp = ip.parents[2]/'plan/plan.json'; plan = None; gaps = []
        if pp.exists():
            if pp not in plans: plans[pp] = read(pp)
            plan = plans[pp]; inputs[str(pp)] = file_sha256(pp)
            wp = pp.parent/'world.sdf'
            if wp.exists():
                digest = file_sha256(wp); inputs[str(wp)] = digest
                if plan.get('files',{}).get('world.sdf') not in (None,digest) or receipt.get('world_sha256') not in (None,digest):
                    raise ValueError('World identity conflict')
            for name in ('sensor_source.sdf',):
                sensor = pp.parent/name
                if sensor.exists(): inputs[str(sensor)] = file_sha256(sensor)
        else: gaps.append('original_plan_path_not_available_at_capture_plan_location')
        annotations = []
        for t,s in zip(truth_for(r),original):
            found = re.search(r'instance-(\d+)-',s['annotation_id'])
            runtime = str(int(found.group(1))) if found else None
            obj = mapping.get(runtime)
            if obj and obj['category'] != t['class_name']: raise ValueError('Instance category conflict')
            if mapping and not obj: raise ValueError('Runtime instance not uniquely resolved')
            annotations.append(dict(local_annotation_id=t['annotation_id'],source_annotation_id=s['annotation_id'],
                class_name=t['class_name'],runtime_label=runtime,object_id=obj['object_id'] if obj else 'unknown',
                identity_status='receipt_mapping_resolved' if obj else 'runtime_only_no_cross_image_instance_claim',
                historical_truncation=s.get('truncation_status','unknown')))
        if original and not mapping: gaps.append('historical_receipt_lacks_instance_mapping')
        if not receipt.get('actual_annotation_mode'): gaps.append('historical_actual_annotation_mode_not_recorded')
        entry = dict(member_id=r['member_id'],subset=r['subset'],source_image=str(ip),source_image_sha256=expected,
            original_rgb_identity=pixels(ip),export_file_identity_different=expected!=r['image_sha256'],full_labels_correspond=True,
            source_view_id=ip.parent.name,source_map=receipt.get('map_id','unknown'),recording=str(ip.parents[2]),
            actual_pose=view.get('actual_pose'),actual_annotation_mode=receipt.get('actual_annotation_mode','unknown'),
            registered_lineage=r['lineage_id'],source_derivation_group=source.get('derivation_group','unknown'),annotations=annotations,gaps=gaps)
        if r['member_id'] in reactor_ids:
            eid = reactor_ids[r['member_id']];d = decisions[eid]
            t = next(t for t in truth_for(r) if t['class_name']=='reactor')
            with Image.open(ip) as im: w,h=im.size
            b=t['bbox_xyxy'];bw=(b[2]-b[0])*640/max(w,h);bh=(b[3]-b[1])*640/max(w,h)
            entry['reactor'] = dict(event_id=eid,width_640=bw,height_640=bh,size_bucket='lt32' if min(bw,bh)<32 else '32to64' if min(bw,bh)<64 else 'ge64',
                identifiable_content=d['identifiable_content'],new_truncation_observation=d['truncation'],
                historical_truncation=next(a['historical_truncation'] for a in annotations if a['class_name']=='reactor'),
                possible_metadata_erratum=d['truncation']!='none' and next(a['historical_truncation'] for a in annotations if a['class_name']=='reactor')=='not_truncated')
        rows.append(entry)
    unique(rows,lambda r:r['member_id'])
    frozen(OUT/'member-source-trace.json',dict(status='all_236_original_RGB_and_full_labels_verified_named_metadata_gaps_retained',rows=rows,inputs=inputs))
    print('TRACE_COMPLETE',len(rows),'gap_members',sum(bool(r['gaps']) for r in rows),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trace',action='store_true');args=parser.parse_args()
    if args.trace: trace_sources()
    else: print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
