"""Read-only source audit. Source integrity never certifies instance visibility."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_supervision_preservation import OUT as PRIOR, ROOT, REFERENCE, read, save, file_sha256, verify_tree, baseline_verify
from scripts.vision.prepare_visual_bridge_training import BASE, ADMISSION, yolo_lines
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record, pose_close
from src.vision.canonical.gates import validate_preflight, validate_point, target_checks

OUT = PRIOR / 'bridge-source-trace-v1'

def require(condition, reason):
    if not condition: raise ValueError(reason)

def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, 'Duplicate identity: ' + row[key])
        result[row[key]] = row
    return result

def pixels_equal(first, second):
    with Image.open(first) as a, Image.open(second) as b:
        return a.size == b.size and a.convert('RGB').tobytes() == b.convert('RGB').tobytes()

def check_sync(capture):
    times = [capture[k] for k in ('rgb_timestamp', 'depth_timestamp', 'truth_timestamp')]
    times.append(capture['actual_pose']['timestamp'])
    require(all(math.isfinite(x) for x in times), 'Nonfinite timestamp')
    require(max(abs(t-times[0]) for t in times) <= .033334+1e-9, 'Synchronization failed')

def objects_with_instances(truth, mapping):
    result = []
    for obj in truth['objects']:
        match = re.search(r'-instance-(\d+)-box-', obj['annotation_id'])
        require(match is not None, 'Unresolved annotation instance')
        label = int(match[1]); require(label in mapping, 'Unmapped annotation instance')
        require(mapping[label]['category'] == obj['class_name'], 'Instance/class conflict')
        box = obj['bbox_xyxy']; w,h = truth['image_width'],truth['image_height']
        result.append(dict(instance_label=label, device_id=mapping[label]['object_id'],
            category=obj['class_name'], bbox_xyxy=box,
            boundary_contact_within_one_pixel=box[0]<=1 or box[1]<=1 or box[2]>=w-1 or box[3]>=h-1,
            visibility_status='unknown', component_status='unknown'))
    require(len({o['instance_label'] for o in result}) == len(result), 'Duplicate instance boxes')
    return result

def main():
    dest = OUT/'trace.json'
    if dest.exists():
        verify_tree(dest); print('VERIFIED_EXISTING', dest); return
    ledger_path = BASE/'visual-bridge-supplement-v2/frozen-positive-ledger.json'
    paths = [PRIOR/'diagnosis.json', REFERENCE/'protocol.json', ADMISSION, ledger_path,
             Path(__file__), ROOT/'tests/test_retained_bridge_trace.py',
             ROOT/'scripts/vision/prepare_visual_bridge_training.py', ROOT/'src/vision/canonical/gates.py', ROOT/'src/vision/canonical/plan.py']
    seen=set()
    for path in paths[:4]: verify_tree(path,seen)
    members = [r for r in read(REFERENCE/'protocol.json')['pool_rows'] if r['subset']=='bridge_positive']
    require(len(members)==39 and len({r['lineage_id'] for r in members})==13, 'Retained scope changed')
    admissions=unique(read(ADMISSION)['entries'],'member_id'); ledger=unique(read(ledger_path)['frames'],'view_id')
    inputs={str(p):file_sha256(p) for p in paths}; results=[]
    def bind(path, expected=None):
        path=Path(path); digest=file_sha256(path)
        require(expected is None or digest==expected, 'Stale bytes: '+str(path))
        inputs[str(path)]=digest
    for member in sorted(members,key=lambda r:r['member_id']):
        result=dict(member_id=member['member_id'], lineage_id=member['lineage_id'], replay_executed=False,
                    visibility_status='unknown', training_admitted=False, promotable=False)
        try:
            admission=admissions[member['member_id']]; frame=ledger[admission['view_id']]
            result.update(view_id=frame['view_id'],variant=frame['variant'],category=frame['category'])
            require(member['lineage_id']==admission['derivation_group']==frame['derivation_group'], 'Lineage conflict')
            rp=Path(frame['receipt_path']); pp=rp.parent.parent/'plan/plan.json'
            bind(rp);bind(pp); receipt=read_record(rp);plan=read_record(pp)
            require(receipt['identity']==frame['receipt_identity'], 'Receipt identity conflict')
            require(plan['identity']==frame['plan_identity']==receipt['plan_identity'], 'Plan identity conflict')
            for name,digest in plan['files'].items(): bind(pp.parent/name,digest)
            view=unique(plan['calibration_views']+plan['pilot_views'],'view_id')[frame['view_id']]
            capture=unique(receipt['views'],'view_id')[frame['view_id']]
            checks,config,mapping=validate_preflight(plan,pp.parent,[view])
            require(checks==receipt['collection_checks'], 'Saved/current preflight conflict')
            require(checks['world_sha256']==frame['world_sha256'], 'World conflict')
            require(plan['files']['sensor_source.sdf']==frame['sensor_sha256'], 'Sensor conflict')
            require(capture['status']=='captured', 'Capture incomplete')
            for kind in ('image','label'): bind(member[kind+'_path'],member[kind+'_sha256'])
            bind(capture['rgb_path'],capture['image_sha256']);bind(capture['depth_path'],capture['depth_sha256'])
            require(capture['image_sha256']==admission['image_sha256']==frame['image_sha256'], 'RGB identity conflict')
            require(pixels_equal(member['image_path'],capture['rgb_path']), 'Training/source RGB differs')
            require(object_sha256(capture['truth'])==admission['label_sha256']==frame['label_sha256'], 'Truth identity conflict')
            require(Path(member['label_path']).read_text()==yolo_lines(capture['truth']), 'Full label export differs')
            objects=objects_with_instances(capture['truth'],mapping)
            require(dict(Counter(o['category'] for o in objects))==member['class_instances'], 'Class count differs')
            target=target_checks(view,capture['raw_truth'],mapping)
            require(target==capture['target_checks'] and target['planned_instance_present'] is True, 'Target check conflict')
            require(sorted(target['observed_instance_labels'])==sorted(o['instance_label'] for o in objects), 'Raw/parsed instance conflict')
            actual=capture['actual_pose']; require(pose_close(actual,view), 'Pose tolerance failed')
            validate_point(actual['position'],config,role='actual carrier')
            optical=[actual['position'][i]+view['camera_position'][i]-view['position'][i] for i in range(3)]
            validate_point(optical,config,role='actual optical center');check_sync(capture)
            result.update(status='source_verified_replay_pending', receipt_path=str(rp),plan_path=str(pp),
                source_rgb=capture['rgb_path'],training_image=member['image_path'],objects=objects,
                actual_pose={k:actual[k] for k in ('position','orientation','timestamp')},
                instance_mapping=checks['instance_mapping'],original_review=frame['review'],
                stability_status='historical_collector_assertion_only_fresh_three_frame_replay_required',
                full_frame_visibility_review='not_certified_by_source_audit')
        except (KeyError,ValueError,OSError) as error:
            result.update(status='source_blocked',reason=f'{type(error).__name__}: {error}')
        results.append(result)
    groups=defaultdict(list)
    for row in results:groups[row['lineage_id']].append(row['member_id'])
    save(dest,dict(status='source_trace_complete_not_visibility_certification',frames=results,groups=dict(groups),
        frame_count=len(results),independent_registered_lineages=len(groups),
        status_counts=dict(Counter(r['status'] for r in results)),
        boundary_contact_boxes=sum(o['boundary_contact_within_one_pixel'] for r in results for o in r.get('objects',[])),
        inputs=inputs,baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        next_step='Replay a deterministic pilot from verified source groups, then review every instance; preserve original labels and all three variants.',
        limitations=['No new replay, collection, visual decisions or training.',
            'Boundary proximity is a review flag, not proof of truncation.',
            'Source equality does not establish sufficient visible content; no eligibility is granted.']))
    print('TRACE_COMPLETE',dest,Counter(r['status'] for r in results))

if __name__=='__main__':main()
