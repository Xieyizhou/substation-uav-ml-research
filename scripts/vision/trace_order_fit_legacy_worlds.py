"""Post-hoc source/world identity, not replay or pixel visibility certification."""
from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.test_body_material_applicability import source_index, model_mapping
from scripts.vision.check_structure_fit_sources import equal_rgb, label_correspondence


def resolve_labels(truth, original, mapping):
    corresponding = label_correspondence(truth, original)
    result = []
    seen = set()
    for current, source in zip(truth, corresponding):
        match = re.search(r'instance-(\d+)-', source['annotation_id'])
        runtime = str(int(match[1])) if match else None
        if runtime not in mapping:
            raise ValueError('saved_world_instance_unresolved')
        obj = mapping[runtime]
        if obj in seen:
            raise ValueError('multiple_full_labels_resolve_to_one_device')
        seen.add(obj)
        result.append(dict(annotation_id=current['annotation_id'], class_name=current['class_name'],
            bbox_xyxy=current['bbox_xyxy'], source_annotation_id=source['annotation_id'],
            runtime_label=runtime, object_id=obj))
    return result


def run():
    dest = OUT / 'legacy-world-source-trace.json'
    if dest.exists():
        record = prior.read(dest); prior.verify(record); return record
    protocol_path = OUT / 'protocol.json'
    p = prior.read(protocol_path); prior.verify(p)
    sources, paths = source_index()
    deps = list(paths) + [protocol_path, Path(__file__).resolve()]
    rows = []
    for m in p['members']:
        if m['member_id'] not in sources:
            continue
        source = sources[m['member_id']]
        row = dict(member_id=m['member_id'], post_hoc_only=True,
            historical_new_gate_pass_claimed=False, pixel_visibility_certified=False,
            training_eligible=False, gaps=[])
        try:
            ip = Path(source.get('source_image_path', source.get('image_path')))
            expected = source.get('source_image_sha256', source.get('image_sha256'))
            if prior.file_sha256(ip) != expected:
                raise ValueError('original_rgb_hash_changed')
            deps.extend([ip, Path(m['image_path']), Path(m['label_path'])])
            if prior.file_sha256(m['image_path']) != m['image_sha256'] or prior.file_sha256(m['label_path']) != m['label_sha256']:
                raise ValueError('current_member_hash_changed')
            equal_rgb(ip, m['image_path'])
            cp = ip.parents[1] / 'collection-receipt.json'
            pp = ip.parents[2] / 'plan/plan.json'
            wp = pp.parent / 'world.sdf'
            deps.extend([cp, pp, wp])
            receipt, plan = prior.read(cp), prior.read(pp)
            world_hash = prior.file_sha256(wp)
            if plan.get('files', {}).get('world.sdf') != world_hash:
                raise ValueError('plan_world_hash_conflict')
            if receipt.get('world_sha256') not in (None, world_hash):
                raise ValueError('receipt_world_hash_conflict')
            views = [v for v in receipt['views'] if v['view_id'] == ip.parent.name]
            if len(views) != 1 or views[0]['image_sha256'] != expected:
                raise ValueError('nonunique_or_stale_receipt_view')
            view = views[0]
            tree = ET.parse(wp); mapping, models = model_mapping(tree)
            annotations = resolve_labels(m['truth'], view['truth']['objects'], mapping)
            inventory = []
            for name, model in models.items():
                if model.find('.//visual[@name="body"]') is not None:
                    inventory.append(dict(object_id=name, pose=model.findtext('pose'),
                        labels=sorted(k for k, v in mapping.items() if v == name),
                        visual_components=[v.get('name') for v in model.findall('.//visual')]))
            if not receipt.get('collection_checks', {}).get('instance_mapping'):
                row['gaps'].append('historical_receipt_mapping_missing_world_mapping_is_post_hoc')
            if not receipt.get('actual_annotation_mode'):
                row['gaps'].append('historical_actual_mode_not_recorded')
            actual = view.get('actual_pose') or {}
            row.update(source_image=str(ip), source_receipt=str(cp), source_plan=str(pp),
                source_world=str(wp), world_sha256=world_hash,
                actual_pose={k: actual[k] for k in ('position', 'orientation', 'timestamp') if k in actual},
                saved_box_modes=[x.text for x in tree.findall('.//box_type')],
                annotations=annotations, body_model_inventory=inventory,
                status='original_rgb_full_labels_and_saved_world_instances_correspond',
                source_recording=str(ip.parents[2]), source_view_id=ip.parent.name)
        except (ValueError, KeyError, FileNotFoundError, ET.ParseError) as exc:
            row['status'] = 'blocked_source_correspondence'
            row['gaps'].append(str(exc))
        rows.append(row)
    return prior.frozen(dest, dict(status='post_hoc_legacy_trace_not_quality_admission', members=rows,
        status_counts=dict(Counter(r['status'] for r in rows)),
        interpretation='Saved world mapping resolves labelled devices only. Unlabelled image structures require separate spatial evidence; missing labels do not prove background semantics.',
        inputs={str(x): prior.file_sha256(x) for x in deps if x.is_file()}))


if __name__ == '__main__':
    print(run()['status_counts'])
