"""Independent identity repair and material candidate gate. Default is read-only."""
import argparse
import re
from pathlib import Path
from collections import Counter
from PIL import Image
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior, paired_truth
from scripts.vision.prepare_development_content_strata import OUT as STRATA
from scripts.vision.prepare_paired_visual_factors import BASE, assert_allowed_world_diff
from src.vision.canonical.gates import instance_mapping, annotation_mode_from_world
from src.ml.artifacts import object_sha256

OUT = RUN.parents[1] / 'material-control-feasibility-v1'
CANDIDATE = OUT.parent / 'closed-body-material-control-v1'


def resolve_target(truth, mapping):
    match = re.search(r'-instance-(\d+)-', truth['annotation_id'])
    if not match: raise ValueError('Unresolved runtime identity')
    label = int(match.group(1))
    if len({v['object_id'] for v in mapping.values()}) != len(mapping):
        raise ValueError('Scene identity collision')
    item = mapping.get(label)
    if item is None or item['category'] != truth['class_name']:
        raise ValueError('Unresolved target or category conflict')
    return dict(runtime_label=label, object_id=item['object_id'], category=item['category'])


def check_candidate_frames(frames):
    keys = [(x['source_review_id'], x['variant']) for x in frames]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate candidate frame')
    return [dict(source=x['source_review_id'], variant=x['variant'],
                 held_labels=x['held_labels'], unboxed_visible_instances=x['unboxed_visible_instances'])
            for x in frames if x['status'] != 'reviewed_candidate_only'
            or x['held_labels'] or x['unboxed_visible_instances']]


def run(write=False):
    paths = [RUN/'protocol.json', STRATA/'review-v2.json', STRATA/'metrics-v2.json']
    config, review, metric = map(prior.read, paths)
    for r in (config, review, metric): prior.verify(r)
    semantic_path = Path(config['evaluation']['paired_review'])
    semantic = prior.read(semantic_path); prior.verify(semantic); paths.append(semantic_path)
    pairs, raw_inputs = paired_truth(semantic['frames']); paths.extend(map(Path, raw_inputs))
    collections = {v: prior.read(next(p for p in raw_inputs if f'/{v}/' in p))
                   for v in ('original','material','background','lighting')}
    mappings = {}
    for v in collections:
        pp = BASE/v/'plan/plan.json'; wp = pp.parent/'world.sdf'
        plan = prior.read(pp); mapping = instance_mapping(plan)
        if {str(k): x for k,x in mapping.items()} != collections[v]['collection_checks']['instance_mapping']:
            raise ValueError('World/collection instance mapping mismatch')
        if prior.file_sha256(wp) != collections[v]['world_sha256'] or annotation_mode_from_world(wp) != 'full_2d':
            raise ValueError('Actual world identity/mode mismatch')
        assert_allowed_world_diff(BASE/'original/plan/world.sdf', wp, v)
        mappings[v] = mapping; paths += [pp, wp]
    old = {(x['view_id'],x['variant'],x['instance_id']):x for x in review['decisions']}
    rows = []; wrong_ids = 0; wrong_crops = 0
    if write: (OUT/'crops').mkdir(parents=True, exist_ok=True)
    for frame, truths in pairs:
        ip = Path(frame['image_path'])
        if prior.file_sha256(ip) != frame['image_sha256']: raise ValueError('Stale RGB')
        paths.append(ip)
        im = Image.open(ip).convert('RGB') if write else None
        for truth in truths:
            identity = resolve_target(truth, mappings[frame['variant']])
            key = (frame['view_id'],frame['variant'],identity['runtime_label'])
            previous = old.get(key)
            if previous:
                wrong_ids += previous['object_id'] != identity['object_id']
                wrong_crops += truth['bbox_xyxy'] != frame['target_bbox_xyxy']
            eid = f"{frame['view_id']}-{frame['variant']}-{identity['object_id']}"
            crop = OUT/'crops'/f'{eid}.png'
            if write:
                im.crop(truth['bbox_xyxy']).save(crop); paths.append(crop)
            rows.append(dict(evidence_id=eid, pair_id=frame['pair_id'],view_id=frame['view_id'],
                variant=frame['variant'], **identity, truth=truth, image_path=str(ip),
                image_sha256=frame['image_sha256'], full_truth_sha256=frame['truth_sha256'],
                crop_path=str(crop) if write else None, crop_sha256=prior.file_sha256(crop) if write else None,
                review_status='pending_individual_review', original_stratum=previous['stratum'] if previous else None,
                visual_certification_transferred=False))
    if len(rows)!=240 or len({x['evidence_id'] for x in rows})!=240:
        raise ValueError('Expected 240 unique target-condition records')
    candidate_paths = [CANDIDATE/n for n in ('semantic-review.json','review-completion.json','mask-coverage.json')]
    for p in candidate_paths: prior.verify(prior.read(p))
    paths += candidate_paths
    candidate = prior.read(candidate_paths[0]); blockers=check_candidate_frames(candidate['frames'])
    # Confirm the legacy integer-key serialization explanation against the exact recorded digest.
    legacy_path = OUT.parent/'reviewed-hold-compensation-control-v1/protocol.json'
    legacy = prior.read(legacy_path); stored = legacy.pop('identity'); loaded = object_sha256(legacy)
    for f in legacy['frames']: f['instance_mapping'] = {int(k):v for k,v in f['instance_mapping'].items()}
    rebuilt = object_sha256(legacy)
    if rebuilt != stored: raise ValueError('Historical identity discrepancy remains unexplained')
    paths.append(legacy_path)
    exposure = {}
    for family in ('R-clean','L-physical'):
        for seed in (7,17,27):
            key = f'{family}-{seed}'; cp = RUN/'training'/key/'completion.json'
            c = prior.read(cp); prior.verify(c); xp = Path(c['exposure_path']); x=prior.read(xp);prior.verify(x)
            if x['draws'] != config['schedules'][key]: raise ValueError('Actual exposure order differs')
            exposure[key] = dict(draws=len(x['draws']), unique_members=len(set(x['draws'])),
                                by_member=dict(Counter(x['draws'])), summary=x['summary'])
            paths += [cp,xp]
    result = dict(status='blocked_at_material_candidate_quality_gate' if blockers else 'identity_trace_complete_visual_review_pending',
        source_frames=48, source_truths=240, corrected_target_records=rows,
        prior_wrong_object_id_records=wrong_ids, prior_wrong_target_crop_records=wrong_crops,
        audit_dependency_bug=dict(consumed=str(STRATA/'review-v2.json'), bound_in_old_metrics=str(STRATA/'review.json')),
        historical_serialization=dict(stored=stored,json_loaded=loaded,integer_keys_reconstructed=rebuilt),
        candidate_frames=candidate['frames'], blockers=blockers, actual_exposure=exposure,
        full_plan_complete=False, new_visual_reviews_complete=False, new_candidate_selected=False,
        training_started=False, next_priority='Independent candidate supervision/data handling proposal before material training.',
        deferred=['four-condition individual visual review','new stratified evaluation','full material coverage census','training replacement protocol'],
        inputs={str(p):prior.file_sha256(p) for p in paths+[Path(__file__).resolve()]})
    if write: prior.frozen(OUT/'initial-gate.json', result)
    print(result['status'], 'targets',len(rows),'prior_wrong_ids',wrong_ids,'prior_wrong_crops',wrong_crops,'held_candidates',len(blockers))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args();run(args.freeze)
