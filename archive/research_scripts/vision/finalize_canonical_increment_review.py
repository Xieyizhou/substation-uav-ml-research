#!/usr/bin/env python3
"""Finalize the specific visually inspected canonical queue, never new reviews."""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from src.vision.canonical.visible_admission import review_frame
from scripts.vision.audit_recovery_reference_groups import read_rows

BASE = ROOT/'data/research/ml_training_recovery_v1'
SOURCE = BASE/'canonical-increment-audit-v1'
OUT = BASE/'canonical-increment-final-v1'
CLASSES = ('transformer','switchgear','capacitor_bank','reactor','no_target')


def coverage(rows):
    counts = Counter(c for r in rows for c in {o['class_name'] for o in r['objects']})
    counts['no_target'] = sum(not r['objects'] for r in rows)
    return {c: counts[c] for c in CLASSES}


def main():
    OUT.mkdir(exist_ok=True)
    inputs, files = {}, {}
    def bind(path):
        path=Path(path)
        inputs[str(path)]=file_sha256(path)
        return path
    def save(name, value, jsonl=False):
        path=OUT/name
        if jsonl:
            path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in value))
        else:
            write_json(path,value)
        files[name]={'path':str(path),'sha256':file_sha256(path)}
    bind(__file__)
    bind(ROOT/'src/vision/canonical/visible_admission.py')
    report=read_record(bind(SOURCE/'report.json'))
    for path,digest in report['inputs'].items():
        if file_sha256(path)!=digest:
            raise ValueError(f'Audited input changed: {path}')
    for spec in report['files'].values():
        if file_sha256(bind(spec['path']))!=spec['sha256']:
            raise ValueError('Audited output changed')
    for path,digest in report['sheet_files'].items():
        if file_sha256(bind(path))!=digest:
            raise ValueError('Inspected contact sheet changed')
    observations=json.loads(bind(SOURCE/'visual-review-observations.json').read_text())
    queue_path=SOURCE/'review-queue.jsonl'
    if file_sha256(queue_path)!=observations['review_queue_sha256']:
        raise ValueError('These visual decisions do not apply to this queue')
    queue=read_rows(queue_path)
    if len(queue)!=226 or {r['review_index'] for r in queue}!=set(range(1,227)):
        raise ValueError('Inspected scope mismatch')
    target=set(observations['target_pass_indices'])
    background=set(observations['background_pass_indices'])
    if target & background or not (target|background)<=set(range(1,227)):
        raise ValueError('Invalid reviewed indices')
    decisions, complete, incomplete = [], [], []
    for row in queue:
        if file_sha256(row['image_path'])!=row['image_sha256']:
            raise ValueError('Reviewed image changed')
        index=row['review_index']
        passed=index in target|background
        objects=row['adapted_truth']['objects']
        if index in target and not objects or index in background and objects:
            raise ValueError('Reviewed target/background disposition mismatch')
        first=(index-1)//12*12+1
        last=min(226,first+11)
        review={'image_sha256':row['image_sha256'],'adapter_identity':row['adapted_truth']['adapter_identity'],
                'all_visible_targets_correct':passed,'no_target_confirmed':index in background,
                'objects':[{'annotation_id':o['annotation_id'],'status':'accepted' if passed else 'unresolved',
                            'visible_extent_confirmed':passed,'class_evidence_confirmed':passed,
                            'framing_status':'accepted' if passed else 'unresolved'} for o in objects]}
        result=review_frame(row['adapted_truth'],review,image_sha256=row['image_sha256'])
        if (result['status']=='semantic_review_passed')!=passed:
            raise ValueError('Semantic review adapter rejected a bound decision')
        disposition=('candidate_pending_other_gates' if row['collection_status']=='complete_pending_review' else 'hold_incomplete_collection') if passed else 'exclude_current_round_unresolved_semantics'
        decisions.append({'id':row['id'],'review_index':index,'view_id':row['view_id'],
                          'collection_identity':row['collection_identity'],'image_sha256':row['image_sha256'],
                          'map_id':row['map_id'],'reviewer':observations['reviewer'],'method':observations['method'],
                          'page_observation':observations['page_observations'][f'{first}-{last}'],
                          'review':review,'semantic_result':result,'current_run_decision':disposition,
                          'collection_status':row['collection_status'],'training_admitted':False})
        if not passed:
            continue
        member={'frame_id':'canonical-'+row['view_id'],'view_id':row['view_id'],'image_sha256':row['image_sha256'],
                'pixel_sha256':row['pixel_sha256'],'perceptual_hash':row['perceptual_hash'],'image_path':row['image_path'],
                'collection':str(Path(row['collection_path']).parent),'collection_identity':row['collection_identity'],
                'collection_status':row['collection_status'],'map_id':row['map_id'],'split':'development',
                'plan_identity':row['plan_identity'],'family':row['family'],'camera_position':row['camera_position'],
                'annotation_revision':'canonical-visible-extrema-v1','adapter_identity':row['adapted_truth']['adapter_identity'],
                'objects':objects,'source_objects':row['source_truth']['objects'],
                'semantic_review_index':index,'training_admitted':False,
                'remaining_gates':['annotation_revision_compatibility','protected_world_and_trajectory_lineage','quota']}
        (complete if row['collection_status']=='complete_pending_review' else incomplete).append(member)
    save('review-decisions.jsonl',decisions,True)
    save('incomplete-collection-hold.jsonl',incomplete,True)
    group_root=BASE/'reference-group-audit-v1'
    prior_report=read_record(bind(group_root/'report.json'))
    prior_path=bind(group_root/'candidate-manifest.json')
    if file_sha256(prior_path)!=prior_report['files']['candidate-manifest.json']['sha256']:
        raise ValueError('Existing candidate manifest changed')
    prior=read_record(prior_path)['selected']
    merged=prior+complete
    if len({r['image_sha256'] for r in merged})!=len(merged) or len({r['pixel_sha256'] for r in merged})!=len(merged):
        raise ValueError('Joint candidate exact duplicate')
    manifest={'schema_version':1,'status':'provisional_candidate_only_not_training_admission',
              'selected':merged,'selected_count':len(merged),'existing_count':len(prior),'canonical_increment_count':len(complete),
              'source_manifest_sha256':file_sha256(prior_path),'review_decisions_sha256':files['review-decisions.jsonl']['sha256'],
              'annotation_revision_compatibility_verified':False,'training_admitted':False}
    manifest['identity']=object_sha256(manifest)
    save('candidate-manifest.json',manifest)

    # Source geometry confirms similarity, not a measured causal attribution or
    # proof that cabinet and switchgear are visually identical.
    geometry=[]
    for map_id in ('simple','medium','complex'):
        plan_path=bind(ROOT/f'data/research/canonical_views_v1/{map_id}-expansion-round1-plan-v1/plan.json')
        plan=read_record(plan_path)
        world=bind(plan_path.parent/'world.sdf')
        if file_sha256(world)!=plan['files']['world.sdf']:
            raise ValueError('Scene geometry evidence changed')
        categories={o['name']:o['category'] for o in plan['objects'] if o['category'] in ('cabinet','switchgear')}
        for model in ET.parse(world).iter('model'):
            name=model.get('name')
            if name not in categories:
                continue
            geometry.append({'map_id':map_id,'object_id':name,'category':categories[name],
                             'world_sha256':file_sha256(world),
                             'visuals':[{'name':v.get('name'),'box_size':v.findtext('geometry/box/size'),
                                         'diffuse':v.findtext('material/diffuse'),'pose':v.findtext('pose')}
                                        for v in model.findall('./link/visual')]})
    save('cabinet-switchgear-geometry-evidence.json',{'status':'appearance_confusability_evidence_not_causal_proof',
                                                   'objects':geometry,'taxonomy_changed':False,'training_admitted':False})
    counts=coverage(merged)
    final={'schema_version':1,'status':'current_round_visual_dispositions_complete','inputs':inputs,'files':files,
           'inspected_frames':len(queue),'semantic_review_passed':len(complete)+len(incomplete),
           'semantic_unresolved_excluded':sum(d['current_run_decision']=='exclude_current_round_unresolved_semantics' for d in decisions),
           'reviewed_target_frames':len(target),'reviewed_background_frames':len(background),
           'new_candidates_from_complete_collections':len(complete),'review_passed_but_incomplete_collection_hold':len(incomplete),
           'new_candidate_coverage':coverage(complete),'incomplete_collection_potential_coverage':coverage(incomplete),
           'existing_candidates':len(prior),'merged_provisional_candidates':len(merged),'coverage_frame_counts':counts,
           'quota_deficits_to_600':{c:max(0,600-counts[c]) for c in CLASSES},
           'training_admitted':False,
           'limits':['This is Codex visual review at contact-sheet resolution, not a human full-resolution annotation audit.',
                     'Unresolved frames are conservatively excluded from this round, not declared wrongly labelled; source images and all source objects are preserved.',
                     'Ordinary cabinets are outside the frozen four-class taxonomy. Their shape resembles switchgear but dimensions/colors differ; visual similarity is a hypothesis for learning difficulty, not measured proof of model error.',
                     'Incomplete pilot collections are held separately even when individual frames pass semantic review.',
                     'Merged candidate counts are provisional: visible-extent/full-object annotation compatibility and historical source/trajectory isolation remain unverified.',
                     'No training entrypoint, official model, taxonomy, collection role or existing framing threshold changed.']}
    final['identity']=object_sha256(final)
    write_json(OUT/'report.json',final)
    print(json.dumps({k:v for k,v in final.items() if k not in ('inputs','files')},indent=2))


if __name__=='__main__':
    main()
