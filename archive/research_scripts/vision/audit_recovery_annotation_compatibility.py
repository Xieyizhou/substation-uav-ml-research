#!/usr/bin/env python3
"""Bind annotation-scope evidence and hold an unverified mixed-scope merge."""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record

BASE=ROOT/'data/research/ml_training_recovery_v1'


def box_modes(path):
    return [s.findtext('camera/box_type') for s in ET.parse(path).iter('sensor')
            if s.get('type')=='boundingbox_camera']


def scope_stats(rows):
    objects=[o for r in rows for o in r['objects']]
    return {'frames':len(rows),'objects':len(objects),
            'objects_with_fractional_coordinates':sum(any(v!=int(v) for v in o['bbox_xyxy']) for o in objects),
            'visibility_status':dict(Counter(o.get('visibility_status') for o in objects)),
            'coordinate_conventions':dict(Counter(o.get('bbox_coordinate_convention') for o in objects))}


def main():
    out=BASE/'annotation-compatibility-audit-v1'
    out.mkdir(exist_ok=True)
    inputs,files={},{}
    def bind(path):
        path=Path(path);inputs[str(path)]=file_sha256(path);return path
    def save(name,value):
        path=out/name;write_json(path,value)
        files[name]={'path':str(path),'sha256':file_sha256(path)}
    bind(__file__)
    final_root=BASE/'canonical-increment-final-v1'
    prior_report=read_record(bind(final_root/'report.json'))
    path=bind(final_root/'candidate-manifest.json')
    if file_sha256(path)!=prior_report['files']['candidate-manifest.json']['sha256']:
        raise ValueError('Merged candidate evidence changed')
    merged=read_record(path)
    legacy_path=bind(BASE/'reference-group-audit-v1/candidate-manifest.json')
    legacy=read_record(legacy_path)['selected']
    if merged['selected'][:len(legacy)]!=legacy:
        raise ValueError('Legacy candidate prefix changed')
    additions=merged['selected'][len(legacy):]
    target_additions=[r for r in additions if r['objects']]
    background_additions=[r for r in additions if not r['objects']]
    plans={}
    source=read_record(bind(BASE/'canonical-increment-audit-v1/report.json'))
    inventory_path=bind(BASE/'canonical-increment-audit-v1/source-inventory.json')
    if file_sha256(inventory_path)!=source['files']['source-inventory.json']['sha256']:
        raise ValueError('Canonical inventory changed')
    for collection in json.loads(inventory_path.read_text())['collections']:
        plans[collection['plan_identity']]=Path(collection['plan_path'])
    scopes=[]
    for identity in sorted({r['plan_identity'] for r in additions}):
        plan_path=bind(plans[identity]);plan=read_record(plan_path)
        world=bind(plan_path.parent/'world.sdf')
        if plan['identity']!=identity or file_sha256(world)!=plan['files']['world.sdf']:
            raise ValueError('Canonical plan or world changed')
        scopes.append({'plan_identity':identity,'world_path':str(world),
                       'declared_annotation_mode':plan.get('annotation_mode'),'sensor_box_modes':box_modes(world)})
    current_sensor=bind(ROOT/'simulation/models/x500_research/model.sdf')
    provenance=[]
    for folder in sorted({r['collection'] for r in legacy}):
        receipt_path=bind(Path(folder)/'collection-receipt.json')
        capture=read_record(receipt_path)
        run_path=Path(folder).parent/'run-receipt.json'
        run=json.loads(bind(run_path).read_text()) if run_path.exists() else {}
        provenance.append({'collection':folder,'collection_identity':capture['identity'],
                           'run_receipt_available':run_path.exists(),
                           'scene_package_identity':run.get('scene_package_identity'),
                           'scene_files_declared':bool(run.get('scene_files')),
                           'historical_sensor_mode_verified':False})
    # This is an explicit hold, not a reinterpretation of the old corpus.
    hold={'schema_version':1,'selected':target_additions,'selected_count':len(target_additions),
          'status':'hold_annotation_scope_compatibility','source_manifest_sha256':file_sha256(path),
          'reason':'Visible-extent labels cannot be assumed equivalent to the legacy target extent; a +1 endpoint conversion does not establish that equivalence.',
          'training_admitted':False}
    hold['identity']=object_sha256(hold);save('canonical-scope-hold.json',hold)
    selection={'schema_version':1,'selected':legacy+background_additions,'selected_count':len(legacy)+len(background_additions),
               'status':'existing_candidates_pending_prior_gates','source_manifest_sha256':file_sha256(legacy_path),
               'background_additions_source_sha256':file_sha256(path),
               'training_admitted':False}
    selection['identity']=object_sha256(selection);save('existing-candidate-manifest.json',selection)
    calibration={'schema_version':1,'status':'specified_not_executed','taxonomy_changed':False,
                 'annotation_comparison':{'views':[{'view_id':r['view_id'],'plan_identity':r['plan_identity'],
                                                  'image_path':r['image_path'],'image_sha256':r['image_sha256']}
                                                 for r in additions if r['objects']],
                                          'paired_modes':['full_2d','visible_2d'],
                                          'requirements':['Bind both modes to the same camera pose and scene snapshot.',
                                                          'Match objects by simulator instance identity, preserve all boxes.',
                                                          'Measure extent differences separately from the endpoint coordinate conversion.',
                                                          'Do not infer a full-object box by enlarging a visible fragment.']},
                 'cabinet_switchgear_comparison':{'maps':['simple','medium','complex'],
                                                  'source_categories':['cabinet','switchgear'],
                                                  'target_classes_unchanged':True,
                                                  'requirements':['Bind displayed cabinet and switchgear to source object IDs and source class mapping.',
                                                                  'Use matched distance, angle and lighting; include front and side views.',
                                                                  'Record dimensions and materials separately from class definitions.',
                                                                  'Treat model confusion as unmeasured until a development-only comparison is run.']},
                 'training_admitted':False}
    save('paired-calibration-specification.json',calibration)
    report={'schema_version':1,'status':'mixed_annotation_scope_merge_blocked','inputs':inputs,'files':files,
            'legacy':scope_stats(legacy),'canonical_additions':scope_stats(additions),
            'canonical_sensor_evidence':scopes,'legacy_collection_provenance':provenance,
            'current_legacy_sensor_modes':box_modes(current_sensor),
            'historical_legacy_sensor_mode_verified':False,
            'coordinate_conversion_validity':'Explicit inclusive-to-half-open conversion is versioned for canonical visible pixels only.',
            'target_extent_compatibility_verified':False,
            'candidate_accounting':{'existing_pending_candidates':len(legacy),'background_additions_without_boxes':len(background_additions),
                                    'current_pending_candidates':len(legacy)+len(background_additions),'canonical_scope_hold':len(target_additions),
                                    'previous_provisional_total':len(merged['selected'])},
            'training_admitted':False,
            'limits':['Fractional coordinates are a descriptor, not proof of historical sensor configuration.',
                      'The current full_2d source file does not certify every historical run.',
                      'Existing candidates retain all earlier quota and isolation blockers.',
                      'Cabinet remains outside the four-class taxonomy; appearance similarity is not proof of mislabelling.',
                      'Paired calibration is specified but no simulator capture, inference or training was performed.']}
    report['identity']=object_sha256(report);write_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','legacy','canonical_additions','candidate_accounting','training_admitted')},indent=2))


if __name__=='__main__':main()
