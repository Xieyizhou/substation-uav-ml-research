#!/usr/bin/env python3
"""Record the completed development-side visual review, without relaxing gates.

This is an evidence-bound receipt for the specific 119-image review, not an
automatic visual reviewer. Reruns validate the inspected frame inventory.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json

REPLAY_PAIR_OBSERVATIONS = {
    'complex-7629-00000207': 'Candidate has a dark horizontal wall band; the replay example has a broad gray ground region and a small edge detail. Composition differs.',
    'complex-780117-00000600': 'Candidate shows a dark perimeter wall and tiled ground; replay example shows a broad gray sloping region. Composition differs.',
    'medium-7627-00000231': 'Candidate shows a corner in the perimeter wall; replay example shows a broad gray ground plane. Composition differs.',
    'medium-780114-00001080': 'Candidate contains wall corners and a tiled ground wedge; replay example has a broad gray region. Composition differs.',
    'simple-7631-00000294': 'Candidate has a dark horizontal strip at the bottom; replay example has a gray diagonal triangular region. Composition differs.',
}
TARGET_FRAMES = {f'medium-7614-{n:08}' for n in (60,162,165,174,201,213)}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    base = ROOT / 'data/research/ml_training_recovery_v1'
    images = ROOT / 'outputs/research/ml_training_recovery_v1/near-review'
    output = base / 'near-review-final-v1'
    source = base / 'pixel-dedup-v1/reviewed-decisions.jsonl'
    pixel_report = json.loads((base / 'pixel-dedup-v1/report.json').read_text())
    if file_sha256(source) != pixel_report['files']['reviewed-decisions.jsonl']['sha256']:
        raise ValueError('Pixel decisions changed')
    source_rows = read(source)
    hits = [r for r in source_rows if r['near_matches']]
    sheet_path = images / 'sheet-membership.json'
    sheet_rows = json.loads(sheet_path.read_text())
    if len(hits) != 119 or len(sheet_rows) != 119:
        raise ValueError('This receipt applies only to the inspected 119 frames')
    expected = {(r['frame_id'],r['image_sha256']) for r in hits}
    if expected != {(r['frame_id'],r['image_sha256']) for r in sheet_rows}:
        raise ValueError('Inspected image inventory mismatch')
    if {r['frame_id'] for r in hits if r['target_status']=='complete_taxonomy_targets_visible'} != TARGET_FRAMES:
        raise ValueError('Target review scope changed')
    if {r['frame_id'] for r in hits if all(m['role']=='replay' for m in r['near_matches'])} != set(REPLAY_PAIR_OBSERVATIONS):
        raise ValueError('Replay pair review scope changed')
    positions = {r['frame_id']: r['index'] for r in sheet_rows}
    decisions = []
    for row in hits:
        if file_sha256(row['image_path']) != row['image_sha256']:
            raise ValueError('Reviewed source bytes changed')
        target = row['frame_id'] in TARGET_FRAMES
        protected = any(m['role']=='protected' for m in row['near_matches'])
        observation = ('Same recording with similar tiled-ground, utility-pole and foreground equipment compositions across the six inspected target frames. No protected image was visually opened.' if target else
                       REPLAY_PAIR_OBSERVATIONS.get(row['frame_id'], 'Development thumbnail dominated by sky, perimeter wall and/or tiled ground, sometimes with a colored ground marker or small edge detail. This observation is not a new annotation audit.'))
        decisions.append({**row, 'reviewer': 'codex_development_visual_review',
                          'review_method': 'Every candidate inspected at 480x270 in contact sheets; five development-only example pairs additionally inspected at 640x360',
                          'review_image_index': positions[row['frame_id']],
                          'review_sheet': f'page-{(positions[row["frame_id"]]-1)//12+1:02}.jpg',
                          'visual_observation': observation,
                          'current_run_decision': 'exclude_protected_near_policy' if protected else 'exclude_replay_near_policy',
                          'duplicate_confirmed_by_visual_review': False,
                          'training_admitted': False,
                          'decision_basis': 'Existing dHash64 Hamming <=2 exclusion policy; no threshold exception granted. Pixel inequality and different seeds do not override this policy.'})
    retained = [r for r in source_rows if not r['near_matches']]
    manifest_path = ROOT / 'data/research/visual_hard_examples_v2_12/curated-run18-run19-full-strict-v1/curated-manifest.json'
    originals = {r['image_sha256']:r for r in json.loads(manifest_path.read_text())['selected']}
    selected = [{**originals[r['image_sha256']], 'semantic_review_image_sha256':r['image_sha256'],
                 'pixel_sha256':r['pixel_sha256'], 'training_admitted':False} for r in retained]
    coverage = Counter(c for r in selected for c in {o['class_name'] for o in r['objects']})
    coverage['no_target'] = sum(not r['objects'] for r in selected)
    output.mkdir(exist_ok=True)
    path = output / 'review-decisions.jsonl'
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in decisions))
    manifest = {'schema_version':1,'selected':selected,'selected_count':len(selected),
                'status':'candidate_only_not_training_admission','training_admitted':False,
                'source_manifest_sha256':file_sha256(manifest_path),'review_decisions_sha256':file_sha256(path)}
    manifest['identity']=object_sha256(manifest)
    write_json(output/'candidate-manifest.json',manifest)
    report = {'status':'current_run_disposition_complete', 'inspected_candidates':119,
              'target_frames':6,'background_frames':113,'development_only_example_pairs_inspected':5,
              'decision_counts':dict(Counter(r['current_run_decision'] for r in decisions)),
              'retained_candidates':len(selected),'coverage_frame_counts':dict(coverage),
              'quota_deficits_to_600':{c:max(0,600-coverage[c]) for c in ('transformer','switchgear','capacitor_bank','reactor','no_target')},
              'pending_current_run_dispositions':0,'confirmed_visual_duplicate_count':0,
              'source_decisions_sha256':file_sha256(source), 'sheet_membership_sha256':file_sha256(sheet_path),
              'evidence_files':{str(p):file_sha256(p) for p in sorted(images.glob('*.jpg'))},
              'output_files':{str(p):file_sha256(p) for p in (path,output/'candidate-manifest.json')},
              'training_admitted':False,
              'limits':['Disposition closed by conservative current-run exclusion, not proof every excluded frame is a duplicate.',
                        'Development pair examples are visibly distinct; policy was not relaxed on this basis.',
                        'Protected images were not visually reviewed; only previously computed exclusion fingerprints used.',
                        'The 7258 replay/protected near hits remain outside this 119-frame visual review.',
                        'Historical reference completeness, lineage isolation and final quotas remain pending.',
                        'Original labels and source images remain unchanged.']}
    report['identity']=object_sha256(report)
    write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evidence_files','output_files')},indent=2))


if __name__ == '__main__':
    main()
