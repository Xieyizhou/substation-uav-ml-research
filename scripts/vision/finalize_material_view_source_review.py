"""Import explicit visual observations; never approve data or run training."""
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import sys
from scripts.vision.establish_material_view_candidates import OUT, prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def validate_observations(review, notes, review_sha):
    if notes['source_review_sha256'] != review_sha:
        raise ValueError('stale_review_binding')
    expected = {e['event_id'] for e in review['events']}
    if len(expected) != len(review['events']):
        raise ValueError('duplicate_source_event')
    ids = [d[0] for d in notes['decisions']]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('missing_or_duplicate_decisions')
    frames = {s['source_review_id'] for s in review['sources']}
    if set(notes['frame_checks']) != frames:
        raise ValueError('missing_frame_review')
    if notes['review_type'] != 'AI辅助审核' or not notes.get('reviewed_at'):
        raise ValueError('review_nature_or_time_missing')
    for _, status, reason in notes['decisions']:
        if status not in ('pending', 'content_observed') or not reason.strip():
            raise ValueError('invalid_explicit_decision')
    return {eid: (status, reason) for eid, status, reason in notes['decisions']}


def summarize_frames(review, notes, decisions):
    rows = []
    for s in review['sources']:
        sid = s['source_review_id']
        held = [e['event_id'] for e in review['events']
                if e['source_review_id'] == sid and decisions[e['event_id']][0] == 'pending']
        coverage_gap = 'identity_pending' in notes['frame_checks'][sid]
        rows.append(dict(source_review_id=sid, source_member_id=s['source_member_id'],
                         lineage_id=s['lineage_id'], pending_events=held,
                         frame_check=notes['frame_checks'][sid],
                         status='held_for_evidence' if held or coverage_gap else 'visual_content_reviewed_only',
                         training_ready=False))
    return rows


def main():
    rp = OUT/'source-review.json'; np = OUT/'visual-observations-v1.json'
    r = prior.read(rp); notes = prior.read(np)
    # Revalidate evidence and original collection dependencies, not just the review page.
    prior.verify(r)
    prior.verify(prior.read(OUT/'source-inventory.json'))
    decisions = validate_observations(r, notes, prior.file_sha256(rp))
    sources = {s['source_review_id']: s for s in r['sources']}
    imported = []
    for e in r['events']:
        s = sources[e['source_review_id']]
        identity = s['instance_mapping'][e['runtime_label']]
        if identity['object_id'] != e['object_id'] or identity['category'] != e['truth']['class_name']:
            raise ValueError('instance_identity_conflict')
        status, reason = decisions[e['event_id']]
        imported.append(dict(e, status=status, reason=reason, review_type=notes['review_type'],
            reviewed_at=notes['reviewed_at'], page_sha256=s['page_sha256'],
            complete_truth=s['source_record']['truth'], instance_mapping=s['instance_mapping'],
            source_world_sha256=prior.file_sha256(s['source_world']),
            source_capture_sha256=prior.file_sha256(s['source_capture']),
            pixel_visibility_certified=False, training_admitted=False, promotable=False))
    materials = []
    for s in r['sources']:
        root = ET.parse(s['source_world'])
        bodies = []
        for m in root.findall('.//world/model'):
            for v in m.findall('./link/visual'):
                if v.get('name') in ('body', 'reactor', 'front_panel'):
                    bodies.append(dict(model=m.get('name'), visual=v.get('name'),
                        ambient=v.findtext('material/ambient'), diffuse=v.findtext('material/diffuse'),
                        mapped_target=any(x['object_id']==m.get('name') for x in s['instance_mapping'].values())))
        materials.append(dict(source_review_id=s['source_review_id'], declared_family=s['material_id'], fields=bodies))
    baseline = baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40:
        raise ValueError('baseline_integrity_failure')
    frames = summarize_frames(r, notes, decisions)
    paths = [rp, np, OUT/'source-inventory.json', Path(__file__).resolve(),
             prior.ROOT/'docs/results/ml_material_view_source_review_20260911.md']
    tests = ['tests.test_material_view_source_review', 'tests.test_material_view_design',
             'tests.test_body_material_applicability', 'tests.test_pixel_duplicates']
    result = subprocess.run([sys.executable, '-m', 'unittest', *tests], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stdout + result.stderr)
    paths += [prior.ROOT/(name.replace('.', '/')+'.py') for name in tests]
    prior.frozen(OUT/'source-review-completion-v1.json', dict(
        status='source_review_complete_candidate_build_blocked', decisions=imported, frames=frames,
        decision_counts=dict(Counter(e['status'] for e in imported)),
        frame_counts=dict(Counter(f['status'] for f in frames)), actual_material_fields=materials,
        baseline=baseline, source_images=12, full_truth_labels=54,
        blockers=['Unresolved nonplanned-label content/ownership risks; whole affected images held',
                  'Unboxed blue structures require correspondence confirmation; cabinet_center is an unmapped blue asset in saved world, not proven label omission',
                  'Eleven source images are material variants, not original-control images',
                  'Unresolved historical source poses limit independence exclusion'],
        training_ready=False, training_started=False, new_capture_started=False,
        pixel_visibility_certified=False, whole_repository_tests_claimed=False,
        regression_output=result.stdout+result.stderr,
        earlier_test_invocation_error='Nonexistent tests.test_material_view_diversity_design; corrected to tests.test_material_view_design',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print(Counter(e['status'] for e in imported), Counter(f['status'] for f in frames))
    print('PINNED40_PASS; NO_TRAINING; CANDIDATE_BUILD_BLOCKED')


if __name__ == '__main__':
    main()
