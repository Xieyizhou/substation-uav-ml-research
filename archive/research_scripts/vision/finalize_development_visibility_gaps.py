"""Validate existing decisions and replay evidence; never create approvals."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.record_development_visibility_review import OUT, prior, validate
from scripts.vision.verify_experiment_baseline import verify as baseline_verify


def main():
    paths = [OUT / name for name in ('protocol.json', 'replay-receipt.json', 'evidence.json', 'review.json')]
    p, r, e, review = [prior.read(path) for path in paths]
    for record in (p, r, e, review):
        prior.verify(record)
    validate(e, review['decisions'])
    if len(p['frames']) != 8 or len(r['results']) != 8 or len(e['events']) != 10 or r['not_attempted']:
        raise ValueError('Incomplete scope')
    for f, unit in zip(p['frames'], r['results'], strict=True):
        rp = Path(unit['receipt'])
        c = prior.read(rp)
        prior.verify(c)  # Includes raw streams, worlds, pose, process and loaded-library evidence.
        paths.append(rp)
        if unit['events'] != [x['event_id'] for x in f['events']]:
            raise ValueError('Frame correspondence changed')
        if c['status'] != 'capture_technical_checks_passed' or not c['process_cleanup_complete'] or not c['original_pixel_visibility_certified']:
            raise ValueError('Replay not certified')
        if len(c['records']) != 3 or any(not x['rgb_exact_vs_original'] or x['skew_ms'] > 33.334001 or max(x['historical_deltas'].values(), default=0) > 1 for x in c['records']):
            raise ValueError('Alignment failed')
        for event in f['events']:
            target = next(x for x in e['events'] if x['event_id'] == event['event_id'])
            masks = [x for x in c['targets'] if x['event_id'] == event['event_id']]
            if len(masks) != 3 or any(x['visible_pixels'] != target['visible_pixels'] or x['visible_bbox'] != target['visible_bbox'] for x in masks):
                raise ValueError('Mask evidence changed')
    suites = ['tests.test_development_visibility_review', 'tests.test_physical_lighting_capture',
              'tests.test_unified_lighting_design', 'tests.test_boundary_roundoff', 'tests.test_unified_lighting_evaluation']
    t = subprocess.run([sys.executable, '-m', 'unittest', *suites], capture_output=True, text=True)
    if t.returncode:
        raise ValueError(t.stdout + t.stderr)
    baseline = baseline_verify(prior.ROOT / 'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40:
        raise ValueError('Baseline integrity failure')
    paths += [Path(__file__).resolve(), prior.ROOT / 'docs/results/ml_development_visibility_gaps_20260910.md']
    paths += [prior.ROOT / (s.replace('.', '/') + '.py') for s in suites]
    prior.frozen(OUT / 'completion.json', dict(status='visibility_diagnosis_complete_content_risks_remain',
        source_frames_verified=8, aligned_replays=8, reviewed_targets=10, positive_instance_masks=10,
        insufficient_content_decisions=10, historical_labels_modified=False, metrics_recomputed_with_exclusions=False,
        all_dataset_risks_cleared=False, training_started=False, baseline=baseline, regression_output=t.stdout+t.stderr,
        whole_repository_tests_claimed=False,
        next_priority='Design prospective recognizability/occlusion strata independently of model scores; retain fixed full-set metrics and separately investigate clear-target confidence regression.',
        inputs={str(path): prior.file_sha256(path) for path in paths}))
    print(t.stdout + t.stderr)
    print('COMPLETE: 8 aligned frames, 10 positive masks, 10 insufficient-content decisions; pinned40 passed')


if __name__ == '__main__':
    main()
