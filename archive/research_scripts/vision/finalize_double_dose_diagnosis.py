"""Verify and close this bounded experiment; never start inference or training."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from scripts.vision import evaluate_compensated_double_dose as ev
from scripts.vision import double_dose_pool_fit as fit
from scripts.vision import compare_double_dose_fit as comparison
from scripts.vision import summarize_double_dose_review as reviews

TESTS = ('test_double_dose_fit_comparison', 'test_double_dose_error_review',
         'test_compensated_double_dose_evaluation', 'test_material_member_fit',
         'test_unified_lighting_evaluation', 'test_order_diagnosis', 'test_fixed_budget_diagnosis')


def recompute_existing(path, fn):
    original = ev.prior.read(path); ev.prior.verify(original)
    def capture(destination, body):
        if Path(destination) != path:
            raise ValueError('Unexpected write in read-only recomputation')
        return body
    with patch.object(ev.prior, 'frozen', capture):
        fresh = fn()
    for k, v in fresh.items():
        if original[k] != v:
            raise ValueError(f'Recomputed result mismatch: {path}:{k}')
    return original


def main():
    root = ev.OUT/'evaluation'
    numerical = recompute_existing(root/'summary.json', ev.finish)
    review = recompute_existing(reviews.DEST/'review-summary.json', reviews.main)
    p = fit.freeze()
    recompute_existing(fit.OUT/'summary.json', lambda: fit.finish(p))
    comparison.main()
    if numerical['matching_conflicts'] != 0:
        raise ValueError('Unresolved matching conflict')
    if review['fp_predictions'] != 14 or review['unique_loss_targets'] != 80 or review['all_transition_count'] != 2160:
        raise ValueError('Error inventory changed')
    # This round failed its frozen numerical gates. Do not silently promote it.
    if any(x['passed'] for x in numerical['policy_results'].values()):
        raise ValueError('Outcome changed; report requires explicit re-analysis')
    tests = subprocess.run([sys.executable, '-m', 'unittest', *('tests.'+x for x in TESTS)],
                           cwd=ev.prior.ROOT, capture_output=True, text=True, timeout=120)
    if tests.returncode != 0:
        raise ValueError('Related regression failed: '+tests.stdout+tests.stderr)
    baseline = ev.baseline_verify(ev.prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40:
        raise ValueError('Baseline integrity failure')
    report = ev.prior.ROOT/'docs/results/ml_compensated_double_dose_evaluation_20260911.md'
    paths = [root/'summary.json', reviews.DEST/'evidence.json', reviews.DEST/'review.json',
             reviews.DEST/'review-summary.json', fit.OUT/'summary.json', fit.OUT/'dose-comparison.json',
             Path(__file__).resolve(), report]
    paths += [ev.prior.ROOT/'tests'/f'{t}.py' for t in TESTS]
    for key in ev.KEYS:
        paths += [root/f'{key}.json', fit.OUT/'inference'/f'{key}.json', ev.OUT/'training'/key/'completion.json']
    body = dict(status='bounded_experiment_complete_model_gates_failed_with_named_content_gaps',
        training_units=6, development_inference_units=6, native_pool_inference_units=6,
        native_pool_members=277, review_decisions=94, unknown_content_ids=review['unknown_content_ids'],
        numerical_passed={'V':False, 'VM':False}, matching_conflicts=0,
        selected_candidate=None, baseline=baseline,
        related_tests=dict(modules=list(TESTS), returncode=tests.returncode, output=tests.stdout+tests.stderr),
        existing_summaries_recomputed=True, historical_artifacts_preserved=True,
        priority='Design controlled condition-transfer diagnosis, not another same-pool exposure increase',
        followup_experiment_started=False, unseen_scene_status='sealed',
        input_scope='Frozen development and training-fit dependencies; not a new whole-history supervision audit',
        inputs={str(p):ev.prior.file_sha256(p) for p in paths})
    target=root/'diagnosis-completion.json'
    if target.exists():
        old=ev.prior.read(target); ev.prior.verify(old)
        for k,v in body.items():
            if k!='related_tests' and old[k]!=v: raise ValueError('Completion changed')
        return old
    return ev.prior.frozen(target, body)


if __name__=='__main__':
    print(main()['status'])
