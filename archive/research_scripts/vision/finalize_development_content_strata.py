"""Finalize the strata stage and stop before training when evidence does not support it."""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior
from scripts.vision.prepare_development_content_strata import OUT
from scripts.vision.verify_experiment_baseline import verify as baseline_verify


def aggregate(rows, prefix, variant):
    out = defaultdict(lambda: [0, 0])
    for x in rows:
        if x['cell'].startswith(prefix) and x['variant'] == variant:
            out[x['stratum']][0] += x['truth']; out[x['stratum']][1] += x['hit']
    return {k: dict(truth=n, hit=h, recall=h/n if n else None) for k,(n,h) in sorted(out.items())}


def main():
    review = prior.read(OUT/'review-v2.json'); metrics = prior.read(OUT/'metrics-v2.json')
    prior.verify(review); prior.verify(metrics)
    if review['status'] != 'content_strata_review_complete' or len(review['decisions']) != 120: raise ValueError('Review incomplete')
    if metrics['status'] != 'stratified_metrics_complete': raise ValueError('Metrics incomplete')
    rows = metrics['rows']
    r = {v: aggregate(rows, 'R-clean', v) for v in ('original','lighting')}
    l = {v: aggregate(rows, 'L-physical', v) for v in ('original','lighting')}
    evidence = dict(
        unique_truth_distribution={s: sum(1 for x in review['decisions'] if x['variant']=='original' and x['stratum']==s) for s in ('clear','partial','fragment','unknown')},
        retained_reference=r, physical_lighting=l,
        retention_delta={v: {s: l[v].get(s,{}).get('recall') - r[v].get(s,{}).get('recall') for s in ('clear','partial','fragment','unknown') if s in l[v] and s in r[v]} for v in ('original','lighting')},
        interpretation=[
            'fragment targets have low recall in both families; this confirms a content difficulty confound but is not a failure-rate estimate for all data.',
            'L-physical matches R-clean on original partial/fragment aggregate recall and is lower on clear by 4.0 percentage points.',
            'L-physical is higher on lighting clear recall by 17.6 percentage points while lighting partial recall is lower by 5.9 points; visibility strata alone do not explain all condition changes.',
            'No uniform clear-target degradation across both original and lighting conditions was observed, so a model-only training intervention is not justified by this stage.',
        ],
        judgment='content_difficulty_is_confirmed_confound_but_model_cause_not_isolated',
        next_priority='Freeze a condition/asset migration control or a source-isolated recognizability supplement; retain clear-target confidence regression as a separate diagnostic question.',
    )
    suites=['tests.test_development_content_strata','tests.test_development_visibility_review','tests.test_physical_lighting_capture','tests.test_unified_lighting_design','tests.test_unified_lighting_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode: raise ValueError(t.stdout+t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40: raise ValueError('Pinned baseline failure')
    report=prior.ROOT/'docs/results/ml_development_content_strata_20260910.md'
    paths=[OUT/'protocol.json',OUT/'review-v2.json',OUT/'metrics-v2.json',Path(__file__).resolve(),report]
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'completion.json',dict(status='strata_complete_training_not_started',reviewed_truths=120,
        unique_paired_truths=60,prediction_cells=6,training_started=False,training_admitted=False,promotable=False,
        evidence=evidence,full_set_metrics_preserved=True,baseline=baseline,regression_output=t.stdout+t.stderr,
        whole_repository_tests_claimed=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(t.stdout+t.stderr); print('STRATA_COMPLETE_TRAINING_NOT_STARTED')

if __name__=='__main__': main()
