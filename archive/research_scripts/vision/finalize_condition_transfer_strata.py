"""Finalize the four-condition strata transfer diagnostic; no training."""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import prior
from scripts.vision.prepare_development_content_strata import OUT
from scripts.vision.verify_experiment_baseline import verify as baseline_verify


def agg(rows, family, variant, stratum='clear'):
    n = h = 0
    for x in rows:
        if x['cell'].startswith(family) and x['variant'] == variant and x['stratum'] == stratum:
            n += x['truth']; h += x['hit']
    return dict(hit=h, truth=n, recall=h/n if n else None)


def main():
    p = OUT/'condition-transfer-metrics.json'; m = prior.read(p); prior.verify(m)
    if m['status'] != 'condition_transfer_strata_complete' or len(m['rows']) != 288:
        raise ValueError('Condition strata incomplete')
    clear = {family: {variant: agg(m['rows'], family, variant) for variant in m['variants']}
             for family in ('R-clean','L-physical')}
    summary = dict(clear=clear, material_clear_delta={family: clear[family]['material']['recall']-clear[family]['original']['recall'] for family in clear},
        background_clear_delta={family: clear[family]['background']['recall']-clear[family]['original']['recall'] for family in clear},
        lighting_clear_delta={family: clear[family]['lighting']['recall']-clear[family]['original']['recall'] for family in clear},
        judgment='material_condition_is_the_strongest_replicated_development_failure; geometry_visibility_does_not_explain_it',
        limits='One complex canonical layout, four existing conditions, six trained cells; no new scene or asset generalization claim.',
        next_priority='Freeze a source-isolated material-coverage control using existing reviewed variants; keep original/background/lighting retention gates fixed.')
    suites=['tests.test_condition_transfer_strata','tests.test_development_content_strata','tests.test_development_visibility_review','tests.test_unified_lighting_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode: raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified'] != 40: raise ValueError('Pinned baseline failure')
    report=prior.ROOT/'docs/results/ml_condition_transfer_strata_20260911.md'
    paths=[p,Path(__file__).resolve(),report]+[prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'condition-transfer-completion.json',dict(status='condition_transfer_diagnostic_complete_material_priority',
        training_started=False,training_admitted=False,promotable=False,summary=summary,baseline=b,
        regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(t.stdout+t.stderr);print('CONDITION_TRANSFER_DIAGNOSTIC_COMPLETE_MATERIAL_PRIORITY')

if __name__=='__main__':main()
