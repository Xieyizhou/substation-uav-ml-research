"""Bind bounded audit findings without repairing historical artifacts."""
import copy,subprocess,sys
from pathlib import Path
from scripts.vision.audit_historical_results import OUT,CURRENT,prior,lr,object_sha256

def main():
    rp=OUT/'results.json';bp=OUT/'baseline-paired-audit.json'
    for path in (rp,bp):prior.verify(prior.read(path))
    old=prior.ROOT/'data/research/ml_training_recovery_v1/paired-visual-factors-v1/evaluation.json'
    r=prior.read(old);identity=r.pop('identity');loaded=object_sha256(r);rebuilt=copy.deepcopy(r)
    for row in rebuilt['development_weight_volatility'].values():row['values']={int(k):v for k,v in row['values'].items()}
    reconstructed=object_sha256(rebuilt)
    if reconstructed!=identity:raise ValueError('Legacy identity explanation failed')
    a,b=prior.read(lr.OUT/'protocol.json'),prior.read(lr.SOURCE/'protocol.json')
    controls={field:a[field]==b[field] for field in ('schedules','brightness_factors','initialization')}
    if not all(controls.values()):raise ValueError('LR-only premise not supported')
    docs=[prior.ROOT/'docs/results'/name for name in ('ml_appearance_background_erratum_20260907.md','ml_project_data_assessment_20260907.md','ml_brightness_lr_retention_20260910.md','ml_unified_lighting_evaluation_20260910.md')]
    paths=[old,rp,bp,lr.OUT/'protocol.json',lr.SOURCE/'protocol.json',CURRENT/'error-review/review.json',Path(__file__).resolve()]+docs
    prior.frozen(OUT/'findings.json',dict(status='bounded_findings_supported',legacy_identity=dict(stored=identity,json_loaded=loaded,integer_seed_keys_reconstructed=reconstructed,
        explanation='Integer seed key sorting differs after JSON string conversion; exact source-compatible reconstruction matches saved identity.',historical_file_modified=False),
        learning_rate_comparison_controls=controls,known_risks='Historical erratum and data assessment are prior findings, not new replay results. Current 10 pending targets remain unresolved.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    suites=['tests.test_historical_results_audit','tests.test_unified_lighting_evaluation','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stderr)
    paths=[rp,bp,OUT/'findings.json',OUT/'protocol.json',prior.ROOT/'docs/results/ml_historical_evidence_audit_20260910.md',Path(__file__).resolve(),prior.ROOT/'scripts/vision/audit_historical_baseline.py']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in suites]
    prior.frozen(OUT/'completion.json',dict(status='bounded_historical_audit_complete_unresolved_label_risks',reports_catalogued=132,modern_units_rescored=33,legacy_weights_rescored=4,
        all_historical_claims_validated=False,global_inflation_percentage=None,tests_output=t.stdout+t.stderr,
        next_priority='Independent confirmation of ten named development visibility/attribution gaps; no new training authorized by this audit.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('AUDIT_COMPLETE_WITH_SCOPE_LIMITS')

if __name__=='__main__':main()
