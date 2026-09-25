"""Bind completed diagnostic evidence and the next bounded, train-only design."""
from pathlib import Path
from scripts.vision.analyze_visibility_quality_results import OUT, ROOT, read, save, file_sha256, verify_tree

def main():
    output=OUT/'completion.json'
    if output.exists(): verify_tree(output); print('VERIFIED_EXISTING'); return
    verify_tree(OUT/'negative-review.json')
    review=read(OUT/'negative-review.json')
    if len(review['items'])!=16 or len({x['item_id'] for x in review['items']})!=16 or any(x['review_status']!='reviewed' for x in review['items']): raise ValueError('Incomplete review')
    import subprocess
    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest','tests.test_visibility_quality_analysis','tests.test_visibility_quality_training','tests.test_instance_visibility_diagnosis','tests.test_canonical_batch_order','tests.test_exposure_diagnosis'],capture_output=True,text=True)
    if result.returncode: raise ValueError(result.stdout+result.stderr)
    paths=[OUT/'analysis.json',OUT/'negative-review.json',ROOT/'docs/results/ml_visibility_quality_error_diagnosis_20260908.md',Path(__file__),ROOT/'tests/test_visibility_quality_analysis.py']
    save(output,dict(status='diagnosis_complete_next_schedule_preflight_required',candidate_selected=False,tests=dict(returncode=result.returncode,output=result.stdout+result.stderr,scope='23 related tests only; global suite not run'),next_design=dict(name='full-instance-exposure-balance-v1',source='Q retained frozen members only',seeds=[7,17,27],steps=[100,300],quota_per_600=dict(base=216,regular=156,bridge_positive=120,hard_negative=108),negative_sequence='unchanged per seed',positive_member_min=1,positive_member_max='2*ceil(subset_quota/subset_member_count)',objective_order=['minimize full-label class instance exposure range','minimize L1 member exposure difference from Q','deterministic seed tie-break'],prerequisites=['integer feasibility and objective independently checked','exact sequences frozen before training','100/300 prefix exact','all evaluation and retention gates unchanged'],fallback='Stop this sampling path if infeasible or no exposure-range improvement; no silent cap relaxation.'),inputs={str(p):file_sha256(p) for p in paths}))
    print('DIAGNOSIS_COMPLETE_NEXT_PREFLIGHT',OUT)

if __name__=='__main__':main()
