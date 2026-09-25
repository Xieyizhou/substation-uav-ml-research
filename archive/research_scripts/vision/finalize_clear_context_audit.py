"""Read existing reviews and fixed gates; never author approval decisions."""
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.freeze_clear_context_training import OUT,checked
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks


def run():
    root=OUT/'evaluation-v1';target=root/'audit-v1.json'
    if target.exists():return checked(target)
    deps=[root/'summary.json',root/'error-review-v1/review.json',root/'positive-review-v1/review.json',OUT/'new-member-fit-v1.json',PRIOR/'protocol.json',Path(__file__)]
    current,fp,positive,fit,policy=[checked(p) for p in deps[:-1]]
    if len(fp['decisions'])!=19 or len(positive['decisions'])!=43:raise ValueError('Review coverage changed')
    if any(current['matching_conflicts'].values()):raise ValueError('Unresolved match conflict')
    refs_paths=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    refs=[checked(p) for p in refs_paths];hp=Path(policy['evaluation']['historical_reference']);hist=checked(hp)
    histpaths=[next(Path(x) for x in hist['inputs'] if x.endswith(f'/historical-A-{s}.json')) for s in (7,17,27)]
    if aggregate([checked(p) for p in histpaths])!=hist['historical_A']:raise ValueError('Historical aggregate differs')
    gate=policy_checks(current['group'],aggregate(refs),hist['historical_A'],policy)
    for c in gate['checks']:
        if c.get('reference')=='same_budget_R':c['reference']='fixed_retained_reference_450_not_same_budget_as_480'
    deps+=refs_paths+[hp]+histpaths
    return write_record(target,dict(status='current_error_review_and_fixed_gate_audit_complete',gates=gate,
        fp_review_events=len(fp['decisions']),positive_loss_review_events=len(positive['decisions']),new_member_training_fit=fit['summary'],
        conclusion='Not a passing candidate. New members fit unevenly across seeds; low-LR control tests update-size sensitivity without assuming a unique cause.',
        training_admitted=False,promotable=False,selected_candidate=None,inputs={str(p.resolve()):file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['gates'])
