"""Independent recount and audit coverage inventory; never publishes readiness."""
from collections import Counter
from pathlib import Path
import subprocess,sys
from scripts.vision.check_risk_capped_redistribution import *
from scripts.vision.prepare_redistribution_review import OUT as PRE
from scripts.vision.prepare_whole_image_hold import baseline_verify

def main():
    p=ready(); doc=read(OUT/'counts.json');verify(doc)
    evidence=read(PRE/'evidence.json');review=read(PRE/'review.json')
    verify(evidence);verify(review)
    if set(doc['results'])!={'7','17','27'}:raise ValueError('Missing seed')
    previous={x['member']['member_id']:x['event_id'] for x in evidence['events']}
    risk=set(doc['risk_members']); idx={r['member_id']:r for r in p['pool_rows']}; increased=set();audit={}
    for seed,x in doc['results'].items():
        old=Counter(p['schedules'][f'reference-450-{seed}']);new=x['counts']
        validate_capped(p['pool_rows'],old,new,set(p['held_member_ids']),risk)
        if tally(p['pool_rows'],new)!=x['totals']:raise ValueError('Stale totals')
        changed={m for m in new if new[m]!=old[m]}
        if len(x['changes'])!=len(changed) or {d['member_id'] for d in x['changes']}!=changed:raise ValueError('Changes mismatch')
        for d in x['changes']:
            m=d['member_id']
            if (d['before'],d['after'],d['delta'])!=(old[m],new[m],new[m]-old[m]):raise ValueError('Stale change')
        added={m for m in new if new[m]>old[m]};increased |= added
        minimum=sum(max(0,new[m]-old[m]) for m in new)
        if minimum!=x['minimum_changed_slots'] or minimum!=sum(max(0,old[m]-new[m]) for m in new):raise ValueError('Slot count mismatch')
        if max(new[m]-old[m] for m in new)!=x['objective']['peak_extra_exposure']:raise ValueError('Peak mismatch')
        lineage_before=Counter();lineage_after=Counter()
        for m,r in idx.items():lineage_before[r['lineage_id']]+=old[m];lineage_after[r['lineage_id']]+=new[m]
        audit[seed]=dict(integer_constraints_verified=True,increased_members=len(added),
            maximum_registered_lineage_before=max(lineage_before.values()),maximum_registered_lineage_after=max(lineage_after.values()),
            registered_lineage_not_independent_scene=True,risk_total_before=sum(old[m] for m in risk),risk_total_after=sum(new[m] for m in risk),
            risk_counts_unchanged=all(old[m]==new[m] for m in risk),minimum_changed_slots=minimum)
    pending=[dict(member=idx[m],reason='Outside preceding 56-image review; requires fresh source and full-label visual review before readiness.') for m in sorted(increased-set(previous))]
    tests=['tests.test_risk_capped_redistribution','tests.test_positive_redistribution','tests.test_redistribution_target_attribution','tests.test_redistribution_occlusion_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths=[OUT/'counts.json',PRE/'evidence.json',PRE/'review.json',Path(__file__),ROOT/'docs/results/ml_risk_capped_redistribution_20260909.md',ROOT/'config/perception/visual_experiment_baseline_v1.json']
    paths.extend(ROOT/(t.replace('.','/')+'.py') for t in tests)
    frozen(OUT/'completion.json',dict(status='count_feasibility_verified_four_additional_reviews_required',audit=audit,
        increased_member_union=len(increased),new_review_required=pending,
        preceding_screened_members=[dict(member_id=m,previous_event_id=previous[m]) for m in sorted(increased&set(previous))],
        prior_screen_is_not_training_admission=True,baseline=baseline,regression_output=result.stderr,whole_repository_tested=False,
        schedule_generated=False,training_started=False,readiness_issued=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('EXACT_COUNTS_VERIFIED',len(increased),'INCREASED_MEMBERS',len(pending),'NEW_REVIEWS_REQUIRED; PINNED40_PASSED; NO_TRAINING')

if __name__=='__main__':main()
