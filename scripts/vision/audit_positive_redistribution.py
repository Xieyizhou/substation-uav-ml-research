"""Independent integer recount and reviewed-content exposure audit, no training."""
from collections import Counter
from pathlib import Path
import subprocess,sys
from scripts.vision.relax_hold_compensation import OUT,RUN,ROOT,ready,read,verify,frozen,file_sha256,validate_counts
from scripts.vision.structure_fit import OUT as FIT

def audit():
    p=ready();r=read(OUT/'counts.json');verify(r)
    fp=read(FIT/'protocol.json');review=read(FIT/'review.json');verify(fp);verify(review)
    ds={d['event_id']:d for d in review['decisions']}
    content={e['member']['member_id']:ds[e['event_id']]['identifiable_content'] for e in fp['reactors']}
    audit={}
    for seed,x in r['results'].items():
        old=Counter(p['schedules'][f'reference-450-{seed}']);new=x['counts']
        validate_counts(p['pool_rows'],old,new,set(p['held_member_ids']))
        if max(new[m]-old[m] for m in new)!=x['objective']['peak_extra_exposure']:raise ValueError('Peak objective mismatch')
        added=sum(max(0,new[m]-old[m]) for m in new);removed=sum(max(0,old[m]-new[m]) for m in new)
        if added!=removed or added!=x['minimum_changed_slots']:raise ValueError('Changed slot bound mismatch')
        audit[seed]=dict(integer_constraints_verified=True,reference_max_member_exposure=max(old.values()),new_max_member_exposure=max(new.values()),
            changed_members=len(x['changes']),increased_members=sum(d['delta']>0 for d in x['changes']),
            retained_reduced_members=sum(d['delta']<0 and d['after']>0 for d in x['changes']),
            reactor_content_exposure_delta={c:sum(new[m]-old[m] for m,v in content.items() if v==c) for c in sorted(set(content.values()))},
            maximum_registered_lineage_before=max(x['registered_lineage_before'].values()),maximum_registered_lineage_after=max(x['registered_lineage_after'].values()),
            lineage_count_is_not_independent_scene_count=True)
    tests=['tests.test_positive_redistribution','tests.test_hold_compensation_feasibility','tests.test_whole_image_hold','tests.test_hold_error_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    paths=[OUT/'counts.json',FIT/'protocol.json',FIT/'review.json',Path(__file__),ROOT/'docs/results/ml_positive_redistribution_feasibility_20260909.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    frozen(OUT/'completion.json',dict(status='count_feasibility_verified_not_ready_for_training',audit=audit,regression_output=result.stderr,
        training_started=False,schedule_generated=False,quality_reaudited=False,whole_repository_tested=False,
        next_gate='Freeze sequence/controls and inspect increased-member quality and provenance before actual loader preflight.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print('COUNT_FEASIBILITY_VERIFIED_NOT_READY_FOR_TRAINING',audit)

if __name__=='__main__':audit()
