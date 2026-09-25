"""Make bounded source-group exclusion explicit and verify current actual exposure."""
from collections import Counter
from pathlib import Path
from scripts.vision.material_control_feasibility import OUT,RUN,CANDIDATE,prior

def require_zero(draws, denied):
    bad={m:n for m,n in Counter(draws).items() if m in denied}
    if bad: raise ValueError(f'Quarantined members exposed: {bad}')

def main():
    dest=OUT/'risk-containment.json'
    if dest.exists():prior.verify(prior.read(dest));print('VALID_CONTAINMENT_REUSED');return
    src=OUT.parent/'annotation-revision-design-v1/source-review-v1/evidence.json'
    p=prior.read(RUN/'protocol.json');old=prior.read(src);prior.verify(old)
    cp=CANDIDATE/'protocol.json';candidate=prior.read(cp);prior.verify(candidate)
    denied={m['member_id'] for m in old['members'] if m['review_id'] in ('T027','T036')}
    lineages={m['lineage_id'] for m in old['members'] if m['review_id'] in ('T027','T036')}
    pool=[r for r in p['pool_rows'] if r['member_id'] in denied or r['lineage_id'] in lineages or r.get('source_member_id') in denied]
    denied.update(r['member_id'] for r in pool)
    paths=[src,cp,RUN/'protocol.json',OUT/'initial-gate.json',Path(__file__).resolve()]
    cells={}
    for family in ('R-clean','L-physical'):
        for seed in (7,17,27):
            key=f'{family}-{seed}';cpath=RUN/'training'/key/'completion.json';c=prior.read(cpath);prior.verify(c)
            xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
            if x['draws']!=p['schedules'][key]:raise ValueError('Actual schedule mismatch')
            require_zero(x['draws'],denied)
            cells[key]={m:x['draws'].count(m) for m in sorted(denied)};paths += [cpath,xp]
    quarantined=[dict(source=f['review_ids'][0],member_id=f['member_id'],lineage_id=f['lineage_id'],
        variants=['source','warm','cool','all_resolved_same_pose_derivatives'],
        action='exclude_whole_image_from_future_proposals',risk_resolved=False)
        for f in candidate['frames'] if f['review_ids'][0] in ('T027','T036')]
    prior.frozen(dest,dict(status='risk_contained_resume_diagnosis_not_training_ready',
        scope='Four known source members, their resolved current-pool derivatives, and T027/T036 warm/cool; not global lineage closure.',
        denied_members=sorted(denied),denied_lineages=sorted(lineages),quarantined_groups=quarantined,
        actual_zero_exposure=cells,labels_modified=False,risks_repaired=False,
        authorization='User requested handling these risks and continuing the plan; whole-image exclusion only, no label repair.',
        correction='These four source members were already excluded in all six current cells. Candidate risks do not prove current training contamination.',
        next_stage='Resume independent four-condition review and coverage census; remaining candidates require explicit budget/coverage checks.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('CONTAINED',len(denied),'MEMBERS; VERIFIED ZERO IN SIX CELLS; DIAGNOSIS RESUMED')

if __name__=='__main__':main()
