"""Import complete explicit observations, without issuing training approval."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior
from scripts.vision import order_fit_risk_observations as notes


def validate(e,obs):
    ids=[x['review_id'] for x in e['events']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(obs):raise ValueError('Missing/duplicate frame review')
    result=[]
    for x in e['events']:
        decisions=obs[x['review_id']]
        if set(decisions)!=set(range(len(x['truth']))):raise ValueError('Missing complete-label decision')
        for i,t in enumerate(x['truth']):
            status,reason=decisions[i]
            if status not in {'content_visible','limited_content','insufficient_content','unknown'} or not reason.strip():
                raise ValueError('Invalid observation')
            result.append(dict(review_id=x['review_id'],member_id=x['member_id'],label_index=i,truth=t,
                image_sha256=x['image_sha256'],label_sha256=x['label_sha256'],page_sha256=x['page_sha256'],
                observation=status,reason=reason,review_nature='AI辅助审核',pixel_visibility_certified=False,
                training_approval=False))
    return result


def run():
    ep=OUT/'risk-reconciliation/evidence.json';e=prior.read(ep);prior.verify(e)
    ds=validate(e,notes.OBS);now=datetime.now(timezone.utc).isoformat()
    for d in ds:d['reviewed_at']=now
    dest=OUT/'risk-reconciliation/review.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='29_frames_current_risk_review_complete_not_admitted',decisions=ds,
        counts=dict(Counter(d['observation'] for d in ds)),
        unresolved_content_members=sorted({d['member_id'] for d in ds if d['observation'] in {'insufficient_content','unknown'}}),
        limited_only_members=sorted({d['member_id'] for d in ds if d['observation']=='limited_content'}-
            {d['member_id'] for d in ds if d['observation'] in {'insufficient_content','unknown'}}),
        limits=['Visual review does not authenticate instance pixels or clear unlabelled-instance risk.',
            'Limited-only frames need explicit eligibility judgment; no automatic approval.',
            'Historical labels unchanged; no threshold or optimizer changed.'],
        inputs={str(x):prior.file_sha256(x) for x in [ep,Path(notes.__file__).resolve(),Path(__file__).resolve()]}))


if __name__=='__main__':
    r=run();print(r['status'],r['counts'],'unresolved members',len(r['unresolved_content_members']),'limited only',len(r['limited_only_members']))
