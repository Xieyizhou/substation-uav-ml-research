"""Combine independent evidence, never manufacture full-frame approval."""
from pathlib import Path
from collections import defaultdict, Counter
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    paths=[OUT/'remaining-review/review.json',OUT/'quality-isolation-v3.json']
    sources=[OUT/'source-spatial-context'/name for name in ['background-review-01.json','spatial-review-02.json','bridge-review-03.json','background-review-04.json','background-review-05.json']]
    sources.append(OUT/'c01-background-review.json')
    review, quarantine=[prior.read(p) for p in paths]
    for r in (review,quarantine):prior.verify(r)
    links=defaultdict(list)
    for path in sources:
        r=prior.read(path);prior.verify(r)
        for d in r['decisions']:
            mid=d.get('member_id',r.get('member_id'))
            if not mid:raise ValueError('Missing member identity')
            links[mid].append(dict(path=str(path),object_id=d['object_id'],status=d['status']))
    q={m['member_id']:m for m in quarantine['members']};rows=[]
    for d in review['decisions']:
        mid=d['member_id']
        rows.append(dict(member_id=mid,review_id=d['review_id'],quarantine_status=q[mid]['status'],
            full_frame_observation=d['full_frame_observation'],named_source_decisions=links[mid],
            label_observation_counts=dict(Counter(x['status'] for x in d['label_observations'])),
            residual_named_question='U33 lower-left fragment not specifically covered by the two cabinet decisions' if d['review_id']=='U33' else None,
            final_eligibility_decision_present=False,training_eligible=False))
    dest=OUT/'full-frame-evidence-index.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    paths+=sources+[Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='51_full_frame_evidence_bundles_indexed_not_approved',members=rows,
        scope='Index retains visual observations, source associations and holds separately; no full-frame pass inference.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    r=run();print(len(r['members']),sum(bool(m['named_source_decisions']) for m in r['members']))
