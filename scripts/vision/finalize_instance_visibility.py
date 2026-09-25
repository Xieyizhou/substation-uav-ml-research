"""Summarize explicit decisions only; this entry point cannot create approvals."""
from collections import Counter
from scripts.vision.instance_visibility_diagnosis import OUT,ROOT,read,save,file_sha256,verify_tree,prepare,Path

STATUSES={'visible_identifiable','visible_content_insufficient','no_instance_pixels','evidence_insufficient'}

def validate_reviews(protocol, reviews):
    expected={e['review_id'] for f in protocol['frames'] for e in f.get('events',[])}
    # Source-blocked rows still require an explicit named gap for every review ID.
    expected.update(rid for f in protocol['frames'] for rid in f['review_ids'])
    if len(reviews)!=len(expected) or {r['review_id'] for r in reviews}!=expected:raise ValueError('Missing/duplicate review')
    for r in reviews:
        if r['status'] not in STATUSES or r['review_nature']!='AI-assisted' or not r.get('reviewed_at') or not r.get('reason'):raise ValueError('Invalid explicit review')
        if r['status']!='evidence_insufficient' and not r.get('original_frame_certified'):raise ValueError('Uncertified decision')
        if r['status']=='no_instance_pixels' and r.get('visible_pixel_count')!=0:raise ValueError('Nonempty mask cannot be absent')
        if r['status'].startswith('visible_') and not (r.get('visible_pixel_count',0)>0):raise ValueError('Empty mask cannot be visible')
        if not r.get('inputs'):raise ValueError('Missing review evidence')
        for p,h in r['inputs'].items():
            if file_sha256(p)!=h:raise ValueError('Stale review evidence')

def main():
    p=prepare();rp=OUT/'review.json';verify_tree(rp);review=read(rp);validate_reviews(p,review['decisions'])
    from scripts.vision.verify_experiment_baseline import verify
    baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if baseline['pinned_files_verified']!=40 or not baseline['integrity_passed']:raise ValueError('Baseline incomplete')
    save(OUT/'baseline-check.json',dict(**baseline,inputs={str(ROOT/'config/perception/visual_experiment_baseline_v1.json'):file_sha256(ROOT/'config/perception/visual_experiment_baseline_v1.json')}))
    inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',rp,OUT/'baseline-check.json',OUT/'frame-dispositions.json',OUT/'tests.json',Path(__file__))}
    dispositions=read(OUT/'frame-dispositions.json')['frames']
    if len(dispositions)!=23 or {f['member_id'] for f in dispositions}!={f['member_id'] for f in p['frames']}:raise ValueError('Incomplete frame dispositions')
    if read(OUT/'tests.json')['passed'] is not True:raise ValueError('Tests not passed')
    save(OUT/'completion.json',dict(status='diagnostic_round_complete_with_named_gaps',source_verified_frames=sum(f['status']=='source_verified' for f in p['frames']),
        reviewed_events=len(review['decisions']),review_counts=dict(Counter(r['status'] for r in review['decisions'])),
        frame_counts=dict(Counter(f['status'] for f in dispositions)),all_visibility_resolved=False,
        all_replays_completed=False,labels_modified=False,data_version_created=False,training_run=False,
        next_decision='Validate native subscription teardown fix under a separately authorized replay budget before expansion; insufficient evidence for full data revision.',inputs=inputs))
    print('COMPLETE_WITH_NAMED_GAPS',dict(Counter(r['status'] for r in review['decisions'])))

if __name__=='__main__':main()
