"""Apply authored whole-frame holds and registered lineage propagation only."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    paths=[OUT/'protocol.json',OUT/'quality-isolation-v3.json',OUT/'authored-quality-dispositions-02.json']
    p,q,r=[prior.read(x) for x in paths]
    for record in (p,q,r):prior.verify(record)
    idx={m['member_id']:m for m in p['members']}
    bad={d['member_id'] for d in r['decisions'] if d['quality_status']=='whole_frame_held_content_evidence_insufficient'}
    if len(bad)!=6 or not bad<=set(idx):raise ValueError('Authored hold set changed')
    groups={idx[mid]['lineage_id'] for mid in bad};rows=[];added=[]
    for original in q['members']:
        row=dict(original);row['reasons']=list(row['reasons']);mid=row['member_id']
        if mid in bad or idx[mid]['lineage_id'] in groups:
            if row['status']!='quarantined':added.append(mid)
            row['status']='quarantined'
            row['reasons'].append('authored_full_frame_content_hold' if mid in bad else 'registered_lineage_of_authored_content_hold')
        row['training_eligible']=False;rows.append(row)
    dest=OUT/'quality-isolation-v4.json'
    if dest.exists():
        result=prior.read(dest);prior.verify(result);return result
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='authored_holds_applied_no_dataset_admission',members=rows,
        counts=dict(Counter(x['status'] for x in rows)),newly_quarantined=added,
        authored_hold_members=sorted(bad),
        unresolved_member_only_lineages=sorted(mid for mid in bad if idx[mid].get('lineage_resolution')=='member_only_no_independence_claim'),
        history_modified=False,dataset_ready=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=run();print(r['counts']);print('newly quarantined',len(r['newly_quarantined']))
