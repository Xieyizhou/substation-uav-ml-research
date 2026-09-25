"""Combine two explicitly reviewed batches; never admits data or starts training."""
from collections import Counter
from pathlib import Path
from scripts.vision.capture_designed_material_triplets import OUT as FIRST, SOURCES as FIRST_SOURCE, prior
from scripts.vision.capture_supplemental_material_triplets import OUT as SECOND, SOURCES as SECOND_SOURCE

OUT=FIRST.parent.parent/'combined-twelve-pose-candidates-v1'


def main():
    members=[];paths=[Path(__file__)];classes=[];decisions=[]
    for folder,source_folder,expected in ((FIRST,FIRST_SOURCE,4),(SECOND,SECOND_SOURCE,8)):
        rp=folder/'reviewed-completion.json';sp=source_folder/'reviewed-completion.json'
        r=prior.read(rp);s=prior.read(sp)
        prior.verify(r);prior.verify(s)
        if r['complete_triplets']!=expected or len(r['members'])!=expected*3:raise ValueError('Incomplete batch')
        if len(s['sources'])!=expected:raise ValueError('Incomplete sources')
        design=source_folder.parent/'protocol.json';p=prior.read(design);prior.verify(p)
        selected={x['probe_id']:x for x in p['selected']}
        sources={x['probe_id']:x for x in s['sources']}
        for m in r['members']:
            sid=m['member_id'].split('-')[0];src=sources[sid];v=selected[sid]
            if m['full_truth']!=src['full_truth'] or m['instance_mapping']!=src['instance_mapping']:raise ValueError('Pair label/mapping conflict')
            if m['actual_pose']!=src['actual_pose']:raise ValueError('Pair pose conflict')
            if prior.file_sha256(m['image_path'])!=m['image_sha256']:raise ValueError('Stale image')
            members.append(dict(m,planned_category=v['category'],planned_object_id=v['object_id'],
                source_review=str(sp),variant_review=str(rp),source_pose_id=sid,
                shared_layout='complex-canonical',shared_assets=True,source_independent=False))
            paths.append(Path(m['image_path']))
            if m['variant']=='original':classes.append(v['category'])
        decisions += s['decisions']+r['decisions'];paths += [rp,sp,design]
    if len(members)!=36 or len({m['member_id'] for m in members})!=36:raise ValueError('Member cardinality')
    groups={m['pair_id'] for m in members}
    if len(groups)!=12 or any(Counter(m['variant'] for m in members if m['pair_id']==g)!=Counter(['original','warm','cool']) for g in groups):raise ValueError('Incomplete pairing')
    if Counter(classes)!=Counter(dict(transformer=3,switchgear=3,capacitor_bank=3,reactor=3)):raise ValueError('Planned class coverage')
    if len(decisions)!=sum(len(m['full_truth']['objects']) for m in members):raise ValueError('Full-label review coverage')
    OUT.mkdir(exist_ok=True)
    prior.frozen(OUT/'reviewed-completion.json',dict(status='36_candidates_reviewed_pending_whole_batch_isolation_and_exposure_checks',
        members=members,decisions=decisions,complete_triplets=12,planned_pose_classes=dict(Counter(classes)),
        training_ready=False,training_started=False,historical_admission_chain_valid=False,
        limitations=['Shared complex layout/assets; 12 poses are not 12 independent scenes',
        'High-angle cabinet views do not establish frontal-panel coverage; reactor distance range remains narrow',
        'Historical admission-chain defect remains disclosed; fresh acquisition is not retrospective historical approval',
        'Full-batch role/exclusion, exposure feasibility and real loader preflight remain required'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('MERGED36 TRIPLETS12 REVIEWS',len(decisions),'NO_TRAINING')


if __name__=='__main__':main()
