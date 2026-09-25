"""Known-source pose and fresh acquisition audit; no protected labels or training."""
from pathlib import Path
from scripts.vision.merge_compensated_material_candidates import OUT,prior
from scripts.vision.establish_material_view_candidates import HISTORY,OUT as PARENT
from src.vision.canonical.plan import pose_close


def main():
    rp=OUT/'reviewed-completion.json';review=prior.read(rp);prior.verify(review)
    paths=[rp,Path(__file__)];known=[];missing=[]
    for file,key in ((HISTORY/'coverage-census.json','members'),(PARENT/'source-inventory.json','records')):
        r=prior.read(file);prior.verify(r);paths.append(file)
        for row in r[key]:
            if row.get('actual_pose'):known.append((str(file),row.get('member_id',row.get('source_member_id')),row['actual_pose']))
            else:missing.append(dict(inventory=str(file),member=row.get('member_id',row.get('source_member_id'))))
    devroot=HISTORY.parent
    for name in ('paired-visual-factors-v1','hard-negative-isolated-v2'):
        for file in sorted((devroot/name).rglob('collection-receipt.json')):
            r=prior.read(file);paths.append(file)
            for v in r.get('views',[]):
                if v.get('actual_pose'):known.append((str(file),v.get('view_id'),v['actual_pose']))
    sources={}
    for member in review['members']:
        group=member['pair_id']
        if group in sources:
            if member['actual_pose']!=sources[group]['actual_pose']:raise ValueError('Derived pose changed')
            continue
        matches=[dict(inventory=p,member_id=mid) for p,mid,pose in known if pose_close(member['actual_pose'],pose)]
        sp=Path(member['source_review']);s=prior.read(sp);prior.verify(s);paths.append(sp)
        candidates=[x for x in s['sources'] if x['lineage_id']==group]
        if len(candidates)!=1:raise ValueError('Nonunique fresh source')
        src=candidates[0];up=sp.parent/src['probe_id']/'protocol.json';u=prior.read(up);prior.verify(u);paths.append(up)
        f=u['frame'];cp=sp.parent/src['probe_id']/'capture/collection-receipt.json';capture=prior.read(cp);paths.append(cp)
        if capture['status']!='complete_pending_review' or len(capture['views'])!=1:raise ValueError('Incomplete acquisition')
        cv=capture['views'][0]
        if cv['actual_pose']!=member['actual_pose'] or cv['truth']!=member['full_truth']:raise ValueError('Source pose/truth mismatch')
        if prior.file_sha256(f['source_image'])!=cv['image_sha256']:raise ValueError('Source RGB changed')
        sources[group]=dict(actual_pose=member['actual_pose'],fresh_source=str(cp),world=f['source_world'],
            source_image=f['source_image'],source_image_sha256=cv['image_sha256'],known_pose_matches=matches,
            instance_mapping_unique=len({x['object_id'] for x in f['instance_mapping'].values()})==len(f['instance_mapping']))
        if not sources[group]['instance_mapping_unique']:raise ValueError('Instance mapping collision')
    if len(sources)!=13:raise ValueError('Incorrect source group count')
    prior.frozen(OUT/'known-source-pose-audit.json',dict(status='known_pose_overlap_found' if any(x['known_pose_matches'] for x in sources.values()) else 'fresh_sources_and_known_pose_exclusion_verified',
        sources=sources,known_pose_records_checked=len(known),reference_records_without_pose=missing,
        protected_labels_read=False,training_ready=False,training_started=False,
        limitations=['Known inventory exclusions are not certification of all historical missing poses',
        'Fresh acquisition source is explicit; shared layout/assets remain disclosed',
        'Historical inventory dependency defect remains uncorrected; old approvals not renewed by this audit'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SOURCES',len(sources),'KNOWN_POSES',len(known),'MATCHES',sum(len(x['known_pose_matches']) for x in sources.values()),'MISSING_POSES',len(missing))


if __name__=='__main__':main()
