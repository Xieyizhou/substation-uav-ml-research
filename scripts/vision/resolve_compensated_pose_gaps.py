"""Posthoc pose provenance resolution; no historical approval is rewritten."""
from pathlib import Path
from scripts.vision.merge_compensated_material_candidates import OUT,prior
from scripts.vision.establish_material_view_candidates import HISTORY,OUT as PARENT
from scripts.vision.structure_fit import OUT as FIT
from scripts.vision.train_frozen_multiscale import contract
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash
from src.vision.canonical.plan import pose_close


def main():
    ap=OUT/'known-source-pose-audit.json';audit=prior.read(ap);prior.verify(audit)
    fp=FIT/'protocol.json';fit=prior.read(fp);prior.verify(fit)
    _,pool,_=contract('fixed-7');lookup={r['member_id']:r for r in pool['pool_rows']}
    sources={r['member']['member_id']:r['source'] for r in fit['negative']}
    ip=PARENT/'source-inventory.json';inventory=prior.read(ip);prior.verify(inventory)
    for r in inventory['records']:
        if not r.get('actual_pose'):sources.setdefault(r['source_member_id'],dict(image_path=r['source_image']))
    paths=[ap,fp,ip,Path(__file__)];rows=[];cache={}
    for gap in audit['reference_records_without_pose']:
        mid=gap['member'];r=dict(gap);src=sources.get(mid)
        if not src:r.update(status='unresolved',reason='No unique original image source');rows.append(r);continue
        image=Path(src['image_path']);receipt=image.parents[1]/'collection-receipt.json'
        try:
            sha=prior.file_sha256(image)
            if src.get('image_sha256',sha)!=sha:raise ValueError('Original image hash changed')
            if receipt not in cache:cache[receipt]=prior.read(receipt)
            record=cache[receipt];vs=[v for v in record['views'] if v['view_id']==image.parent.name]
            if len(vs)!=1 or vs[0]['image_sha256']!=sha or not vs[0].get('actual_pose'):raise ValueError('Capture mapping/hash/pose conflict')
            v=vs[0]
            if mid in lookup:
                current=lookup[mid]
                if prior.file_sha256(current['image_path'])!=current['image_sha256'] or pixel_hash(image)!=pixel_hash(current['image_path']):raise ValueError('Current image differs from original RGB')
                paths.append(Path(current['image_path']))
            matches=[g for g,s in audit['sources'].items() if pose_close(s['actual_pose'],v['actual_pose'])]
            r.update(status='posthoc_original_capture_pose_verified',source_image=str(image),source_image_sha256=sha,
                source_receipt=str(receipt),source_view_id=v['view_id'],actual_pose=v['actual_pose'],new_group_matches=matches)
            paths += [image,receipt]
        except (ValueError,KeyError,FileNotFoundError) as ex:r.update(status='unresolved',reason=str(ex))
        rows.append(r)
    gaps=[r for r in rows if r['status']=='unresolved'];matches=[r for r in rows if r.get('new_group_matches')]
    prior.frozen(OUT/'resolved-pose-gaps.json',dict(status='posthoc_missing_pose_exclusion_verified' if not gaps and not matches else 'named_gaps_remain',
        rows=rows,unresolved=gaps,pose_overlaps=matches,training_ready=False,training_started=False,
        historical_approvals_modified=False,protected_labels_read=False,
        limitations=['Posthoc provenance check only; does not assert old acquisitions passed new semantic gates','Shared layout and assets are not independent scenes'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('CHECKED',len(rows),'UNRESOLVED',len(gaps),'MATCHES',len(matches),flush=True)


if __name__=='__main__':main()
