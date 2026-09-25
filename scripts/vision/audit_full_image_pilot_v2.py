"""Read-only reference audit. Never open sealed images or any protected labels."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from scripts.vision.build_full_image_pose_review import OUT, read, save, file_sha256, verify_tree
from scripts.vision.exact_dedup_semantic_review import ROOT, _reference_specurations, _read_rows
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.verify_pixel_duplicates import rgb_digest

def pixels(path):
    return rgb_digest(Path(path).read_bytes())[0]

def main():
    dest=OUT/'review/intake-audit-v2.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    rp=OUT/'review/decisions.json';verify_tree(rp);review=read(rp)
    frames=review['frames'];current={str(Path(f['image_path']).resolve()) for f in frames}
    inputs={str(rp):file_sha256(rp),str(Path(__file__)):file_sha256(Path(__file__))}
    refs=[];seen=set();gaps=[]
    base=ROOT/'data/research/ml_training_recovery_v1'
    cache=base/'pixel-dedup-v1/pixel-fingerprints.jsonl'
    report=base/'pixel-dedup-v1/report.json'
    if file_sha256(cache)!=read(report)['files']['pixel-fingerprints.jsonl']['sha256']:
        raise ValueError('Pixel cache changed')
    inputs[str(cache)]=file_sha256(cache);inputs[str(report)]=file_sha256(report)
    cached={r['image_sha256']:r['pixel_sha256'] for r in _read_rows(cache)}
    # Protected comparisons use pre-existing index metadata only.
    for spec in _reference_specurations():
        p=spec['path'];inputs[str(p)]=file_sha256(p)
        for row in _read_rows(p,spec.get('row_key')):
            refs.append(dict(path=str(p),image_sha256=row.get('image_sha256',row.get('payload_sha256')),
              perceptual_hash=row.get('perceptual_hash'),role=spec['role'],pixel_sha256=row.get('pixel_sha256') or cached.get(row.get('image_sha256',row.get('payload_sha256')))))
    supplemental=base/'reference-group-audit-v1/supplemental-protected-fingerprints.jsonl'
    inputs[str(supplemental)]=file_sha256(supplemental)
    refs.extend({**r,'path':str(supplemental),'role':'protected'} for r in _read_rows(supplemental))
    for base in (ROOT/'data/research/canonical_views_v1',ROOT/'data/research/ml_training_recovery_v1'):
        for p in sorted(base.rglob('collection-receipt.json')):
            if OUT in p.parents or any(s in str(p).lower() for s in ('protected','sealed','unseen','new-scene','new_scene')):continue
            r=read(p);inputs[str(p)]=file_sha256(p)
            for row in r.get('views',[]):
                raw=row.get('rgb_path')
                if not raw or row.get('status')!='captured':continue
                image=Path(raw).resolve()
                if str(image) in current or image in seen:continue
                seen.add(image)
                if not image.is_file():gaps.append('missing historical RGB: '+str(image));continue
                digest=file_sha256(image)
                if digest!=row.get('image_sha256'):raise ValueError('Stale historical RGB: '+str(image))
                inputs[str(image)]=digest
                refs.append(dict(path=str(image),image_sha256=digest,pixel_sha256=pixels(image),
                  perceptual_hash=dhash64(image),role='historical_development',view_id=row.get('view_id')))
    results=[]
    for f in frames:
        pix=pixels(f['image_path']);ph=dhash64(f['image_path'])
        exact=[r for r in refs if r['image_sha256']==f['image_sha256'] or r.get('pixel_sha256')==pix]
        distances=[((int(ph,16)^int(r['perceptual_hash'],16)).bit_count(),r) for r in refs if r.get('perceptual_hash')]
        nearest=min(distances,key=lambda x:x[0]) if distances else None
        near=[{'distance':d,**r} for d,r in distances if d<=2]
        results.append(dict(frame_id=f['frame_id'],pixel_sha256=pix,perceptual_hash=ph,exact_matches=exact,
          near_matches=near,nearest=({'distance':nearest[0],**nearest[1]} if nearest else None),
          shared_source_group=f['shared_source_group'],derived_from_view_id=f['derived_from_view_id'],
          status='held_duplicate_or_similarity_review' if exact or near else 'no_detected_reference_overlap'))
    internal=[]
    for i,a in enumerate(results):
        for b in results[i+1:]:
            d=(int(a['perceptual_hash'],16)^int(b['perceptual_hash'],16)).bit_count()
            if d<=2 or a['pixel_sha256']==b['pixel_sha256']:
                internal.append(dict(frames=[a['frame_id'],b['frame_id']],distance=d,
                  same_source=a['shared_source_group']==b['shared_source_group'],pixel_equal=a['pixel_sha256']==b['pixel_sha256']))
    # Inventory omissions stay explicit; no expansion permit from partial checks.
    missing_protected_pixels=sum(r['role']=='protected' and not r.get('pixel_sha256') for r in refs)
    if missing_protected_pixels:gaps.append(f'{missing_protected_pixels} protected index rows lack original-size pixel hashes; no protected images opened.')
    if any(not x['same_source'] or x['pixel_equal'] for x in internal):gaps.append('Internal similarity needs explicit relationship review.')
    from scripts.vision.verify_experiment_baseline import verify
    save(dest,dict(status='held_pending_intake_resolution' if gaps or any(r['exact_matches'] or r['near_matches'] for r in results) else 'reference_checks_complete_pending_source_isolation',
      frames=results,internal_similarity=internal,reference_count=len(refs),gaps=gaps,
      lineage_policy='All share complex map/assets. Preserve source and parent links; no independent-scene claim or automatic role assignment.',
      baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs))
    print(json.dumps({'status':read(dest)['status'],'gaps':gaps,'frames':[{k:r[k] for k in ('frame_id','status')} for r in results]},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
