"""Variant fingerprint audit preserving designed siblings as one source group."""
from pathlib import Path
from scripts.vision.capture_full_image_appearance import OUT,ORIGINAL,read,save,file_sha256,verify_tree
from scripts.vision.audit_full_image_pilot_v2 import pixels
from scripts.vision.exact_dedup_semantic_review import ROOT,_reference_specurations,_read_rows
from src.vision.training.hard_example_curator import dhash64

def main():
    dest=OUT/'review/intake-audit.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    rp=OUT/'review/decisions.json';ap=ORIGINAL/'review/intake-audit-v2.json'
    verify_tree(rp);verify_tree(ap);r=read(rp);a=read(ap)
    inputs={str(rp):file_sha256(rp),str(ap):file_sha256(ap),str(Path(__file__)):file_sha256(Path(__file__))}
    base=ROOT/'data/research/ml_training_recovery_v1';cache=base/'pixel-dedup-v1/pixel-fingerprints.jsonl'
    cached={x['image_sha256']:x['pixel_sha256'] for x in _read_rows(cache)};refs=[]
    for spec in _reference_specurations():
        p=spec['path'];inputs[str(p)]=file_sha256(p)
        for x in _read_rows(p,spec.get('row_key')):
            digest=x.get('image_sha256',x.get('payload_sha256'))
            refs.append(dict(path=str(p),image_sha256=digest,pixel_sha256=x.get('pixel_sha256') or cached.get(digest),perceptual_hash=x.get('perceptual_hash')))
    supplemental=base/'reference-group-audit-v1/supplemental-protected-fingerprints.jsonl'
    inputs[str(supplemental)]=file_sha256(supplemental)
    refs.extend({**x,'path':str(supplemental)} for x in _read_rows(supplemental))
    for raw,digest in a['inputs'].items():
        p=Path(raw)
        if p.suffix.lower() not in ('.ppm','.png','.jpg','.jpeg'):continue
        if file_sha256(p)!=digest:raise ValueError('Reference changed')
        refs.append(dict(path=str(p),image_sha256=digest,pixel_sha256=pixels(p),perceptual_hash=dhash64(p)))
    originals=read(ORIGINAL/'review/decisions.json')['frames'];allframes=originals+r['frames']
    fingerprints=[dict(frame_id=f['frame_id'],image_sha256=f['image_sha256'],pixel_sha256=pixels(f['image_path']),
      perceptual_hash=dhash64(f['image_path']),shared_source_group=f['shared_source_group']) for f in allframes]
    checks=[]
    for f in fingerprints[8:]:
        exact=[x['path'] for x in refs if x['image_sha256']==f['image_sha256'] or x.get('pixel_sha256')==f['pixel_sha256']]
        near=[x['path'] for x in refs if x.get('perceptual_hash') and (int(x['perceptual_hash'],16)^int(f['perceptual_hash'],16)).bit_count()<=2]
        peers=[x for x in fingerprints if x['frame_id']!=f['frame_id'] and (x['pixel_sha256']==f['pixel_sha256'] or (int(x['perceptual_hash'],16)^int(f['perceptual_hash'],16)).bit_count()<=2)]
        unexpected=[x for x in peers if x['shared_source_group']!=f['shared_source_group']]
        checks.append(dict(**f,reference_exact=exact,reference_near=near,designed_similarity=[x['frame_id'] for x in peers if x not in unexpected],unexpected_similarity=unexpected))
    held=any(x['reference_exact'] or x['reference_near'] or x['unexpected_similarity'] for x in checks)
    save(dest,dict(status='held_for_similarity_review' if held else 'variant_fingerprint_checks_passed',checks=checks,
      reference_count=len(refs),source_groups=7,scene_count=1,
      limitation='Designed variants are correlated, retained together and not independent samples. This receipt is not training admission.',inputs=inputs))
    print('VARIANT_AUDIT',read(dest)['status'],'held',sum(bool(x['reference_exact'] or x['reference_near'] or x['unexpected_similarity']) for x in checks),flush=True)

if __name__=='__main__':main()
