"""Lossless candidate export and reference exclusion. Never starts training."""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.capture_designed_material_triplets import OUT as MATERIAL,prior
from scripts.vision.train_frozen_multiscale import contract
from scripts.vision.freeze_visual_augmentation_240_v2 import REFERENCE_INDEXES,pixel_hash


def label_text(truth,width,height):
    lines=[]
    for obj in truth['objects']:
        cls=obj['class_id'];x1,y1,x2,y2=obj['bbox_xyxy']
        if cls not in (0,1,2,3) or not all(math.isfinite(v) for v in (x1,y1,x2,y2)):raise ValueError('Invalid class/coordinates')
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Invalid full box bounds')
        values=((x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height)
        line=str(cls)+' '+' '.join(f'{v:.12f}' for v in values)
        c,x,y,w,h=map(float,line.split());back=((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height)
        if c!=cls or max(abs(a-b) for a,b in zip(back,(x1,y1,x2,y2)))>1e-5:raise ValueError('Label roundtrip differs')
        lines.append(line)
    if not lines:raise ValueError('Positive candidate lacks full supervision')
    return '\n'.join(lines)+'\n'


def export(review_path,out):
    r=prior.read(review_path);prior.verify(r);members=r['members']
    if len(members)!=len({m['member_id'] for m in members}):raise ValueError('Duplicate member ID')
    _,pool,_=contract('fixed-7');references=[];paths=[review_path,Path(__file__)]
    for m in pool['pool_rows']:references.append(dict(role='current_pool',image_path=m['image_path'],image_sha256=m['image_sha256']))
    for key in ('paired_review','negative_review'):
        p=Path(pool['evaluation'][key]);q=prior.read(p);paths.append(p)
        for f in q['frames']:references.append(dict(role=key,image_path=f['image_path'],image_sha256=f['image_sha256']))
    file_hashes={};pixel_hashes={}
    for ref in references:
        ip=Path(ref['image_path']);sha=prior.file_sha256(ip)
        if sha!=ref['image_sha256']:raise ValueError('Stale reference image')
        file_hashes.setdefault(sha,[]).append(ref['role']);pixel_hashes.setdefault(pixel_hash(ip),[]).append(ref['role']);paths.append(ip)
    protected_count=0
    for p in REFERENCE_INDEXES:
        paths.append(p)
        for line in p.read_text().splitlines():
            x=json.loads(line);protected_count+=1
            if x.get('image_sha256'):file_hashes.setdefault(x['image_sha256'],[]).append('registered_reference_fingerprint')
            if x.get('pixel_sha256'):pixel_hashes.setdefault(x['pixel_sha256'],[]).append('registered_reference_fingerprint')
    rows=[];seen_pixels={};gaps=[]
    out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'labels').mkdir(exist_ok=True)
    for m in members:
        ip=Path(m['image_path']);sha=prior.file_sha256(ip)
        if sha!=m['image_sha256']:raise ValueError('Candidate image hash changed')
        px=pixel_hash(ip);overlap=file_hashes.get(sha,[])+pixel_hashes.get(px,[])
        if px in seen_pixels:overlap.append('candidate_exact_pixel_duplicate:'+seen_pixels[px])
        seen_pixels[px]=m['member_id']
        if overlap:gaps.append(dict(member_id=m['member_id'],overlap=overlap))
        image=Image.open(ip).convert('RGB');text=label_text(m['full_truth'],*image.size)
        op=out/'images'/(m['member_id']+'.png');lp=out/'labels'/(m['member_id']+'.txt')
        if op.exists():
            if pixel_hash(op)!=px:raise ValueError('Existing export pixels differ')
        else:image.save(op)
        if lp.exists():
            if lp.read_text()!=text:raise ValueError('Existing exported labels differ')
        else:lp.write_text(text)
        if pixel_hash(op)!=px:raise ValueError('Lossless image roundtrip failure')
        rows.append(dict(member_id=m['member_id'],pair_id=m['pair_id'],variant=m['variant'],image_path=str(op),image_sha256=prior.file_sha256(op),pixel_sha256=px,
            label_path=str(lp),label_sha256=prior.file_sha256(lp),source_image=str(ip),full_truth=m['full_truth'],
            class_instances=dict(Counter(o['class_name'] for o in m['full_truth']['objects'])),data_role='development_training_candidate',
            layout='complex-canonical',asset_family='canonical-complex-equipment',source_independent=False,
            exact_reference_overlap=overlap,training_admitted=False,promotable=False))
        paths += [ip,op,lp]
    dest=out/'manifest.json'
    record=dict(status='blocked_reference_overlap' if gaps else 'pixel_exclusion_and_lossless_label_export_passed_candidate_only',members=rows,
        reference_images_checked=len(references),reference_fingerprint_rows=protected_count,reference_overlap_gaps=gaps,
        protected_labels_read=False,training_ready=False,training_started=False,
        limitations=['Shared layout/assets disclosed; exact-pixel exclusion is not independent-scene certification','Near similarity within same-pose variants is intentional and grouped','Historical source-role chain and complete 36-image scope still require final freeze'],
        inputs={str(p):prior.file_sha256(p) for p in paths})
    if dest.exists():prior.verify(prior.read(dest))
    else:prior.frozen(dest,record)
    print('EXPORTED',len(rows),'OVERLAP_GAPS',len(gaps),'REFERENCE_IMAGES',len(references),'FINGERPRINTS',protected_count)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--review-path',type=Path,default=MATERIAL/'reviewed-completion.json');ap.add_argument('--output',type=Path,default=MATERIAL/'candidate-export-v1');a=ap.parse_args();export(a.review_path,a.output)
