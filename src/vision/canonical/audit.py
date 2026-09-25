"""Conservative review/duplicate audit. Candidate counts are never training release."""
from collections import Counter
from pathlib import Path
import json
from src.ml.artifacts import file_sha256
from src.vision.training.hard_example_curator import hamming_hex
from .plan import read_record, write_record

CLASSES=('transformer','switchgear','capacitor_bank','reactor')

def duplicate(a,b):
    return a['image_sha256']==b['image_sha256'] or hamming_hex(a['perceptual_hash'],b['perceptual_hash'])<=2

def select_new(rows,development,protected):
    retained=[];rejected=[]
    for row in sorted(rows,key=lambda r:(r['image_sha256'],r['view_id'])):
        if any(duplicate(row,p) for p in protected):reason='protected_partition_duplicate'
        elif any(duplicate(row,p) for p in development):reason='historical_development_duplicate'
        elif any(duplicate(row,p) for p in retained):reason='within_batch_duplicate'
        else:retained.append(row);continue
        rejected.append({'view_id':row['view_id'],'reason':reason})
    return retained,rejected

def audit(receipt_path, output, review_path=None, development=(), protected=()):
    receipt=read_record(receipt_path); rows=[r for r in receipt['views'] if r['status']=='captured']
    for row in rows:
        if row['split']!='development':raise ValueError('Non-development view')
        if file_sha256(row['rgb_path'])!=row['image_sha256'] or file_sha256(row['depth_path'])!=row['depth_sha256']:raise ValueError('Captured image/depth mismatch')
    review={};review_identity=None
    if review_path:
        record=read_record(review_path)
        if record['collection_identity']!=receipt['identity'] or record['plan_identity']!=receipt['plan_identity']:raise ValueError('Review belongs to another capture')
        review_identity=record['identity']
        for item in record['views']:
            if item['view_id'] in review:raise ValueError('Duplicate review decision')
            review[item['view_id']]=item
    references={};pools=[]
    for label,paths in [('development',development),('protected',protected)]:
        pool=[]
        for p in paths:
            source=json.loads(Path(p).read_text()); references[str(Path(p).resolve())]=file_sha256(p)
            for row in source['selected']:
                if label=='development' and row.get('split')!='development':raise ValueError('Invalid development reference')
                if label=='protected' and row.get('split')=='development':raise ValueError('Invalid protected reference')
                if not row.get('perceptual_hash'):raise ValueError('Reference missing perceptual hash')
                pool.append(row)
        pools.append(pool)
    candidates=[];ambiguous=[];ignored=[]
    for row in rows:
        item=review.get(row['view_id'],{})
        if item.get('decision')=='ignored':ignored.append(row['view_id']);continue
        objects=row['truth']['objects']
        if item.get('decision')!='accepted' or item.get('image_sha256')!=row['image_sha256'] or item.get('all_visible_targets_correct') is not True or (not objects and item.get('no_target_confirmed') is not True):
            ambiguous.append(row['view_id']);continue
        candidates.append(row)
    new,rejections=select_new(candidates,*pools)
    counts=Counter()
    for r in new:
        names={o['class_name'] for o in r['truth']['objects']}
        if not names:counts['no_target']+=1
        else:counts.update(names)
    rate=len(new)/max(1,len(rows))
    blockers=[]
    if receipt['status']!='complete_pending_review':blockers.append('collection_incomplete')
    if ambiguous:blockers.append('unreviewed_or_ambiguous_frames')
    if not development or not protected:blockers.append('complete_reference_pools_required')
    if len(new)<20:blockers.append('fewer_than_20_new_views')
    if rate<.2:blockers.append('incremental_rate_below_20_percent')
    return write_record(output,{'schema_version':1,'status':'blocked' if blockers else 'pilot_audit_passed','collection_identity':receipt['identity'],'review_identity':review_identity,'references':references,'visited_views':receipt['view_count'],'captured_frames':len(rows),'review_accepted_frames':len(candidates),'new_frames':len(new),'incremental_rate':rate,'new_candidate_coverage':dict(counts),'ambiguous':ambiguous,'ignored':ignored,'duplicate_rejections':rejections,'selected':new,'blocked_reasons':blockers,'training_admitted':False,'training_blocker':'Full three-map calibration and unified 600-per-category curation required; this pilot alone cannot release training.'})
