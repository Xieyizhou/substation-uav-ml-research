#!/usr/bin/env python3
"""Bounded development-only proposal attribution; never admit training data."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256
from src.vision.training.hard_example_curator import load_collection
from src.vision.evaluation.detection_metrics import box_iou

def main():
    from ultralytics import YOLO
    output = ROOT / 'data/research/run19-product-scope-v1/development-proposal-audit-v1'
    output.mkdir(exist_ok=False)
    source = ROOT / 'data/research/visual_hard_examples_v2_12/curated-run18-run19-full-strict-v1/curated-manifest.json'
    pool = json.loads(source.read_text())
    groups = defaultdict(lambda: defaultdict(list))
    for row in pool['selected']:
        if row['split'] == 'development' and row['map_id'] in ('simple', 'medium', 'complex'):
            groups[(row['map_id'], bool(row['objects']))][row['collection']].append(row)
    selected = []
    for key, recordings in sorted(groups.items()):
        queues = [sorted(v, key=lambda r: r['image_sha256']) for _, v in sorted(recordings.items())]
        count = 0
        while queues and count < 20:
            for queue in queues:
                if count == 20:
                    break
                selected.append(queue.pop(0)); count += 1
            queues = [q for q in queues if q]
    model_path = ROOT / 'models/equipment/visual-yolo11n-proposal-v2.12-candidate-run18/weights/best.pt'
    model_hash = file_sha256(model_path)
    model = YOLO(str(model_path))
    classes = ['transformer', 'switchgear', 'capacitor_bank', 'reactor']
    cache = {}; frames = []; review = []; skipped = []
    totals = Counter(); map_counts = defaultdict(Counter)
    for index, member in enumerate(selected):
        collection = member['collection']
        if collection not in cache:
            run_path = Path(collection).parent / 'run-receipt.json'
            run = json.loads(run_path.read_text())
            if run.get('status') != 'complete' or run.get('landing_confirmed') is not True:
                cache[collection] = {}
            else:
                candidates, _, _ = load_collection(collection, classes)
                cache[collection] = {c.frame_id: c for c in candidates}
        candidate = cache[collection].get(member['frame_id'])
        if candidate is None:
            skipped.append({'frame_id':member['frame_id'],'reason':'ineligible_run_or_missing_synchronized_truth'}); continue
        if candidate.split != 'development' or candidate.collection_identity != member['collection_identity']:
            raise ValueError('Source identity or split mismatch')
        image_path = Path(collection) / candidate.rgb_path
        if file_sha256(image_path) != candidate.image_sha256:
            raise ValueError('Image identity mismatch')
        truths = [o['bbox_xyxy'] for o in candidate.objects]
        with Image.open(image_path) as im:
            result = model.predict(source=im.convert('RGB'), imgsz=640, conf=.01, iou=.7,
                                   agnostic_nms=True, max_det=300, batch=1,
                                   rect=False, device='cpu', verbose=False)[0]
        predictions = [{'bbox': b, 'confidence': float(c)} for b,c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        predictions.sort(key=lambda p: (-p['confidence'], p['bbox']))
        used = set(); counts = Counter()
        for rank, prediction in enumerate(predictions):
            box = prediction['bbox']; overlaps = [box_iou(box,t) for t in truths]
            best = max(overlaps, default=0.0)
            target = overlaps.index(best) if overlaps else None
            area = max(0,box[2]-box[0])*max(0,box[3]-box[1])
            containment = max((max(0,min(box[2],t[2])-max(box[0],t[0])) * max(0,min(box[3],t[3])-max(box[1],t[1])) / max(area,1e-9) for t in truths), default=0)
            if best >= .5:
                category = 'duplicate_target_box' if target in used else 'matched_target'
                used.add(target)
            elif containment >= .8:
                category = 'possible_equipment_part'
            elif best >= .1:
                category = 'localization_or_partial_overlap'
            else:
                category = 'unmatched_background_candidate_requires_review'
            prediction.update({'rank':rank+1,'within_runtime_cap':rank<16,'best_truth_iou':best,'proposal_containment':containment,'attribution':category})
            if rank < 16:
                counts[category] += 1
                if category != 'matched_target':
                    review.append({'frame_id':candidate.frame_id,'image_path':str(image_path),'map_id':candidate.map_id,'truth':candidate.objects,'prediction':prediction,'review_status':'pending'})
        record = {'member':candidate.record(),'truth_count_before_bbox_filter':len(truths),'nms_proposal_count':len(predictions),'runtime_cap_counts':dict(counts),'proposals':predictions}
        frames.append(record); totals.update(counts); map_counts[candidate.map_id].update(counts)
        print(f'{index+1}/{len(selected)} {candidate.map_id}: {len(predictions)} proposals',flush=True)
    review.sort(key=lambda r:(-r['prediction']['confidence'],r['frame_id'],r['prediction']['rank']))
    report = {'schema_version':1,'status':'complete_pending_visual_review','evidence_level':'sampled_development_diagnostic_not_acceptance','source_identity':pool['identity'],'model_sha256':model_hash,'script_sha256':file_sha256(Path(__file__)),'configuration':{'backend':'pytorch_cpu','imgsz':640,'batch':1,'confidence':.01,'agnostic_nms_iou':.7,'diagnostic_max_det':300,'runtime_cap':16},'frame_count':len(frames),'selected_count':len(selected),'skipped':skipped,'runtime_cap_counts':dict(totals),'per_map_counts':{k:dict(v) for k,v in map_counts.items()},'frames':frames,'review_queue':review,'training_admission':False,'limitations':['Development samples are not independent validation.','Unmatched proposals are not confirmed false positives.','Geometric containment is not a semantic equipment-part judgment.','Up to 300 post-NMS proposals are observed, not raw network boxes.','No comparison to official v2.11 has been performed.']}
    report['identity']=object_sha256(report)
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(output),'frames':len(frames),'counts':dict(totals),'identity':report['identity']}),flush=True)
if __name__ == '__main__':
    main()
