"""Recheck explicit review, identify matching competition, benchmark evidence rendering."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import time
from concurrent.futures import ProcessPoolExecutor
from scripts.vision import build_closed_source_error_review as builder
from scripts.vision.record_closed_source_error_review import validate
from scripts.vision.evaluate_closed_source_control import OUT
from scripts.vision.analyze_recovery_paired_calibration import iou

DEST=builder.DEST;prior=builder.prior


def main():
    dest=DEST/'diagnostic-confirmation.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    ep,rp,sp=[DEST/n for n in ('evidence.json','review.json','review-summary.json')]
    e,r,s=[prior.read(p) for p in (ep,rp,sp)]
    for record in (e,r,s):prior.verify(record)
    validate(e,r['decisions']);competitions=[];deps=[ep,rp,sp,Path(__file__).resolve()]
    for t in s['loss_details']:
        if not t['miss']['formal_matching_competition']:continue
        path=OUT/'evaluation'/f"M-{t['seed']}.json";evaluation=prior.read(path);prior.verify(evaluation);deps.append(path)
        row=next(x for x in evaluation['rows'] if x['pair_id']==t['pair_id'] and x['variant']==t['variant'])
        evidence=[]
        for j,p in enumerate(row['predictions']):
            overlap=iou(p['bbox_xyxy'],t['truth']['bbox_xyxy'])
            if p['class_name']!=t['truth']['class_name'] or overlap<.5:continue
            assignments=[m for m in row['matches'] if m['prediction_index']==j]
            if len(assignments)!=1:raise ValueError('Unexplained competition')
            other=row['truth'][assignments[0]['truth_index']]
            if other==t['truth']:raise ValueError('Lost target actually matched')
            evidence.append(dict(prediction=p,iou_to_missed=overlap,assigned_match=assignments[0],assigned_truth=other))
        if not evidence:raise ValueError('Missing competition evidence')
        competitions.append(dict(decision_id=t['review_decision'],seed=t['seed'],variant=t['variant'],truth=t['truth'],
            original_operational_reason=t['miss']['reason'],interpretation='Qualifying formal prediction assigned to different overlapping truth; not simple confidence absence.',evidence=evidence))
    with TemporaryDirectory(prefix='closed-review-render-') as tmp:
        start=time.monotonic()
        with patch.object(builder,'DEST',Path(tmp)):
            rendered=[builder.render(dict(x)) for x in e['events']]
        serial=time.monotonic()-start
        # JSON freezing sorts dictionary keys. FP captions use str(dict), so
        # compare serial/parallel on the SAME reloaded input, not pre-JSON cards.
        original_differences=[b['event_id'] for a,b in zip(rendered,e['events'],strict=True) if a['page_sha256']!=b['page_sha256']]
        with patch.object(builder,'DEST',Path(tmp)):
            start=time.monotonic()
            with ProcessPoolExecutor(max_workers=2) as pool:
                parallel=list(pool.map(builder.render,[dict(x) for x in e['events']]))
            parallel_seconds=time.monotonic()-start
        if any(a['page_sha256']!=b['page_sha256'] for a,b in zip(rendered,parallel,strict=True)):raise ValueError('Parallel render changes evidence')
    noncompetition=[x for x in s['loss_details'] if not x['miss']['formal_matching_competition']]
    return prior.frozen(dest,dict(status='review_complete_with_named_content_limits_no_candidate',
        review_decisions=len(r['decisions']),pending_content_ids=r['pending_ids'],matching_competitions=competitions,
        noncompetition_loss_reasons=dict(Counter(x['miss']['reason'] for x in noncompetition)),
        clear_body_loss_reasons=dict(Counter(x['miss']['reason'] for x in s['loss_details'] if x['review_content']=='body_identifiable')),
        render_benchmark=dict(serial_seconds=serial,two_process_seconds=parallel_seconds,original_generation_seconds=e['render_seconds'],exact_image_bytes=True,
            pre_json_caption_order_differences=original_differences,
            preferred_for_this_batch='serial' if serial<parallel_seconds else 'two_process',
            limitation='One comparison, includes two-process startup, OS cache/order effects possible; no claim of universal speedup.'),
        full_metrics_unchanged=True,selected_candidate=None,training_admitted=False,promotable=False,
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':
    r=main();print('REVIEW',r['review_decisions'],'PENDING',len(r['pending_content_ids']),'COMPETITION',len(r['matching_competitions']),r['render_benchmark'])
