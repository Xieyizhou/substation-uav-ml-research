"""Import only explicit observations and keep empty/unknown content unresolved."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.review_cool_light_errors import DEST,prior
from scripts.vision.cool_light_error_notes import parse
from src.ml.artifacts import object_sha256

def validate(e,ds):
    want={(x['event_id'],i):(x,i) for x in e['events'] for i in range(len(x['events']) if x['kind']=='FP' else 1)}
    if len(ds)!=len(want) or {(d['event_id'],d['index']) for d in ds}!=set(want):raise ValueError('Review missing/duplicate')
    for d in ds:
        x,i=want[d['event_id'],d['index']]
        if d['event_sha256']!=object_sha256(x) or d['page_sha256']!=prior.file_sha256(x['page_path']) or d['image_sha256']!=prior.file_sha256(x['source']['image_path']):raise ValueError('Stale review')
        if not d['reason'] or d['review_nature']!='AI-assisted' or not d['reviewed_at']:raise ValueError('Invalid decision')
        if d['training_admitted'] or d['promotable'] or d['pixel_visibility_certified']:raise ValueError('Unsupported certification')
        if d['status']!=('unresolved_content' if d['content'] in ('unknown','empty_roi') else 'observed_not_admitted'):raise ValueError('Unknown approved')

def main():
    ep=DEST/'evidence.json';e=prior.read(ep);prior.verify(e);notes=parse();ds=[]
    if set(notes)!={x['event_id'] for x in e['events']}:raise ValueError('Missing explicit notes')
    for x in e['events']:
        if len(notes[x['event_id']])!=(len(x['events']) if x['kind']=='FP' else 1):raise ValueError('Per-box coverage')
        for i,(content,reason) in enumerate(notes[x['event_id']]):
            if (i in x['empty_crop_indices'])!=(content=='empty_roi'):raise ValueError('Empty crop misrepresented')
            ds.append(dict(event_id=x['event_id'],index=i,kind=x['kind'],content=content,reason=reason,
                image_sha256=x['source']['image_sha256'],page_sha256=x['page_sha256'],event_sha256=object_sha256(x),
                review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
                status='unresolved_content' if content in ('unknown','empty_roi') else 'observed_not_admitted',
                pixel_visibility_certified=False,training_admitted=False,promotable=False))
    validate(e,ds)
    deps=[ep,Path(__file__).resolve(),Path(__file__).with_name('cool_light_error_notes.py')]
    return prior.frozen(DEST/'review.json',dict(status='explicit_review_complete_with_named_content_gaps',decisions=ds,
        counts={kind:dict(Counter(d['content'] for d in ds if d['kind']==kind)) for kind in ('FP','LOSS')},
        unresolved=[f"{d['event_id']}:{d['index']}" for d in ds if d['status']=='unresolved_content'],
        full_metrics_unchanged=True,inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':print(main()['counts'])

