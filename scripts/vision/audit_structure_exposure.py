"""Read-only source/label/exposure census. ROI themes are not exhaustive contents."""
from collections import Counter
from pathlib import Path
import hashlib
from PIL import Image
from scripts.vision.exposure_order_retention import OUT as SOURCE, ROOT, KEYS, read, frozen, file_sha256, verify_tree
from scripts.vision.hard_negative_coverage import OUT as COVERAGE
from scripts.vision.run_exposure_diagnosis import checked_cell

OUT=SOURCE/'structure-supervision-coverage-v1'

def pixels(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        return (im.size,hashlib.sha256(im.tobytes()).hexdigest())

def label_counts(path,names):
    result=Counter()
    for line in Path(path).read_text().splitlines():
        values=list(map(float,line.split()))
        if len(values)!=5 or not values[0].is_integer() or not 0<=values[0]<len(names):raise ValueError('Invalid label')
        if not all(0<=v<=1 for v in values[1:]) or min(values[3:])<=0:raise ValueError('Invalid coordinates')
        result[names[int(values[0])]]+=1
    return dict(result)

def main():
    dest=OUT/'census.json'
    if dest.exists():verify_tree(dest);return
    OUT.mkdir(exist_ok=True)
    pp=SOURCE/'protocol.json';rp=SOURCE/'post-training-review-v1/review.json';ap=COVERAGE/'final-admission.json'
    verify_tree(rp);verify_tree(pp)
    p=read(pp);a=read(ap)
    from scripts.vision.exposure_protocol import verify
    verify(a)
    decisions={d['view_id']:d for d in a['decisions']}
    sources={pixels(f['image_path']):(f,decisions[f['view_id']]) for f in a['frames']}
    if len(sources)!=len(a['frames']):raise ValueError('Ambiguous source')
    inputs={str(x):file_sha256(x) for x in (pp,rp,ap,Path(__file__))}
    from scripts.vision.prepare_visual_bridge_negative_v2 import BASE as BRIDGE
    bridge={}
    for stage in ('pilot-v1','remaining-v1'):
        path=BRIDGE/stage/'review-v1/semantic-review.json';doc=read(path);verify(doc);inputs[str(path)]=file_sha256(path)
        for f in doc['frames']:bridge[pixels(f['image_path'])]=f
    counts={};rows=[]
    for key in KEYS:
        cp=SOURCE/'training'/key/'completion.json';checked_cell(cp,p)
        inputs[str(cp)]=file_sha256(cp);counts[key]=Counter(p['schedules'][key])
    for r in p['pool_rows']:
        for kind in ('image','label'):
            path=r[kind+'_path']
            if file_sha256(path)!=r[kind+'_sha256']:raise ValueError('Stale member')
            inputs[path]=r[kind+'_sha256']
        classes=label_counts(r['label_path'],p['names'])
        if classes!=r['class_instances']:raise ValueError('Full label census differs')
        item=dict(member_id=r['member_id'],subset=r['subset'],lineage_id=r['lineage_id'],
            lineage_resolution=r.get('lineage_resolution','unspecified_no_independence_claim'),image_path=r['image_path'],image_sha256=r['image_sha256'],
            class_instances=classes,exposures={k:c[r['member_id']] for k,c in counts.items()})
        if r['subset']=='hard_negative':
            if classes:raise ValueError('Nonempty negative')
            match=sources.get(pixels(r['image_path']))
            if match:
                f,d=match
                if d['decision']!='accepted' or d['image_sha256']!=f['image_sha256'] or file_sha256(f['image_path'])!=f['image_sha256']:raise ValueError('Invalid prior decision')
                inputs[f['image_path']]=f['image_sha256']
                item.update(source_status='existing_AI_review_linked_by_exact_RGB_pixels',source_view_id=f['view_id'],
                    roi_contents=sorted({v['content'] for v in d['rois']}),source_reason=d['reason'],
                    correlation_group=f['correlation_group_id'],rois=d['rois'])
            elif pixels(r['image_path']) in bridge:
                f=bridge[pixels(r['image_path'])]
                if f['decision']!='accepted' or f['pair_id']!=r['pair_id'] or file_sha256(f['image_path'])!=f['image_sha256']:raise ValueError('Invalid bridge source')
                inputs[f['image_path']]=f['image_sha256']
                item.update(source_status='bridge_review_exact_RGB_and_pair_linked',source_view_id=f['view_id'],roi_contents=['subject:'+f['subject_family']],source_reason=f['reason'])
            else:item.update(source_status='unresolved',roi_contents=['unknown'])
        if classes.get('reactor'):
            with Image.open(r['image_path']) as im:w,h=im.size
            sizes=[]
            for idx,line in enumerate(Path(r['label_path']).read_text().splitlines()):
                cls,x,y,bw,bh=map(float,line.split())
                if p['names'][int(cls)]=='reactor':sizes.append(dict(label_line_index=idx,width_640=bw*w*640/max(w,h),height_640=bh*h*640/max(w,h)))
            item.update(reactor_boxes=sizes,occlusion='unknown',identifiable_content='not_reaudited',pixel_visibility_certified=False)
        rows.append(item)
    summary={}
    for key,c in counts.items():
        neg=Counter();supervision={s:Counter() for s in ('base','regular','bridge_positive','hard_negative')}
        for r in rows:
            n=c[r['member_id']]
            for cls,v in r['class_instances'].items():supervision[r['subset']][cls]+=n*v
            if r['subset']=='hard_negative':
                for content in r['roi_contents']:neg[content]+=n
        summary[key]=dict(negative_reviewed_ROI_exposures=dict(neg),class_instance_exposure_by_subset={s:dict(v) for s,v in supervision.items()})
    frozen(dest,dict(status='census_complete_fine_structure_and_bridge_trace_pending',rows=rows,summary=summary,
        limits=['ROI content counts are existing review subjects, not exhaustive per-frame objects.',
            'Labels certify supervision counts, not visibility or independent geometry.',
            'No new training, data admission, collection, threshold change, or sealed evaluation.'],inputs=inputs))
    print('CENSUS',len(rows),'NEG_SOURCE',Counter(r.get('source_status') for r in rows if r['subset']=='hard_negative'))
    for key in ('staged-450-7','staged-450-17','staged-450-27'):print(key,summary[key])

if __name__=='__main__':main()
