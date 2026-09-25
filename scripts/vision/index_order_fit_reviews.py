"""Locate review evidence by exact identity, without converting hits to approval."""
import json
import subprocess
from collections import defaultdict,Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior


def nodes(value,path='$'):
    if isinstance(value,dict):
        yield path,value
        for k,v in value.items():yield from nodes(v,path+'.'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from nodes(v,f'{path}[{i}]')


def run():
    p=freeze();root=prior.ROOT/'data/research/ml_training_recovery_v1'
    names=subprocess.check_output(['rg','--files',str(root)],text=True).splitlines()
    paths=[Path(n) for n in names if Path(n).name in {'quality-review.json','review.json','visual-review.json','label-review.json','review-ledger.json','reviewed-completion.json'} and not str(OUT) in n]
    images=defaultdict(list);ids=set();hits=defaultdict(list);files=[];deps=[OUT/'protocol.json',Path(__file__).resolve()]
    for m in p['members']:images[m['image_sha256']].append(m['member_id']);ids.add(m['member_id'])
    for path in sorted(paths):
        r=prior.read(path);matches=[]
        for loc,n in nodes(r):
            mids=set(images.get(n.get('image_sha256'),[]))
            if isinstance(n.get('member_id'),str) and n['member_id'] in ids:mids.add(n['member_id'])
            if mids:matches.append((loc,n,mids))
        if not matches:continue
        try:prior.verify(r);valid=True;error=None
        except Exception as exc:valid=False;error=str(exc)
        files.append(dict(path=str(path),valid=valid,error=error,status=r.get('status')));deps.append(path)
        for loc,n,mids in matches:
            for mid in mids:
                hits[mid].append(dict(record=str(path),json_location=loc,record_valid=valid,
                    status=n.get('status'),record_status=r.get('status'),
                    binds_exact_image=n.get('image_sha256')==next(m['image_sha256'] for m in p['members'] if m['member_id']==mid),
                    decision_fields={k:n[k] for k in ('event_id','object_id','annotation_id','reason','review_nature','label_sha256','evidence_identity') if k in n}))
    result=dict(status='review_identity_index_not_quality_approval',records=files,
        members=[dict(member_id=m['member_id'],potential_review_links=hits[m['member_id']],
            status='requires_semantic_coverage_validation' if hits[m['member_id']] else 'requires_provenance_trace') for m in p['members']],
        counts=dict(Counter(bool(hits[m['member_id']]) for m in p['members'])),
        inputs={str(x):prior.file_sha256(x) for x in deps})
    dest=OUT/'review-index.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,result)


if __name__=='__main__':
    r=run();print(r['status'],r['counts'],'records',len(r['records']))
