"""Reuse only identity-valid, explicit full-label evidence; never synthesize approvals."""
from pathlib import Path
from collections import Counter
from scripts.vision import train_material_retention_coverage as reference
from scripts.vision import import_transfer_pilot_review as pilot, import_transfer_expansion_review as expansion
from scripts.vision.audit_material_transfer_scope import CAND,resolve_truth
from scripts.vision.structure_fit import truth_for

prior=reference.prior
OUT=reference.OUT.parent/'same-source-material-dose-control-v1'

def main():
    p,_,_=reference.contract('T-7');cp=CAND/'reviewed-completion.json';c=prior.read(cp);prior.verify(c)
    originals={m['member_id']:m for m in c['members']};paths=[cp,reference.OUT/'protocol.json',Path(__file__).resolve()]
    legacy={};gray={}
    for name in sorted({m[k] for m in c['members'] for k in ('source_review','variant_review') if k in m}):
        f=Path(name);r=prior.read(f);prior.verify(r);legacy[name]=r;paths.append(f)
    for mod in (pilot,expansion):
        ep=mod.DEST/'evidence.json';rp=mod.DEST/'label-review.json';e,r=prior.read(ep),prior.read(rp)
        prior.verify(e);prior.verify(r);mod.validate(e,r['decisions']);paths.extend((ep,rp))
        ds={d['event_id']:d for d in r['decisions']}
        for x in e['events']:
            if x['condition']=='gray_target_body':gray.setdefault(x['source_id'],[]).append((x,ds[x['event_id']],rp))
    selected=[r for r in p['pool_rows'] if 'full_truth' in r and r.get('variant') in ('original','warm','cool','gray_target_body') and r['member_id'].split('-')[0] in {f'G{i:02}' for i in range(1,5)}|{f'S{i:02}' for i in range(1,9)}]
    if len(selected)!=48:raise ValueError('Expected 48 members')
    matrix=[];gaps=[]
    for row in selected:
        mid=row['member_id'];sid=mid.split('-')[0];variant=row['variant']
        for kind in ('image','label'):
            path=Path(row[kind+'_path'])
            if prior.file_sha256(path)!=row[kind+'_sha256']:raise ValueError('Member hash drift')
            paths.append(path)
        if Counter(t['class_name'] for t in truth_for(row))!=Counter(row['class_instances']):raise ValueError('Full label drift')
        src=originals[sid+'-original'];resolved=resolve_truth(src['full_truth'],src['instance_mapping'])
        decisions=[]
        for truth,obj in zip(src['full_truth']['objects'],resolved,strict=True):
            if variant=='gray_target_body':
                candidates=[(d,str(rp)) for x,d,rp in gray[sid] if x['object_id']==obj['object_id'] and x['truth']==truth]
            else:
                review=src['source_review'] if variant=='original' else originals[mid]['variant_review']
                candidates=[(d,review) for d in legacy[review]['decisions'] if d.get('image_sha256')==row['image_sha256'] and d.get('truth')==truth and d.get('object_id')==obj['object_id']]
            if len(candidates)!=1:
                gaps.append(dict(member_id=mid,object_id=obj['object_id'],reason='missing_or_ambiguous_exact_review',matches=len(candidates)));continue
            d,rp=candidates[0]
            status=d.get('status',d.get('decision'))
            if status not in ('source_content_review_passed','variant_content_review_passed','diagnostic_content_reviewed'):
                gaps.append(dict(member_id=mid,object_id=obj['object_id'],reason='review_pending_or_rejected',status=status))
            for k in ('crop','image'):
                if k+'_path' in d and prior.file_sha256(d[k+'_path'])!=d[k+'_sha256']:raise ValueError('Stale review pixels')
            decisions.append(dict(object_id=obj['object_id'],review_path=rp,decision=d))
        matrix.append(dict(member_id=mid,pair_id=row['pair_id'],variant=variant,image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],label_count=len(resolved),reviews=decisions,
            disposition='pending' if any(g['member_id']==mid for g in gaps) else 'existing_explicit_review_bound_not_new_admission'))
    OUT.mkdir(exist_ok=True);dest=OUT/'quality.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='quality_binding_blocked' if gaps else 'review_bindings_complete_further_semantic_gate_required',members=matrix,gaps=gaps,
        training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=main();print(r['status'],len(r['members']),len(r['gaps']));print(r['gaps'][:5])
