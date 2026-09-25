"""Summarize technical replay evidence, without semantic approval."""
from scripts.vision.replay_pilot_audit import OUT, freeze
from scripts.vision.replay_pilot_audit_remaining import checked
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256
from pathlib import Path

def run():
    p=freeze();rows=[];deps={}
    for f in p['frames']:
        paths=sorted((OUT/'replay'/f['review_ids'][0]).glob('attempt-*/receipt.json'))
        valid=[]
        for path in paths:
            r=checked(path);deps[str(path.resolve())]=file_sha256(path)
            if r['status']=='original_pixel_evidence_certified':valid.append((path,r))
        if len(valid)!=1:raise ValueError('Missing or ambiguous certified replay')
        path,r=valid[0]
        rows.append(dict(frame=f['review_ids'][0],receipt=str(path.resolve()),targets=r['records'][0]['targets'],review_status='mask_overlay_review_pending'))
    for path in (OUT/'protocol.json',Path(__file__)):
        deps[str(path.resolve())]=file_sha256(path)
    return write_record(OUT/'technical-summary.json',dict(status='exact_replay_complete_semantic_review_pending',frames=rows,inputs=deps,training_admitted=False,promotable=False))

if __name__=='__main__':print(run()['status'])
