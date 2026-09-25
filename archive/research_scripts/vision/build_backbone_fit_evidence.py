"""Build current residual pages, without importing historical review decisions."""
from pathlib import Path
from scripts.vision import review_lr_material_fit as builder
from scripts.vision.diagnose_routed_backbone_fit import OUT, KEYS
from scripts.vision.routed_backbone_control import checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    ep=OUT/'residual-review-v1/evidence.json'
    if not ep.exists():
        builder.OUT=OUT
        builder.KEYS=KEYS
        builder.main()
    evidence=checked(ep)
    count=sum(len(f['events']) for f in evidence['frames'])
    expected=sum(len(r['misses']) for key in KEYS for r in checked(OUT/f'{key}.json')['rows'])
    if count!=expected:raise ValueError('Residual coverage mismatch')
    receipt=ep.parent/'build-receipt.json'
    if receipt.exists():return checked(receipt)
    paths=[ep,Path(__file__),Path(builder.__file__)]
    return write_record(receipt,dict(status='residual_evidence_ready_review_pending',unique_images=len(evidence['frames']),events=count,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run())
