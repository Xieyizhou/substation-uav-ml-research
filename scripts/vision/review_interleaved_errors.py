"""Independent review evidence for interleaved endpoints, no old review reuse."""
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,prior
from scripts.vision.train_interleaved_small_scale import complete

DEST=OUT/'evaluation/error-review-v1'


def evidence():
    from scripts.vision import build_small_scale_fp_review as builder
    builder.OUT=OUT; builder.KEYS=KEYS; builder.DEST=DEST; builder.complete=complete
    r=builder.build()
    path=DEST/'adapter-binding.json'
    if path.exists(): prior.verify(prior.read(path))
    else:
        deps=[DEST/'evidence.json',Path(__file__).resolve()]
        prior.frozen(path,dict(status='interleaved_evidence_bound',inputs={str(d):prior.file_sha256(d) for d in deps}))
    return r


if __name__=='__main__':
    r=evidence(); print('FALSE_POSITIVE_EVIDENCE',r['predictions'],r['unique_images'],len(r['pages']))
