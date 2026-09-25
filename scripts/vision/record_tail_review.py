"""Bind explicit current observations with the existing strict review validator."""
from scripts.vision.review_tail_interleaving import DEST,prior
from scripts.vision import record_bn_review as recorder
from scripts.vision.tail_review_notes import parse
from pathlib import Path

def main():
    recorder.DEST=DEST;recorder.parse=parse
    # The reused importer binds its own source; this additional receipt binds the
    # new observations and adapter without modifying any historical artifact.
    r=recorder.main()
    deps=[DEST/'review.json',Path(__file__).resolve(),Path(__file__).with_name('tail_review_notes.py')]
    prior.frozen(DEST/'review-binding.json',dict(status='explicit_current_review_bound',inputs={str(p):prior.file_sha256(p) for p in deps}))
    print(r['counts']);print('UNRESOLVED',r['unresolved'])
if __name__=='__main__':main()
