from pathlib import Path
from scripts.vision.review_interleaved_light import DEST,prior
from scripts.vision.interleaved_light_review_notes import parse
from scripts.vision import record_bn_review as recorder
def main():
    recorder.DEST=DEST;recorder.parse=parse;r=recorder.main()
    deps=[DEST/'review.json',DEST/'evidence.json',Path(__file__).resolve(),Path(__file__).with_name('interleaved_light_review_notes.py')]
    prior.frozen(DEST/'completion.json',dict(status='review_complete_with_named_development_gaps',unresolved=r['unresolved'],selected_candidate=None,inputs={str(d):prior.file_sha256(d) for d in deps}))
    print(r['counts'],r['unresolved'])
if __name__=='__main__':main()
