"""Capture the frozen source-isolated reactor coverage plan."""
import argparse, asyncio
from pathlib import Path
from src.vision.canonical.collect import collect
from src.vision.canonical.plan import read_record
from scripts.vision.prepare_reactor_source_isolated import BASE, run as prepare

async def main(mode):
    plan=prepare(); out=BASE/mode
    if out.exists():
        # Failed/blocked attempts are immutable.  Continue in a new attempt
        # directory rather than mixing partial frames with a retry.
        for n in range(2, 4):
            candidate=BASE/f'{mode}-attempt-{n:03}'
            if not candidate.exists():
                out=candidate;break
        else:
            print('ATTEMPT_CAP_EXHAUSTED',out);return 2
    review=BASE/'calibration-review.json' if mode=='pilot' else None
    receipt=await collect(BASE/'plan.json',out,mode=mode,review_path=review)
    print(receipt['status'],len(receipt.get('views',[])),flush=True)
    return 0 if receipt['status']=='complete_pending_review' else 2

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('calibration','pilot'),required=True);a=ap.parse_args()
    raise SystemExit(asyncio.run(main(a.mode)))
