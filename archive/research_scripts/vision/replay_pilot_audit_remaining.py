"""Expand only after the first affected frame has exact certified replay."""
import asyncio
from scripts.vision.replay_pilot_audit import freeze, OUT, replay
from src.vision.canonical.plan import read_record
from src.ml.artifacts import file_sha256

def checked(path):
    r=read_record(path)
    for p,h in r['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Changed replay evidence')
    if not r['process_cleanup_complete']:raise ValueError('Incomplete cleanup')
    return r

async def run():
    p=freeze();replay.OUT=OUT.resolve()
    pilot=[checked(x) for x in (OUT/'replay/frame-00').glob('attempt-*/receipt.json')]
    if not any(r['status']=='original_pixel_evidence_certified' for r in pilot):raise ValueError('Pilot not certified')
    for f in p['frames'][1:]:
        result=None
        for n in range(1,4):
            folder=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}'
            if folder.exists():
                if not (folder/'receipt.json').exists():continue
                result=checked(folder/'receipt.json')
            else:result=await replay.attempt(f,n)
            if result['status']!='technical_failure':break
        if result is None or result['status']!='original_pixel_evidence_certified':
            print('EXPANSION_HELD',f['review_ids'],flush=True);return
    print('ALL_AFFECTED_FRAMES_REPLAYED_REVIEW_PENDING',flush=True)

if __name__=='__main__':asyncio.run(run())
