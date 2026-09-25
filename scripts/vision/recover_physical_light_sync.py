"""Reanalyze preserved raw capture using the inherited RGB-reference protocol."""
from pathlib import Path
import shutil
from scripts.vision import run_physical_lighting_capture as old
from scripts.vision import run_physical_lighting_capture_v2 as new

def main():
    p=new.freeze();src=old.OUT/'replay/L01-original/attempt-01';rp=src/'receipt.json';r=new.prior.read(rp);new.prior.verify(r)
    if r['reason']!='Clock/synchronization gate failed' or not r['process_cleanup_complete']:raise ValueError('Different failure')
    dest=new.OUT/'replay/L01-original/attempt-01';dest.mkdir(parents=True,exist_ok=False)
    for x in src.iterdir():
        if x.is_file() and x.name!='receipt.json':shutil.copy2(x,dest/x.name)
    f=next(f for f in p['frames'] if f['review_ids']==['L01-original']);clock=new.prior.read(src/'clock-preflight.json');new.prior.verify(clock)
    fence=new.base.message_timestamp(clock['samples'][-1]['message'])
    result=new.analyze(dest,f,fence)
    result.update(process_cleanup_complete=True,replayed=False,source_failed_receipt=str(rp),
        metric_correction='Restore inherited per-stream absolute time difference to RGB, while retaining all-stream span as diagnostic; tolerance remains 33.334ms.',
        inputs={str(x):new.prior.file_sha256(x) for x in [rp,old.OUT/'protocol.json',new.OUT/'protocol.json',Path(__file__).resolve()]+[x for x in dest.iterdir() if x.is_file()]})
    new.prior.frozen(dest/'receipt.json',result);print(result['status'])

if __name__=='__main__':main()
